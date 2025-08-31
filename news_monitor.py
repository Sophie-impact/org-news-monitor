#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import html
import time
import logging
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
from zoneinfo import ZoneInfo
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import tldextract

# --- LLM (OpenAI) ---
try:
    import openai  # pip install openai>=1.40.0
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

KST = ZoneInfo("Asia/Seoul")

# ---------- 공통 유틸 ----------
def now_kst():
    return datetime.now(tz=KST)

def parse_datetime(dt_str):
    try:
        dt = dtparser.parse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def to_kst_str(dt):
    if dt is None:
        return ""
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M")

def strip_html(text):
    text = html.unescape(text or "")
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)

def domain_from_url(url):
    try:
        ext = tldextract.extract(url)
        parts = [p for p in [ext.domain, ext.suffix] if p]
        return ".".join(parts) if parts else ""
    except Exception:
        return ""

def norm_title(t):
    t = strip_html(t or "").lower()
    t = re.sub(r"[\[\]【】()（）〈〉<>『』「」]", " ", t)
    t = re.sub(r"[^\w가-힣\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ---------- 조회 구간 계산 ----------
def compute_window_utc(now=None):
    """
    매일 09:00 KST 실행 기준 조회 구간:
    - 화~금: 전날 09:00 ~ 오늘 09:00
    - 월: 금요일 09:00 ~ 월요일 09:00
    """
    now = now or datetime.now(tz=KST)
    anchor_kst = now.astimezone(KST).replace(hour=9, minute=0, second=0, microsecond=0)

    days = 3 if anchor_kst.weekday() == 0 else 1
    start_kst = anchor_kst - timedelta(days=days)
    end_kst = anchor_kst

    return start_kst.astimezone(timezone.utc), end_kst.astimezone(timezone.utc)

# ---------- 시트 읽기 ----------
def fetch_org_list():
    sheet_url = os.environ.get("SHEET_CSV_URL", "").strip()
    if not sheet_url:
        raise RuntimeError("SHEET_CSV_URL env var is not set.")
    resp = requests.get(sheet_url, timeout=30)
    resp.raise_for_status()

    csv_text = resp.content.decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(csv_text))

    if "조직명" not in df.columns:
        raise RuntimeError("CSV에 '조직명' 열이 필요합니다.")

    orgs = []
    for _, row in df.iterrows():
        org = str(row["조직명"]).strip()
        if not org or org.lower() == "nan":
            continue
        query = str(row["검색어"]).strip() if "검색어" in df.columns and str(row["검색어"]).strip() else org
        item = {"org": org, "query": query}
        if item not in orgs:
            orgs.append(item)
    return orgs

# ---------- 네이버 뉴스 검색 ----------
def search_naver(query, display=5):
    cid = os.environ.get("NAVER_CLIENT_ID", "")
    csec = os.environ.get("NAVER_CLIENT_SECRET", "")
    if not cid or not csec:
        return []
    endpoint = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec}
    params = {"query": f"{query}", "display": display, "start": 1, "sort": "date"}
    try:
        r = requests.get(endpoint, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        items = r.json().get("items", [])
        results = []
        for it in items:
            title = strip_html(it.get("title"))
            url = it.get("originallink") or it.get("link")
            pub = parse_datetime(it.get("pubDate"))
            if not url or not title:
                continue
            src = domain_from_url(url) or "naver"
            results.append({
                "title": title,
                "url": url,
                "source": src,
                "published_at": pub,
                "origin": "naver",
                "summary": strip_html(it.get("description", "")),
            })
        return results
    except:
        return []

# ---------- NewsAPI ----------
def search_newsapi(query, window_from_utc, window_to_utc, language="ko"):
    key = os.environ.get("NEWSAPI_KEY", "")
    if not key:
        return []
    endpoint = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{query}",
        "from": window_from_utc.isoformat().replace("+00:00","Z"),
        "to": window_to_utc.isoformat().replace("+00:00","Z"),
        "sortBy": "publishedAt",
        "pageSize": 50,
        "language": language,
        "apiKey": key,
    }
    try:
        r = requests.get(endpoint, params=params, timeout=20)
        r.raise_for_status()
        arts = r.json().get("articles", [])
        results = []
        for a in arts:
            title = strip_html(a.get("title"))
            url = a.get("url")
            pub = parse_datetime(a.get("publishedAt"))
            src = (a.get("source") or {}).get("name") or domain_from_url(url)
            if not url or not title:
                continue
            results.append({
                "title": title,
                "url": url,
                "source": src,
                "published_at": pub,
                "origin": "newsapi",
                "summary": strip_html(a.get("description") or a.get("content") or ""),
            })
        return results
    except:
        return []

# ---------- 라벨 규칙 (폴백용) ----------
NEG_KW = ["횡령","배임","사기","고발","기소","구속","수사","압수수색","소송","고소","분쟁","리콜","결함","징계","제재","벌금","과징금","부실","파산","부도","중단","연기","오염","사망","부상","폭발","화재","추락","유출","해킹","랜섬웨어","침해","악성코드","담합","독점","불매","논란","갑질","표절","혐의","불법","위법","취소","철회","부정","적자","감소","급락","하락","경고","경보","리스크","소환","징역"]
WATCH_KW = ["의혹","점검","조사","심사","검토","논의","잠정","연구결과","유예","우려","관심","주시","잠정치","공정위","국감","지적","요구","연장","변동성","불확실성"]
POS_KW = ["투자유치","시리즈","라운드","유치","수상","선정","혁신","신기록","최대","상승","증가","호조","호재","확대","진출","오픈","출시","공개","협력","파트너십","MOU","계약","수주","달성","성과","흑자","흑자전환","개최"]

def rule_label(title, summary):
    text = f"{title} {summary}".lower()
    if any(k.lower() in text for k in NEG_KW): return "🔴"
    if any(k.lower() in text for k in WATCH_KW): return "🟡"
    if any(k.lower() in text for k in POS_KW): return "🔵"
    return "🟢"

# ---------- LLM 라벨러 ----------
def llm_enabled():
    return bool(os.environ.get("LLM_ENABLE", "").strip()) and bool(os.environ.get("OPENAI_API_KEY", "").strip()) and _HAS_OPENAI

def llm_label(org, title, summary):
    if not llm_enabled(): return None
    try:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        prompt = f"""다음 기사가 조직에 미치는 영향을 분류하세요.
조직: {org}
제목: {title}
요약: {summary}

라벨 중 하나만 출력:
- 🔵 (긍정)
- 🟢 (중립)
- 🟡 (팔로업 필요: 잠재 리스크 가능)
- 🔴 (부정/리스크)
반드시 기호만 출력하세요."""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            max_tokens=4,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out if out in {"🔵","🟢","🟡","🔴"} else None
    except:
        return None

# ---------- 오탐 필터 ----------
ALLOWED_BRAND_KWS = ["브라이언임팩트","사이드임팩트","brian impact","side impact"]
BLOCK_COMMON = {"피아니스트","피플","시공간","끝까지간다"}

def is_relevant(org, title, summary):
    text = f"{title} {summary}".lower()
    org_l = org.lower()
    if org in BLOCK_COMMON:
        return any(kw.lower() in text for kw in ALLOWED_BRAND_KWS)
    return org_l in text

# ---------- Slack ----------
def post_to_slack(lines):
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    channel = os.environ.get("SLACK_CHANNEL", "").strip()
    if not token or not channel:
        raise RuntimeError("SLACK_BOT_TOKEN or SLACK_CHANNEL missing.")
    client = WebClient(token=token)
    text = "\n".join(lines) if lines else "오늘은 신규로 감지된 기사가 없습니다."
    client.chat_postMessage(channel=channel, text=text)

# ---------- main ----------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ✅ 토/일이면 아무 것도 하지 않고 종료 (이중 안전장치)
    if now_kst().weekday() in (5, 6):  # 5=토, 6=일
        logging.info("Weekend (Sat/Sun) – skipping run.")
        return

    window_from_utc, window_to_utc = compute_window_utc()
    logging.info("Window UTC: %s ~ %s", window_from_utc, window_to_utc)

    max_per_org = int(os.environ.get("MAX_RESULTS_PER_ORG", "1"))

    org_rows = fetch_org_list()
    logging.info("Loaded %d organizations.", len(org_rows))

    all_lines = []
    for idx, row in enumerate(org_rows, start=1):
        org, query = row["org"], row["query"]
        logging.info("(%d/%d) Searching: %s | %s", idx, len(org_rows), org, query)

        naver_items = search_naver(query, display=max(10, max_per_org*4))
        time.sleep(0.25)
        newsapi_items = search_newsapi(query, window_from_utc, window_to_utc, language="ko")

        items = []
        for it in (naver_items + newsapi_items):
            it["org"] = org
            items.append(it)

        items = [it for it in items if is_relevant(it["org"], it["title"], it.get("summary",""))]
        items = [it for it in items if it["published_at"] and window_from_utc <= it["published_at"] < window_to_utc]

        items.sort(key=lambda x: x["published_at"], reverse=True)

        seen = set(); uniq = []
        for it in items:
            key = (norm_title(it["title"]), domain_from_url(it["url"]))
            if key not in seen and it["url"]:
                uniq.append(it); seen.add(key)

        take = uniq[:max_per_org]

        for art in take:
            label = llm_label(art["org"], art["title"], art.get("summary","")) or rule_label(art["title"], art.get("summary",""))
            src = art["source"]
            when_str = to_kst_str(art["published_at"])
            line = f"[{art['org']}] <{art['url']}|{art['title']}> ({src})({when_str}) [{label}]"
            all_lines.append(line)

    post_to_slack(all_lines)
    logging.info("Posted %d lines to Slack.", len(all_lines))

if __name__ == "__main__":
    main()
