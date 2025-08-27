#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily news monitor: Naver Search API + NewsAPI
- Reads an organization list from Google Sheets CSV (SHEET_CSV_URL)
- Fetches latest articles for each org from both sources
- Simple rule-based "risk sentiment" labeling
- Posts a formatted summary to Slack

Environment variables required:
  NAVER_CLIENT_ID, NAVER_CLIENT_SECRET, NEWSAPI_KEY
  SLACK_BOT_TOKEN, SLACK_CHANNEL
  SHEET_CSV_URL
Optional:
  MAX_RESULTS_PER_ORG (default: 1)  # number of headlines per org
  LOOKBACK_HOURS (default: 24)      # time window for recent news
"""
import os
import re
import html
import time
import logging
import requests
import pandas as pd
import tldextract
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
from zoneinfo import ZoneInfo
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

KST = ZoneInfo("Asia/Seoul")

# -------- utils --------
def now_kst():
    return datetime.now(tz=KST)

def parse_datetime(dt_str):
    """Robust datetime parser (RFC1123, ISO8601). Returns aware UTC dt."""
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
        import tldextract
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

# -------- org list --------
def fetch_org_list():
    sheet_url = os.environ.get("SHEET_CSV_URL", "").strip()
    if not sheet_url:
        raise RuntimeError("SHEET_CSV_URL env var is not set.")
    resp = requests.get(sheet_url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    if df.shape[1] == 0:
        raise RuntimeError("CSV has no columns.")
    # prefer a column named 'org' or '조직명' else first column
    col = None
    for cand in ["org", "Org", "ORG", "조직명"]:
        if cand in df.columns:
            col = cand
            break
    if col is None:
        col = df.columns[0]
    orgs = [str(x).strip() for x in df[col].tolist() if str(x).strip() and str(x).strip().lower() != "nan"]
    # dedup while preserving order
    seen = set(); uniq = []
    for o in orgs:
        if o not in seen:
            uniq.append(o); seen.add(o)
    return uniq

# -------- Naver News Search --------
def search_naver(org, display=5):
    cid = os.environ.get("NAVER_CLIENT_ID", "")
    csec = os.environ.get("NAVER_CLIENT_SECRET", "")
    if not cid or not csec:
        logging.warning("NAVER credentials missing; skipping Naver for %s", org)
        return []
    endpoint = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec}
    params = {
        "query": f"\"{org}\"",
        "display": display,
        "start": 1,
        "sort": "date",
    }
    try:
        r = requests.get(endpoint, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", [])
        results = []
        for it in items:
            title = strip_html(it.get("title"))
            url = it.get("originallink") or it.get("link")
            pub = parse_datetime(it.get("pubDate"))
            if not url or not title:
                continue
            src = domain_from_url(url) or "Naver"
            results.append({
                "org": org,
                "title": title,
                "url": url,
                "source": src,
                "published_at": pub,
                "origin": "naver",
                "summary": strip_html(it.get("description", "")),
            })
        return results
    except Exception as e:
        logging.exception("Naver search failed for %s: %s", org, e)
        return []

# -------- NewsAPI --------
def search_newsapi(org, page_size=5, lookback_hours=24, language="ko"):
    key = os.environ.get("NEWSAPI_KEY", "")
    if not key:
        logging.warning("NEWSAPI_KEY missing; skipping NewsAPI for %s", org)
        return []
    endpoint = "https://newsapi.org/v2/everything"
    to_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    frm = to_utc - timedelta(hours=lookback_hours)
    params = {
        "q": f"\"{org}\"",
        "from": frm.isoformat().replace("+00:00","Z"),
        "to": to_utc.isoformat().replace("+00:00","Z"),
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "language": language,
        "apiKey": key,
    }
    try:
        r = requests.get(endpoint, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        arts = data.get("articles", [])
        results = []
        for a in arts:
            title = strip_html(a.get("title"))
            url = a.get("url")
            pub = parse_datetime(a.get("publishedAt"))
            src = (a.get("source") or {}).get("name") or domain_from_url(url)
            if not url or not title:
                continue
            results.append({
                "org": org,
                "title": title,
                "url": url,
                "source": src,
                "published_at": pub,
                "origin": "newsapi",
                "summary": strip_html(a.get("description") or a.get("content") or ""),
            })
        return results
    except Exception as e:
        logging.exception("NewsAPI search failed for %s: %s", org, e)
        return []

# -------- sentiment / risk labeling --------
NEG_KW = [
    "횡령","배임","사기","고발","기소","구속","수사","압수수색","소송","고소","분쟁","리콜","결함",
    "징계","제재","벌금","과징금","부실","파산","부도","중단","연기","오염","사망","부상","폭발","화재",
    "추락","유출","해킹","랜섬웨어","침해","악성코드","담합","독점","불매","논란","갑질","표절","혐의",
    "불법","위법","취소","철회","부정","적자","적자전환","감소","급락","하락","경고","경보","리스크",
    "부정적","경찰","검찰","당국","소환","징역","금지","퇴출"
]
WATCH_KW = [
    "의혹","점검","조사","심사","검토","논의","잠정","연구결과","유예","우려","관심","주시","잠정치",
    "하한","상한","공정위","국감","지적","요구","연장","연동","변동성","경쟁 심화","불확실성"
]
POS_KW = [
    "투자유치","시리즈","라운드","유치","유치 성공","수상","선정","최우수","혁신","신기록","최대",
    "상승","증가","호조","호재","확대","진출","오픈","출시","공개","협력","파트너십","MOU","계약",
    "수주","달성","달성했다","달성해","선보여","개최","성과","매출 성장","흑자","흑자전환","수익성 개선"
]

def label_sentiment(title, summary):
    text = f"{title} {summary}".lower()
    def any_kw(kws):
        for kw in kws:
            if kw.lower() in text:
                return True
        return False
    if any_kw(NEG_KW):
        return "🔴"
    if any_kw(WATCH_KW):
        return "🟡"
    if any_kw(POS_KW):
        return "🔵"
    return "🟢"

# -------- Slack --------
def post_to_slack(lines):
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    channel = os.environ.get("SLACK_CHANNEL", "").strip()  # '#news-monitor' or channel ID 'C...'
    if not token or not channel:
        raise RuntimeError("SLACK_BOT_TOKEN or SLACK_CHANNEL missing.")
    client = WebClient(token=token)
    text = "\n".join(lines)
    try:
        client.chat_postMessage(channel=channel, text=text)
    except SlackApiError as e:
        raise RuntimeError(f"Slack API error: {e.response.get('error')}")

# -------- main flow --------
def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    lookback = int(os.environ.get("LOOKBACK_HOURS", "24"))
    max_per_org = int(os.environ.get("MAX_RESULTS_PER_ORG", "1"))
    # 1) organizations
    orgs = fetch_org_list()
    logging.info("Loaded %d organizations.", len(orgs))

    all_lines = []
    for idx, org in enumerate(orgs, start=1):
        logging.info("(%d/%d) Searching: %s", idx, len(orgs), org)
        naver_items = search_naver(org, display=max(3, max_per_org*2))
        time.sleep(0.25)  # be polite
        newsapi_items = search_newsapi(org, page_size=max(3, max_per_org*2), lookback_hours=lookback, language="ko")
        items = naver_items + newsapi_items

        # filter by time window
        cutoff = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(hours=lookback)
        items_recent = [it for it in items if (it["published_at"] is None or it["published_at"] >= cutoff)]
        items = items_recent or items  # fallback if none recent

        # sort by published_at desc
        items.sort(key=lambda x: x["published_at"] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

        # de-dup by normalized title + domain
        seen_keys = set()
        uniq = []
        for it in items:
            key = (norm_title(it["title"]), domain_from_url(it["url"]))
            if key not in seen_keys and it["url"]:
                uniq.append(it); seen_keys.add(key)

        # take top N per org
        take = uniq[:max_per_org]

        for art in take:
            label = label_sentiment(art["title"], art.get("summary",""))
            src = art["source"]
            when_str = to_kst_str(art["published_at"] or now_kst())
            line = f"[{art['org']}] <{art['url']}|{art['title']}> ({src})({when_str}) [{label}]"
            all_lines.append(line)

    if not all_lines:
        all_lines.append("오늘은 신규로 감지된 기사가 없습니다.")

    # group message
    post_to_slack(all_lines)

if __name__ == "__main__":
    main()
