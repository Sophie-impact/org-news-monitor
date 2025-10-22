#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
News Monitor – 페이징 강화 & 안정화 버전
- 네이버 뉴스 API 5페이지(최대 100건) 페이징 수집 (NAVER_PAGES/NAVER_DISPLAY로 조절)
- 시트 규칙(MUST_ALL/MUST_ANY/BLOCK/검색어) 기반 1차 필터
- 기간 필터: 화~금 전일 09:00~오늘 09:00, 월요일은 금 09:00~월 09:00
- 주말(토/일) 자동 스킵
- 제목 기준 중복 제거
- 조직별 대표 최대 3건(예외: 카카오/브라이언임팩트/김범수는 무제한)
- LLM 라벨러(옵션, OPENAI_API_KEY + LLM_ENABLE=true) / 규칙 라벨러 폴백
- 슬랙 전송
"""

from __future__ import annotations

import os
import re
import html
import time
import json
import logging
import hashlib
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

# trafilatura는 선택(없으면 requests+BeautifulSoup 폴백)
try:
    import trafilatura  # pip install trafilatura>=1.9.0
    _HAS_TRAFILATURA = True
except Exception:
    _HAS_TRAFILATURA = False

# LLM (OpenAI)는 선택
try:
    import openai  # pip install openai>=1.40.0
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

KST = ZoneInfo("Asia/Seoul")

# ────────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ────────────────────────────────────────────────────────────────────────────────
def now_kst() -> datetime:
    return datetime.now(tz=KST)

def parse_datetime(dt_str: str | None) -> datetime | None:
    if not dt_str:
        return None
    try:
        dt = dtparser.parse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def to_kst_str(dt: datetime | None) -> str:
    if dt is None:
        return ""
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M")

def strip_html(text: str | None) -> str:
    text = html.unescape(text or "")
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)

def domain_from_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        ext = tldextract.extract(url)
        parts = [p for p in [ext.domain, ext.suffix] if p]
        return ".".join(parts) if parts else ""
    except Exception:
        return ""

def norm_title(t: str | None) -> str:
    t = strip_html(t or "").lower()
    t = re.sub(r"[\[\]【】()（）〈〉<>『』「」]", " ", t)
    t = re.sub(r"[^\w가-힣\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _get_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default

# ────────────────────────────────────────────────────────────────────────────────
# 조회 구간 계산 (09:00 KST 기준)
# ────────────────────────────────────────────────────────────────────────────────
def compute_window_utc(now: datetime | None = None) -> tuple[datetime, datetime]:
    """
    매일 09:00 KST 실행 기준
    - 화~금: 전날 09:00 ~ 오늘 09:00
    - 월: 금요일 09:00 ~ 월요일 09:00
    """
    now = now or datetime.now(tz=KST)
    anchor_kst = now.astimezone(KST).replace(hour=9, minute=0, second=0, microsecond=0)
    days = 3 if anchor_kst.weekday() == 0 else 1  # 0=월
    start_kst = anchor_kst - timedelta(days=days)
    end_kst = anchor_kst
    return start_kst.astimezone(timezone.utc), end_kst.astimezone(timezone.utc)

# ────────────────────────────────────────────────────────────────────────────────
# 시트 읽기 & 규칙
# ────────────────────────────────────────────────────────────────────────────────
def _split_list(val) -> list[str]:
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [x.strip().lower() for x in str(val).split(",") if x.strip()]

def _query_tokens_from(q: str) -> list[str]:
    if not q:
        return []
    parts = re.split(r'\bOR\b', q, flags=re.IGNORECASE)
    tokens = []
    for p in parts:
        t = p.strip().strip('"').strip("'").lower()
        if t:
            tokens.append(t)
    return tokens

def fetch_org_list() -> list[dict]:
    """
    CSV 시트 구조(예시):
      - 조직명(또는 표시명)
      - 검색어(선택)
      - 유형(ORG|PERSON, 선택)
      - MUST_ALL, MUST_ANY, BLOCK (쉼표 분리, 선택)
    """
    sheet_url = os.environ.get("SHEET_CSV_URL", "").strip()
    if not sheet_url:
        raise RuntimeError("SHEET_CSV_URL env var is not set.")

    resp = requests.get(sheet_url, timeout=30)
    resp.raise_for_status()
    csv_text = resp.content.decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(csv_text))

    name_col = None
    for candidate in ["조직명", "표시명"]:
        if candidate in df.columns:
            name_col = candidate
            break
    if not name_col:
        raise RuntimeError("CSV에는 반드시 '조직명' 또는 '표시명' 열이 필요합니다.")

    rows: list[dict] = []
    for _, r in df.iterrows():
        display = str(r[name_col]).strip()
        if not display or display.lower() == "nan":
            continue

        query = (str(r.get("검색어", "")) or "").strip() or display
        kind  = str(r.get("유형", "ORG")).strip().upper() or "ORG"

        must_all = _split_list(r.get("MUST_ALL", ""))
        must_any = _split_list(r.get("MUST_ANY", ""))
        block    = _split_list(r.get("BLOCK", ""))

        rows.append({
            "display": display,
            "query": query,
            "kind": kind,
            "must_all": must_all,
            "must_any": must_any,
            "block": block,
            "query_tokens": _query_tokens_from(query),
        })

    # (표시명, 검색어) 기준 중복 제거
    seen = set(); uniq = []
    for it in rows:
        key = (it["display"], it["query"])
        if key not in seen:
            uniq.append(it); seen.add(key)
    return uniq

def _contains_all(text: str, toks: list[str]) -> bool:
    return all(t in text for t in toks) if toks else True

def _contains_any(text: str, toks: list[str]) -> bool:
    return any(t in text for t in toks) if toks else True

def _contains_none(text: str, toks: list[str]) -> bool:
    return all(t not in text for t in toks) if toks else True

def is_relevant_by_rule(row_cfg: dict, title: str, summary: str) -> bool:
    text = f"{title} {summary}".lower()
    if row_cfg.get("query_tokens") and not _contains_any(text, row_cfg["query_tokens"]):
        return False
    if not _contains_all(text, row_cfg.get("must_all", [])):
        return False
    ma = row_cfg.get("must_any", [])
    if ma and not _contains_any(text, ma):
        return False
    if not _contains_none(text, row_cfg.get("block", [])):
        return False
    return True

# ────────────────────────────────────────────────────────────────────────────────
# 본문 추출
# ────────────────────────────────────────────────────────────────────────────────
def fetch_article_text(url: str, timeout: int = 20) -> str:
    if not url:
        return ""
    if _HAS_TRAFILATURA:
        try:
            downloaded = trafilatura.fetch_url(url, no_ssl=True, timeout=timeout)
            if downloaded:
                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    include_formatting=False,
                    favor_recall=True,
                    deduplicate=True,
                ) or ""
                return text.strip()
        except Exception:
            pass
    # 폴백
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return strip_html(r.text)[:8000].strip()
    except Exception:
        return ""

# ────────────────────────────────────────────────────────────────────────────────
# 검색기 – 페이징 강화 (네이버)
# ────────────────────────────────────────────────────────────────────────────────
def search_naver(query: str, display: int = 20, pages: int = 5) -> list[dict]:
    """
    네이버 뉴스 API 페이징 수집
    - display: 페이지당 결과 수 (기본 20)
    - pages  : 페이지 수 (기본 5 → 최대 100건)
    환경변수:
      NAVER_DISPLAY (기본 20)
      NAVER_PAGES   (기본 5)
    """
    cid  = os.environ.get("NAVER_CLIENT_ID", "")
    csec = os.environ.get("NAVER_CLIENT_SECRET", "")
    if not cid or not csec:
        return []

    display = _get_int_env("NAVER_DISPLAY", display)
    pages   = _get_int_env("NAVER_PAGES", pages)

    endpoint = "https://openapi.naver.com/v1/search/news.json"
    headers  = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec}

    results: list[dict] = []
    for p in range(pages):
        start = 1 + p * display  # 1, 21, 41, ...
        params = {
            "query": f"{query}",
            "display": display,
            "start": start,
            "sort": "date",
        }
        try:
            r = requests.get(endpoint, headers=headers, params=params, timeout=20)
            r.raise_for_status()
            items = r.json().get("items", [])
            if not items:
                break
            for it in items:
                title = strip_html(it.get("title"))
                url   = it.get("originallink") or it.get("link")
                pub   = parse_datetime(it.get("pubDate"))
                if not url or not title:
                    continue
                src = domain_from_url(url) or "naver"
                results.append({
                    "title": title,
                    "url": url,
                    "source": src,
                    "published_at": pub,
                    "origin": "naver",
                    "summary": strip_html(it.get("description") or ""),
                })
        except Exception:
            # 한 페이지 실패는 건너뛰고 다음 페이지 시도
            pass

        time.sleep(0.2)  # 레이트리밋 완화

    return results

# NewsAPI는 그대로(옵션)
def search_newsapi(query: str, window_from_utc: datetime, window_to_utc: datetime,
                   language: str = "ko") -> list[dict]:
    key = os.environ.get("NEWSAPI_KEY", "")
    if not key:
        return []
    endpoint = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{query}",
        "from": window_from_utc.isoformat().replace("+00:00", "Z"),
        "to": window_to_utc.isoformat().replace("+00:00", "Z"),
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
            url   = a.get("url")
            pub   = parse_datetime(a.get("publishedAt"))
            src   = (a.get("source") or {}).get("name") or domain_from_url(url)
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
    except Exception:
        return []

# ────────────────────────────────────────────────────────────────────────────────
# 라벨러 (규칙 + 선택적 LLM)
# ────────────────────────────────────────────────────────────────────────────────
NEG_KW = [
    "횡령","배임","사기","고발","기소","구속","수사","압수수색","소송","징계","제재",
    "벌금","과징금","리콜","결함","사망","부상","폭발","화재","추락","유출","해킹","랜섬웨어",
    "부도","파산","영업정지","퇴출"
]
WATCH_KW = [
    "의혹","조사","점검","심사","검토","국감","감사","연기","지연","유예","우려","경고","리스크","변동성","불확실성"
]
POS_KW = [
    "투자유치","수상","선정","혁신","신기록","최대","달성","성과","협력","파트너십","mou","계약","수주",
    "후원","지원","기부","기탁","장학금","봉사","확대","증가","상승","호조","진출","오픈","출시","공개"
]

def rule_label(title: str, summary: str) -> str:
    """간단 규칙 라벨러 (과도한 빨강 완화)"""
    text = f"{title} {summary}".lower()
    neg  = any(k in text for k in NEG_KW)
    pos  = any(k in text for k in POS_KW)
    watch= any(k in text for k in WATCH_KW)

    # 긍정 신호가 존재하고 직접적 부정 키워드가 없으면 중립/긍정
    if pos and not neg:
        return "🔵"
    # 직접 부정은 없고 주의 키워드만 있으면 노랑
    if watch and not neg:
        return "🟡"
    # 직접 부정이 있으면 빨강
    if neg:
        return "🔴"
    # 기본은 초록
    return "🟢"

def llm_enabled() -> bool:
    flag = os.environ.get("LLM_ENABLE", "").strip().lower()
    enabled = flag in {"1","true","yes","on"}
    return enabled and bool(os.environ.get("OPENAI_API_KEY", "").strip()) and _HAS_OPENAI

def llm_label(display: str, title: str, summary: str, content: str) -> str | None:
    """선택적 LLM 라벨러. 실패 시 None."""
    if not llm_enabled():
        return None
    body = (content or "").strip()
    if len(body) > 3500:
        body = body[:3500]
    try:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        prompt = f"""당신은 위기관리 분석가입니다. 아래 기사가 조직에 미치는 영향을 한 가지로만 분류하세요.
조직: {display}
제목: {title}
요약: {summary}
본문(일부): {body}

가능한 라벨: 🔵(positive), 🟢(neutral), 🟡(monitor), 🔴(negative)
정확히 해당 기호 하나만 출력하세요."""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=6,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out if out in {"🔵","🟢","🟡","🔴"} else None
    except Exception:
        return None

# ────────────────────────────────────────────────────────────────────────────────
# Slack
# ────────────────────────────────────────────────────────────────────────────────
def post_to_slack(lines: list[str]) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    channel = os.environ.get("SLACK_CHANNEL", "").strip()
    if not token or not channel:
        raise RuntimeError("SLACK_BOT_TOKEN or SLACK_CHANNEL missing.")
    client = WebClient(token=token)
    text = "\n".join(lines) if lines else "오늘은 신규로 감지된 기사가 없습니다."
    client.chat_postMessage(channel=channel, text=text)

# ────────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────────
EXEMPT_UNCAPPED = {"카카오", "브라이언임팩트", "김범수"}  # 무제한
PER_ORG_CAP = 3  # 그 외 최대 3건

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 주말 스킵
    if now_kst().weekday() in (5, 6):  # 5=토, 6=일
        logging.info("Weekend (Sat/Sun) – skipping run.")
        return

    window_from_utc, window_to_utc = compute_window_utc()
    logging.info("Window UTC: %s ~ %s", window_from_utc, window_to_utc)

    rows = fetch_org_list()
    logging.info("Loaded %d targets.", len(rows))

    all_lines: list[str] = []

    for idx, row in enumerate(rows, start=1):
        display = row["display"]
        query   = row["query"]
        logging.info("(%d/%d) Searching: %s | %s", idx, len(rows), display, query)

        # 네이버 5페이지 수집 (환경변수로 조절 가능)
        naver_items = search_naver(
            query,
            display=_get_int_env("NAVER_DISPLAY", 20),
            pages=_get_int_env("NAVER_PAGES", 5),
        )
        time.sleep(0.25)
        newsapi_items = search_newsapi(query, window_from_utc, window_to_utc, language="ko")
        logging.info("  raw: naver=%d, newsapi=%d", len(naver_items), len(newsapi_items))

        items: list[dict] = []
        for it in (naver_items + newsapi_items):
            it["display"] = display
            it["row_cfg"] = row
            items.append(it)

        # 1) 관련성 필터
        before_rel = len(items)
        items = [it for it in items if is_relevant_by_rule(it["row_cfg"], it["title"], it.get("summary",""))]
        logging.info("  after relevance: %d -> %d", before_rel, len(items))

        # 2) 기간 필터
        before_win = len(items)
        items = [it for it in items if it["published_at"] and window_from_utc <= it["published_at"] < window_to_utc]
        logging.info("  after window: %d -> %d", before_win, len(items))

        # 3) 최신순 + 제목 기준 dedup
        items.sort(key=lambda x: x["published_at"], reverse=True)
        seen_titles = set(); uniq = []
        for it in items:
            tkey = norm_title(it["title"])
            if tkey and it["url"] and tkey not in seen_titles:
                uniq.append(it); seen_titles.add(tkey)

        # 4) 조직별 상한 적용 (예외는 무제한)
        if display not in EXEMPT_UNCAPPED:
            uniq = uniq[:PER_ORG_CAP]

        # 5) 라벨링 & 메시지 생성
        for art in uniq:
            content = ""  # 속도 위해 본문은 기본 미사용 (LLM 쓰면 활성화)
            label = None

            if llm_enabled():
                # 필요 시 본문 추출
                content = fetch_article_text(art["url"])
                label = llm_label(art["display"], art["title"], art.get("summary",""), content)

            if not label:
                label = rule_label(art["title"], art.get("summary",""))

            src = art["source"]
            when_str = to_kst_str(art["published_at"])
            line = f"[{art['display']}] <{art['url']}|{art['title']}> ({src})({when_str}) [{label}]"
            all_lines.append(line)

    post_to_slack(all_lines)
    logging.info("Posted %d lines to Slack.", len(all_lines))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error")
        raise
