#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
News Monitor – 본문 추출 + LLM(JSON) 판단 + 규칙/키워드 보조 + 주말 스킵
- 시트 규칙(MUST_ALL/MUST_ANY/BLOCK/검색어)로 1차 필터
- trafilatura로 본문 추출 후 LLM이 조직 관점 영향(긍정/중립/모니터/부정)을 JSON으로 판정
- 강부정 키워드/리스크 점수로 보수적 보정 (과한 노란색 방지 로직 포함)
- 동일 제목(정규화) 중복 제거
"""

from __future__ import annotations

import os
import re
import html
import time
import json
import logging
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
from zoneinfo import ZoneInfo
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError  # noqa: F401 (예비)
import tldextract
import trafilatura

# --- LLM (OpenAI) ---
try:
    import openai  # pip install openai>=1.40.0
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

KST = ZoneInfo("Asia/Seoul")

# =========================
# 공통 유틸
# =========================
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

# =========================
# 조회 구간 (09:00 KST 기준)
# =========================
def compute_window_utc(now: datetime | None = None) -> tuple[datetime, datetime]:
    """
    매일 09:00 KST 실행 기준
    - 화~금: 전날 09:00 ~ 오늘 09:00
    - 월: 금요일 09:00 ~ 월요일 09:00
    """
    now = now or datetime.now(tz=KST)
    anchor_kst = now.astimezone(KST).replace(hour=9, minute=0, second=0, microsecond=0)
    days = 3 if anchor_kst.weekday() == 0 else 1
    start_kst = anchor_kst - timedelta(days=days)
    end_kst = anchor_kst
    return start_kst.astimezone(timezone.utc), end_kst.astimezone(timezone.utc)

# =========================
# 시트 파싱 유틸/로더
# =========================
def _split_list(val) -> list[str]:
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [x.strip().lower() for x in str(val).split(",") if x.strip()]

def _query_tokens_from(q: str) -> list[str]:
    # 'A' OR 'B' → ["a", "b"] (따옴표 제거, OR 분리)
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
    CSV 시트 구조(헤더 예시):
      - 조직명(또는 표시명)
      - 검색어(선택)
      - 유형(ORG|PERSON, 선택)
      - MUST_ALL, MUST_ANY, BLOCK (쉼표로 구분, 선택)
    반환: [{
      'display', 'query', 'kind', 'must_all', 'must_any', 'block', 'query_tokens'
    }, ...]
    """
    sheet_url = os.environ.get("SHEET_CSV_URL", "").strip()
    if not sheet_url:
        raise RuntimeError("SHEET_CSV_URL env var is not set.")

    resp = requests.get(sheet_url, timeout=30)
    resp.raise_for_status()
    csv_text = resp.content.decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(csv_text))

    # 필수 이름 컬럼: '조직명' 또는 '표시명'
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

        # 검색어 NaN/공백/문자열 'nan' 모두 방지
        raw_q = r.get("검색어", "")
        if pd.isna(raw_q) or str(raw_q).strip() == "" or str(raw_q).strip().lower() == "nan":
            query = display
        else:
            query = str(raw_q).strip()

        kind = str(r.get("유형", "ORG")).strip().upper() or "ORG"
        must_all = _split_list(r.get("MUST_ALL", ""))
        must_any = _split_list(r.get("MUST_ANY", ""))
        block    = _split_list(r.get("BLOCK", ""))

        item = {
            "display": display,
            "query": query,
            "kind": kind,
            "must_all": must_all,
            "must_any": must_any,
            "block": block,
            "query_tokens": _query_tokens_from(query),
        }
        rows.append(item)

    # 중복 제거(표시명+검색어 기준)
    seen = set(); uniq = []
    for it in rows:
        key = (it["display"], it["query"])
        if key not in seen:
            uniq.append(it); seen.add(key)
    return uniq

# =========================
# 본문 추출
# =========================
def fetch_article_text(url: str, timeout: int = 20) -> str:
    """
    trafilatura 우선, 실패 시 requests + HTML 제거 폴백
    """
    if not url:
        return ""
    # 1) trafilatura
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
            if text:
                return text.strip()
    except Exception:
        pass

    # 2) requests + strip
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return strip_html(r.text)[:8000].strip()
    except Exception:
        return ""

# =========================
# 뉴스 검색기
# =========================
def search_naver(query: str, display: int = 20) -> list[dict]:
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
    except Exception:
        return []

def search_newsapi(query: str, window_from_utc: datetime, window_to_utc: datetime,
                   language: str = "ko") -> list[dict]:
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
    except Exception:
        return []

# =========================
# 규칙 라벨(폴백/참고)
# =========================
NEG_KW = ["횡령","배임","사기","고발","기소","구속","수사","압수수색","소송","고소","분쟁","리콜","결함","징계","제재",
          "벌금","과징금","부실","파산","부도","중단","연기","오염","사망","부상","폭발","화재","추락","유출",
          "해킹","랜섬웨어","침해","악성코드","담합","독점","불매","논란","갑질","표절","혐의","불법","위법",
          "취소","철회","부정","적자","감소","급락","하락","경고","경보","리스크","소환","징역"]
WATCH_KW = ["의혹","점검","조사","심사","검토","논의","잠정","연구결과","유예","우려","관심","주시","잠정치","공정위","국감",
            "지적","요구","연장","변동성","불확실성"]
POS_KW = ["투자유치","시리즈","라운드","유치","수상","선정","혁신","신기록","최대","상승","증가","호조","호재","확대",
          "진출","오픈","출시","공개","협력","파트너십","MOU","계약","수주","달성","성과","흑자","흑자전환","개최"]

def rule_label(title: str, summary: str) -> str:
    text = f"{title} {summary}".lower()
    if any(k.lower() in text for k in NEG_KW): return "🔴"
    if any(k.lower() in text for k in WATCH_KW): return "🟡"
    if any(k.lower() in text for k in POS_KW): return "🔵"
    return "🟢"

# =========================
# 행 규칙 기반 관련성 필터
# =========================
def _contains_all(text: str, toks: list[str]) -> bool:
    return all(t in text for t in toks) if toks else True

def _contains_any(text: str, toks: list[str]) -> bool:
    return any(t in text for t in toks) if toks else True

def _contains_none(text: str, toks: list[str]) -> bool:
    return all(t not in text for t in toks) if toks else True

def is_relevant_by_rule(row_cfg: dict, title: str, summary: str) -> bool:
    """
    1) query_tokens(검색어) 중 하나라도 포함 (없으면 패스)
    2) MUST_ALL 모두 포함
    3) MUST_ANY 중 최소 1개 포함 (없으면 패스)
    4) BLOCK 포함 시 제외
    """
    text = f"{title} {summary}".lower()

    if row_cfg.get("query_tokens") and not _contains_any(text, row_cfg["query_tokens"]):
        return False
    if not _contains_all(text, row_cfg.get("must_all", [])):
        return False

    must_any = row_cfg.get("must_any", [])
    if must_any and not _contains_any(text, must_any):
        return False

    if not _contains_none(text, row_cfg.get("block", [])):
        return False
    return True

# =========================
# 보조 리스크 신호(점수화)
# =========================
STRONG_NEG = {"수사","고발","특검","기소","압수수색","제재","리콜","결함","사망","부상","폭발","화재","유출","랜섬웨어"}

def calc_risk_signals(title: str, summary: str, content: str):
    text = f"{title} {summary} {content}".lower()
    neg_hits   = [kw for kw in NEG_KW   if kw.lower() in text]
    watch_hits = [kw for kw in WATCH_KW if kw.lower() in text]
    pos_hits   = [kw for kw in POS_KW   if kw.lower() in text]
    strong_neg = [kw for kw in STRONG_NEG if kw.lower() in text]

    score = 2*len(neg_hits) + 1*len(watch_hits) - 1*len(pos_hits)  # 부정2, 주의1, 긍정-1
    hints = []
    if neg_hits:   hints.append(f"부정:{', '.join(neg_hits[:8])}")
    if watch_hits: hints.append(f"주의:{', '.join(watch_hits[:8])}")
    if pos_hits:   hints.append(f"긍정:{', '.join(pos_hits[:8])}")
    return {"score": score, "strong_neg": strong_neg, "hints_text": " / ".join(hints) if hints else ""}

# =========================
# LLM(JSON) 라벨러
# =========================
IMPACT_MAP = {"positive":"🔵","neutral":"🟢","monitor":"🟡","negative":"🔴"}

def _safe_load_json(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def llm_enabled() -> bool:
    flag = os.environ.get("LLM_ENABLE", "").strip().lower()
    enabled = flag in {"1","true","yes","on"}
    return enabled and bool(os.environ.get("OPENAI_API_KEY", "").strip()) and _HAS_OPENAI

def llm_label(display_name: str, title: str, summary: str, content: str, hints_text: str = ""):
    """
    JSON으로 {impact, confidence, evidence[]} 반환 → {label, confidence, raw} or None
    """
    if not llm_enabled():
        return None

    body = (content or "").strip()
    if len(body) > 3500:
        body = body[:3500]

    try:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

        prompt = f"""역할: 당신은 위기관리 분석가입니다. 다음 기사가 '조직'에 주는 '영향'만 평가하세요.
- 긍정/중립/모니터링 필요/부정 네 가지 중 하나로 결정합니다.
- 조직 관점의 '실질적 영향'에 집중합니다(산업 일반 논평은 가중치 낮음).
- 불확실하거나 양가적이면 '모니터링 필요'를 권장합니다.

JSON으로만 출력:
{{
  "impact": "positive|neutral|monitor|negative",
  "confidence": 0.0-1.0,
  "evidence": ["짧은 근거 1-2개"]
}}

[조직]
{display_name}

[제목]
{title}

[요약]
{summary}

[본문(일부)]
{body}

[키워드 힌트]
{hints_text or "없음"}
"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0,
            max_tokens=220,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = _safe_load_json(raw)
        if not data:
            return None

        impact = str(data.get("impact","")).lower()
        conf = float(data.get("confidence", 0.5))
        label = IMPACT_MAP.get(impact)
        if not label:
            return None
        return {"label": label, "confidence": conf, "raw": data}
    except Exception:
        return None

# =========================
# Slack
# =========================
def post_to_slack(lines: list[str]) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    channel = os.environ.get("SLACK_CHANNEL", "").strip()
    if not token or not channel:
        raise RuntimeError("SLACK_BOT_TOKEN or SLACK_CHANNEL missing.")
    client = WebClient(token=token)
    text = "\n".join(lines) if lines else "오늘은 신규로 감지된 기사가 없습니다."
    client.chat_postMessage(channel=channel, text=text)

# =========================
# main
# =========================
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

        naver_items = search_naver(query, display=20)
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

        # 3) 최신순 + 제목 기준 dedup (도메인 상관없이 동일 제목 1개)
        items.sort(key=lambda x: x["published_at"], reverse=True)
        seen_titles = set(); uniq = []
        for it in items:
            title_key = norm_title(it["title"])
            if title_key and it["url"] and title_key not in seen_titles:
                uniq.append(it); seen_titles.add(title_key)

        # 4) 전부 처리 (조직별 제한 없음)
        for art in uniq:
            content = fetch_article_text(art["url"])
            signals = calc_risk_signals(art["title"], art.get("summary",""), content)

            # LLM 우선
            llm_out = llm_label(
                art["display"], art["title"], art.get("summary",""),
                content, hints_text=signals["hints_text"]
            )
            if llm_out:
                label = llm_out["label"]
                conf  = llm_out["confidence"]
            else:
                label = rule_label(art["title"], art.get("summary",""))
                conf  = 0.5

            # 강한 부정 신호 있으면 보수적 오버라이드(🔴)
            if signals["strong_neg"] and label in {"🟢","🟡"}:
                label = "🔴"

            # 리스크 점수 기반 보정: LLM이 🔵/🟢이고 확신 낮고(score 높음) → 🟡
            if label in {"🔵","🟢"} and conf < 0.7 and signals["score"] >= 3:
                label = "🟡"

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
