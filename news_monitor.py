#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
News Monitor – LLM(본문 기반) + 위험 힌트 강화 버전

필요 env:
- SHEET_CSV_URL
- NAVER_CLIENT_ID / NAVER_CLIENT_SECRET (네이버 뉴스)
- NEWSAPI_KEY (NewsAPI)
- SLACK_BOT_TOKEN / SLACK_CHANNEL (채널 ID 또는 #채널명)
- OPENAI_API_KEY (LLM 사용 시)
- LLM_ENABLE ("1" / "true" / "on" 등 truthy)
- LLM_MODEL (선택, 기본 gpt-4o-mini)
"""

from __future__ import annotations

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
import trafilatura

# --- LLM (OpenAI) ---
try:
    import openai  # pip install openai>=1.40.0
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

KST = ZoneInfo("Asia/Seoul")


# ---------- 공통 유틸 ----------
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


# ---------- 조회 구간 계산 ----------
def compute_window_utc(now: datetime | None = None) -> tuple[datetime, datetime]:
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


# ---------- 시트 파싱 유틸 ----------
def _split_list(val) -> list[str]:
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [x.strip().lower() for x in str(val).split(",") if x.strip()]

def _query_tokens_from(q: str) -> list[str]:
    # 'A' OR 'B' → ["a", "b"] (따옴표 제거)
    if not q:
        return []
    parts = re.split(r'\bOR\b', q, flags=re.IGNORECASE)
    tokens = []
    for p in parts:
        t = p.strip().strip('"').strip("'").lower()
        if t:
            tokens.append(t)
    return tokens


# ---------- 시트 읽기 ----------
def fetch_org_list() -> list[dict]:
    """
    반환: 리스트[{
      'display': 슬랙 표시명,
      'query': 검색어(없으면 display),
      'kind': 'ORG' | 'PERSON',
      'must_all': [...], 'must_any': [...], 'block': [...],
      'query_tokens': [...]
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

        query = str(r.get("검색어", "")).strip() or display
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
    seen = set()
    uniq = []
    for it in rows:
        key = (it["display"], it["query"])
        if key not in seen:
            uniq.append(it)
            seen.add(key)
    return uniq


# ---------- 본문 추출 ----------
def fetch_article_text(url: str, timeout: int = 20) -> str:
    """
    주어진 URL의 본문 텍스트를 최대한 깔끔하게 추출.
    trafilatura 우선, 실패 시 간단한 HTML->텍스트 폴백.
    """
    if not url:
        return ""
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


# ---------- 네이버 뉴스 검색 ----------
def search_naver(query: str, display: int = 10) -> list[dict]:
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


# ---------- NewsAPI ----------
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


# ---------- 규칙 라벨(보조용, 선택) ----------
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


# ---------- 위험 힌트(키워드) ----------
def _make_risk_hints(title: str, summary: str, content: str) -> str:
    text = f"{title} {summary} {content}".lower()
    neg_hits = [kw for kw in NEG_KW if kw.lower() in text]
    watch_hits = [kw for kw in WATCH_KW if kw.lower() in text]
    hints = []
    if neg_hits:
        hints.append(f"부정 키워드: {', '.join(neg_hits[:8])}")
    if watch_hits:
        hints.append(f"주의 키워드: {', '.join(watch_hits[:8])}")
    return " / ".join(hints)


# ---------- LLM 라벨러(본문 기반, 노랑 쪽 가중) ----------
def llm_enabled() -> bool:
    flag = os.environ.get("LLM_ENABLE", "").strip().lower()
    enabled = flag in {"1","true","yes","on"}
    return enabled and bool(os.environ.get("OPENAI_API_KEY", "").strip()) and _HAS_OPENAI

def llm_label(display_name: str, title: str, summary: str, content: str, risk_hints: str = "") -> str | None:
    """
    기사 본문까지 고려한 라벨링. 결과는 {🔵,🟢,🟡,🔴} 중 하나.
    프롬프트를 강화해 '약한 부정/우려'는 🟡로 기울도록 설계.
    """
    if not llm_enabled():
        return None

    body = (content or "").strip()
    if len(body) > 3500:  # 비용/속도 최적화
        body = body[:3500]

    try:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

        prompt = f"""당신은 언론/위기관리 분석가입니다. 아래 기사 내용이 '조직'에 미치는 영향을 평가하세요.
모호하거나 정보가 불충분할 때는 '중립(🟢)'이 아니라 '팔로업 필요(🟡)'로 분류하세요.

[조직]
{display_name}

[기사 제목]
{title}

[요약/리드]
{summary}

[기사 본문(일부)]
{body}

[신호 힌트(키워드 탐지 결과)]
{risk_hints if risk_hints else "없음"}

판단 규칙(중요, 우선순위 적용):
1) 🔴 부정/리스크: 비판/논란/고발/수사/소송/제재/특검/리콜/결함/중대사고 등으로
   조직의 평판·법무·재정 위험이 현실화되었거나 매우 높음.
2) 🟡 팔로업 필요: 직접적 피해는 아니지만 부정적 뉘앙스·우려·논란 가능성·규제/조사 가능성 등
   잠재 리스크가 존재하거나, 파장이 불확실하여 추적 모니터링이 필요한 경우.
   (모호·정보부족·부정적 인상 → 🟡로 기울기)
3) 🟢 중립: 긍/부정 효과가 뚜렷하지 않은 사실 전달/일반 보도.
4) 🔵 긍정: 수상·성과·투자·협력·호재 등 조직에 명확히 유리.

출력 형식: 라벨 기호 하나만 출력 (오직 🔴,🟡,🟢,🔵 중 하나)
"""

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=4,
        )
        out = (resp.choices[0].message.content or "").strip()
        return out if out in {"🔵","🟢","🟡","🔴"} else None
    except Exception:
        return None


# ---------- 행 규칙 기반 관련성 필터 ----------
def _contains_all(text: str, toks: list[str]) -> bool:  return all(t in text for t in toks) if toks else True
def _contains_any(text: str, toks: list[str]) -> bool:  return any(t in text for t in toks) if toks else True
def _contains_none(text: str, toks: list[str]) -> bool: return all(t not in text for t in toks) if toks else True

def is_relevant_by_rule(row_cfg: dict, title: str, summary: str) -> bool:
    """
    1) query_tokens 중 하나는 반드시 포함
    2) MUST_ALL 모두 포함
    3) MUST_ANY 중 최소 1개 포함
    4) BLOCK 단어가 포함되면 제외
    """
    text = f"{title} {summary}".lower()
    if row_cfg["query_tokens"] and not _contains_any(text, row_cfg["query_tokens"]):
        return False
    if not _contains_all(text, row_cfg["must_all"]):
        return False
    if not _contains_any(text, row_cfg["must_any"]):
        return False
    if not _contains_none(text, row_cfg["block"]):
        return False
    return True


# ---------- Slack ----------
def post_to_slack(lines: list[str]) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    channel = os.environ.get("SLACK_CHANNEL", "").strip()
    if not token or not channel:
        raise RuntimeError("SLACK_BOT_TOKEN or SLACK_CHANNEL missing.")
    client = WebClient(token=token)
    text = "\n".join(lines) if lines else "오늘은 신규로 감지된 기사가 없습니다."
    client.chat_postMessage(channel=channel, text=text)


# ---------- main ----------
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # ✅ 토/일이면 스킵
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

        # 검색
        naver_items = search_naver(query, display=max(10, 20))
        time.sleep(0.25)
        newsapi_items = search_newsapi(query, window_from_utc, window_to_utc, language="ko")
        logging.info("  raw: naver=%d, newsapi=%d", len(naver_items), len(newsapi_items))

        # 합치기
        items: list[dict] = []
        for it in (naver_items + newsapi_items):
            it["display"] = display
            it["row_cfg"] = row
            items.append(it)

        # 관련성 필터
        before_rel = len(items)
        items = [it for it in items if is_relevant_by_rule(it["row_cfg"], it["title"], it.get("summary",""))]
        logging.info("  after relevance: %d -> %d", before_rel, len(items))

        # 기간 필터
        before_win = len(items)
        items = [it for it in items if it["published_at"] and window_from_utc <= it["published_at"] < window_to_utc]
        logging.info("  after window: %d -> %d", before_win, len(items))

        # 최신순 + 중복 제거(제목만 기준)
        items.sort(key=lambda x: x["published_at"], reverse=True)
        seen_titles = set()
        uniq = []
        for it in items:
            title_key = norm_title(it["title"])
            if title_key and it["url"] and title_key not in seen_titles:
                uniq.append(it)
                seen_titles.add(title_key)

        # 제한 없이 전부 전송
        for art in uniq:
            # 1) 본문 추출 + 힌트 생성
            content = fetch_article_text(art["url"])
            hints = _make_risk_hints(art["title"], art.get("summary",""), content)

            # 2) LLM 판별 (우선) – 실패 시 중립
            label = llm_label(
                art["display"],
                art["title"],
                art.get("summary",""),
                content,
                risk_hints=hints
            ) or "🟢"

            # 3) (옵션) 보수적 승격: 중립인데 힌트가 있으면 🟡로 승격
            if label == "🟢" and hints:
                label = "🟡"

            src = art["source"]
            when_str = to_kst_str(art["published_at"])
            line = f"[{art['display']}] <{art['url']}|{art['title']}> ({src})({when_str}) [{label}]"
            all_lines.append(line)

    post_to_slack(all_lines)
    logging.info("Posted %d lines to Slack.", len(all_lines))


if __name__ == "__main__":
    main()
