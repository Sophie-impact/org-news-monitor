#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
News Monitor – Gemini+Claude 통합 개선 버전
- 컨텍스트 기반 키워드 분석 (직접부정/상황부정/모니터링/긍정)
- LLM 프롬프트 강화 (primary_reason, org_relevance, confidence)
- 다층 검증: 규칙 → LLM(신뢰도) → 보수적 검증 → 과도한 🔴 완화
- 주말 스킵, 조회구간(월=3일, 그외=1일), 본문 추출(trafilatura)
- 제목 기준 dedup, 캐싱(제목+본문 해시)로 재분석 방지
- Slack 전송 시 신뢰도 표시(!/?), 주요 근거 요약
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
from collections import defaultdict
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError  # noqa: F401
import tldextract

# 본문 추출 (옵션)
try:
    import trafilatura
    _HAS_TRAFILATURA = True
except Exception:
    _HAS_TRAFILATURA = False

# OpenAI LLM (옵션)
try:
    import openai  # pip install openai>=1.40.0
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

KST = ZoneInfo("Asia/Seoul")

# =========================================================
# 컨텍스트 기반 키워드
# =========================================================
DIRECT_NEGATIVE = {
    "법적": ["횡령", "배임", "사기", "고발", "기소", "구속", "수사", "압수수색", "징역", "실형", "특검"],
    "사업": ["리콜", "결함", "파산", "부도", "영업정지", "사업중단", "퇴출"],
    "안전": ["사망", "부상", "폭발", "화재", "추락", "유출", "해킹", "랜섬웨어", "개인정보유출"],
}
CONTEXTUAL_NEGATIVE = {
    "경영": ["적자", "손실", "감소", "하락", "부실"],
    "규제": ["제재", "벌금", "과징금", "징계", "처분"],
    "논란": ["논란", "비판", "갑질", "불법", "위법", "부정"],
}
MONITORING_KEYWORDS = {
    "조사": ["의혹", "조사", "점검", "심사", "검토", "국감", "감사"],
    "불확실": ["연기", "지연", "유예", "잠정", "검토중", "불확실성"],
    "주의": ["우려", "경고", "리스크", "변동성", "관심", "주시"],
}
POSITIVE_KEYWORDS = {
    "성과": ["수상", "선정", "혁신", "신기록", "달성", "성과", "흑자전환", "최대"],
    "성장": ["투자유치", "상승", "증가", "호조", "확대", "진출", "성장"],
    "협력": ["협력", "파트너십", "mou", "계약", "수주", "제휴", "연합"],
    "사회공헌": ["후원", "지원", "기부", "기증", "기탁", "장학금", "봉사"],
}

# =========================================================
# 유틸
# =========================================================
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
    if not dt:
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

def content_hash(title: str, content: str) -> str:
    return hashlib.md5(f"{title}:{content[:1000]}".encode()).hexdigest()

# =========================================================
# 조회 구간 (09:00 KST 기준)
# =========================================================
def compute_window_utc(now: datetime | None = None) -> tuple[datetime, datetime]:
    """
    실행 시간 09:00 KST 기준
    - 화~금: 전날 09:00 ~ 오늘 09:00
    - 월: 금 09:00 ~ 월 09:00
    """
    now = now or now_kst()
    anchor_kst = now.astimezone(KST).replace(hour=9, minute=0, second=0, microsecond=0)
    days = 3 if anchor_kst.weekday() == 0 else 1
    start_kst = anchor_kst - timedelta(days=days)
    end_kst = anchor_kst
    return start_kst.astimezone(timezone.utc), end_kst.astimezone(timezone.utc)

# =========================================================
# 시트 로딩 / 관련성 필터
# =========================================================
def _split_list(val) -> list[str]:
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [x.strip().lower() for x in str(val).split(",") if x.strip()]

def _query_tokens_from(q: str) -> list[str]:
    if not q:
        return []
    parts = re.split(r"\bOR\b", q, flags=re.IGNORECASE)
    tokens = []
    for p in parts:
        t = p.strip().strip('"').strip("'").lower()
        if t:
            tokens.append(t)
    return tokens

def fetch_org_list() -> list[dict]:
    """
    CSV 헤더(예):
    - 조직명(또는 표시명), 검색어(옵션), 유형(ORG|PERSON 옵션),
      MUST_ALL, MUST_ANY, BLOCK (쉼표구분, 옵션)
    """
    sheet_url = os.environ.get("SHEET_CSV_URL", "").strip()
    if not sheet_url:
        raise RuntimeError("SHEET_CSV_URL env var is not set.")

    resp = requests.get(sheet_url, timeout=30)
    resp.raise_for_status()
    csv_text = resp.content.decode("utf-8", errors="replace")
    df = pd.read_csv(StringIO(csv_text))

    name_col = None
    for c in ["조직명", "표시명"]:
        if c in df.columns:
            name_col = c
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

        rows.append({
            "display": display,
            "query": query,
            "kind": kind,
            "must_all": must_all,
            "must_any": must_any,
            "block": block,
            "query_tokens": _query_tokens_from(query),
        })

    # 표시명+검색어 기준 중복 제거
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
    """
    1) query_tokens 중 하나라도 포함(있을 때)
    2) MUST_ALL 모두 포함
    3) MUST_ANY 중 최소 1개 포함(있을 때)
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

# =========================================================
# 본문 추출
# =========================================================
def fetch_article_text(url: str, timeout: int = 20) -> str:
    if not url:
        return ""
    # trafilatura 우선
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
                if text:
                    return text.strip()
        except Exception:
            pass
    # 폴백: requests + HTML 제거
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return strip_html(r.text)[:8000].strip()
    except Exception:
        return ""

# =========================================================
# 검색기
# =========================================================
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

# =========================================================
# 컨텍스트 분석 & 규칙 라벨
# =========================================================
def analyze_context_signals(title: str, summary: str, content: str, org_name: str) -> dict:
    text = f"{title} {summary} {content}".lower()
    org_lower = org_name.lower()
    org_involvement = "direct" if org_lower in text else "indirect"

    signals = {
        "direct": [],
        "context": [],
        "mon": [],
        "pos": [],
        "org": org_involvement,
        "score": 0,
        "conf": 0.5,
    }

    # 직접 부정: 높은 가중치
    for category, kws in DIRECT_NEGATIVE.items():
        found = [kw for kw in kws if kw in text]
        if found:
            signals["direct"].extend((category, kw) for kw in found)
            signals["score"] += 3 * len(found)

    # 상황 부정: 조직 연관성에 따라 가중치
    for category, kws in CONTEXTUAL_NEGATIVE.items():
        found = [kw for kw in kws if kw in text]
        if found:
            weight = 2 if org_involvement == "direct" else 1
            signals["context"].extend((category, kw) for kw in found)
            signals["score"] += weight * len(found)

    # 모니터링
    for category, kws in MONITORING_KEYWORDS.items():
        found = [kw for kw in kws if kw in text]
        if found:
            signals["mon"].extend((category, kw) for kw in found)
            signals["score"] += 1 * len(found)

    # 긍정: 점수 차감
    for category, kws in POSITIVE_KEYWORDS.items():
        found = [kw for kw in kws if kw in text]
        if found:
            signals["pos"].extend((category, kw) for kw in found)
            signals["score"] -= 1 * len(found)

    total = len(signals["direct"]) + len(signals["context"]) + len(signals["mon"]) + len(signals["pos"])
    if total:
        signals["conf"] = (min(0.9, 0.5 + 0.1 * total)
                           if org_involvement == "direct"
                           else min(0.7, 0.3 + 0.05 * total))
    return signals

def rule_label_from_signals(sig: dict) -> str:
    if sig["direct"] and sig["conf"] > 0.6:
        return "🔴"
    if sig["context"] and sig["org"] == "direct" and sig["score"] > 4:
        return "🔴"
    if sig["mon"] and sig["score"] > 2:
        return "🟡"
    if sig["pos"] and sig["score"] < 0:
        return "🔵"
    if sig["score"] <= 2:
        return "🟢"
    return "🟡"

# =========================================================
# LLM 라벨링
# =========================================================
def llm_enabled() -> bool:
    flag = os.environ.get("LLM_ENABLE", "").strip().lower()
    return (flag in {"1", "true", "yes", "on"}) and bool(os.environ.get("OPENAI_API_KEY", "")) and _HAS_OPENAI

def _format_signals_for_llm(sig: dict) -> str:
    parts = []
    if sig["direct"]:
        parts.append("직접부정: " + ", ".join(kw for _, kw in sig["direct"]))
    if sig["context"]:
        parts.append("상황부정: " + ", ".join(kw for _, kw in sig["context"]))
    if sig["mon"]:
        parts.append("모니터링: " + ", ".join(kw for _, kw in sig["mon"]))
    if sig["pos"]:
        parts.append("긍정: " + ", ".join(kw for _, kw in sig["pos"]))
    parts.append(f"조직연관성: {sig['org']} / 위험점수: {sig['score']}")
    return " | ".join(parts)

def enhanced_llm_label(display: str, title: str, summary: str, content: str, sig: dict) -> dict | None:
    if not llm_enabled():
        return None

    body = (content or "").strip()
    if len(body) > 4000:
        body = body[:4000]
    signal_text = _format_signals_for_llm(sig)

    prompt = f"""
당신은 기업 위기관리 전문 분석가입니다. 다음 뉴스가 '{display}'에 미치는 영향을 평가하세요.

[판단기준]
- positive(🔵): 명확한 긍정 영향(수상, 투자, 협력, 후원 등)
- neutral(🟢): 영향 적음(업계 일반 동향, 단순 언급)
- monitor(🟡): 주의 필요(조사/불확실/잠재 리스크)
- negative(🔴): 명확한 부정 영향(법적 문제, 사고, 사업 타격)

[원칙]
- 강한 부정 키워드가 있어도 '{display}'와 직접 관련 없으면 부정으로 판단하지 말 것
- 추측 금지, 기사 내용 기반 판단
- 애매하면 monitor

[조직] {display}
[제목] {title}
[요약] {summary or "없음"}
[본문(일부)] {body}
[자동 분석 신호] {signal_text}

JSON 응답만 출력:
{{
  "impact": "positive|neutral|monitor|negative",
  "confidence": 0.0-1.0,
  "primary_reason": "주요 판단 근거",
  "evidence": ["근거1","근거2"],
  "org_relevance": "direct|indirect|minimal"
}}
"""

    try:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=320,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"```json\s*|\s*```", "", raw)  # 코드블록 제거
        data = json.loads(raw)

        im = str(data.get("impact", "")).lower()
        if im not in {"positive", "neutral", "monitor", "negative"}:
            return None
        label_map = {"positive": "🔵", "neutral": "🟢", "monitor": "🟡", "negative": "🔴"}
        return {
            "label": label_map[im],
            "confidence": float(data.get("confidence", 0.5)),
            "primary_reason": data.get("primary_reason", ""),
            "org_relevance": data.get("org_relevance", ""),
            "raw": data,
        }
    except Exception as e:
        logging.error(f"LLM labeling failed: {e}")
        return None

# =========================================================
# 통합 라벨링
# =========================================================
def integrated_labeling(display: str, title: str, summary: str, content: str) -> dict:
    sig = analyze_context_signals(title, summary, content, display)
    rule = rule_label_from_signals(sig)
    llm = enhanced_llm_label(display, title, summary, content, sig)

    out = {
        "label": rule,
        "confidence": sig["conf"],
        "method": "rule",
        "signals": sig,
        "llm": llm,
    }

    # LLM 신뢰 가능하면 우선 적용
    if llm and llm["confidence"] > 0.6:
        # 직접 부정이 뚜렷한데 LLM이 🟢/🔵 → 보수적 오버라이드
        if sig["direct"] and llm["label"] in {"🟢", "🔵"} and sig["org"] == "direct":
            out["label"] = "🔴"
            out["method"] = "conservative_override"
        else:
            out["label"] = llm["label"]
            out["confidence"] = llm["confidence"]
            out["method"] = "llm"

    # 과도한 🔴 완화: 직접 부정 없고 점수 낮을때
    if out["label"] == "🔴" and not sig["direct"] and sig["score"] < 5:
        out["label"] = "🟡"
        out["method"] += "_moderated"

    return out

# =========================================================
# Slack
# =========================================================
def post_to_slack(lines: list[str]) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    channel = os.environ.get("SLACK_CHANNEL", "").strip()
    if not token or not channel:
        raise RuntimeError("SLACK_BOT_TOKEN or SLACK_CHANNEL missing.")
    client = WebClient(token=token)
    text = "\n".join(lines) if lines else "오늘은 신규로 감지된 기사가 없습니다."
    client.chat_postMessage(channel=channel, text=text)

# =========================================================
# 메인
# =========================================================
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # 주말 스킵(이중 안전장치)
    if now_kst().weekday() in (5, 6):
        logging.info("Weekend (Sat/Sun) – skipping run.")
        return

    window_from_utc, window_to_utc = compute_window_utc()
    logging.info("Window UTC: %s ~ %s", window_from_utc, window_to_utc)

    rows = fetch_org_list()
    logging.info("Loaded %d targets.", len(rows))

    # 캐시: 같은 기사(제목+본문) 재분석 방지
    analysis_cache: dict[str, dict] = {}
    all_lines: list[str] = []

    for idx, row in enumerate(rows, start=1):
        display = row["display"]
        query = row["query"]
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
        items = [it for it in items if is_relevant_by_rule(it["row_cfg"], it["title"], it.get("summary", ""))]
        logging.info("  after relevance: %d -> %d", before_rel, len(items))

        # 2) 기간 필터
        before_win = len(items)
        items = [it for it in items if it["published_at"] and window_from_utc <= it["published_at"] < window_to_utc]
        logging.info("  after window: %d -> %d", before_win, len(items))

        # 3) 최신순 + 제목 기준 dedup (도메인 무관)
        items.sort(key=lambda x: x["published_at"], reverse=True)
        seen_titles = set(); uniq = []
        for it in items:
            tk = norm_title(it["title"])
            if tk and it["url"] and tk not in seen_titles:
                uniq.append(it); seen_titles.add(tk)

        # 4) 라벨링
        for art in uniq:
            content = fetch_article_text(art["url"])

            # 캐시 체크
            ck = content_hash(art["title"], content)
            if ck in analysis_cache:
                result = analysis_cache[ck].copy()
                logging.info("  Cache hit: %s", art["title"][:60])
            else:
                result = integrated_labeling(art["display"], art["title"], art.get("summary", ""), content)
                analysis_cache[ck] = result
                logging.info("  Analyzed: %s -> %s (method=%s, conf=%.2f)",
                             art["title"][:60], result["label"], result["method"], result["confidence"])

            # Slack 라인 생성
            label = result["label"]
            conf = result["confidence"]
            conf_mark = "!" if conf > 0.8 else ("?" if conf < 0.5 else "")
            src = art["source"]
            when_str = to_kst_str(art["published_at"])

            extra = []
            llm = result.get("llm") or {}
            if llm.get("primary_reason"):
                extra.append(f"이유:{llm['primary_reason']}")
            sig = result.get("signals", {})
            if sig.get("direct"):
                extra.append(f"직접위험:{len(sig['direct'])}")
            if sig.get("pos"):
                extra.append(f"긍정신호:{len(sig['pos'])}")

            extra_txt = f" ({', '.join(extra)})" if extra else ""
            line = f"[{art['display']}] <{art['url']}|{art['title']}> ({src})({when_str}) [{label}{conf_mark}]{extra_txt}"
            all_lines.append(line)

    # 결과 라벨 분포 로그
    counts = defaultdict(int)
    for ln in all_lines:
        for emo in ("🔴", "🟡", "🟢", "🔵"):
            if emo in ln:
                counts[emo] += 1
                break
    logging.info("Label distribution: 🔴%d 🟡%d 🟢%d 🔵%d",
                 counts["🔴"], counts["🟡"], counts["🟢"], counts["🔵"])

    # Slack 전송
    post_to_slack(all_lines)
    logging.info("Posted %d lines to Slack.", len(all_lines))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error")
        raise
