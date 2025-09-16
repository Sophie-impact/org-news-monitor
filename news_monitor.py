#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
News Monitor – 개선된 라벨링 시스템 (Claude + Gemini 피드백 반영)
- 컨텍스트 기반 키워드 분석으로 오탐 감소
- LLM 프롬프트 개선으로 일관성 향상
- 다층 검증 + 보수적 조정으로 과도한 '🔴' 방지
- 캐싱으로 중복 분석 방지
- 주말(토/일) 자동 스킵
- 카카오/브라이언임팩트/김범수는 중복 허용, 그 외 조직은 같은 제목(정규화) 1건만 대표 표시
"""

from __future__ import annotations

import os
import re
import html
import time
import json
import hashlib
import logging
import requests
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
from zoneinfo import ZoneInfo
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError  # noqa: F401
import tldextract
import trafilatura
from collections import defaultdict

# --- LLM (OpenAI) ---
try:
    import openai  # pip install openai>=1.40.0
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

KST = ZoneInfo("Asia/Seoul")

# =========================
# 개선된 키워드 시스템
# =========================

DIRECT_NEGATIVE = {
    "법적": ["횡령", "배임", "사기", "고발", "기소", "구속", "수사", "압수수색", "특검", "징역", "실형"],
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
    "성과": ["수상", "선정", "혁신", "신기록", "최대", "달성", "성과", "흑자전환"],
    "성장": ["투자유치", "시리즈", "상승", "증가", "호조", "확대", "진출", "성장"],
    "협력": ["협력", "파트너십", "mou", "계약", "수주", "제휴", "연합"],
    "사회공헌": ["후원", "지원", "기부", "기증", "기탁", "장학금", "봉사"],
}

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

def content_hash(title: str, content: str) -> str:
    combined = f"{title}:{content[:1000]}"
    return hashlib.md5(combined.encode()).hexdigest()

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
# 컨텍스트 분석
# =========================
def analyze_context_signals(title: str, summary: str, content: str, org_name: str) -> dict:
    full_text = f"{title} {summary} {content}".lower()
    org_lower = org_name.lower()
    org_mentioned = org_lower in full_text

    signals = {
        "direct_negative": [],
        "contextual_negative": [],
        "monitoring": [],
        "positive": [],
        "org_involvement": "direct" if org_mentioned else "indirect",
        "severity_score": 0,
        "confidence": 0.5,
    }

    # 직접 부정(강 부정)
    for category, keywords in DIRECT_NEGATIVE.items():
        found = [kw for kw in keywords if kw in full_text]
        if found:
            signals["direct_negative"].extend([(category, kw) for kw in found])
            signals["severity_score"] += len(found) * 3

    # 상황 부정(맥락 의존)
    for category, keywords in CONTEXTUAL_NEGATIVE.items():
        found = [kw for kw in keywords if kw in full_text]
        if found:
            weight = 2 if _is_org_related_context(full_text, found, org_lower) else 1
            signals["contextual_negative"].extend([(category, kw) for kw in found])
            signals["severity_score"] += len(found) * weight

    # 모니터링 신호
    for category, keywords in MONITORING_KEYWORDS.items():
        found = [kw for kw in keywords if kw in full_text]
        if found:
            signals["monitoring"].extend([(category, kw) for kw in found])
            signals["severity_score"] += len(found)

    # 긍정 신호
    for category, keywords in POSITIVE_KEYWORDS.items():
        found = [kw for kw in keywords if kw in full_text]
        if found:
            signals["positive"].extend([(category, kw) for kw in found])
            signals["severity_score"] -= len(found)

    # 신뢰도 산정
    total = (
        len(signals["direct_negative"])
        + len(signals["contextual_negative"])
        + len(signals["monitoring"])
        + len(signals["positive"])
    )
    if total > 0:
        if signals["org_involvement"] == "direct":
            signals["confidence"] = min(0.9, 0.5 + total * 0.1)
        else:
            signals["confidence"] = min(0.7, 0.3 + total * 0.05)

    return signals

def _is_org_related_context(text: str, keywords: list[str], org_name: str) -> bool:
    if not org_name:
        return False
    org_positions = [m.start() for m in re.finditer(re.escape(org_name), text)]
    for kw in keywords:
        kw_positions = [m.start() for m in re.finditer(re.escape(kw), text)]
        for org_pos in org_positions:
            for kw_pos in kw_positions:
                if abs(org_pos - kw_pos) <= 100:  # 100자 이내
                    return True
    return False

def enhanced_rule_label(signals: dict) -> str:
    score, confidence = signals["severity_score"], signals["confidence"]
    if signals["direct_negative"] and confidence > 0.6:
        return "🔴"
    if signals["contextual_negative"] and signals["org_involvement"] == "direct" and score > 4:
        return "🔴"
    if signals["monitoring"] and score > 2:
        return "🟡"
    if signals["positive"] and score < 0:
        return "🔵"
    if score <= 2:
        return "🟢"
    return "🟡"

# =========================
# LLM 프롬프트 (강화)
# =========================
def enhanced_llm_label(display_name: str, title: str, summary: str, content: str, signals: dict) -> dict | None:
    if not llm_enabled():
        return None

    body = (content or "").strip()
    if len(body) > 4000:
        body = body[:4000]

    signal_summary = _format_signals_for_llm(signals)

    try:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

        prompt = f"""당신은 기업 위기관리 전문 분석가입니다. 다음 뉴스 기사가 '{display_name}'에 미치는 영향을 평가하세요.

평가 기준:
- positive(🔵): 명확한 긍정적 영향 (수상/투자/성과/협력 등)
- neutral(🟢): 영향도가 낮거나 중립
- monitor(🟡): 조사/검토/불확실 등 잠재 리스크
- negative(🔴): 법적 문제/사고/직접 비판/사업 타격 등 명확한 부정

원칙:
1) 업계 일반론보다 해당 조직에 대한 직접 영향 우선
2) 과도한 추정 금지, 기사 내용 기반 판단
3) 애매하면 'monitor', 명확히 긍정이면 'positive'

조직명: {display_name}
제목: {title}
요약: {summary or "없음"}
본문(일부): {body}

자동 분석 요약: {signal_summary}

JSON만 반환:
{{
  "impact": "positive|neutral|monitor|negative",
  "confidence": 0.0-1.0,
  "primary_reason": "주요 판단 근거 (한 줄)",
  "evidence": ["근거1", "근거2"],
  "org_relevance": "direct|indirect|minimal"
}}"""

        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = _safe_load_json(raw)
        if not data or "impact" not in data:
            return None

        impact = str(data.get("impact", "")).lower()
        if impact not in ["positive", "neutral", "monitor", "negative"]:
            return None

        conf = float(data.get("confidence", 0.5))
        label_map = {"positive": "🔵", "neutral": "🟢", "monitor": "🟡", "negative": "🔴"}

        return {
            "label": label_map[impact],
            "confidence": conf,
            "raw": data,
            "primary_reason": data.get("primary_reason", ""),
            "org_relevance": data.get("org_relevance", "unknown"),
        }
    except Exception as e:
        logging.error(f"LLM labeling failed: {e}")
        return None

def _format_signals_for_llm(signals: dict) -> str:
    parts = []
    if signals["direct_negative"]:
        parts.append(f"직접적 부정: {', '.join([kw for _, kw in signals['direct_negative']])}")
    if signals["contextual_negative"]:
        parts.append(f"상황적 부정: {', '.join([kw for _, kw in signals['contextual_negative']])}")
    if signals["monitoring"]:
        parts.append(f"모니터링: {', '.join([kw for _, kw in signals['monitoring']])}")
    if signals["positive"]:
        parts.append(f"긍정: {', '.join([kw for _, kw in signals['positive']])}")
    parts.append(f"조직 연관성: {signals['org_involvement']}")
    parts.append(f"위험도 점수: {signals['severity_score']}")
    return " | ".join(parts)

def _safe_load_json(s: str):
    try:
        s = re.sub(r'```json\s*|\s*```', '', s)
        return json.loads(s)
    except Exception:
        return None

# =========================
# 통합 라벨링
# =========================
def integrated_labeling(display_name: str, title: str, summary: str, content: str) -> dict:
    signals = analyze_context_signals(title, summary, content, display_name)
    rule_label = enhanced_rule_label(signals)
    llm_result = enhanced_llm_label(display_name, title, summary, content, signals)

    result = {
        "label": rule_label,
        "confidence": signals["confidence"],
        "method": "rule_based",
        "signals": signals,
        "llm_result": llm_result or {},  # 항상 dict
    }

    # LLM이 충분히 확신 있으면 우선
    if llm_result and llm_result.get("confidence", 0) > 0.6:
        if signals["direct_negative"] and llm_result["label"] in {"🟢", "🔵"} and signals["org_involvement"] == "direct":
            result["label"] = "🔴"
            result["method"] = "conservative_override"
        else:
            result["label"] = llm_result["label"]
            result["confidence"] = llm_result["confidence"]
            result["method"] = "llm_primary"

    # 과도한 🔴 완화
    if result["label"] == "🔴" and not signals["direct_negative"] and signals["severity_score"] < 5:
        result["label"] = "🟡"
        result["method"] += "_moderated"

    return result

# =========================
# 시트 로딩
# =========================
def _split_list(val) -> list[str]:
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [x.strip().lower() for x in str(val).split(",") if x.strip()]

def _query_tokens_from(q: str) -> list[str]:
    if not q:
        return []
    parts = re.split(r"\bOR\b", q, flags=re.IGNORECASE)
    return [p.strip().strip('"').strip("'").lower() for p in parts if p.strip()]

def fetch_org_list() -> list[dict]:
    sheet_url = os.environ.get("SHEET_CSV_URL", "").strip()
    if not sheet_url:
        raise RuntimeError("SHEET_CSV_URL not set")

    resp = requests.get(sheet_url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.content.decode("utf-8", errors="replace")))

    name_col = None
    for cand in ["조직명", "표시명"]:
        if cand in df.columns:
            name_col = cand
            break
    if not name_col:
        raise RuntimeError("CSV에 '조직명'/'표시명' 필요")

    rows = []
    for _, r in df.iterrows():
        display = str(r[name_col]).strip()
        if not display or display.lower() == "nan":
            continue
        query = str(r.get("검색어", "")).strip() or display
        kind = str(r.get("유형", "ORG")).strip().upper() or "ORG"
        rows.append(
            {
                "display": display,
                "query": query,
                "kind": kind,
                "must_all": _split_list(r.get("MUST_ALL", "")),
                "must_any": _split_list(r.get("MUST_ANY", "")),
                "block": _split_list(r.get("BLOCK", "")),
                "query_tokens": _query_tokens_from(query),
            }
        )

    uniq, seen = [], set()
    for it in rows:
        key = (it["display"], it["query"])
        if key not in seen:
            uniq.append(it)
            seen.add(key)
    return uniq

# =========================
# 본문 추출 / 검색
# =========================
def fetch_article_text(url: str, timeout: int = 20) -> str:
    if not url:
        return ""
    try:
        dl = trafilatura.fetch_url(url, no_ssl=True, timeout=timeout)
        if dl:
            txt = trafilatura.extract(
                dl,
                include_comments=False,
                include_tables=False,
                include_formatting=False,
                favor_recall=True,
                deduplicate=True,
            ) or ""
            return txt.strip()
    except Exception:
        pass
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return strip_html(r.text)[:8000].strip()
    except Exception:
        return ""

def search_naver(query: str, display: int = 20) -> list[dict]:
    cid, cs = os.environ.get("NAVER_CLIENT_ID", ""), os.environ.get("NAVER_CLIENT_SECRET", "")
    if not cid or not cs:
        return []
    endpoint = "https://openapi.naver.com/v1/search/news.json"
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": cs}
    params = {"query": query, "display": display, "start": 1, "sort": "date"}
    try:
        r = requests.get(endpoint, headers=headers, params=params, timeout=20)
        r.raise_for_status()
        items = r.json().get("items", [])
        res = []
        for it in items:
            title = strip_html(it.get("title"))
            url = it.get("originallink") or it.get("link")
            pub = parse_datetime(it.get("pubDate"))
            if not url or not title:
                continue
            src = domain_from_url(url) or "naver"
            res.append(
                {
                    "title": title,
                    "url": url,
                    "source": src,
                    "published_at": pub,
                    "origin": "naver",
                    "summary": strip_html(it.get("description", "")),
                }
            )
        return res
    except Exception:
        return []

def search_newsapi(query: str, window_from_utc: datetime, window_to_utc: datetime, language: str = "ko") -> list[dict]:
    key = os.environ.get("NEWSAPI_KEY", "")
    if not key:
        return []
    endpoint = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
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
        res = []
        for a in arts:
            title = strip_html(a.get("title"))
            url = a.get("url")
            pub = parse_datetime(a.get("publishedAt"))
            src = (a.get("source") or {}).get("name") or domain_from_url(url)
            if not url or not title:
                continue
            res.append(
                {
                    "title": title,
                    "url": url,
                    "source": src,
                    "published_at": pub,
                    "origin": "newsapi",
                    "summary": strip_html(a.get("description") or a.get("content") or ""),
                }
            )
        return res
    except Exception:
        return []

# =========================
# 관련성 필터
# =========================
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
    must_any = row_cfg.get("must_any", [])
    if must_any and not _contains_any(text, must_any):
        return False
    if not _contains_none(text, row_cfg.get("block", [])):
        return False
    return True

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
# 조직별 대표 1건 dedup (예외: 카카오/브라이언임팩트/김범수)
# =========================
EXEMPT_DEDUP = {"카카오", "브라이언임팩트", "김범수"}

def collapse_by_display(items: list[dict]) -> list[dict]:
    """
    같은 display(조직)에서 제목(정규화)이 같은 기사가 여러 매체에 있을 경우
    최신 1건만 남긴다. 단, EXEMPT_DEDUP에 있는 display는 제외.
    """
    by_disp: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_disp[it["display"]].append(it)

    final = []
    for display, group in by_disp.items():
        if display in EXEMPT_DEDUP:
            final.extend(group)
            continue
        # 제목 키로 최신 하나씩
        latest_by_title: dict[str, dict] = {}
        for it in group:
            key = norm_title(it["title"])
            old = latest_by_title.get(key)
            if (old is None) or (it.get("published_at") and old.get("published_at") and it["published_at"] > old["published_at"]):
                latest_by_title[key] = it
        final.extend(latest_by_title.values())
    return final

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

    # 중복 분석 캐시
    analysis_cache: dict[str, dict] = {}

    collected: list[dict] = []
    for idx, row in enumerate(rows, start=1):
        display = row["display"]
        query = row["query"]
        logging.info("(%d/%d) Searching: %s | %s", idx, len(rows), display, query)

        naver_items = search_naver(query, display=20)
        time.sleep(0.25)
        newsapi_items = search_newsapi(query, window_from_utc, window_to_utc, language="ko")
        logging.info("  raw: naver=%d, newsapi=%d", len(naver_items), len(newsapi_items))

        items = []
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
        items = [it for it in items if it.get("published_at") and window_from_utc <= it["published_at"] < window_to_utc]
        logging.info("  after window: %d -> %d", before_win, len(items))

        # 3) 최신순 정렬 + 같은 제목 dedup (도메인 무관)
        items.sort(key=lambda x: x["published_at"], reverse=True)
        seen_title = set()
        uniq = []
        for it in items:
            key = norm_title(it["title"])
            if key and it.get("url") and key not in seen_title:
                uniq.append(it)
                seen_title.add(key)

        collected.extend(uniq)

    # 4) 조직별 대표 1건(예외 조직 제외)
    collected = collapse_by_display(collected)

    # 5) 기사별 분석 + 라벨링
    lines: list[str] = []
    for art in sorted(collected, key=lambda x: x["published_at"] or datetime(1970,1,1,tzinfo=timezone.utc), reverse=True):
        content = fetch_article_text(art["url"])

        # 캐시 체크
        cache_key = content_hash(art["title"], content)
        if cache_key in analysis_cache:
            result = analysis_cache[cache_key].copy()
            logging.info("  Cache hit: %s", art["title"][:50])
        else:
            result = integrated_labeling(art["display"], art["title"], art.get("summary", ""), content)
            analysis_cache[cache_key] = result
            logging.info("  Analyzed: %s -> %s (method=%s, conf=%.2f)",
                         art["title"][:50], result["label"], result["method"], result["confidence"])

        label = result["label"]
        confidence = float(result.get("confidence", 0.5))

        conf_mark = "!" if confidence > 0.8 else ("?" if confidence < 0.5 else "")
        src = art["source"]
        when_str = to_kst_str(art["published_at"])

        extra_bits = []
        llm = result.get("llm_result") or {}
        pr = llm.get("primary_reason")
        if pr:
            extra_bits.append(f"이유: {pr}")
        sig = result.get("signals") or {}
        if sig.get("direct_negative"):
            extra_bits.append(f"직접위험:{len(sig['direct_negative'])}")
        if sig.get("positive"):
            extra_bits.append(f"긍정신호:{len(sig['positive'])}")
        extra = f" ({', '.join(extra_bits)})" if extra_bits else ""

        line = f"[{art['display']}] <{art['url']}|{art['title']}> ({src})({when_str}) [{label}{conf_mark}]{extra}"
        lines.append(line)

    # 요약 로깅
    label_counts = defaultdict(int)
    for line in lines:
        for emoji in ["🔴", "🟡", "🟢", "🔵"]:
            if emoji in line:
                label_counts[emoji] += 1
                break
    logging.info("Label distribution: 🔴%d 🟡%d 🟢%d 🔵%d",
                 label_counts["🔴"], label_counts["🟡"], label_counts["🟢"], label_counts["🔵"])

    # Slack 전송
    post_to_slack(lines)
    logging.info("Posted %d lines to Slack.", len(lines))


def llm_enabled() -> bool:
    flag = os.environ.get("LLM_ENABLE", "").strip().lower()
    enabled = flag in {"1", "true", "yes", "on"}
    return enabled and bool(os.environ.get("OPENAI_API_KEY", "").strip()) and _HAS_OPENAI


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error")
        raise
