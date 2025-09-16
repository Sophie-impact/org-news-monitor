#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
News Monitor – 컨텍스트+LLM 기반 라벨링 / 주말 스킵 / 대표 3건 제한(일부 제외)
요청 반영:
- 예방/상시활동 안전신호
- 규칙 라벨러의 과도한 🔴 완화
- LLM 통합 단계에서 긍정우세+직접부정없음 완화
- 조직별 대표 최대 3건(카카오/브임/김범수 제외)
- 슬랙 보조정보 제거
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
from typing import Optional, Tuple, List, Dict, Any

# --- LLM (OpenAI) ---
try:
    import openai  # pip install openai>=1.40.0
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

KST = ZoneInfo("Asia/Seoul")

# 대표 3건 제한 제외 대상
PRIORITY_ORGS = {"카카오", "브라이언임팩트", "김범수"}

# =========================
# 키워드(컨텍스트 분류)
# =========================
DIRECT_NEGATIVE = {
    "법적": ["횡령", "배임", "사기", "고발", "기소", "구속", "수사", "압수수색", "특검", "징역", "실형"],
    "사업": ["리콜", "결함", "파산", "부도", "영업정지", "사업중단", "퇴출"],
    "안전": ["사망", "부상", "폭발", "화재", "추락", "유출", "해킹", "랜섬웨어", "개인정보유출"]
}
CONTEXTUAL_NEGATIVE = {
    "경영": ["적자", "손실", "감소", "하락", "부실"],
    "규제": ["제재", "벌금", "과징금", "징계", "처분"],
    "논란": ["논란", "비판", "갑질", "불법", "위법", "부정"]
}
MONITORING_KEYWORDS = {
    "조사": ["의혹", "조사", "심사", "검토", "국감", "감사", "불확실성", "연기", "지연", "유예", "잠정", "검토중", "우려", "경고", "리스크", "변동성", "관심", "주시"]
}
# 예방/상시 활동 → 안전 신호(강한 긍정 가중)
SAFE_ACTIVITY_KEYWORDS = [
    "예방", "정기점검", "상시점검", "안전점검", "실태 점검", "안전 교육", "안전교육",
    "훈련", "모의훈련", "리허설", "캠페인", "자율점검", "점검 실시", "점검 진행", "안전 보강",
    "보건관리", "사고 예방", "특별 점검", "현장 점검", "집중 점검"
]
POSITIVE_KEYWORDS = {
    "성과": ["수상", "선정", "혁신", "신기록", "최대", "달성", "성과", "흑자전환"],
    "성장": ["투자유치", "시리즈", "상승", "증가", "호조", "확대", "진출", "성장"],
    "협력": ["협력", "파트너십", "mou", "계약", "수주", "제휴", "연합"],
    "사회공헌": ["후원", "지원", "기부", "기증", "기탁", "장학금", "봉사"]
}

# =========================
# 공통 유틸
# =========================
def now_kst() -> datetime:
    return datetime.now(tz=KST)

def parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    try:
        dt = dtparser.parse(dt_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

def to_kst_str(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    return dt.astimezone(KST).strftime("%Y-%m-%d %H:%M")

def strip_html(text: Optional[str]) -> str:
    text = html.unescape(text or "")
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)

def domain_from_url(url: Optional[str]) -> str:
    if not url:
        return ""
    try:
        ext = tldextract.extract(url)
        parts = [p for p in [ext.domain, ext.suffix] if p]
        return ".".join(parts) if parts else ""
    except Exception:
        return ""

def norm_title(t: Optional[str]) -> str:
    t = strip_html(t or "").lower()
    t = re.sub(r"[\[\]【】()（）〈〉<>『』「」]", " ", t)
    t = re.sub(r"[^\w가-힣\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def content_hash(title: str, content: str) -> str:
    combined = f"{title}:{content[:1000]}"
    return hashlib.md5(combined.encode()).hexdigest()

# =========================
# 조회 구간
# =========================
def compute_window_utc(now: Optional[datetime] = None) -> Tuple[datetime, datetime]:
    """
    09:00 KST 기준 실행
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
# 시트 로더
# =========================
def _split_list(val) -> List[str]:
    if pd.isna(val) or str(val).strip() == "":
        return []
    return [x.strip().lower() for x in str(val).split(",") if x.strip()]

def _query_tokens_from(q: str) -> List[str]:
    if not q:
        return []
    parts = re.split(r'\bOR\b', q, flags=re.IGNORECASE)
    tokens = []
    for p in parts:
        t = p.strip().strip('"').strip("'").lower()
        if t:
            tokens.append(t)
    return tokens

def fetch_org_list() -> List[Dict[str, Any]]:
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

    rows: List[Dict[str, Any]] = []
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

    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return strip_html(r.text)[:8000].strip()
    except Exception:
        return ""

# =========================
# 검색기
# =========================
def search_naver(query: str, display: int = 20) -> List[Dict[str, Any]]:
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
                   language: str = "ko") -> List[Dict[str, Any]]:
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

# =========================
# 관련성 필터(행 규칙)
# =========================
def _contains_all(text: str, toks: List[str]) -> bool:
    return all(t in text for t in toks) if toks else True

def _contains_any(text: str, toks: List[str]) -> bool:
    return any(t in text for t in toks) if toks else True

def _contains_none(text: str, toks: List[str]) -> bool:
    return all(t not in text for t in toks) if toks else True

def is_relevant_by_rule(row_cfg: Dict[str, Any], title: str, summary: str) -> bool:
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
# 컨텍스트 분석 → 규칙 라벨
# =========================
def _is_org_related_context(text: str, keywords: List[str], org_name: str) -> bool:
    if not org_name:
        return False
    org = org_name.lower()
    org_positions = [m.start() for m in re.finditer(re.escape(org), text)]
    for kw in keywords:
        for m in re.finditer(re.escape(kw), text):
            for pos in org_positions:
                if abs(pos - m.start()) <= 100:
                    return True
    return False

def analyze_context_signals(title: str, summary: str, content: str, org_name: str) -> Dict[str, Any]:
    full_text = f"{title} {summary} {content}".lower()
    org_mentioned = org_name.lower() in full_text if org_name else False

    signals = {
        "direct_negative": [],
        "contextual_negative": [],
        "monitoring": [],
        "positive": [],
        "safe_activity": [],
        "org_involvement": "direct" if org_mentioned else "indirect",
        "severity_score": 0,
        "confidence": 0.5
    }

    # 직접 부정(가중치 3)
    for category, kws in DIRECT_NEGATIVE.items():
        found = [kw for kw in kws if kw in full_text]
        if found:
            signals["direct_negative"].extend([(category, kw) for kw in found])
            signals["severity_score"] += len(found) * 3

    # 상황 부정(조직 근접 시 가중치 2, 아니면 1)
    for category, kws in CONTEXTUAL_NEGATIVE.items():
        found = [kw for kw in kws if kw in full_text]
        if found:
            weight = 2 if _is_org_related_context(full_text, found, org_name.lower()) else 1
            signals["contextual_negative"].extend([(category, kw) for kw in found])
            signals["severity_score"] += len(found) * weight

    # 모니터링 키워드(가중치 1)
    for category, kws in MONITORING_KEYWORDS.items():
        found = [kw for kw in kws if kw in full_text]
        if found:
            signals["monitoring"].extend([(category, kw) for kw in found])
            signals["severity_score"] += len(found) * 1

    # 예방/상시 활동(강한 긍정 가중치 -2)
    sa_found = [kw for kw in SAFE_ACTIVITY_KEYWORDS if kw in full_text]
    if sa_found:
        signals["safe_activity"].extend(sa_found)
        signals["severity_score"] -= len(sa_found) * 2

    # 일반 긍정(가중치 -1)
    for category, kws in POSITIVE_KEYWORDS.items():
        found = [kw for kw in kws if kw in full_text]
        if found:
            signals["positive"].extend([(category, kw) for kw in found])
            signals["severity_score"] -= len(found) * 1

    # 신뢰도
    total = (len(signals["direct_negative"]) + len(signals["contextual_negative"])
             + len(signals["monitoring"]) + len(signals["positive"]) + len(signals["safe_activity"]))
    if total > 0:
        if signals["org_involvement"] == "direct":
            signals["confidence"] = min(0.9, 0.5 + total * 0.1)
        else:
            signals["confidence"] = min(0.7, 0.3 + total * 0.05)

    return signals

def enhanced_rule_label(signals: Dict[str, Any]) -> str:
    score = signals["severity_score"]
    conf = signals["confidence"]

    # 긍정/안전 신호 우세 판단
    pos_cnt = len(signals["positive"]) + len(signals["safe_activity"])
    neg_cnt = len(signals["direct_negative"]) + len(signals["contextual_negative"]) + len(signals["monitoring"])
    positive_dominant = pos_cnt > neg_cnt

    # 🔴 조건 완화: 직접부정 + 직접연관 + 점수≥6 + 긍정우세 아님
    if (signals["direct_negative"]
        and signals["org_involvement"] == "direct"
        and score >= 6
        and not positive_dominant
        and conf > 0.55):
        return "🔴"

    # 상황부정으로만 🔴는 더 보수적으로
    if (signals["contextual_negative"]
        and signals["org_involvement"] == "direct"
        and score > 7
        and not positive_dominant):
        return "🔴"

    # 명백한 긍정(점수<0 또는 안전신호 존재)
    if positive_dominant and score <= 1:
        return "🔵"

    # 모니터링 위주
    if signals["monitoring"] and score > 2 and not positive_dominant:
        return "🟡"

    # 기본 중립
    if score <= 2 or positive_dominant:
        return "🟢"

    return "🟡"

# =========================
# LLM(JSON) 라벨
# =========================
IMPACT_MAP = {"positive": "🔵", "neutral": "🟢", "monitor": "🟡", "negative": "🔴"}

def _format_signals_for_llm(signals: Dict[str, Any]) -> str:
    parts = []
    if signals.get("direct_negative"):
        parts.append("직접부정: " + ", ".join([kw for _, kw in signals["direct_negative"]]))
    if signals.get("contextual_negative"):
        parts.append("상황부정: " + ", ".join([kw for _, kw in signals["contextual_negative"]]))
    if signals.get("monitoring"):
        parts.append("모니터: " + ", ".join([kw for _, kw in signals["monitoring"]]))
    if signals.get("safe_activity"):
        parts.append("예방/상시활동: " + ", ".join(signals["safe_activity"]))
    if signals.get("positive"):
        parts.append("긍정: " + ", ".join([kw for _, kw in signals["positive"]]))
    parts.append(f"연관성: {signals.get('org_involvement','indirect')}")
    parts.append(f"위험점수: {signals.get('severity_score',0)}")
    return " | ".join(parts) if parts else "특이 신호 없음"

def _safe_load_json(s: str) -> Optional[Dict[str, Any]]:
    try:
        s = re.sub(r'```json\s*|\s*```', '', s)
        return json.loads(s)
    except Exception:
        try:
            m = re.search(r'\{[^}]*"impact"\s*:\s*"([^"]+)"[^}]*\}', s)
            if m:
                return {"impact": m.group(1), "confidence": 0.5}
        except Exception:
            pass
        return None

def llm_enabled() -> bool:
    flag = os.environ.get("LLM_ENABLE", "").strip().lower()
    enabled = flag in {"1", "true", "yes", "on"}
    return enabled and bool(os.environ.get("OPENAI_API_KEY", "").strip()) and _HAS_OPENAI

def enhanced_llm_label(display_name: str, title: str, summary: str, content: str,
                       signals: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not llm_enabled():
        return None

    body = (content or "").strip()
    if len(body) > 4000:
        body = body[:4000]

    signal_summary = _format_signals_for_llm(signals)

    try:
        client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"].strip())
        model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

        prompt = f"""당신은 기업 위기관리 전문 분석가입니다. 다음 뉴스가 '조직'에 미치는 영향을 평가하세요.

[평가기준]
- positive(🔵): 수상/투자/성과/협력/예방·상시안전활동 등 명확한 호재
- neutral(🟢): 영향 미미, 단순 언급/일반 동향
- monitor(🟡): 조사/검토/불확실 등 잠재 리스크
- negative(🔴): 법적·사고 등 직접 악재/사업 타격

[원칙]
1) 조직에 대한 '직접 영향' 우선, 업계 일반론은 가중치 낮음
2) 기사에 명시된 사실 기반, 과도한 추측 금지
3) 애매하면 monitor, 명백히 긍정은 positive
4) 예방/상시 안전활동은 긍정 또는 중립으로 분류(부정 아님)

[조직명] {display_name}
[제목] {title}
[요약] {summary or "없음"}

[본문(일부)]
{body}

[자동 분석 요약]
{signal_summary}

JSON으로만 답변:
{{
  "impact": "positive|neutral|monitor|negative",
  "confidence": 0.0,
  "primary_reason": "한 줄 근거",
  "evidence": ["근거1","근거2"],
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
        if not data:
            return None

        impact = str(data.get("impact", "")).lower()
        if impact not in IMPACT_MAP:
            return None

        conf = float(data.get("confidence", 0.5))
        return {
            "label": IMPACT_MAP[impact],
            "confidence": conf,
            "raw": data,
            "primary_reason": data.get("primary_reason", ""),
            "org_relevance": data.get("org_relevance", "unknown"),
        }
    except Exception as e:
        logging.error(f"LLM labeling failed: {e}")
        return None

# =========================
# 통합 라벨링
# =========================
def integrated_labeling(display_name: str, title: str, summary: str, content: str) -> Dict[str, Any]:
    signals = analyze_context_signals(title, summary, content, display_name)
    rule_lab = enhanced_rule_label(signals)
    llm_res = enhanced_llm_label(display_name, title, summary, content, signals)

    result = {
        "label": rule_lab,
        "confidence": signals["confidence"],
        "method": "rule_based",
        "signals": signals,
        "llm_result": llm_res
    }

    # LLM이 신뢰도 충분하면 우선 적용
    if llm_res and llm_res.get("confidence", 0.0) > 0.6:
        if (signals["direct_negative"]
            and llm_res["label"] in {"🔵", "🟢"}
            and signals["org_involvement"] == "direct"):
            result["label"] = "🔴"
            result["method"] = "conservative_override"
        else:
            result["label"] = llm_res["label"]
            result["confidence"] = llm_res.get("confidence", result["confidence"])
            result["method"] = "llm_primary"

    # 보수 완화: 직접부정 없고 긍정 우세면 한 단계 완화
    pos_cnt = len(signals["positive"]) + len(signals["safe_activity"])
    neg_cnt = len(signals["direct_negative"]) + len(signals["contextual_negative"]) + len(signals["monitoring"])
    positive_dominant = (pos_cnt > neg_cnt)

    if not signals["direct_negative"] and positive_dominant:
        if result["label"] == "🔴":
            result["label"] = "🟡"
            result["method"] += "_pos_dominant_relief"
        elif result["label"] == "🟡":
            result["label"] = "🟢"
            result["method"] += "_pos_dominant_relief"

    # 과도한 🔴 완화(직접부정 없고 점수 낮음)
    if (result["label"] == "🔴"
            and not signals["direct_negative"]
            and signals["severity_score"] < 5):
        result["label"] = "🟡"
        result["method"] += "_moderated"

    return result

# =========================
# Slack
# =========================
def post_to_slack(lines: List[str]) -> None:
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
    if now_kst().weekday() in (5, 6):
        logging.info("Weekend (Sat/Sun) – skipping run.")
        return

    window_from_utc, window_to_utc = compute_window_utc()
    logging.info("Window UTC: %s ~ %s", window_from_utc, window_to_utc)

    rows = fetch_org_list()
    logging.info("Loaded %d targets.", len(rows))

    analysis_cache: Dict[str, Dict[str, Any]] = {}
    all_lines: List[str] = []

    for idx, row in enumerate(rows, start=1):
        display = row["display"]
        query = row["query"]
        logging.info("(%d/%d) Searching: %s | %s", idx, len(rows), display, query)

        naver_items = search_naver(query, display=20)
        time.sleep(0.25)
        newsapi_items = search_newsapi(query, window_from_utc, window_to_utc, language="ko")
        logging.info("  raw: naver=%d, newsapi=%d", len(naver_items), len(newsapi_items))

        items: List[Dict[str, Any]] = []
        for it in (naver_items + newsapi_items):
            it["display"] = display
            it["row_cfg"] = row
            items.append(it)

        # 관련성 필터
        before_rel = len(items)
        items = [it for it in items if is_relevant_by_rule(it["row_cfg"], it["title"], it.get("summary", ""))]
        logging.info("  after relevance: %d -> %d", before_rel, len(items))

        # 기간 필터
        before_win = len(items)
        items = [it for it in items if it["published_at"] and window_from_utc <= it["published_at"] < window_to_utc]
        logging.info("  after window: %d -> %d", before_win, len(items))

        # 최신순 + 제목 dedup
        items.sort(key=lambda x: x["published_at"], reverse=True)
        seen_titles = set()
        uniq: List[Dict[str, Any]] = []
        for it in items:
            title_key = norm_title(it["title"])
            if title_key and it["url"] and title_key not in seen_titles:
                uniq.append(it)
                seen_titles.add(title_key)

        # 대표 건수 제한(우선순위 제외)
        if display not in PRIORITY_ORGS:
            uniq = uniq[:3]

        for art in uniq:
            content = fetch_article_text(art["url"])
            cache_key = content_hash(art["title"], content)

            if cache_key in analysis_cache:
                result = dict(analysis_cache[cache_key])  # copy
                logging.info("  Cache hit: %s", art["title"][:50])
            else:
                result = integrated_labeling(
                    art["display"], art["title"], art.get("summary", ""), content
                )
                analysis_cache[cache_key] = result
                logging.info(
                    "  Analyzed: %s -> %s (method=%s, conf=%.2f)",
                    art["title"][:50], result["label"], result["method"], result["confidence"]
                )

            # 신뢰도 표시(?, !)
            conf = float(result.get("confidence", 0.5))
            indicator = "?" if conf < 0.5 else ("!" if conf > 0.8 else "")

            src = art["source"]
            when_str = to_kst_str(art["published_at"])
            line = f"[{art['display']}] <{art['url']}|{art['title']}> ({src})({when_str}) [{result['label']}{indicator}]"
            all_lines.append(line)

    # 분포 로그
    counts = defaultdict(int)
    for ln in all_lines:
        for em in ["🔴", "🟡", "🟢", "🔵"]:
            if em in ln:
                counts[em] += 1
                break
    logging.info("Label distribution: 🔴%d 🟡%d 🟢%d 🔵%d",
                 counts["🔴"], counts["🟡"], counts["🟢"], counts["🔵"])

    post_to_slack(all_lines)
    logging.info("Posted %d lines to Slack.", len(all_lines))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error")
        raise
