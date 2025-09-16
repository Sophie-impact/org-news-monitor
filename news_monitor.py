#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
News Monitor – 개선된 라벨링 시스템 (Claude + Gemini 피드백 반영)
- 컨텍스트 기반 키워드 분석으로 오탐 감소
- LLM 프롬프트 개선으로 일관성 향상
- 다층 검증 시스템으로 과도한 위험 신호 방지
- 캐싱으로 중복 분석 방지
- 카카오/브라이언임팩트/김범수 제외, 나머지는 중복 기사 하나만 대표 표시
"""

from __future__ import annotations

import os, re, html, time, json, logging, requests, pandas as pd, hashlib
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
from zoneinfo import ZoneInfo
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError  # noqa: F401
import tldextract, trafilatura
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
    "법적": ["횡령","배임","사기","고발","기소","구속","수사","압수수색","특검","징역","실형"],
    "사업": ["리콜","결함","파산","부도","영업정지","사업중단","퇴출"],
    "안전": ["사망","부상","폭발","화재","추락","유출","해킹","랜섬웨어","개인정보유출"],
}

CONTEXTUAL_NEGATIVE = {
    "경영": ["적자","손실","감소","하락","부실"],
    "규제": ["제재","벌금","과징금","징계","처분"],
    "논란": ["논란","비판","갑질","불법","위법","부정"],
}

MONITORING_KEYWORDS = {
    "조사": ["의혹","조사","점검","심사","검토","국감","감사"],
    "불확실": ["연기","지연","유예","잠정","검토중","불확실성"],
    "주의": ["우려","경고","리스크","변동성","관심","주시"],
}

POSITIVE_KEYWORDS = {
    "성과": ["수상","선정","혁신","신기록","최대","달성","성과","흑자전환"],
    "성장": ["투자유치","시리즈","상승","증가","호조","확대","진출","성장"],
    "협력": ["협력","파트너십","mou","계약","수주","제휴","연합"],
    "사회공헌": ["후원","지원","기부","기증","기탁","장학금","봉사"],
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
# 조회 구간
# =========================
def compute_window_utc(now: datetime | None = None) -> tuple[datetime, datetime]:
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
    for category, keywords in DIRECT_NEGATIVE.items():
        found = [kw for kw in keywords if kw in full_text]
        if found:
            signals["direct_negative"].extend([(category, kw) for kw in found])
            signals["severity_score"] += len(found) * 3
    for category, keywords in CONTEXTUAL_NEGATIVE.items():
        found = [kw for kw in keywords if kw in full_text]
        if found:
            weight = 2 if _is_org_related_context(full_text, found, org_lower) else 1
            signals["contextual_negative"].extend([(category, kw) for kw in found])
            signals["severity_score"] += len(found) * weight
    for category, keywords in MONITORING_KEYWORDS.items():
        found = [kw for kw in keywords if kw in full_text]
        if found:
            signals["monitoring"].extend([(category, kw) for kw in found])
            signals["severity_score"] += len(found)
    for category, keywords in POSITIVE_KEYWORDS.items():
        found = [kw for kw in keywords if kw in full_text]
        if found:
            signals["positive"].extend([(category, kw) for kw in found])
            signals["severity_score"] -= len(found)
    total_signals = len(signals["direct_negative"]) + len(signals["contextual_negative"]) + len(signals["monitoring"]) + len(signals["positive"])
    if total_signals > 0:
        if signals["org_involvement"] == "direct":
            signals["confidence"] = min(0.9, 0.5 + total_signals * 0.1)
        else:
            signals["confidence"] = min(0.7, 0.3 + total_signals * 0.05)
    return signals

def _is_org_related_context(text: str, keywords: list[str], org_name: str) -> bool:
    if not org_name:
        return False
    org_positions = [m.start() for m in re.finditer(re.escape(org_name), text)]
    for kw in keywords:
        kw_positions = [m.start() for m in re.finditer(re.escape(kw), text)]
        for org_pos in org_positions:
            for kw_pos in kw_positions:
                if abs(org_pos - kw_pos) <= 100:
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
# LLM 프롬프트
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
        prompt = f"""당신은 기업 위기관리 전문 분석가입니다. 기사가 {display_name}에 미치는 영향을 평가하세요.

평가 기준:
- positive(🔵): 명확한 긍정적 영향
- neutral(🟢): 영향도 낮음/중립
- monitor(🟡): 주의깊게 관찰 필요
- negative(🔴): 명확한 부정적 영향

조직명: {display_name}
제목: {title}
요약: {summary or "없음"}
본문(일부): {body}
자동 분석: {signal_summary}

JSON만 반환:
{{
  "impact": "positive|neutral|monitor|negative",
  "confidence": 0.0-1.0,
  "primary_reason": "주요 판단 근거",
  "evidence": ["근거1","근거2"],
  "org_relevance": "direct|indirect|minimal"
}}"""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = _safe_load_json(raw)
        if not data or "impact" not in data:
            return None
        impact = str(data.get("impact","")).lower()
        if impact not in ["positive","neutral","monitor","negative"]:
            return None
        conf = float(data.get("confidence",0.5))
        label_map = {"positive":"🔵","neutral":"🟢","monitor":"🟡","negative":"🔴"}
        return {
            "label": label_map[impact],
            "confidence": conf,
            "raw": data,
            "primary_reason": data.get("primary_reason",""),
            "org_relevance": data.get("org_relevance","unknown"),
        }
    except Exception as e:
        logging.error(f"LLM labeling failed: {e}")
        return None

def _format_signals_for_llm(signals: dict) -> str:
    parts = []
    if signals["direct_negative"]:
        parts.append(f"직접적 부정: {', '.join([kw for _,kw in signals['direct_negative']])}")
    if signals["contextual_negative"]:
        parts.append(f"상황적 부정: {', '.join([kw for _,kw in signals['contextual_negative']])}")
    if signals["monitoring"]:
        parts.append(f"모니터링: {', '.join([kw for _,kw in signals['monitoring']])}")
    if signals["positive"]:
        parts.append(f"긍정: {', '.join([kw for _,kw in signals['positive']])}")
    parts.append(f"조직 연관성: {signals['org_involvement']}")
    parts.append(f"위험도: {signals['severity_score']}")
    return " | ".join(parts)

def _safe_load_json(s: str):
    try:
        s = re.sub(r'```json\s*|\s*```','',s)
        return json.loads(s)
    except Exception:
        return None

# =========================
# 통합 라벨링
# =========================
def integrated_labeling(display_name: str, title: str, summary: str, content: str) -> dict:
    signals = analyze_context_signals(title, summary, content, display_name)
    rule_label = enhanced_rule_label(signals)
    llm_result = enhanced_llm_label(display_name,title,summary,content,signals)
    result = {
        "label": rule_label,
        "confidence": signals["confidence"],
        "method": "rule_based",
        "signals": signals,
        "llm_result": llm_result or {},   # 항상 dict
    }
    if llm_result and llm_result["confidence"]>0.6:
        if signals["direct_negative"] and llm_result["label"] in {"🟢","🔵"} and signals["org_involvement"]=="direct":
            result["label"]="🔴"; result["method"]="conservative_override"
        else:
            result["label"]=llm_result["label"]; result["confidence"]=llm_result["confidence"]; result["method"]="llm_primary"
    if result["label"]=="🔴" and not signals["direct_negative"] and signals["severity_score"]<5:
        result["label"]="🟡"; result["method"]+="_moderated"
    return result

# =========================
# 시트 로딩
# =========================
def _split_list(val)->list[str]:
    if pd.isna(val) or str(val).strip()=="":
        return []
    return [x.strip().lower() for x in str(val).split(",") if x.strip()]

def _query_tokens_from(q:str)->list[str]:
    if not q: return []
    parts=re.split(r'\bOR\b',q,flags=re.IGNORECASE)
    return [p.strip().strip('"').strip("'").lower() for p in parts if p.strip()]

def fetch_org_list()->list[dict]:
    sheet_url=os.environ.get("SHEET_CSV_URL","").strip()
    if not sheet_url: raise RuntimeError("SHEET_CSV_URL not set")
    resp=requests.get(sheet_url,timeout=30); resp.raise_for_status()
    df=pd.read_csv(StringIO(resp.content.decode("utf-8",errors="replace")))
    name_col=None
    for cand in ["조직명","표시명"]:
        if cand in df.columns: name_col=cand; break
    if not name_col: raise RuntimeError("CSV에 '조직명'/'표시명' 필요")
    rows=[]
    for _,r in df.iterrows():
        display=str(r[name_col]).strip()
        if not display or display.lower()=="nan": continue
        query=str(r.get("검색어","")).strip() or display
        kind=str(r.get("유형","ORG")).strip().upper() or "ORG"
        rows.append({
            "display":display,"query":query,"kind":kind,
            "must_all":_split_list(r.get("MUST_ALL","")),
            "must_any":_split_list(r.get("MUST_ANY","")),
            "block":_split_list(r.get("BLOCK","")),
            "query_tokens":_query_tokens_from(query),
        })
    uniq=[]; seen=set()
    for it in rows:
        key=(it["display"],it["query"])
        if key not in seen: uniq.append(it); seen.add(key)
    return uniq

# =========================
# 본문 추출 / 검색
# =========================
def fetch_article_text(url:str,timeout:int=20)->str:
    if not url: return ""
    try:
        dl=trafilatura.fetch_url(url,no_ssl=True,timeout=timeout)
        if dl:
            txt=trafilatura.extract(dl,include_comments=False,include_tables=False,include_formatting=False,favor_recall=True,deduplicate=True) or ""
            return txt.strip()
    except: pass
    try:
        r=requests.get(url,timeout=timeout,headers={"User-Agent":"Mozilla/5.0"}); r.raise_for_status()
        return strip_html(r.text)[:8000].strip()
    except: return ""

def search_naver(query:str,display:int=20)->list[dict]:
    cid,cs=os.environ.get("NAVER_CLIENT_ID",""),os.environ.get("NAVER_CLIENT_SECRET","")
    if not cid or not cs: return []
    endpoint="https://openapi.naver.com/v1/search/news.json"
    headers={"X-Naver-Client-Id":cid,"X-Naver-Client-Secret":cs}
    params={"query":query,"display":display,"start":1,"sort":"date"}
    try:
        r=requests.get(endpoint,headers=headers,params=params,timeout=20); r.raise_for_status()
        items=r.json().get("items",[]); res=[]
        for it in items:
            title=strip_html(it.get("title")); url=it.get("originallink") or it.get("link"); pub=parse_datetime(it.get("pubDate"))
            if not url or not title: continue
            src=domain_from_url(url) or "naver"
            res.append({"title":title,"url":url,"source":src,"published_at":pub,"origin":"naver","summary":strip_html(it.get("description",""))})
        return res
    except: return []

def search_newsapi(query:str,window_from_utc:datetime,window_to_utc:datetime,language:str="ko")->list[dict]:
    key=os.environ.get("NEWSAPI_KEY",""); 
    if not key: return []
    endpoint="https://newsapi.org/v2
