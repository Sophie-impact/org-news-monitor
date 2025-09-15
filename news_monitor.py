#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
News Monitor – 컨텍스트+LLM 통합 라벨링 (예외 조직별 중복제거 정책 포함)
- 시트 규칙(MUST_ALL/MUST_ANY/BLOCK/검색어) 1차 필터
- 본문 추출(trafilatura) → 컨텍스트 기반 신호 분석 → 개선된 LLM(JSON) 판단
- 다층 통합: 규칙/신호/LLM 결과를 일관성 있게 결합
- 캐시로 동일 기사 재분석 방지, 상세 로깅/통계
- 주말 자동 스킵
- [중복정책] 카카오/브라이언임팩트/김범수는 중복 제거 없이 모두 노출, 그 외 조직은 같은 제목(정규화) 1개만 노출
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

# ============================================================
# 예외 조직 (제목 중복 제거를 적용하지 않음)
# ============================================================
DEDUP_EXEMPT_ORGS = {"카카오", "브라이언임팩트", "김범수"}

# ============================================================
# 컨텍스트 키워드 세트
# ============================================================
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

# ============================================================
# 유틸
# ============================================================
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
    return hashlib.md5(f"{title}:{content[:1000]}".encode()).hexdigest()

# ============================================================
# 조회 구간 (09:00 KST 실행 기준)
# ============================================================
def compute_window_utc(now: datetime | None = None) -> tuple[datetime, datetime]:
    now = now or datetime.now(tz=KST)
    anchor_kst = now.astimezone(KST).replace(hour=9, minute=0, second=0, microsecond=0)
    days = 3 if anchor_kst.weekday() == 0 else 1  # 월요일이면 3일 커버
    start_kst = anchor_kst - timedelta(days=days)
    end_kst = anchor_kst
    return start_kst.astimezone(timezone.utc), end_kst.astimezone(timezone.utc)

# ============================================================
# 시트 로더
# ============================================================
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

    seen = set(); uniq = []
    for it in rows:
        key = (it["display"], it["query"])
        if key not in seen:
            uniq.append(it); seen.add(key)
    return uniq

# ============================================================
# 본문 추출
# ============================================================
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

# ============================================================
# 뉴스 검색기
# ============================================================
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

# ============================================================
# 관련성 필터(시트 규칙)
# ============================================================
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

# ============================================================
# 컨텍스트 신호 분석 + 규칙 라벨
# ============================================================
def _is_org_related_context(text: str, keywords: list[str], org_name: str) -> bool:
    if not org_name:
        return False
    org_positions = [m.start() for m in re.finditer(re.escape(org_name), text)]
    for kw in keywords:
        for kp in [m.start() for m in re.finditer(re.escape(kw), text)]:
            for op in org_positions:
                if abs(op - kp) <= 100:
                    return True
    return False

def analyze_context_signals(title: str, summary: str, content: str, org_name: str) -> dict:
    full = f"{title} {summary} {content}".lower()
    org_lower = org_name.lower()
    org_direct = org_lower in full

    signals = {
        "direct_negative": [],
        "contextual_negative": [],
        "monitoring": [],
        "positive": [],
        "org_involvement": "direct" if org_direct else "indirect",
        "severity_score": 0,
        "confidence": 0.5,
    }

    for cat, kws in DIRECT_NEGATIVE.items():
        found = [kw for kw in kws if kw in full]
        if found:
            signals["direct_negative"].extend([(cat, kw) for kw in found])
            signals["severity_score"] += len(found) * 3

    for cat, kws in CONTEXTUAL_NEGATIVE.items():
        found = [kw for kw in kws if kw in full]
        if found:
            weight = 2 if _is_org_related_context(full, found, org_lower) else 1
            signals["contextual_negative"].extend([(cat, kw) for kw in found])
            signals["severity_score"] += len(found) * weight

    for cat, kws in MONITORING_KEYWORDS.items():
        found = [kw for kw in kws if kw in full]
        if found:
            signals["monitoring"].extend([(cat, kw) for kw in found])
            signals["severity_score"] += len(found) * 1

    for cat, kws in POSITIVE_KEYWORDS.items():
        found = [kw for kw in kws if kw in full]
        if found:
            signals["positive"].extend([(cat, kw) for kw in found])
            signals["severity_score"] -= len(found) * 1

    total = (len(signals["direct_negative"]) + len(signals["contextual_negative"]) +
             len(signals["monitoring"]) + len(signals["positive"]))
    if total > 0:
        if signals["org_involvement"] == "direct":
            signals["confidence"] = min(0.9, 0.5 + total * 0.1)
        else:
            signals["confidence"] = min(0.7, 0.3 + total * 0.05)
    return signals

def enhanced_rule_label(signals: dict) -> str:
    score = signals["severity_score"]
    conf  = signals["confidence"]

    if signals["direct_negative"] and conf > 0.6:
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

# ============================================================
# LLM 라벨러 (개선 프롬프트)
# ============================================================
IMPACT_MAP = {"positive": "🔵", "neutral": "🟢", "monitor": "🟡", "negative": "🔴"}

def _format_signals_for_llm(s: dict) -> str:
    parts = []
    if s["direct_negative"]:
        parts.append(f"직접부정:{', '.join([kw for _, kw in s['direct_negative']])}")
    if s["contextual_negative"]:
        parts.append(f"상황부정:{', '.join([kw for _, kw in s['contextual_negative']])}")
    if s["monitoring"]:
        parts.append(f"모니터링:{', '.join([kw for _, kw in s['monitoring']])}")
    if s["positive"]:
        parts.append(f"긍정:{', '.join([kw for _, kw in s['positive']])}")
    parts.append(f"연관성:{s['org_involvement']}")
    parts.append(f"위험도:{s['severity_score']}")
    return " | ".join(parts) if parts else "특별한 신호 없음"

def _safe_load_json(s: str):
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

        prompt = f"""당신은 기업 위기관리 분석가입니다. 아래 기사가 조직에 미치는 '영향'만 평가하세요.

평가 기준:
- positive(🔵): 명확한 긍정적 영향 (수상/투자/계약/협력/사회공헌 등)
- neutral(🟢): 중립 또는 영향 미미 (업계 동향, 단순 언급 등)
- monitor(🟡): 주의 필요 (조사/검토/불확실/잠재 리스크)
- negative(🔴): 명확한 부정 (법적 문제/사고/직접 비판/사업 타격)

원칙:
1) 조직이 주요 대상인지(직접) vs 단순 언급(간접) 구분
2) 기사에 명시된 사실에 기반, 과도한 추정 금지
3) 산업 일반론보다 조직 직접 영향 우선
4) 애매하면 보수적(🟡)이되, 명백한 호재는 positive

조직: {display_name}
제목: {title}
요약: {summary or "없음"}

본문(일부):
{body}

자동 분석 요약:
{signal_summary}

JSON으로만 응답:
{{
  "impact": "positive|neutral|monitor|negative",
  "confidence": 0.0-1.0,
  "primary_reason": "주요 판단 근거 한 줄",
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
        if not data or "impact" not in data:
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

# ============================================================
# 통합 라벨링
# ============================================================
def integrated_labeling(display_name: str, title: str, summary: str, content: str) -> dict:
    signals = analyze_context_signals(title, summary, content, display_name)
    rule_label = enhanced_rule_label(signals)
    llm_result = enhanced_llm_label(display_name, title, summary, content, signals)

    result = {
        "label": rule_label,
        "confidence": signals["confidence"],
        "method": "rule_based",
        "signals": signals,
        "llm_result": llm_result,
    }

    if llm_result and llm_result["confidence"] > 0.6:
        # 직접부정이 있는데 LLM이 긍정/중립이면 보수적
        if signals["direct_negative"] and llm_result["label"] in {"🟢", "🔵"} and signals["org_involvement"] == "direct":
            result["label"] = "🔴"
            result["method"] = "conservative_override"
        else:
            result["label"] = llm_result["label"]
            result["confidence"] = llm_result["confidence"]
            result["method"] = "llm_primary"

    # 과도한 🔴 방지
    if result["label"] == "🔴" and not signals["direct_negative"] and signals["severity_score"] < 5:
        result["label"] = "🟡"
        result["method"] += "_moderated"

    return result

# ============================================================
# Slack
# ============================================================
def post_to_slack(lines: list[str]) -> None:
    token = os.environ.get("SLACK_BOT_TOKEN", "").strip()
    channel = os.environ.get("SLACK_CHANNEL", "").strip()
    if not token or not channel:
        raise RuntimeError("SLACK_BOT_TOKEN or SLACK_CHANNEL missing.")
    client = WebClient(token=token)
    text = "\n".join(lines) if lines else "오늘은 신규로 감지된 기사가 없습니다."
    client.chat_postMessage(channel=channel, text=text)

# ============================================================
# main
# ============================================================
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

    analysis_cache: dict[str, dict] = {}
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

        # 1) 시트 규칙 관련성
        before_rel = len(items)
        items = [it for it in items if is_relevant_by_rule(it["row_cfg"], it["title"], it.get("summary", ""))]
        logging.info("  after relevance: %d -> %d", before_rel, len(items))

        # 2) 기간 필터
        before_win = len(items)
        items = [it for it in items if it["published_at"] and window_from_utc <= it["published_at"] < window_to_utc]
        logging.info("  after window: %d -> %d", before_win, len(items))

        # 3) 최신순 + 제목 중복 제거 (예외 조직은 제거하지 않음)
        items.sort(key=lambda x: x["published_at"], reverse=True)
        seen_titles = set()
        uniq = []
        for it in items:
            title_key = norm_title(it["title"])
            if not title_key or not it["url"]:
                continue
            if it["display"] in DEDUP_EXEMPT_ORGS:
                # 예외 조직: 중복 허용
                uniq.append(it)
            else:
                if title_key not in seen_titles:
                    uniq.append(it)
                    seen_titles.add(title_key)

        # 4) 통합 라벨링
        for art in uniq:
            content = fetch_article_text(art["url"])
            cache_key = content_hash(art["title"], content)
            if cache_key in analysis_cache:
                result = analysis_cache[cache_key].copy()
                logging.info("  Cache hit: %s", art["title"][:60])
            else:
                result = integrated_labeling(art["display"], art["title"], art.get("summary", ""), content)
                analysis_cache[cache_key] = result
                logging.info("  Analyzed: %s -> %s (%s, conf=%.2f)",
                             art["title"][:60], result["label"], result["method"], result["confidence"])

            label = result["label"]
            confidence = result["confidence"]
            method = result["method"]

            conf_mark = "!" if confidence > 0.8 else ("?" if confidence < 0.5 else "")
            src = art["source"]
            when_str = to_kst_str(art["published_at"])

            extra_info = []
            if result.get("llm_result", {}).get("primary_reason"):
                extra_info.append(f"이유: {result['llm_result']['primary_reason']}")
            sig = result.get("signals", {})
            if sig.get("direct_negative"):
                extra_info.append(f"직접위험:{len(sig['direct_negative'])}")
            if sig.get("positive"):
                extra_info.append(f"긍정신호:{len(sig['positive'])}")
            extra = f" ({', '.join(extra_info)})" if extra_info else ""

            line = f"[{art['display']}] <{art['url']}|{art['title']}> ({src})({when_str}) [{label}{conf_mark}]{extra}"
            all_lines.append(line)

    # 요약 로깅
    label_counts = defaultdict(int)
    for line in all_lines:
        for emoji in ["🔴", "🟡", "🟢", "🔵"]:
            if emoji in line:
                label_counts[emoji] += 1
                break
    logging.info("Label distribution: 🔴%d 🟡%d 🟢%d 🔵%d",
                 label_counts["🔴"], label_counts["🟡"], label_counts["🟢"], label_counts["🔵"])

    post_to_slack(all_lines)
    logging.info("Posted %d lines to Slack.", len(all_lines))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("Fatal error")
        raise
