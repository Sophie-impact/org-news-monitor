#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
News Monitor – 본문 추출 + LLM(JSON) 판단 + 키워드 보조 + 주말 스킵
- 시트 규칙으로 1차 필터
- trafilatura 본문 추출 → LLM이 조직 관점 영향(긍정/중립/모니터/부정)을 JSON으로 판정
- 키워드는 힌트/보조 용도로만 사용 (과도한 override 제거)
- 동일 제목(정규화) 중복 제거
"""

from __future__ import annotations
import os, re, html, time, json, logging, requests, pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparser
from zoneinfo import ZoneInfo
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError  # noqa: F401
import tldextract, trafilatura

try:
    import openai  # pip install openai>=1.40.0
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

KST = ZoneInfo("Asia/Seoul")

# -------------------------
# 공통 유틸
# -------------------------
def now_kst(): return datetime.now(tz=KST)

def parse_datetime(dt_str: str | None):
    if not dt_str: return None
    try:
        dt = dtparser.parse(dt_str)
        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception: return None

def to_kst_str(dt): return "" if not dt else dt.astimezone(KST).strftime("%Y-%m-%d %H:%M")

def strip_html(text: str | None): return BeautifulSoup(html.unescape(text or ""), "html.parser").get_text(" ", strip=True)

def domain_from_url(url: str | None):
    if not url: return ""
    try:
        ext = tldextract.extract(url)
        return ".".join(p for p in [ext.domain, ext.suffix] if p)
    except Exception: return ""

def norm_title(t: str | None):
    t = strip_html(t or "").lower()
    t = re.sub(r"[\[\]()<>『』「」]", " ", t)
    t = re.sub(r"[^\w가-힣\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

# -------------------------
# 조회 구간
# -------------------------
def compute_window_utc(now=None):
    now = now or now_kst()
    anchor = now.astimezone(KST).replace(hour=9, minute=0, second=0, microsecond=0)
    days = 3 if anchor.weekday() == 0 else 1
    return (anchor - timedelta(days=days)).astimezone(timezone.utc), anchor.astimezone(timezone.utc)

# -------------------------
# 시트 로더
# -------------------------
def _split_list(val): return [] if pd.isna(val) or not str(val).strip() else [x.strip().lower() for x in str(val).split(",") if x.strip()]

def _query_tokens_from(q): 
    if not q: return []
    return [p.strip().strip('"').strip("'").lower() for p in re.split(r'\bOR\b', q, flags=re.I) if p.strip()]

def fetch_org_list():
    sheet_url = os.environ.get("SHEET_CSV_URL", "").strip()
    if not sheet_url: raise RuntimeError("SHEET_CSV_URL not set.")
    df = pd.read_csv(StringIO(requests.get(sheet_url, timeout=30).content.decode("utf-8", "replace")))
    name_col = next((c for c in ["조직명","표시명"] if c in df.columns), None)
    if not name_col: raise RuntimeError("CSV에 '조직명' 또는 '표시명' 필요")
    rows=[]; seen=set()
    for _,r in df.iterrows():
        display=str(r[name_col]).strip()
        if not display or display.lower()=="nan": continue
        item={
            "display":display,"query":str(r.get("검색어","")).strip() or display,
            "kind":str(r.get("유형","ORG")).strip().upper() or "ORG",
            "must_all":_split_list(r.get("MUST_ALL","")),
            "must_any":_split_list(r.get("MUST_ANY","")),
            "block":_split_list(r.get("BLOCK","")),
        }
        item["query_tokens"]=_query_tokens_from(item["query"])
        key=(item["display"],item["query"])
        if key not in seen: rows.append(item); seen.add(key)
    return rows

# -------------------------
# 본문 추출
# -------------------------
def fetch_article_text(url, timeout=20):
    if not url: return ""
    try:
        d = trafilatura.fetch_url(url, no_ssl=True, timeout=timeout)
        if d: return (trafilatura.extract(d, favor_recall=True, deduplicate=True) or "").strip()
    except Exception: pass
    try: return strip_html(requests.get(url,timeout=timeout,headers={"User-Agent":"Mozilla/5.0"}).text)[:8000].strip()
    except Exception: return ""

# -------------------------
# 뉴스 검색기
# -------------------------
def search_naver(query, display=20):
    cid,csec=os.environ.get("NAVER_CLIENT_ID",""),os.environ.get("NAVER_CLIENT_SECRET","")
    if not cid or not csec: return []
    try:
        r=requests.get("https://openapi.naver.com/v1/search/news.json",
            headers={"X-Naver-Client-Id":cid,"X-Naver-Client-Secret":csec},
            params={"query":query,"display":display,"start":1,"sort":"date"},timeout=20).json()
        return [{"title":strip_html(it.get("title")),"url":it.get("originallink") or it.get("link"),
                 "source":domain_from_url(it.get("originallink") or it.get("link")) or "naver",
                 "published_at":parse_datetime(it.get("pubDate")),"origin":"naver",
                 "summary":strip_html(it.get("description",""))}
                for it in r.get("items",[]) if it.get("title")]
    except Exception: return []

def search_newsapi(query, f,t,language="ko"):
    key=os.environ.get("NEWSAPI_KEY","")
    if not key: return []
    try:
        r=requests.get("https://newsapi.org/v2/everything",
            params={"q":query,"from":f.isoformat().replace("+00:00","Z"),
                    "to":t.isoformat().replace("+00:00","Z"),
                    "sortBy":"publishedAt","pageSize":50,"language":language,"apiKey":key},timeout=20).json()
        return [{"title":strip_html(a.get("title")),"url":a.get("url"),
                 "source":(a.get("source") or {}).get("name") or domain_from_url(a.get("url")),
                 "published_at":parse_datetime(a.get("publishedAt")),"origin":"newsapi",
                 "summary":strip_html(a.get("description") or a.get("content") or "")}
                for a in r.get("articles",[]) if a.get("title")]
    except Exception: return []

# -------------------------
# 키워드 세트 (정비된 버전)
# -------------------------
STRONG_NEG=[
  "횡령","배임","사기","고발","기소","구속","수사","압수수색","소송","고소","분쟁",
  "리콜","결함","징계","제재","벌금","과징금","부실","파산","부도","오염","사망","부상",
  "폭발","화재","추락","유출","해킹","랜섬웨어","침해","악성코드","담합","독점","불매",
  "논란","갑질","표절","혐의","불법","위법","취소","철회","부정","적자","급락","하락","징역"
]
POS_KW=["수상","선정","후원","지원","기부","투자","파트너십","mou","계약","수주","인증","승인"]

def calc_risk_signals(title, summary, content):
    text=f"{title} {summary} {content}".lower()
    strong=[kw for kw in STRONG_NEG if kw.lower() in text]
    pos=[kw for kw in POS_KW if kw.lower() in text]
    return {"score":len(strong)*2-len(pos),"strong_neg":strong,"hints_text":("부정:"+",".join(strong)) if strong else ""}

# -------------------------
# LLM 라벨러
# -------------------------
IMPACT_MAP={"positive":"🔵","neutral":"🟢","monitor":"🟡","negative":"🔴"}

def llm_enabled(): 
    return os.environ.get("LLM_ENABLE","").lower() in {"1","true","yes"} and os.environ.get("OPENAI_API_KEY") and _HAS_OPENAI

def llm_label(display,title,summary,content,hints):
    if not llm_enabled(): return None
    body=(content or "").strip()[:3500]
    strong_info = f"강한 부정 키워드 감지: {hints}" if hints else "없음"
    prompt=f"""당신은 위기관리 분석가입니다. 다음 기사가 '{display}'에 주는 영향을 평가하세요.
- 분류: positive(긍정), neutral(중립), monitor(모니터링 필요), negative(부정)
- 긍정 신호(수상/선정/후원/지원/기부/투자/파트너십 등)는 기본적으로 긍정적.
- 강한 부정 키워드가 기사에 포함되더라도, '{display}'와 직접적 관련이 없다면 부정으로 판단하지 마세요.
- 애매하면 monitor로 분류하되, 과도하게 보수적일 필요는 없습니다.

JSON 예시:
{{"impact":"positive|neutral|monitor|negative","confidence":0.0-1.0,"evidence":["근거1","근거2"]}}

[제목]{title}
[요약]{summary}
[본문]{body}
[부정 키워드]{strong_info}"""
    try:
        resp=openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"]).chat.completions.create(
            model=os.environ.get("LLM_MODEL","gpt-4o-mini"),
            messages=[{"role":"user","content":prompt}],temperature=0,max_tokens=220)
        raw=(resp.choices[0].message.content or "").strip()
        data=json.loads(raw)
        return {"label":IMPACT_MAP.get(data.get("impact",""),None),"confidence":float(data.get("confidence",0.5)),"raw":data}
    except Exception: return None

# -------------------------
# Slack
# -------------------------
def post_to_slack(lines):
    token,channel=os.environ.get("SLACK_BOT_TOKEN",""),os.environ.get("SLACK_CHANNEL","")
    if not token or not channel: raise RuntimeError("Slack token/channel missing.")
    WebClient(token=token).chat_postMessage(channel=channel,text="\n".join(lines) if lines else "오늘은 신규 기사가 없습니다.")

# -------------------------
# main
# -------------------------
def main():
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    if now_kst().weekday() in (5,6): return
    f,t=compute_window_utc(); rows=fetch_org_list(); all_lines=[]
    for row in rows:
        items=search_naver(row["query"],20)+search_newsapi(row["query"],f,t)
        uniq={}; 
        for it in items:
            key=norm_title(it["title"])
            if key and it.get("published_at") and f<=it["published_at"]<t: uniq[key]=it|{"display":row["display"],"row_cfg":row}
        for art in uniq.values():
            content=fetch_article_text(art["url"])
            signals=calc_risk_signals(art["title"],art.get("summary",""),content)
            out=llm_label(art["display"],art["title"],art.get("summary",""),content,signals["hints_text"])
            if out: label,conf=out["label"],out["confidence"]
            else: label,conf="🟢",0.5
            # 보정: 확신 매우 낮고(score 높음) → monitor
            if label in {"🔵","🟢"} and conf<0.4 and signals["score"]>=5: label="🟡"
            all_lines.append(f"[{art['display']}] <{art['url']}|{art['title']}> ({art['source']})({to_kst_str(art['published_at'])}) [{label}]")
    post_to_slack(all_lines)

if __name__=="__main__":
    main()
