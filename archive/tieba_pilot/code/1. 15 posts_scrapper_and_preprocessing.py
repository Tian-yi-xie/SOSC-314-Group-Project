"""
scraper_and_preprocessing.py  (pilot, not used in the final analysis)
Scrape 15 hand-picked Tieba threads and prepare the posts for analysis.
  1. Scrape every page of each thread through the Tieba client API (one CSV per thread, reused if it exists).
  2. Clean: keep Chinese characters only, drop posts shorter than MIN_CHARS.
  3. Tokenize with jieba, remove stopwords and one-character words.
Input:  Tieba (live)
Output: archive/tieba_pilot/data/tieba_posts.xls
"""
import hashlib
import json
import random
import re
import time
from pathlib import Path 
import jieba
import pandas as pd
import requests
 
# ---------- folders (relative to the archive/tieba_pilot/ folder of the repository) ----------
DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(parents=True, exist_ok=True)
OUT = DATA / "tieba_posts.xlsx"
 
# ---------- settings ----------
THREADS = [9974816726, 10285208826, 10573053816, 10850977454, 11002684373,
           9800612477, 9591629357, 9871474906, 9869009983, 10577521084,
           9437024525, 6392973752, 8570015684, 4327033981, 10337395520]     # 15 hand-picked threads
MIN_CHARS = 5
MEDIA_TYPES = {2, 3, 4, 5, 9, 10, 11, 20}          # emoji, image, video, gif, ...: not text
 
def load_stopwords(path=DATA / "stopwords-zh.json"):
    if not path.exists():
        url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-zh/master/stopwords-zh.json"
        path.write_text(json.dumps(requests.get(url, timeout=30).json(), ensure_ascii=False), encoding="utf-8")
    return set(json.loads(path.read_text(encoding="utf-8")))
 
STOPWORDS = load_stopwords()
 
# ============================================================
# 1. SCRAPE
# ============================================================
def get_page(tid, page):
    """One page of a thread from the Tieba client API (signed request)."""
    params = {"_client_id": "wappc_1534235498291_633", "_client_type": "2",
              "_client_version": "9.7.8.0", "_phone_imei": "000000000000000",
              "kz": str(tid), "pn": str(page), "rn": "30", "r": "0", "lz": "0", "st": "0", "z": "0"}
    raw = "".join(f"{k}={params[k]}" for k in sorted(params)) + "tiebaclient!!!"
    params["sign"] = hashlib.md5(raw.encode()).hexdigest().upper()
    r = requests.post("http://c.tieba.baidu.com/c/f/pb/page", data=params, timeout=20,
                      headers={"User-Agent": "bdtb for Android 9.7.8.0"})
    r.raise_for_status()
    return r.json()
 
def post_text(post):
    return "".join(item.get("text", "") for item in post.get("content") or []
                   if isinstance(item, dict) and item.get("type") not in MEDIA_TYPES
                   and isinstance(item.get("text"), str)).strip()
 
def scrape_thread(tid):
    records, total_pages = [], None
    for page in range(1, 201):
        try:
            data = get_page(tid, page)
        except Exception as error:
            print(f"{tid}: ERROR on page {page} ({error})")
            break
        total_pages = total_pages or data.get("page", {}).get("total_page")
        posts = data.get("post_list") or []
        if not posts:
            break
        records += [{"thread_id": str(tid), "text": post_text(p)} for p in posts]
        if total_pages and page >= total_pages:
            break
        time.sleep(random.uniform(0.5, 1.2))
    return records
 
records = []
for tid in THREADS:
    thread = scrape_thread(tid)
    print(f"{tid}: {len(thread):,} posts")
    records += thread
 
# ============================================================
# 2-3. CLEAN AND TOKENIZE
# ============================================================
df = pd.DataFrame(records)
n_raw = len(df)
df["clean"] = df["text"].fillna("").map(lambda t: re.sub(r"[^一-鿿]", "", str(t)))   # Chinese only
df = df[df["clean"].str.len() >= MIN_CHARS]
n_long = len(df)
df["tokens_joined"] = df["clean"].map(lambda t: " ".join(w for w in jieba.lcut(t)
                                                         if w.strip() and w not in STOPWORDS and len(w) > 1))
df = df[df["tokens_joined"] != ""]
 
with pd.ExcelWriter(OUT) as writer:
    df[["thread_id", "text", "clean", "tokens_joined"]].to_excel(writer, sheet_name="posts", index=False)
    df.groupby("thread_id").size().reset_index(name="post_count").to_excel(writer, sheet_name="summary", index=False)
print(f"Posts: {n_raw:,} scraped | {n_long:,} after length filter | {len(df):,} after stopwords")
