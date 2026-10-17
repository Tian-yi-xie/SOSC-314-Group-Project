# ============================================================
# 1. SETTINGS
# ============================================================
import hashlib, json, os, random, re, time
from pathlib import Path
from urllib.parse import quote
import pandas as pd
import requests
 
# ---------- where to look ----------
SEARCH_KEYWORDS = ["AI伴侣", "AI男友", "AI女友", "AI老公", "AI老婆", "AI恋人", "人机恋",
                   "和AI谈恋爱", "AI陪伴", "星野 AI", "猫箱 AI", "筑梦岛", "豆包 智能体",
                   "AI 角色扮演", "虚拟恋人", "虚拟男友", "虚拟女友", "电子男友", "电子女友",
                   "赛博恋人", "人工智能 恋爱"]
COMPANION_BARS = ["星野", "猫箱", "筑梦岛", "ai伴侣", "人机恋"] 
GENERAL_BARS = ["豆包", "deepseek", "智能体", "minimax"] 
TARGET_BARS = COMPANION_BARS + GENERAL_BARS
PAGES, MAX_THREADS, MIN_CHARS, SCROLLS = 10, 3000, 5, 15
USER_AGENT = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/152.0.0.0 Safari/537.36'
SALT = os.environ.get("TIEBA_SALT", "")          # private salt for hashing author IDs
RAW_DIR = Path("bigdata/raw_threads")
RAW_DIR.mkdir(parents=True, exist_ok=True)
INDEX_XLSX = "bigdata/thread_index.xlsx"
POSTS_XLSX = "bigdata/posts.xlsx"
 
# ---------- which titles count as AI companion ----------
COMPANION_TERMS = ["ai伴侣", "ai男友", "ai女友", "ai老公", "ai老婆", "ai恋人", "ai对象",
                   "虚拟恋人", "虚拟男友", "虚拟女友", "电子男友", "电子女友", "赛博恋人",
                   "人机恋", "赛博恋爱", "ai陪伴", "情感陪伴", "陪聊", "角色扮演", "纸片人"]
APP_NAMES = ["ai", "星野", "minimax", "talkie", "猫箱", "筑梦岛", "豆包", "deepseek", "小冰"]
RELATION_WORDS = ["聊天", "恋", "男友", "女友", "老公", "老婆", "爱上", "陪", "感情",
                  "角色", "人设", "对象", "分手", "表白", "下架", "导出"]
EXCLUDE_TERMS = ["招聘", "兼职", "代练", "出售", "转让", "加群", "广告",
                 "创业", "实战", "课程", "训练营", "变现", "教程", "搭建", "开发"]
 
def is_companion(title, bar=""):
    title = str(title).lower()
    if any(word in title for word in EXCLUDE_TERMS):
        return 0
    if str(bar).lower() in COMPANION_BARS:
        return 1
    if any(word in title for word in COMPANION_TERMS):
        return 1
    return int(any(w in title for w in APP_NAMES) and any(w in title for w in RELATION_WORDS))
 
# ---------- Tieba client API ----------
API_BASE = "http://c.tieba.baidu.com"
BASE_PARAMS = {"_client_id": "wappc_1534235498291_633", "_client_type": "2",
               "_client_version": "9.7.8.0", "_phone_imei": "000000000000000"}
HEADERS = {"User-Agent": "bdtb for Android 9.7.8.0",
           "Content-Type": "application/x-www-form-urlencoded"}
 
def api_post(path, extra):
    params = {**BASE_PARAMS, **{k: str(v) for k, v in extra.items()}}
    raw = "".join(f"{k}={params[k]}" for k in sorted(params)) + "tiebaclient!!!"
    params["sign"] = hashlib.md5(raw.encode()).hexdigest().upper()
    data = requests.post(API_BASE + path, data=params, headers=HEADERS, timeout=20).json()
    if str(data.get("error_code", "0")) != "0":
        raise RuntimeError(f"error {data.get('error_code')}: {data.get('error_msg', '')}")
    return data
 
# ---------- small helpers ----------
def get_id(obj):
    return str(obj.get("id") or obj.get("tid") or obj.get("spid") or "")
 
def to_date(value):
    value = str(value or "")
    if re.match(r"\d{4}-\d{2}-\d{2}", value):
        return value[:10]
    if value.isdigit(): 
        return pd.Timestamp(int(value[:10]), unit="s", tz="Asia/Shanghai").strftime("%Y-%m-%d")
    return ""
 
def pause():
    time.sleep(random.uniform(1.0, 2.0))
 
print("settings ok")
 
# ============================================================
# 2. FIND + SCRAPE + COMBINE
# ============================================================
from concurrent.futures import ThreadPoolExecutor
 
SCROLLS = 15 
HEADLESS = False        # False = see the Chrome window (and can solve a captcha if one appears)
 
def pick_threads(data, keyword, out, stats):
    """Find every thread (tid + title) in the data the search page loads."""
    if isinstance(data, dict):
        tid, title = data.get("tid") or data.get("thread_id"), data.get("title")
        if tid and isinstance(title, str):
            stats["candidates"] += 1
            forum = data.get("forum_name") or data.get("fname") or ""
            if not forum and isinstance(data.get("forum"), dict):
                forum = data["forum"].get("forum_name") or data["forum"].get("name") or ""
            text = re.sub(r"<[^>]+>", "", f"{title} {data.get('content', '')} {data.get('abstract', '')}")
            if all(part.lower() in text.lower() for part in keyword.split()):   # keyword really in the post
                out.append({"thread_id": str(tid), "title": re.sub(r"<[^>]+>", "", title),
                            "bar": str(forum).removesuffix("吧"), "source": "search",
                            "date": to_date(data.get("create_time") or data.get("time"))})
        for value in data.values():
            pick_threads(value, keyword, out, stats)
    elif isinstance(data, list):
        for value in data:
            pick_threads(value, keyword, out, stats)
 
def browser_search(keywords):
    from playwright.sync_api import sync_playwright
    found, current, samples = [], {"keyword": ""}, []
    stats = {"responses": 0, "candidates": 0}
 
    def on_response(response):
        if "json" not in (response.headers.get("content-type") or "") and "search" not in response.url:
            return
        try:
            data = response.json()
        except Exception:
            return
        stats["responses"] += 1
        if len(samples) < 30:                                   # kept for checking what Tieba sends
            samples.append({"keyword": current["keyword"], "url": response.url.split("?")[0], "data": data})
        pick_threads(data, current["keyword"], found, stats)
 
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        context = browser.new_context(user_agent=USER_AGENT)
        context.add_cookies([{"name": n, "value": v, "domain": ".baidu.com", "path": "/"}
                             for n, v in [("BAIDUID", BAIDUID), ("BDUSS", BDUSS), ("STOKEN", STOKEN)] if v])
        page = context.new_page()
        page.on("response", on_response)
        for keyword in keywords:
            current["keyword"], before = keyword, len(found)
            stats.update(responses=0, candidates=0)
            try:
                page.goto(f"https://tieba.baidu.com/f/search/res?ie=utf-8&qw={quote(keyword)}", timeout=30000)
                page.wait_for_timeout(3000)
                page.get_by_text("贴", exact=True).first.click(timeout=5000)    # posts tab
                page.wait_for_timeout(2000)
            except Exception as error:
                print(f"search {keyword}: {str(error)[:80]}")
            for _ in range(SCROLLS):                                          # load more results
                page.mouse.wheel(0, 6000)
                page.wait_for_timeout(1500)
            print(f"search {keyword}: {len(found) - before} threads "
                  f"({stats['responses']} data responses, {stats['candidates']} candidates)")
        browser.close()
    Path("bigdata/search_debug.json").write_text(json.dumps(samples, ensure_ascii=False)[:2_000_000], encoding="utf-8")
    return found
 
def bar_page(bar, page):
    data = api_post("/c/f/frs/page", {"kw": bar, "pn": page, "rn": 50})
    return [{"thread_id": get_id(t), "title": t.get("title", ""), "bar": bar,
             "date": to_date(t.get("create_time")), "source": "bar"}
            for t in data.get("thread_list") or []]
 
# ---------- one thread ----------
def post_text(obj):
    content = obj.get("content")
    if isinstance(content, str):
        return content
    return "".join(item.get("text", "") if str(item.get("type", "0")) == "0"       # text
                   else f"[{item.get('c', '')}]" if str(item.get("type")) == "2"   # emoticon
                   else "" for item in content or []).strip()
 
def make_record(thread, obj, post_type, parent_id):
    author = obj.get("author_id") or (obj.get("author") or {}).get("id") or ""
    return {"post_id": get_id(obj), "parent_post_id": parent_id,
            "thread_id": thread["thread_id"], "bar": thread["bar"], "post_type": post_type,
            "author_hash": hashlib.sha256(f"{SALT}{author}".encode()).hexdigest()[:12] if author else "",
            "date": to_date(obj.get("time")), "thread_title": thread["title"], "text": post_text(obj)}
 
def get_replies(thread_id, post):
    sub = post.get("sub_post_list") or {}
    replies = sub.get("sub_post_list", []) if isinstance(sub, dict) else sub
    if int(post.get("sub_post_number") or 0) <= len(replies):          # preview is complete
        return replies
    replies = []
    for page in range(1, PAGES + 1):
        items = api_post("/c/f/pb/floor", {"kz": thread_id, "pid": get_id(post),
                                           "pn": page, "rn": 30}).get("subpost_list") or []
        replies += items
        if len(items) < 30:
            break
        pause()
    return replies
 
# original = 主楼 (no parent) | comment = 楼层 (parent = 主楼) | reply = 楼中楼 (parent = 楼层)
def scrape_thread(thread):
    records, seen = [], set()
    for page in range(1, PAGES + 1):
        data = api_post("/c/f/pb/page", {"kz": thread["thread_id"], "pn": page, "rn": 30,
                                         "r": 0, "lz": 0, "with_floor": 1})
        posts = ([data["first_floor"]] if page == 1 and data.get("first_floor") else []) \
                + (data.get("post_list") or [])
        new_posts = 0
        for post in posts:
            if get_id(post) in seen:
                continue
            seen.add(get_id(post))
            new_posts += 1
            op_id = records[0]["post_id"] if records else ""
            records.append(make_record(thread, post, "comment" if op_id else "original", op_id))
            for reply in get_replies(thread["thread_id"], post):
                records.append(make_record(thread, reply, "reply", get_id(post)))
        if new_posts == 0 or page >= int((data.get("page") or {}).get("total_page") or PAGES):
            break
        pause()
    return records
 
# ---------- find threads (saved list is reused; delete thread_index.xlsx to search again) ----------
if Path(INDEX_XLSX).exists():
    index = pd.read_excel(INDEX_XLSX, dtype={"thread_id": str}).fillna("")
    print(f"Using saved {INDEX_XLSX} -- NO new search. Delete that file to search again.")
else:
    with ThreadPoolExecutor() as pool:                    # Playwright runs in its own thread
        found = pool.submit(browser_search, SEARCH_KEYWORDS).result() if SEARCH_KEYWORDS else []
    for bar in TARGET_BARS:
        before = len(found)
        for page in range(1, PAGES + 1):
            try:
                threads = bar_page(bar, page)
            except Exception as error:
                print(f"bar {bar} page {page}: {error}")
                break
            if not threads:
                break
            found += threads
            pause()
        print(f"bar {bar}: {len(found) - before} threads")
    index = pd.DataFrame(found)
 
# ---------- screen ----------
# search results: keyword is in the post -> keep (unless spam) | bar threads: companion bar or companion title
if "source" not in index:
    index["source"] = "bar"
spam = index["title"].str.lower().map(lambda t: any(w in t for w in EXCLUDE_TERMS))
index["include"] = [int(s == "search") or is_companion(t, str(b).removesuffix("吧"))
                    for s, t, b in zip(index["source"], index["title"], index["bar"])]
index.loc[spam, "include"] = 0
index.loc[(index["date"] != "") & (index["date"] < START_DATE), "include"] = 0
index = index.sort_values("include", ascending=False).drop_duplicates("thread_id")
index.to_excel(INDEX_XLSX, index=False)
print(f"\nAI-companion threads: {index['include'].sum():,} of {len(index):,} "
      f"(from search: {(index['include'].astype(bool) & index['source'].eq('search')).sum():,})")
 
# ---------- pick threads: take bars in turn, so one bar can't fill the whole cap ----------
eligible = index[index["include"] == 1]
todo = eligible.assign(turn=eligible.groupby("bar").cumcount()).sort_values("turn", kind="stable").head(MAX_THREADS)
 
# ---------- scrape (already scraped threads are skipped) ----------
for i, thread in enumerate(todo.to_dict("records"), 1):
    path = RAW_DIR / f"{thread['thread_id']}.jsonl"
    if path.exists():
        continue
    if not SALT:                                        # only needed when new threads are scraped
        raise SystemExit("Set TIEBA_SALT (a private string) before scraping, so author IDs are hashed.")
    try:
        records = scrape_thread(thread)
    except Exception as error:
        print(f"[{i}/{len(todo)}] {thread['thread_id']}: ERROR {error}")
        continue
    if not records:                                     # don't save empty files, so a rerun retries
        continue
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    print(f"[{i}/{len(todo)}] {thread['thread_id']}: {len(records):,} posts | {thread['title'][:30]}")
    pause()
 
# ---------- combine + clean ----------
def clean_text(text):
    text = re.sub(r"^回复\s*[^:：]{0,30}[:：]", "", str(text))
    text = re.sub(r"[^一-鿿A-Za-z\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()
 
keep = set(todo["thread_id"])
records = [json.loads(line) for path in RAW_DIR.glob("*.jsonl") if path.stem in keep
           for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
df = pd.DataFrame(records).drop_duplicates("post_id")
df["clean"] = df["text"].fillna("").map(clean_text)
df = df[df["clean"].str.len() >= MIN_CHARS]
df["year"] = df["date"].str[:4]
df[["text", "thread_title"]] = df[["text", "thread_title"]].replace(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", regex=True)
df.to_excel(POSTS_XLSX, index=False)
 
print(f"\nThreads: {df['thread_id'].nunique():,} | Posts: {len(df):,}")
print(df["post_type"].value_counts().to_string())
print(df["bar"].value_counts().head(10).to_string())
print(f"Saved: {POSTS_XLSX}")
