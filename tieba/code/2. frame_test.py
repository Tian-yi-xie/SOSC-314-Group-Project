import json, re
from pathlib import Path
import jieba
import matplotlib.pyplot as plt
import pandas as pd
import requests
from statsmodels.stats.proportion import proportions_ztest
 
TIEBA = Path(__file__).resolve().parents[1]
DATA, FIGURES = TIEBA / "data", TIEBA / "figures"
for folder in (DATA, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)
 
SRC, TOKENS_OUT = DATA / "posts.xlsx", DATA / "posts_tokenized_frames.xlsx"
TABLE_OUT, FIG_OUT = DATA / "frames_test.xlsx", FIGURES / "frames_test.png"
STOP_PATH = DATA / "stopwords-zh.json"
CUTOFF = "2026-04-10"                    # rules issued: before = earlier, after = this day onward
 
# ---------- the four frames (seed words) ----------
FRAMES = {
    "genuine_attachment":     ["爱", "心动", "谈恋爱", "真的喜欢", "伴侣"],
    "emotional_substitution": ["替代", "比真人", "情绪价值", "倾诉", "不如ai"],
    "playful_performance":    ["整活", "捏人", "乐子", "玩梗", "笑死", "哈哈哈哈"],
    "dangerous_dependence":   ["上瘾", "戒不掉", "停不下来", "离不开", "崩溃", "哭死"],
}
COLORS = {"genuine_attachment": "#e586ab", "emotional_substitution": "#dfa935",
          "playful_performance": "#6fa2df", "dangerous_dependence": "#9c89db"}
SEEDS = [w for words in FRAMES.values() for w in words]
 
# ---------- tokenizer: keep app names, slang and seed words as one word ----------
CUSTOM_WORDS = ["智能体", "猫箱", "星野", "筑梦岛", "包打听", "破甲", "酒馆", "豆包", "千问",
                "元宝", "小艺", "哈基米", "鲸鱼娘", "人机恋", "聊天记录", "防沉迷", "飞机杯", "笑眼", "客户端"]
CUSTOM_STOPWORDS = {"这是", "这件", "包括", "一次", "两个", "东西", "事情", "里", "次", "出", "需", "挺",
                    "回复", "楼主", "楼下", "顶", "镇楼", "插个", "图片", "我发", "发了", "站", "贴", "帖", "主", "务"}
for word in CUSTOM_WORDS + [w for w in SEEDS if len(w) > 1]:
    jieba.add_word(word)
 
def load_stopwords(path):
    if not path.exists():
        url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-zh/master/stopwords-zh.json"
        path.write_text(json.dumps(requests.get(url, timeout=30).json(), ensure_ascii=False), encoding="utf-8")
    return set(json.loads(path.read_text(encoding="utf-8")))
 
STOPWORDS = (load_stopwords(STOP_PATH) | CUSTOM_STOPWORDS) - set(SEEDS)     # seed words are never dropped
 
def tokenize(text):
    return " ".join(t for t in jieba.lcut(str(text).lower())
                    if t.strip() and t not in STOPWORDS
                    and not (len(t) == 1 and t.isascii()) and not t.isdigit()
                    and (re.search(r"[一-鿿]", t) or len(t) <= 8))
 
# ---------- 1. tokenize and split into periods ----------
df = pd.read_excel(SRC, dtype=str)
df["tokenized"] = df["clean"].fillna("").map(tokenize)
df = df[df["tokenized"] != ""].copy()
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["period"] = "no date"
df.loc[df["date"] < CUTOFF, "period"] = "before"
df.loc[df["date"] >= CUTOFF, "period"] = "after"
df[["post_id", "date", "period", "clean", "tokenized"]].to_excel(TOKENS_OUT, sheet_name="posts", index=False)
print("posts:", df["period"].value_counts().to_dict())
 
# ---------- 2. which posts carry each frame ----------
for frame, words in FRAMES.items():
    df[frame] = df["tokenized"].map(lambda t: any(w in t for w in words))
 
# ---------- 3. share of posts before vs after, with a two-proportion z-test ----------
before, after = df[df["period"] == "before"], df[df["period"] == "after"]
rows = []
for frame in FRAMES:
    z, p = proportions_ztest([after[frame].sum(), before[frame].sum()], [len(after), len(before)])
    rows.append({"frame": frame, "before_%": 100 * before[frame].mean(), "after_%": 100 * after[frame].mean(),
                 "z": z, "p": p})
share = pd.DataFrame(rows).set_index("frame").round(3)
share["change"] = (share["after_%"] - share["before_%"]).round(2)
print("\n% of posts using each frame (|z| > 1.96 = clear difference):\n", share.to_string())
print(f"\nPosts matching none of the four frames: {(~df[list(FRAMES)].any(axis=1)).mean():.1%}")
 
words = pd.DataFrame([{"frame": frame, "word": w,
                       "before": before["tokenized"].str.contains(w, regex=False).sum(),
                       "after": after["tokenized"].str.contains(w, regex=False).sum()}
                      for frame, ws in FRAMES.items() for w in ws])
print("\nPosts per word:\n", words.to_string(index=False))
 
with pd.ExcelWriter(TABLE_OUT) as w:
    share.to_excel(w, sheet_name="share")
    words.to_excel(w, sheet_name="words", index=False)
 
# ---------- 4. bar chart ----------
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC", "SimHei"]
fig, ax = plt.subplots(figsize=(9, 5))
for i, frame in enumerate(FRAMES):
    for dx, col, alpha, label in [(-0.2, "before_%", 0.45, "before"), (0.2, "after_%", 1.0, "after")]:
        value = share.loc[frame, col]
        ax.bar(i + dx, value, 0.4, color=COLORS[frame], alpha=alpha, label=label if i == 0 else None)
        ax.text(i + dx, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(FRAMES)), [f.replace("_", "\n") for f in FRAMES])
ax.set_ylabel("% of posts")
ax.set_title("Four frames before vs after the rules (10 Apr 2026)", loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(FIG_OUT, dpi=200)
plt.show()
