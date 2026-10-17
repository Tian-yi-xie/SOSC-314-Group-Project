import re
import subprocess
import sys
from pathlib import Path
import jieba
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from sklearn.feature_extraction.text import CountVectorizer
from utils import load_stopwords
 
# ---------- folders (relative to the tieba/ folder of the repository) ----------
TIEBA = Path(__file__).resolve().parents[1]
DATA, FIGURES = TIEBA / "data", TIEBA / "figures"
FW_DIR = FIGURES / "fightin_words"
FW_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC", "SimHei"]
 
# ---------- Fightin' Words code (github.com/jmhessel/FightingWords), downloaded once ----------
if not (DATA / "FightingWords").exists():
    subprocess.run(["git", "clone", "https://github.com/jmhessel/FightingWords.git"], check=True, cwd=DATA)
sys.path.insert(0, str(DATA))
from FightingWords import fighting_words_py3
 
# ---------- settings ----------
START, CUTOFF = "2022-11-01", "2026-04-10"          # posts from START; before / after the rules at CUTOFF
TOP = 15                                            # words per side in each chart
COLORS = {"A": "#6fa2df", "F": "#ef9265", "G": "#4fbf9f", "C": "#dfa935", "E": "#e586ab",
          "B": "#7cbf73", "D": "#9c89db", "H": "#e4e584", "I": "#a3785c"}
 
# app names / slang jieba should keep as ONE word (otherwise 智能体 -> 智能 + 体)
CUSTOM_WORDS = ["智能体", "猫箱", "星野", "筑梦岛", "包打听", "破甲", "酒馆", "豆包", "千问",
                "元宝", "小艺", "哈基米", "鲸鱼娘", "人机恋", "聊天记录", "防沉迷", "飞机杯", "笑眼", "客户端"]
CUSTOM_STOPWORDS = {"这是", "这件", "包括", "一次", "两个", "东西", "事情", "里", "次", "出", "需", "挺",
                    "回复", "楼主", "楼下", "顶", "镇楼", "插个", "图片", "我发", "发了", "站", "贴", "帖", "主", "务", "大佬"}
for word in CUSTOM_WORDS:
    jieba.add_word(word)
STOPWORDS = load_stopwords() | CUSTOM_STOPWORDS
 
def tokenize(text):
    return " ".join(t for t in jieba.lcut(str(text).lower())
                    if t.strip() and t not in STOPWORDS
                    and not (len(t) == 1 and t.isascii()) and not t.isdigit()
                    and (re.search(r"[一-鿿]", t) or len(t) <= 8))
 
# ============================================================
# 1-2. TOKENIZE, THEN ATTACH THEME AND PERIOD (one row per post)
# ============================================================
themes = pd.read_excel(DATA / "cluster_profiles.xlsx")[["cluster", "theme"]].dropna()
if themes.empty:
    raise SystemExit("No themes yet: write a theme (A-I) for each cluster in cluster_profiles.xlsx (see 05).")
posts = pd.read_excel(DATA / "posts.xlsx", dtype={"post_id": str})[["post_id", "date", "clean"]]
clusters = pd.read_excel(DATA / "embedding_clusters_all.xlsx", dtype={"post_id": str})[["post_id", "cluster_all"]]
 
posts["tokenized"] = posts["clean"].fillna("").map(tokenize)
df = (posts[posts["tokenized"] != ""]
      .merge(clusters, on="post_id")
      .merge(themes, left_on="cluster_all", right_on="cluster"))       # post -> cluster -> theme
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["date"] >= START]
df["period"] = np.where(df["date"] >= CUTOFF, "after", "before")
df = df.drop_duplicates("tokenized")                                        # copy-pasted / repeated posts
df = df[~df["tokenized"].str.contains("magnet|复制到|点开|链接", regex=True)]   # link-spam posts
print(pd.crosstab(df["theme"], df["period"])[["before", "after"]].to_string())
 
# ============================================================
# 3. FIGHTIN' WORDS PER THEME
# ============================================================
with pd.ExcelWriter(DATA / "fightin_words_by_theme.xlsx") as writer:
    for theme in sorted(df["theme"].unique()):
        part = df[df["theme"] == theme]
        after_texts = part.loc[part["period"] == "after", "tokenized"]
        before_texts = part.loc[part["period"] == "before", "tokenized"]
 
        # word counts per post (binary), words in at least 10 posts; result sorted from most BEFORE to most AFTER
        cv = CountVectorizer(token_pattern=r"\S+", min_df=10, max_features=5000, binary=True, dtype=np.int8)
        fw = pd.DataFrame(fighting_words_py3.bayes_compare_language(after_texts, before_texts, cv=cv),
                          columns=["word", "z-score"])
        fw.to_excel(writer, sheet_name=theme[0], index=False)                # one sheet per theme: A, B, C...
        print(f"\n===== {theme}  ({len(before_texts):,} posts before, {len(after_texts):,} after) =====")
        print("more typical AFTER: ", ", ".join(fw["word"][-20:][::-1]))
        print("more typical BEFORE:", ", ".join(fw["word"][:20]))
 
        # chart: strongest words on each side (light = before, full colour = after, grey = weak, |z| < 1.96)
        before = fw[fw["z-score"] < 0].head(TOP)                             # most "before" first
        after = fw[fw["z-score"] > 0].tail(TOP)[::-1]                        # most "after" first
        col = COLORS[theme[0]]
        lim = max(fw["z-score"].abs().max(), 3) * 1.35                       # room for the word labels
        fig, ax = plt.subplots(figsize=(10, 7.5), facecolor="white")
        ax.set_facecolor(to_rgba(col, 0.07))
        for words, shade in [(before, 0.45), (after, 1.0)]:
            for i, (w, z) in enumerate(zip(words["word"], words["z-score"])):
                clear = abs(z) > 1.96
                ax.barh(i, z, color=to_rgba(col, shade) if clear else "#dcdbd6", height=0.72)
                ax.text(z + (1 if z > 0 else -1) * lim * 0.015, i, w, va="center",
                        ha="left" if z > 0 else "right", fontsize=10,
                        color="#2f2e2b" if clear else "#a8a7a2", fontweight="bold" if clear and z > 0 else "normal")
        ax.axvline(0, color=col, lw=1.5)
        for x in (-1.96, 1.96):
            ax.axvline(x, color="#b8b7b2", ls=":", lw=1)                     # beyond these lines = clear difference
        ax.set_xlim(-lim, lim); ax.set_ylim(TOP - 0.3, -1.4)
        ax.text(0.01, 0.985, f"← more typical BEFORE ({START} → {CUTOFF})", transform=ax.transAxes,
                fontsize=9, color="#7a7974", va="top")
        ax.text(0.99, 0.985, f"more typical AFTER ({CUTOFF} → data collection) →", transform=ax.transAxes,
                fontsize=9, color="#7a7974", va="top", ha="right")
        ax.set_yticks([]); ax.tick_params(axis="x", colors="#9b9a95", labelsize=8)
        ax.set_xlabel("Fightin' Words z-score   (grey = weak difference, |z| < 1.96)", color="#7a7974", fontsize=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{theme}\n", loc="left", fontsize=15, fontweight="bold", color="#3d3c39")
        ax.text(0, 1.02, f"{len(before_texts):,} posts before  ·  {len(after_texts):,} posts after",
                transform=ax.transAxes, fontsize=9, color="#7a7974")
        plt.tight_layout()
        plt.savefig(FW_DIR / f"{theme[0]}_fightin_words.png", dpi=200, bbox_inches="tight"); plt.close()
 
# ============================================================
# 4. ALL THEME CHARTS IN ONE IMAGE (overview for slides)
# ============================================================
files = sorted(FW_DIR.glob("?_fightin_words.png"))
COLS = 4
rows = -(-len(files) // COLS)                                                # rounded up
fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 10, rows * 7.5), squeeze=False, facecolor="white")
for ax in axes.flat:
    ax.axis("off")
for ax, f in zip(axes.flat, files):
    ax.imshow(mpimg.imread(f))
plt.tight_layout(pad=0.5)
plt.savefig(FW_DIR / "all_themes.png", dpi=150, facecolor="white"); plt.close()
print(f"\nSaved fightin_words_by_theme.xlsx and {len(files)} theme charts + all_themes.png")
