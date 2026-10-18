"""
baidu_dictionary_coding.py  (pilot, not used in the final analysis)
Code each pilot post on the four frames with a first keyword list (D1-D4).
A post has a frame if its cleaned text contains any of the frame's keywords (multi-label).
  1. Code every post on the four frames and count how many frames it matches.
  2. Posts matching exactly one frame.
  3. Posts matching two or more frames, with their dominant frame (most keywords matched).
Many of these keywords turned out to be too general (e.g. 真的, 没有, 东西), which is why the final
analysis uses the refined seed dictionary in dictionaries/.
Input:  archive/tieba_pilot/data/tieba_posts.xlsx                       (from scraper_and_preprocessing.py)
Output: archive/tieba_pilot/data/one_dimension_posts.xlsx, two_plus_dimension_posts.xlsx

"""
from pathlib import Path
import pandas as pd
 
DATA = Path(__file__).resolve().parents[1] / "data"
df = pd.read_excel(DATA / "tieba_posts.xlsx", sheet_name="posts").fillna("")
 
# ---------- first keyword list for the four frames ----------
D_CODES = {
    "D1_Genuine_Attachment":     ["愿意", "喜欢", "记忆", "感觉", "朋友", "情感", "真的", "觉得"],
    "D2_Emotional_Substitution": ["人类", "机器人", "社会", "情感", "工作", "用户", "没有", "只能"],
    "D3_Playful_Performance":    ["游戏", "剧情", "设定", "小说", "直接", "回答", "东西", "事情", "凉凉", "哈哈"],
    "D4_Dangerous_Dependence":   ["无限", "只能", "不能", "不会", "复活", "记忆", "未来", "地狱"],}
LABELS = {"D1_Genuine_Attachment": "D1 Genuine attachment", "D2_Emotional_Substitution": "D2 Emotional substitution",
          "D3_Playful_Performance": "D3 Playful performance", "D4_Dangerous_Dependence": "D4 Dangerous dependence"}
DIMS = list(D_CODES)
df = pd.read_excel(DATA / "tieba_posts.xlsx", sheet_name="posts").fillna("")
 
# ============================================================
# 1. CODE POSTS
# ============================================================
for dim, keywords in D_CODES.items():
    df[dim + "_terms"] = df["clean"].map(lambda text: [k for k in keywords if k in text])   # matched keywords
    df[dim] = (df[dim + "_terms"].str.len() > 0).astype(int)
df["n_dimensions"] = df[DIMS].sum(axis=1)
 
# ============================================================
# 2. POSTS MATCHING ONE FRAME
# ============================================================
one = df[df["n_dimensions"] == 1].copy()
one_counts = one[DIMS].sum()                                     # each of these posts has exactly one frame
 
fig, ax = plt.subplots(figsize=(6.2, 4.7))
bars = ax.bar([LABELS[d] for d in DIMS], one_counts, color=["#ffee8c", "#6fa3c8", "#ff8da1", "#d97979"])
ax.bar_label(bars)
ax.set_ylabel("Number of posts")
ax.set_title("Posts with One Dimension")
plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig(FIGURES / "posts_with_one_dimension.png", dpi=300); plt.close()
 
# ============================================================
# 3. POSTS MATCHING TWO OR MORE FRAMES: dominant frame (most keywords matched; ties all counted)
# ============================================================
multi = df[df["n_dimensions"] >= 2].copy()
counts = pd.DataFrame({dim: multi[dim + "_terms"].str.len() for dim in DIMS})   # keywords matched per frame
dominant_counts = counts.eq(counts.max(axis=1), axis=0).sum()   # how often each frame has the most keywords
 
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([LABELS[d] for d in DIMS], dominant_counts, color=["#9b8acb", "#7fa9cc", "#c995c4", "#d98b8b"])
ax.bar_label(bars)
ax.set_ylabel("Number of posts")
ax.set_title("Relationship Dimensions in 2+ Dimension Posts")
plt.xticks(rotation=15)
plt.tight_layout(); plt.savefig(FIGURES / "posts_with_two_plus_dimensions.png", dpi=300); plt.close()
 
print(f"Posts: {len(df):,} | one frame: {len(one):,} | two or more frames: {len(multi):,}")
print("Saved posts_with_one_dimension.png and posts_with_two_plus_dimensions.png")
