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
 
DATA = Path(__file__).resolve().parents[1] / "data"      # archive/tieba_pilot/data/
df = pd.read_excel(DATA / "tieba_posts.xlsx", sheet_name="posts").fillna("")
 
D_CODES = {
    "D1_Genuine_Attachment": ["愿意", "喜欢", "记忆", "感觉", "朋友", "情感", "真的", "觉得"],
    "D2_Emotional_Substitution": ["人类", "机器人", "社会", "情感", "工作", "用户", "没有", "只能"],
    "D3_Playful_Performance": ["游戏", "剧情", "设定", "小说", "直接", "回答", "东西", "事情", "凉凉", "哈哈"],
    "D4_Dangerous_Dependence": ["无限", "只能", "不能", "不会", "复活", "记忆", "未来", "地狱"],}
 
LABELS = {
    "D1_Genuine_Attachment": "D1 Genuine attachment",
    "D2_Emotional_Substitution": "D2 Emotional substitution",
    "D3_Playful_Performance": "D3 Playful performance",
    "D4_Dangerous_Dependence": "D4 Dangerous dependence",}
 
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
one["dimension"] = one[DIMS].idxmax(axis=1).map(LABELS)
one[["thread_id", "text", "clean", "dimension"] + DIMS].to_excel(DATA / "one_dimension_posts.xlsx", index=False)
 
# ============================================================
# 3. POSTS MATCHING TWO OR MORE FRAMES (dominant = most keywords matched; ties listed together)
# ============================================================
multi = df[df["n_dimensions"] >= 2].copy()
counts = pd.DataFrame({dim: multi[dim + "_terms"].str.len() for dim in DIMS})
multi["dominant_dimension"] = [" | ".join(LABELS[d] for d in DIMS if row[d] == row.max())
                               for _, row in counts.iterrows()]
multi[["thread_id", "text", "clean"] + DIMS + ["dominant_dimension"]].to_excel(
    DATA / "two_plus_dimension_posts.xlsx", index=False)
 
print(f"Posts: {len(df):,} | one frame: {len(one):,} | two or more frames: {len(multi):,}")
