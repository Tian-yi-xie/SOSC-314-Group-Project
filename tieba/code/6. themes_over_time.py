from pathlib import Path
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
# ---------- folders (relative to the tieba/ folder of the repository) ----------
TIEBA = Path(__file__).resolve().parents[1]
DATA, FIGURES = TIEBA / "data", TIEBA / "figure"
FIGURES.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Noto Sans CJK SC", "SimHei"]
 
# one colour per theme letter (same colours as in cluster_profiles.xlsx)
COLORS = {"A": "#6fa2df", "F": "#ef9265", "G": "#4fbf9f", "C": "#dfa935", "E": "#e586ab",
          "B": "#7cbf73", "D": "#9c89db", "H": "#e4e584", "I": "#a3785c"}
EVENTS = {"2022-11-30": "ChatGPT", "2023-08-17": "Doubao", "2026-04-10": "Rules issued"}
 
# ---------- post -> cluster -> theme, with the post's date ----------
themes = pd.read_excel(DATA / "cluster_profiles.xlsx")[["cluster", "theme"]].dropna()
if themes.empty:
    raise SystemExit("No themes yet: write a theme (A-I) for each cluster in cluster_profiles.xlsx (see 05).")
clusters = pd.read_excel(DATA / "embedding_clusters_all.xlsx", dtype={"post_id": str})[["post_id", "cluster_all"]]
dates = pd.read_excel(DATA / "posts.xlsx", dtype={"post_id": str})[["post_id", "date"]]
df = clusters.merge(dates, on="post_id").merge(themes, left_on="cluster_all", right_on="cluster")
df = df[df["theme"].str[0].isin(COLORS)]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df[df["date"] >= "2022-01-01"]
df["year"] = df["date"].dt.year
 
# ---------- posts per theme per year ----------
years = range(int(df["year"].min()), int(df["year"].max()) + 1)
counts = pd.crosstab(df["year"], df["theme"]).reindex(years, fill_value=0)
counts.to_excel(DATA / "themes_per_year.xlsx")
print(counts.to_string())
plot = counts.astype(float)
plot[counts.sum(axis=1) == 0] = np.nan              # years with no posts become gaps in the lines
last = df["date"].max()
 
# ---------- line chart ----------
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(15, 7))
ax.axvspan(last.year - 0.5, last.year + 0.5, color="#ecebe8", zorder=0)           # incomplete last year
ax.text(last.year, 0.02, f"incomplete\n(to {last:%d %b})", transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=8, color="#7a7974")
for theme in sorted(counts.columns, key=lambda t: list(COLORS).index(t[0])):
    ax.plot(plot.index, plot[theme], color=COLORS[theme[0]], lw=2.2, marker="o", ms=4, label=theme)
for day, name in EVENTS.items():                                                   # events as dotted lines
    x = pd.Timestamp(day).year - 0.5 + pd.Timestamp(day).dayofyear / 365
    ax.axvline(x, color="#9b9a95", ls=":", lw=1)
    ax.text(x, 0.98, name, transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="top",
            fontsize=8, color="#7a7974")
gap = [y for y in years if counts.loc[y].sum() == 0]
if gap:
    ax.text((gap[0] + gap[-1]) / 2, 0.5, f"no posts collected\n{gap[0]}–{gap[-1]}",
            transform=ax.get_xaxis_transform(), ha="center", va="center", fontsize=9, color="#9b9a95")
ax.set_xticks(list(years))
ax.tick_params(axis="x", rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.set_xlim(years[0] - 0.6, years[-1] + 0.6)
ax.set_ylim(0, counts.to_numpy().max() * 1.12)
ax.grid(color="#ecebe8", lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("Year"); ax.set_ylabel("Number of posts")
ax.set_title("Tieba posts per year, by theme", loc="left", fontsize=15, fontweight="bold")
handles, names = ax.get_legend_handles_labels()
rank = np.argsort([-counts[n].sum() for n in names])                               # biggest theme first
ax.legend([handles[i] for i in rank], [names[i] for i in rank], title="Theme",
          loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, title_fontsize=10)
plt.tight_layout(); plt.savefig(FIGURES / "themes_per_year_lines.png", dpi=200); plt.close()
print("Saved themes_per_year.xlsx and themes_per_year_lines.png")
