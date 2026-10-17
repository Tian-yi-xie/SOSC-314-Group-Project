from collections import Counter
from pathlib import Path
 
import jieba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from sklearn.metrics.pairwise import cosine_similarity
 
from utils import load_stopwords
 
# ---------- folders (relative to the tieba/ folder of the repository) ----------
TIEBA = Path(__file__).resolve().parents[1]
DATA, FIGURES = TIEBA / "data", TIEBA / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
PROFILES = DATA / "cluster_profiles.xlsx"
 
clusters = pd.read_excel(DATA / "embedding_clusters_all.xlsx", dtype={"post_id": str})
labels = clusters["cluster_all"].to_numpy()
ids = sorted(set(labels) - {-1})                    # cluster numbers, noise (-1) left out
 
# ============================================================
# 1. CLUSTER SIMILARITY
# ============================================================
emb = np.load(DATA / "embeddings.npy")
emb = emb - emb.mean(axis=0)                        # centre: remove what every post shares (being about AI on Tieba)
centre = np.vstack([emb[labels == k].mean(axis=0) for k in ids])     # one average vector per cluster
sim = pd.DataFrame(cosine_similarity(centre), index=ids, columns=ids).round(2)
sim.to_excel(DATA / "cluster_similarity.xlsx")
 
# similarity tree (average linkage); its leaf order puts related clusters side by side
Z = linkage(centre, "average", metric="cosine", optimal_ordering=True)
order = [ids[i] for i in leaves_list(Z)]
 
# heatmap: darker = more similar; families of clusters show up as dark blocks on the diagonal
m = sim.loc[order, order].to_numpy(dtype=float).copy()
np.fill_diagonal(m, np.nan)                         # hide self-similarity (always 1)
plt.figure(figsize=(11, 10))
plt.imshow(m, cmap="Reds", vmin=np.nanmin(m), vmax=np.nanmax(m))
plt.xticks(range(len(order)), order, fontsize=8)
plt.yticks(range(len(order)), order, fontsize=8)
plt.colorbar(shrink=0.7, label="cosine similarity")
plt.title("Cosine similarity between clusters")
plt.tight_layout(); plt.savefig(FIGURES / "cluster_similarity.png", dpi=200); plt.close()
 
# 3) Export each cluster's three most similar clusters (by cosine similarity) to Excel for theme grouping
rows = []
for k in ids:
    row = {"cluster": k}
    for i, (j, v) in enumerate(sim.loc[k].drop(k).nlargest(3).items(), start=1):
        row[f"neighbor_{i}"], row[f"similarity_{i}"] = j, v
    rows.append(row)
pd.DataFrame(rows).to_excel(DATA / "cluster_top_neighbors.xlsx", index=False)
 
# ============================================================
# 2. NINE GROUPS: cut the same tree into 9 groups
# ============================================================
group = dict(zip(ids, fcluster(Z, t=9, criterion="maxclust")))
print("Clusters in each of the 9 groups:")
for g in sorted(set(group.values())):
    print(f"  group {g}: {[k for k in order if group[k] == g]}")
 
# ============================================================
# 3. CLUSTER PROFILES: what does each cluster say?
# ============================================================
N_WORDS, N_POSTS = 15, 15       # per cluster: top distinctive words, and example posts to read
 
if PROFILES.exists():
    print(f"\n{PROFILES.name} already exists (it holds our hand-written topics and themes), so it is not overwritten.")
else:
    posts = pd.read_excel(DATA / "posts.xlsx", dtype={"post_id": str})
    df = clusters.merge(posts[["post_id", "text", "clean"]], on="post_id").fillna("")
    stop = load_stopwords() | {"回复", "楼主", "顶"}
    df["tokens"] = df["clean"].str.lower().map(lambda t: [w for w in jieba.lcut(t) if w.strip() and w not in stop])
    total = Counter(w for ws in df["tokens"] for w in ws)
    n_total = sum(total.values())
 
    def distinctive_words(tokens, prior=1000):
        """Fightin' Words: words typical of this cluster compared with all other posts."""
        a = Counter(w for ws in tokens for w in ws)
        words = [w for w in a if total[w] >= 5]                       # ignore very rare words
        ya = np.array([a[w] for w in words], float)                   # count in this cluster
        yb = np.array([total[w] - a[w] for w in words], float)        # count everywhere else
        alpha = (ya + yb) / n_total * prior                           # informative prior
        na, nb = ya.sum(), n_total - ya.sum()
        z = (np.log((ya + alpha) / (na + prior - ya - alpha)) - np.log((yb + alpha) / (nb + prior - yb - alpha))) \
            / np.sqrt(1 / (ya + alpha) + 1 / (yb + alpha))
        return [words[i] for i in np.argsort(-z)[:N_WORDS]]
 
    summary, examples = [], []
    for c in order:                                                   # same order as the heatmap
        g = df[df["cluster_all"] == c]
        summary.append({"group": group[c], "cluster": c, "posts": len(g), "share": f"{len(g) / len(df):.1%}",
                        "words": ", ".join(distinctive_words(g["tokens"])), "topic": "", "theme": ""})
        top = (g[g["text"].str.len() >= 15].sort_values("cluster_prob", ascending=False)
               .drop_duplicates("text").head(N_POSTS))
        examples += [{"group": group[c], "cluster": c, "post": t[:200]} for t in top["text"]]
 
    summary = pd.DataFrame(summary).sort_values(["group", "posts"], ascending=[True, False])
    with pd.ExcelWriter(PROFILES) as writer:
        summary.to_excel(writer, sheet_name="clusters", index=False)
        pd.DataFrame(examples).to_excel(writer, sheet_name="posts", index=False)
    print("\n" + summary[["group", "cluster", "posts", "share", "words"]].to_string(index=False))
    print(f"\nSaved {PROFILES.name}: now write the topic and theme of each cluster by hand.")
