from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
 
# ---------- folders (relative to the tieba/ folder of the repository) ----------
TIEBA = Path(__file__).resolve().parents[1]
DATA, FIGURES = TIEBA / "data", TIEBA / "figure"
for folder in (DATA, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)
 
K, AGREE = 15, 0.8          # 15 nearest clustered posts, at least 80% must agree
 
df = pd.read_excel(DATA / "embedding_clusters.xlsx", dtype={"post_id": str, "thread_id": str})
space = np.load(DATA / "cluster_space.npy")
assert len(df) == len(space)
 
# ---------- 1. assign noise posts when their neighbours agree ----------
labels = df["cluster"].to_numpy()
clustered, noise_idx = labels != -1, np.where(labels == -1)[0]
_, idx = NearestNeighbors(n_neighbors=K).fit(space[clustered]).kneighbors(space[noise_idx])
assigned = labels.copy()
for i, neighbours in zip(noise_idx, labels[clustered][idx]):
    values, counts = np.unique(neighbours, return_counts=True)
    if counts.max() / K >= AGREE:
        assigned[i] = values[counts.argmax()]
df["cluster_all"] = assigned
df["from_noise"] = (labels == -1) & (assigned != -1)
df.to_excel(DATA / "embedding_clusters_all.xlsx", index=False)
print(f"Noise before: {(labels == -1).mean():.0%} | after: {(assigned == -1).mean():.0%}")
 
# ---------- 2. final map (2-D positions from script 03; recomputed if missing) ----------
if (DATA / "umap_2d.npy").exists():
    coords = np.load(DATA / "umap_2d.npy")
else:
    import umap
    coords = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.5, metric="cosine",
                       random_state=123).fit_transform(np.load(DATA / "embeddings.npy"))
lo, hi = np.percentile(coords, [1, 99], axis=0)               # zoom: ignore far-away stragglers
pad = (hi - lo) * 0.05
clusters = sorted(c for c in set(assigned) if c != -1)
colors = plt.cm.turbo(np.linspace(0, 1, len(clusters)))
np.random.default_rng(0).shuffle(colors)                       # neighbouring clusters get different colours
 
fig, ax = plt.subplots(figsize=(12, 10))
ax.scatter(*coords[assigned == -1].T, c="#dddddd", s=1, alpha=0.3, rasterized=True)    # noise
for color, c in zip(colors, clusters):
    pts = coords[assigned == c]
    ax.scatter(*pts.T, color=color, s=2, alpha=0.5, rasterized=True)
    ax.text(*np.median(pts, axis=0), str(c), fontsize=8, weight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
ax.set_aspect("equal"); ax.axis("off")
ax.set_title(f"{len(clusters)} clusters | noise {(assigned == -1).mean():.0%}")
plt.tight_layout(); plt.savefig(FIGURES / "embedding_clusters_all.png", dpi=300)
