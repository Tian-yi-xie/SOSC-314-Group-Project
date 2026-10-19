from pathlib import Path
import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
 
TIEBA = Path(__file__).resolve().parents[1]
DATA, FIGURES = TIEBA / "data", TIEBA / "figure"
for folder in (DATA, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)
 
df = pd.read_excel(DATA / "posts.xlsx", dtype=str)
 
# ---------- 1. sentence embeddings ----------
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
embeddings = model.encode(df["clean"].fillna("").tolist(), batch_size=32,
                          normalize_embeddings=True, show_progress_bar=True)
np.save(DATA / "embeddings.npy", embeddings)

# [FIX] Our hand-written themes (cluster_profiles.xlsx) are tied to the saved cluster numbers.
# Re-running UMAP/HDBSCAN can renumber the clusters, so by default stop after the embeddings
# (script 5 needs embeddings.npy). Set RECLUSTER = True only to redo the clustering from scratch.
RECLUSTER = False
if not RECLUSTER and (DATA / "embedding_clusters_all.xlsx").exists():
    raise SystemExit("Saved embeddings.npy. Clusters already exist (embedding_clusters_all.xlsx), so they were not redone.")
 
# ---------- 2. UMAP to 10 dimensions (similar posts packed tightly) ----------
space = umap.UMAP(
    n_components=10,
    n_neighbors=30,
    min_dist=0.0,
    metric="cosine",
    random_state=123).fit_transform(embeddings)
np.save(DATA / "cluster_space.npy", space)
 
# ---------- 3. HDBSCAN: clusters of at least 200 posts, the rest is noise ----------
clusterer = hdbscan.HDBSCAN(   # [FIX] removed duplicated word
    min_cluster_size=200,
    min_samples=10,
    cluster_selection_method="eom")
df["cluster"] = clusterer.fit_predict(space)
df["cluster_prob"] = clusterer.probabilities_
df["outlier_score"] = clusterer.outlier_scores_
df[["post_id", "thread_id", "bar", "thread_title", "text", "clean",
    "cluster", "cluster_prob", "outlier_score"]].to_excel(DATA / "embedding_clusters.xlsx", index=False)   # [FIX] was "bigdata/..."
 
# ---------- 4. 2-D map (for the picture only) ----------
reduced = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.5, metric="cosine",
                   random_state=123).fit_transform(embeddings)
np.save(DATA / "umap_2d.npy", reduced)
noise = (df["cluster"] == -1).to_numpy()
plt.figure(figsize=(10, 8))
plt.scatter(*reduced[noise].T, c="lightgrey", s=5, alpha=0.3)
plt.scatter(*reduced[~noise].T, c=df.loc[~noise, "cluster"], cmap="tab20", s=5, alpha=0.6)
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2"); plt.title("Semantic clusters from embeddings")
plt.tight_layout(); plt.savefig(FIGURES / "embedding_clusters.png", dpi=300)   # [FIX] was "bigdata/..."
 
print(f"Embeddings: {embeddings.shape} | Clusters: {df['cluster'].nunique() - 1} | Noise: {noise.mean():.1%}")
