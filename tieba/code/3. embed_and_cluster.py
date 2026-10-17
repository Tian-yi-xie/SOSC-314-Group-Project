import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
 
df = pd.read_excel("bigdata/posts.xlsx", dtype=str)
 
# ---------- 1. sentence embeddings ----------
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
embeddings = model.encode(df["clean"].fillna("").tolist(), batch_size=32,
                          normalize_embeddings=True, show_progress_bar=True)
np.save("bigdata/embeddings.npy", embeddings)
 
# ---------- 2. UMAP to 10 dimensions (similar posts packed tightly) ----------
space = umap.UMAP(
    n_components=10,
    n_neighbors=30,
    min_dist=0.0,
    metric="cosine",
    random_state=123).fit_transform(embeddings)
np.save("bigdata/cluster_space.npy", space)
 
# ---------- 3. HDBSCAN: clusters of at least 200 posts, the rest is noise ----------
clusterer clusterer = hdbscan.HDBSCAN(
    min_cluster_size=200,
    min_samples=10,
    cluster_selection_method="eom")
df["cluster"] = clusterer.fit_predict(space)
df["cluster_prob"] = clusterer.probabilities_
df["outlier_score"] = clusterer.outlier_scores_
df[["post_id", "thread_id", "bar", "thread_title", "text", "clean",
    "cluster", "cluster_prob", "outlier_score"]].to_excel("bigdata/embedding_clusters.xlsx", index=False)
 
# ---------- 4. 2-D map (for the picture only) ----------
reduced = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.5, metric="cosine",
                   random_state=123).fit_transform(embeddings)
np.save("bigdata/umap_2d.npy", reduced)
noise = (df["cluster"] == -1).to_numpy()
plt.figure(figsize=(10, 8))
plt.scatter(*coords[noise].T, c="lightgrey", s=5, alpha=0.3)
plt.scatter(*coords[~noise].T, c=df.loc[~noise, "cluster"], cmap="tab20", s=5, alpha=0.6)
plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2"); plt.title("Semantic clusters from embeddings")
plt.tight_layout(); plt.savefig("bigdata/embedding_clusters.png", dpi=300)
 
print(f"Embeddings: {embeddings.shape} | Clusters: {df['cluster'].nunique() - 1} | Noise: {noise.mean():.1%}")
