from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage


# ---------------------------------------------------------
# Path setup
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "src" / "data" / "raw_placement_data.csv"
FIGURES = ROOT / "reports" / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv(DATA)


# ---------------------------------------------------------
# Select features for clustering
# ---------------------------------------------------------
features = [
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count",
]

X = df[features].copy()


# ---------------------------------------------------------
# Convert college tier to numeric
# ---------------------------------------------------------
X["college_tier"] = (
    X["college_tier"]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
)

X["college_tier"] = pd.to_numeric(
    X["college_tier"],
    errors="coerce"
)


# ---------------------------------------------------------
# Remove missing values
# ---------------------------------------------------------
X = X.dropna()


# ---------------------------------------------------------
# Standardize features
# ---------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---------------------------------------------------------
# Function to calculate clustering scores
# ---------------------------------------------------------
def clustering_scores(labels, data):
    """
    Calculate silhouette score and Davies-Bouldin score.
    DBSCAN noise points (-1) are ignored.
    """

    labels = np.asarray(labels)

    mask = labels != -1

    filtered_labels = labels[mask]
    filtered_data = data[mask]

    unique_labels = np.unique(filtered_labels)

    if len(unique_labels) < 2:
        return np.nan, np.nan, len(unique_labels)

    sample_size = min(2000, len(filtered_data))

    silhouette = silhouette_score(
        filtered_data,
        filtered_labels,
        sample_size=sample_size,
        random_state=42,
    )

    db_score = davies_bouldin_score(
        filtered_data,
        filtered_labels
    )

    return silhouette, db_score, len(unique_labels)


# ---------------------------------------------------------
# K-Means: Find optimal K using Silhouette Score
# ---------------------------------------------------------
k_values = range(2, 11)

inertias = []
silhouette_scores = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)

    sample_size = min(2000, len(X_scaled))

    score = silhouette_score(
        X_scaled,
        labels,
        sample_size=sample_size,
        random_state=42,
    )

    silhouette_scores.append(score)


# ---------------------------------------------------------
# K-Means Elbow Plot
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Method")

plt.grid(True)

plt.savefig(
    FIGURES / "clustering_kmeans_elbow.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# K-Means Silhouette Plot
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("K-Means Silhouette Scores")

plt.grid(True)

plt.savefig(
    FIGURES / "clustering_kmeans_silhouette.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# Select optimal K
# ---------------------------------------------------------
best_index = int(np.argmax(silhouette_scores))
best_k = list(k_values)[best_index]

print("Optimal K:", best_k)
print(
    "Best Silhouette Score:",
    silhouette_scores[best_index]
)


# ---------------------------------------------------------
# Final K-Means
# ---------------------------------------------------------
kmeans = KMeans(
    n_clusters=best_k,
    n_init=10,
    random_state=42
)

kmeans_labels = kmeans.fit_predict(X_scaled)

kmeans_silhouette, kmeans_db, kmeans_clusters = (
    clustering_scores(
        kmeans_labels,
        X_scaled
    )
)


# ---------------------------------------------------------
# Hierarchical Agglomerative Clustering
# ---------------------------------------------------------
sample_size = min(1000, len(X_scaled))

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    len(X_scaled),
    size=sample_size,
    replace=False
)

X_hierarchical = X_scaled[sample_indices]

hierarchical = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)

hierarchical_labels = hierarchical.fit_predict(
    X_hierarchical
)

hierarchical_silhouette, hierarchical_db, hierarchical_clusters = (
    clustering_scores(
        hierarchical_labels,
        X_hierarchical
    )
)


# ---------------------------------------------------------
# Hierarchical Dendrogram
# ---------------------------------------------------------
sample_size = min(1000, len(X_scaled))

rng = np.random.default_rng(42)

sample_indices = rng.choice(
    len(X_scaled),
    size=sample_size,
    replace=False
)

X_sample = X_scaled[sample_indices]

linked = linkage(
    X_sample,
    method="ward"
)

plt.figure(figsize=(10, 6))

dendrogram(
    linked,
    truncate_mode="lastp",
    p=30
)

plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample / Cluster")
plt.ylabel("Distance")

plt.savefig(
    FIGURES / "hierarchical_dendrogram.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------
dbscan = DBSCAN(
    eps=0.8,
    min_samples=5
)

dbscan_labels = dbscan.fit_predict(X_scaled)

dbscan_silhouette, dbscan_db, dbscan_clusters = (
    clustering_scores(
        dbscan_labels,
        X_scaled
    )
)


# ---------------------------------------------------------
# Comparison Summary
# ---------------------------------------------------------
comparison = pd.DataFrame(
    {
        "Algorithm": [
            "K-Means",
            "Agglomerative",
            "DBSCAN"
        ],
        "Clusters": [
            kmeans_clusters,
            hierarchical_clusters,
            dbscan_clusters
        ],
        "Silhouette Score": [
            kmeans_silhouette,
            hierarchical_silhouette,
            dbscan_silhouette
        ],
        "Davies-Bouldin Score": [
            kmeans_db,
            hierarchical_db,
            dbscan_db
        ],
    }
)

comparison.to_csv(
    FIGURES / "clustering_comparison.csv",
    index=False
)


# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------
print("\nClustering Comparison:")
print(comparison)

print(
    "\nResults saved to:",
    FIGURES
)