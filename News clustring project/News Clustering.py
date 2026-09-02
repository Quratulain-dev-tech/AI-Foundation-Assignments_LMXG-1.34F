"""
News Title Clustering Pipeline
================================
Applies K-Means, DBSCAN and Hierarchical (Agglomerative) clustering on a
CSV file of news titles, and automatically finds the "optimal" parameter
for each algorithm:

    - K-Means            -> optimal k via Elbow Method + Silhouette Score
    - DBSCAN             -> optimal eps via k-distance graph (knee point)
                             + optimal min_samples via silhouette search
    - Hierarchical (Agg) -> optimal number of clusters via Silhouette Score
                             + dendrogram

USAGE (from VS Code terminal):
    pip install -r requirements.txt
    python news_clustering.py --file "Untitled_spreadsheet_-_Sheet1.csv" --column Title

All plots are saved as PNG files in the "outputs" folder, and clustered
data is saved as CSV files (one per algorithm) + a combined comparison CSV.
"""

import argparse
import os
import re
import warnings

import matplotlib
matplotlib.use("Agg")  # so it works even without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

warnings.filterwarnings("ignore")

OUT_DIR = "outputs"


# --------------------------------------------------------------------------- #
# 1. TEXT CLEANING
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)     # keep letters only
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# 2. VECTORIZATION
# --------------------------------------------------------------------------- #
def vectorize(titles, max_features=3000, svd_components=100):
    """TF-IDF -> TruncatedSVD (dense, reduced-dim, good for clustering + speed)."""
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words=list(ENGLISH_STOP_WORDS),
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
    )
    X_tfidf = tfidf.fit_transform(titles)

    n_comp = min(svd_components, X_tfidf.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)
    print(f"[Vectorize] TF-IDF shape: {X_tfidf.shape} | "
          f"SVD-reduced shape: {X_reduced.shape} | "
          f"explained variance: {svd.explained_variance_ratio_.sum():.2%}")
    return X_tfidf, X_reduced, tfidf, svd


# --------------------------------------------------------------------------- #
# 3. K-MEANS: optimal k via Elbow + Silhouette
# --------------------------------------------------------------------------- #
def find_optimal_kmeans(X, k_min=2, k_max=15):
    inertias, sil_scores, ks = [], [], list(range(k_min, k_max + 1))

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        print(f"  k={k:2d} | inertia={km.inertia_:10.2f} | silhouette={sil_scores[-1]:.4f}")

    # --- knee point of elbow curve (max distance from the line joining ends) ---
    elbow_k = ks[_find_knee(inertias)]
    # --- best k by silhouette ---
    silhouette_k = ks[int(np.argmax(sil_scores))]

    # ---- PLOT 1: Elbow Method (Inertia vs k) ----
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, "bo-", linewidth=2, markersize=7)
    plt.axvline(elbow_k, color="b", linestyle="--", alpha=0.6)
    elbow_y = inertias[ks.index(elbow_k)]
    plt.scatter([elbow_k], [elbow_y], color="red", s=150, zorder=5,
                label=f"Elbow point -> k={elbow_k}")
    plt.annotate(f"Elbow (best k by this method)\nk={elbow_k}, inertia={elbow_y:.1f}",
                 xy=(elbow_k, elbow_y), xytext=(elbow_k + 0.6, elbow_y + (max(inertias) - min(inertias)) * 0.12),
                 arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (sum of squared distances within clusters)")
    plt.title("Elbow Method: Inertia vs Number of Clusters (k)\n"
              "Lower inertia = tighter clusters. We look for the 'bend' where\n"
              "adding more clusters stops giving much improvement.")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/kmeans_elbow.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- PLOT 2: Silhouette Score vs k ----
    plt.figure(figsize=(8, 5))
    plt.plot(ks, sil_scores, "gs-", linewidth=2, markersize=7)
    sil_y = sil_scores[ks.index(silhouette_k)]
    plt.scatter([silhouette_k], [sil_y], color="red", s=150, zorder=5,
                label=f"Best k -> k={silhouette_k}")
    plt.annotate(f"Highest silhouette score\nk={silhouette_k}, score={sil_y:.3f}",
                 xy=(silhouette_k, sil_y), xytext=(silhouette_k + 0.6, sil_y - 0.05),
                 arrowprops=dict(arrowstyle="->", color="red"), fontsize=9, color="red")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score (-1 to 1)")
    plt.title("Silhouette Score vs Number of Clusters (k)\n"
              "Higher score = points fit better in their own cluster vs other clusters.\n"
              "We pick the k with the highest score (marked in red).")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/kmeans_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Prefer the silhouette-optimal k (more reliable than elbow alone);
    # elbow_k is reported too, for reference / comparison in the plot.
    optimal_k = silhouette_k
    print(f"[K-Means] Elbow suggests k={elbow_k} | Silhouette suggests k={silhouette_k} "
          f"-> using k={optimal_k}")
    return optimal_k, {"ks": ks, "inertias": inertias, "silhouettes": sil_scores,
                        "elbow_k": elbow_k, "silhouette_k": silhouette_k}


def _find_knee(values):
    """Find index of the knee/elbow point using the max-distance-from-line method."""
    values = np.array(values, dtype=float)
    n = len(values)
    x = np.arange(n)
    # normalize
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_norm = (values - values.min()) / (values.max() - values.min() + 1e-12)
    # line from first to last point
    p1, p2 = np.array([x_norm[0], y_norm[0]]), np.array([x_norm[-1], y_norm[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    distances = []
    for xi, yi in zip(x_norm, y_norm):
        p = np.array([xi, yi]) - p1
        proj_len = np.dot(p, line_vec_norm)
        proj_point = proj_len * line_vec_norm
        distances.append(np.linalg.norm(p - proj_point))
    return int(np.argmax(distances))


# --------------------------------------------------------------------------- #
# 4. DBSCAN: optimal eps via k-distance graph + best min_samples via silhouette
# --------------------------------------------------------------------------- #
def find_optimal_dbscan(X, min_samples_range=range(3, 11)):
    # k-distance graph for an initial min_samples (rule of thumb: 2*dim)
    k = max(min_samples_range)
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    k_distances = np.sort(distances[:, -1])

    knee_idx = _find_knee(k_distances)
    eps_guess = k_distances[knee_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(k_distances)
    plt.axhline(eps_guess, color="r", linestyle="--", label=f"eps (knee) = {eps_guess:.4f}")
    plt.axvline(knee_idx, color="g", linestyle="--", alpha=0.5)
    plt.xlabel("Points sorted by distance")
    plt.ylabel(f"{k}-th nearest neighbor distance")
    plt.title("DBSCAN: k-distance Graph (Elbow -> eps)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/dbscan_kdistance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # search a small grid around the eps guess and over min_samples
    eps_candidates = eps_guess * np.array([0.7, 0.85, 1.0, 1.15, 1.3])
    best = {"score": -1, "eps": None, "min_samples": None, "labels": None}
    results = []
    for eps in eps_candidates:
        for ms in min_samples_range:
            labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = np.mean(labels == -1)
            if n_clusters < 2 or n_clusters > 60 or noise_ratio > 0.5:
                continue
            mask = labels != -1
            try:
                score = silhouette_score(X[mask], labels[mask])
            except ValueError:
                continue
            results.append((eps, ms, n_clusters, noise_ratio, score))
            if score > best["score"]:
                best.update(score=score, eps=eps, min_samples=ms, labels=labels)

    if best["eps"] is None:
        print("[DBSCAN] No good eps/min_samples combo found in grid; falling back to knee eps guess.")
        best["eps"], best["min_samples"] = eps_guess, min(min_samples_range)
        best["labels"] = DBSCAN(eps=best["eps"], min_samples=best["min_samples"]).fit_predict(X)
    else:
        print(f"[DBSCAN] Best combo -> eps={best['eps']:.4f}, min_samples={best['min_samples']}, "
              f"silhouette={best['score']:.4f}")

    for eps, ms, nc, nr, sc in sorted(results, key=lambda r: -r[4])[:10]:
        print(f"  eps={eps:.4f} | min_samples={ms} | clusters={nc} | noise={nr:.1%} | silhouette={sc:.4f}")

    return best["eps"], best["min_samples"], {"k_distances": k_distances, "grid_results": results}


# --------------------------------------------------------------------------- #
# 5. HIERARCHICAL / AGGLOMERATIVE: optimal #clusters via Silhouette + dendrogram
# --------------------------------------------------------------------------- #
def find_optimal_hierarchical(X, k_min=2, k_max=15, sample_for_dendrogram=300):
    sil_scores, ks = [], list(range(k_min, k_max + 1))
    for k in ks:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X)
        score = silhouette_score(X, labels)
        sil_scores.append(score)
        print(f"  k={k:2d} | silhouette={score:.4f}")

    optimal_k = ks[int(np.argmax(sil_scores))]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, sil_scores, "go-")
    plt.axvline(optimal_k, color="r", linestyle="--", label=f"Best k={optimal_k}")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Hierarchical Clustering: Silhouette Score vs k")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/hierarchical_silhouette.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Dendrogram on a random sample (full 5000-row dendrogram is unreadable)
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=min(sample_for_dendrogram, X.shape[0]), replace=False)
    Z = linkage(X[idx], method="ward")
    plt.figure(figsize=(12, 6))
    dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90)
    plt.title(f"Dendrogram (sample of {len(idx)} titles, last 30 merges)")
    plt.xlabel("Cluster size / sample index")
    plt.ylabel("Distance (Ward)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/hierarchical_dendrogram.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[Hierarchical] Optimal number of clusters (silhouette) = {optimal_k}")
    return optimal_k, {"ks": ks, "silhouettes": sil_scores}


# --------------------------------------------------------------------------- #
# 6. CLUSTER INTERPRETATION (top TF-IDF terms per cluster)
# --------------------------------------------------------------------------- #
def top_terms_per_cluster(tfidf_matrix, tfidf_vectorizer, labels, n_terms=8):
    terms = np.array(tfidf_vectorizer.get_feature_names_out())
    summary = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            summary["Noise (-1)"] = ["-- noise points, no coherent topic --"]
            continue
        mask = labels == cluster_id
        mean_tfidf = np.asarray(tfidf_matrix[mask].mean(axis=0)).ravel()
        top_idx = mean_tfidf.argsort()[::-1][:n_terms]
        summary[f"Cluster {cluster_id}"] = list(terms[top_idx])
    return summary


# --------------------------------------------------------------------------- #
# 7. 2D VISUALIZATION OF CLUSTERS (via SVD -> 2 components)
# --------------------------------------------------------------------------- #
def plot_clusters_2d(X, labels, title, filename):
    svd2 = TruncatedSVD(n_components=2, random_state=42)
    X2 = svd2.fit_transform(X) if X.shape[1] > 2 else X
    plt.figure(figsize=(9, 7))
    unique_labels = sorted(set(labels))
    try:
        cmap = matplotlib.colormaps["tab20"].resampled(max(len(unique_labels), 1))
    except AttributeError:
        cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 1))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        color = "lightgrey" if lab == -1 else cmap(i)
        label_name = "Noise" if lab == -1 else f"Cluster {lab}"
        plt.scatter(X2[mask, 0], X2[mask, 1], s=10, color=color, label=label_name, alpha=0.7)
    plt.title(title)
    plt.xlabel("SVD Component 1")
    plt.ylabel("SVD Component 2")
    if len(unique_labels) <= 20:
        plt.legend(markerscale=2, fontsize=8, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{filename}", dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def resolve_file_path(file_path):
    """If the given path doesn't exist, try to auto-find a CSV in the
    current folder (handles filename mismatches like spaces vs underscores)."""
    if os.path.exists(file_path):
        return file_path

    print(f"  [!] '{file_path}' not found in current folder ({os.getcwd()}).")
    csv_files = [f for f in os.listdir(".") if f.lower().endswith(".csv")]

    if len(csv_files) == 1:
        print(f"  [i] Found exactly one CSV file in this folder instead: '{csv_files[0]}' -> using it.")
        return csv_files[0]
    elif len(csv_files) > 1:
        print(f"  [i] Multiple CSV files found in this folder: {csv_files}")
        print(f"      Please re-run with the exact one you want, e.g.:")
        print(f"      python \"News Clustering.py\" --file \"{csv_files[0]}\"")
        raise FileNotFoundError(
            f"'{file_path}' not found, and multiple CSVs exist in {os.getcwd()} — "
            f"specify --file explicitly."
        )
    else:
        raise FileNotFoundError(
            f"'{file_path}' not found, and no CSV files exist in {os.getcwd()}. "
            f"Make sure the CSV is in the same folder as the script, or pass "
            f"the full path with --file \"C:\\full\\path\\to\\file.csv\"."
        )


def main(file_path, column, max_features, svd_components, k_min, k_max):
    os.makedirs(OUT_DIR, exist_ok=True)

    file_path = resolve_file_path(file_path)
    print(f"\n[1/6] Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")
    df = df.dropna(subset=[column]).reset_index(drop=True)
    df["clean_title"] = df[column].apply(clean_text)
    df = df[df["clean_title"].str.strip() != ""].reset_index(drop=True)
    print(f"  Loaded {len(df)} valid titles.")

    print("\n[2/6] Vectorizing titles (TF-IDF + TruncatedSVD)...")
    X_tfidf, X_reduced, tfidf_vectorizer, svd = vectorize(
        df["clean_title"], max_features=max_features, svd_components=svd_components
    )

    # ---------------- K-MEANS ----------------
    print("\n[3/6] K-Means: searching for optimal k ...")
    best_k, kmeans_info = find_optimal_kmeans(X_reduced, k_min=k_min, k_max=k_max)
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["kmeans_cluster"] = kmeans_final.fit_predict(X_reduced)
    plot_clusters_2d(X_reduced, df["kmeans_cluster"].values,
                      f"K-Means Clusters (k={best_k})", "kmeans_clusters_2d.png")
    kmeans_terms = top_terms_per_cluster(X_tfidf, tfidf_vectorizer, df["kmeans_cluster"].values)

    # ---------------- DBSCAN ----------------
    print("\n[4/6] DBSCAN: searching for optimal eps & min_samples ...")
    best_eps, best_min_samples, dbscan_info = find_optimal_dbscan(X_reduced)
    dbscan_final = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    df["dbscan_cluster"] = dbscan_final.fit_predict(X_reduced)
    n_dbscan_clusters = len(set(df["dbscan_cluster"])) - (1 if -1 in df["dbscan_cluster"].values else 0)
    print(f"  DBSCAN found {n_dbscan_clusters} clusters "
          f"({(df['dbscan_cluster'] == -1).mean():.1%} noise).")
    plot_clusters_2d(X_reduced, df["dbscan_cluster"].values,
                      f"DBSCAN Clusters (eps={best_eps:.3f}, min_samples={best_min_samples})",
                      "dbscan_clusters_2d.png")
    dbscan_terms = top_terms_per_cluster(X_tfidf, tfidf_vectorizer, df["dbscan_cluster"].values)

    # ---------------- HIERARCHICAL ----------------
    print("\n[5/6] Hierarchical/Agglomerative: searching for optimal number of clusters ...")
    best_h_k, hier_info = find_optimal_hierarchical(X_reduced, k_min=k_min, k_max=k_max)
    hier_final = AgglomerativeClustering(n_clusters=best_h_k, linkage="ward")
    df["hierarchical_cluster"] = hier_final.fit_predict(X_reduced)
    plot_clusters_2d(X_reduced, df["hierarchical_cluster"].values,
                      f"Hierarchical Clusters (k={best_h_k})", "hierarchical_clusters_2d.png")
    hier_terms = top_terms_per_cluster(X_tfidf, tfidf_vectorizer, df["hierarchical_cluster"].values)

    # ---------------- SAVE RESULTS ----------------
    print("\n[6/6] Saving results to CSV ...")
    out_cols = [column, "kmeans_cluster", "dbscan_cluster", "hierarchical_cluster"]
    df[out_cols].to_csv(f"{OUT_DIR}/clustered_titles.csv", index=False)

    with open(f"{OUT_DIR}/cluster_top_terms.txt", "w", encoding="utf-8") as f:
        for algo_name, terms_dict in [("K-MEANS", kmeans_terms),
                                       ("DBSCAN", dbscan_terms),
                                       ("HIERARCHICAL", hier_terms)]:
            f.write(f"\n===== {algo_name} =====\n")
            for cluster, terms in terms_dict.items():
                f.write(f"{cluster}: {', '.join(terms)}\n")

    summary = pd.DataFrame({
        "Algorithm": ["K-Means", "DBSCAN", "Hierarchical"],
        "Optimal_Parameter": [f"k={best_k}", f"eps={best_eps:.4f}, min_samples={best_min_samples}", f"k={best_h_k}"],
        "N_Clusters_Found": [
            best_k,
            n_dbscan_clusters,
            best_h_k,
        ],
        "Silhouette_Score": [
            silhouette_score(X_reduced, df["kmeans_cluster"]),
            (silhouette_score(X_reduced[df["dbscan_cluster"] != -1], df.loc[df["dbscan_cluster"] != -1, "dbscan_cluster"])
             if n_dbscan_clusters >= 2 else float("nan")),
            silhouette_score(X_reduced, df["hierarchical_cluster"]),
        ],
    })
    summary.to_csv(f"{OUT_DIR}/algorithm_comparison_summary.csv", index=False)

    print("\n================ DONE ================")
    print(summary.to_string(index=False))
    print(f"\nAll outputs saved in the '{OUT_DIR}/' folder:")
    print("  - clustered_titles.csv           (original titles + cluster labels for all 3 algorithms)")
    print("  - algorithm_comparison_summary.csv")
    print("  - cluster_top_terms.txt          (top keywords per cluster, for interpretation)")
    print("  - kmeans_elbow.png                (Inertia vs k -> Elbow Method)")
    print("  - kmeans_silhouette.png           (Silhouette Score vs k)")
    print("  - dbscan_kdistance.png")
    print("  - hierarchical_silhouette.png / hierarchical_dendrogram.png")
    print("  - kmeans_clusters_2d.png / dbscan_clusters_2d.png / hierarchical_clusters_2d.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster news titles with K-Means, DBSCAN, Hierarchical.")
    parser.add_argument("--file", type=str, default="Untitled_spreadsheet_-_Sheet1.csv",
                         help="Path to the CSV file containing news titles.")
    parser.add_argument("--column", type=str, default="Title",
                         help="Name of the column containing the news titles.")
    parser.add_argument("--max_features", type=int, default=3000,
                         help="Max TF-IDF vocabulary size.")
    parser.add_argument("--svd_components", type=int, default=100,
                         help="Number of TruncatedSVD components (dimensionality reduction before clustering).")
    parser.add_argument("--k_min", type=int, default=2, help="Minimum clusters to test for K-Means/Hierarchical.")
    parser.add_argument("--k_max", type=int, default=15, help="Maximum clusters to test for K-Means/Hierarchical.")
    args = parser.parse_args()

    main(args.file, args.column, args.max_features, args.svd_components, args.k_min, args.k_max)