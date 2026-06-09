"""
03_compute_metrics.py — Compute self-similarity, MEV, and clustering metrics.

Usage:
    python 03_compute_metrics.py

Reads:
    data/embeddings_{concept}.npz files from step 02
    data/embeddings_random_baseline.npz for anisotropy correction

Outputs:
    results/metrics_summary.csv — self-similarity, MEV, cluster count per concept
    results/metrics_detail.json — full metric details including cluster info
"""

import numpy as np
import json
import csv
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial.distance import cosine
from itertools import combinations
import warnings
warnings.filterwarnings("ignore")

# --- Configuration ---

DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

CONCEPTS = ["reaction_time", "working_memory", "well_being", "resilience", "self_esteem"]

# GMM settings
MAX_K = 6  # Max clusters to try
N_GMM_INITS = 50  # Number of random initializations for robustness
N_ARI_RUNS = 100  # Number of ARI comparisons for stability check
PCA_DIMS_TO_TRY = [2, 3, 5, 10]  # PCA dimensions to try for GMM


def compute_anisotropy_baseline(baseline_path):
    """
    Compute the average cosine similarity between random sentence pairs.
    This is the anisotropy correction factor (Ethayarajh 2019).
    """
    data = np.load(baseline_path, allow_pickle=True)
    embs = data["embeddings"]

    # Sample pairs (compute on a subset for speed)
    n = min(len(embs), 300)
    sims = []
    for i, j in combinations(range(n), 2):
        sim = 1 - cosine(embs[i], embs[j])
        sims.append(sim)

    baseline = np.mean(sims)
    print(f"Anisotropy baseline (avg random cosine): {baseline:.4f}")
    return baseline


def compute_self_similarity(embeddings, anisotropy_baseline=0.0):
    """
    Average pairwise cosine similarity between all occurrences.
    Returns both raw and anisotropy-corrected values.
    """
    n = len(embeddings)
    if n < 2:
        return {"raw": np.nan, "corrected": np.nan, "n_pairs": 0}

    # For large n, sample pairs
    if n > 300:
        indices = np.random.choice(n, 300, replace=False)
        subset = embeddings[indices]
    else:
        subset = embeddings

    sims = []
    for i, j in combinations(range(len(subset)), 2):
        sim = 1 - cosine(subset[i], subset[j])
        sims.append(sim)

    raw = np.mean(sims)
    corrected = raw - anisotropy_baseline

    return {
        "raw": float(raw),
        "corrected": float(corrected),
        "std": float(np.std(sims)),
        "n_pairs": len(sims),
    }


def compute_mev(embeddings):
    """
    Maximum Explained Variance: proportion of variance explained by PC1.
    """
    if len(embeddings) < 3:
        return {"mev": np.nan, "top_3_explained": []}

    pca = PCA(n_components=min(10, len(embeddings) - 1))
    pca.fit(embeddings)

    return {
        "mev": float(pca.explained_variance_ratio_[0]),
        "top_3_explained": [float(v) for v in pca.explained_variance_ratio_[:3]],
        "cumulative_5": float(sum(pca.explained_variance_ratio_[:5])),
    }


def compute_gmm_clustering(embeddings, max_k=6, pca_dims=5, n_inits=50):
    """
    Fit GMMs with varying K, select best by silhouette score.
    Check robustness via ARI across multiple runs.
    """
    if len(embeddings) < 10:
        return {"best_k": 1, "silhouette": np.nan, "ari_stability": np.nan}

    # PCA reduction first
    actual_pca_dims = min(pca_dims, len(embeddings) - 1, embeddings.shape[1])
    pca = PCA(n_components=actual_pca_dims)
    reduced = pca.fit_transform(embeddings)

    # Try different K values
    best_k = 1
    best_sil = -1
    results_by_k = {}

    for k in range(2, min(max_k + 1, len(embeddings) // 5)):
        try:
            gmm = GaussianMixture(n_components=k, n_init=n_inits, random_state=42)
            labels = gmm.fit_predict(reduced)
            sil = silhouette_score(reduced, labels)
            results_by_k[k] = {"silhouette": float(sil), "bic": float(gmm.bic(reduced))}

            if sil > best_sil:
                best_sil = sil
                best_k = k
        except Exception:
            continue

    # ARI stability check for best K
    ari_scores = []
    if best_k > 1:
        reference_gmm = GaussianMixture(n_components=best_k, n_init=10, random_state=0)
        reference_labels = reference_gmm.fit_predict(reduced)

        for run in range(N_ARI_RUNS):
            gmm = GaussianMixture(n_components=best_k, n_init=1, random_state=run + 1)
            labels = gmm.fit_predict(reduced)
            ari = adjusted_rand_score(reference_labels, labels)
            ari_scores.append(ari)

    return {
        "best_k": best_k,
        "silhouette": float(best_sil) if best_sil > -1 else None,
        "ari_stability_mean": float(np.mean(ari_scores)) if ari_scores else None,
        "ari_stability_std": float(np.std(ari_scores)) if ari_scores else None,
        "results_by_k": results_by_k,
        "pca_dims_used": actual_pca_dims,
    }


def css_score(self_sim_corrected):
    """
    Convert anisotropy-corrected self-similarity to a 0-100 CSS.
    Higher = more stable. Maps the typical range (-0.1 to 0.5) onto 0-100.
    """
    if np.isnan(self_sim_corrected):
        return np.nan
    # Clamp and rescale: self-sim corrected of ~0.5 = very stable (100),
    # self-sim corrected of ~-0.1 = very unstable (0)
    score = (self_sim_corrected + 0.1) / 0.6 * 100
    return float(max(0, min(100, score)))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load anisotropy baseline
    baseline_path = DATA_DIR / "embeddings_random_baseline.npz"
    if baseline_path.exists():
        aniso_baseline = compute_anisotropy_baseline(baseline_path)
    else:
        print("WARNING: No anisotropy baseline found. Using 0.")
        aniso_baseline = 0.0

    all_metrics = []

    for concept_slug in CONCEPTS:
        print(f"\n--- Computing metrics for: {concept_slug} ---")

        emb_path = DATA_DIR / f"embeddings_{concept_slug}.npz"
        if not emb_path.exists():
            print(f"  WARNING: {emb_path} not found, skipping")
            continue

        data = np.load(emb_path, allow_pickle=True)
        embeddings = data["embeddings"]
        print(f"  Loaded {embeddings.shape[0]} embeddings ({embeddings.shape[1]}d)")

        # Self-similarity
        self_sim = compute_self_similarity(embeddings, aniso_baseline)
        print(f"  Self-similarity: raw={self_sim['raw']:.4f}, corrected={self_sim['corrected']:.4f}")

        # MEV
        mev = compute_mev(embeddings)
        print(f"  MEV (PC1): {mev['mev']:.4f}")

        # GMM clustering
        gmm = compute_gmm_clustering(embeddings)
        print(f"  Best K: {gmm['best_k']}, Silhouette: {gmm['silhouette']}")
        if gmm["ari_stability_mean"] is not None:
            print(f"  ARI stability: {gmm['ari_stability_mean']:.3f} +/- {gmm['ari_stability_std']:.3f}")

        # CSS score
        css = css_score(self_sim["corrected"])
        print(f"  CSS (0-100): {css:.1f}")

        metrics = {
            "concept": concept_slug,
            "n_embeddings": int(embeddings.shape[0]),
            "self_similarity": self_sim,
            "mev": mev,
            "clustering": gmm,
            "css_score": css,
        }
        all_metrics.append(metrics)

    # Save detailed results
    detail_path = RESULTS_DIR / "metrics_detail.json"
    with open(detail_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nDetailed metrics saved to {detail_path}")

    # Save summary CSV
    summary_path = RESULTS_DIR / "metrics_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "concept", "n_embeddings", "css_score",
            "self_sim_raw", "self_sim_corrected",
            "mev", "best_k", "silhouette", "ari_stability"
        ])
        for m in all_metrics:
            writer.writerow([
                m["concept"],
                m["n_embeddings"],
                f"{m['css_score']:.1f}",
                f"{m['self_similarity']['raw']:.4f}",
                f"{m['self_similarity']['corrected']:.4f}",
                f"{m['mev']['mev']:.4f}",
                m["clustering"]["best_k"],
                f"{m['clustering']['silhouette']:.4f}" if m["clustering"]["silhouette"] else "N/A",
                f"{m['clustering']['ari_stability_mean']:.3f}" if m["clustering"]["ari_stability_mean"] else "N/A",
            ])

    print(f"Summary saved to {summary_path}")
    print("\n=== Metric computation complete ===")


if __name__ == "__main__":
    main()
