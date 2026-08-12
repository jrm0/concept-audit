"""
tdm_03_metrics.py — Metric suite inside TDM Studio (identical to local
03_compute_metrics.py: anisotropy-corrected self-similarity, MEV, GMM
clustering with silhouette-selected K and ARI stability).

Reads export/embeddings_{slug}.npz (+ random baseline); writes
export/metrics_summary.csv and export/metrics_detail.json — both derived
statistics, safe to export.

Usage:
    python3 tdm_03_metrics.py
"""

import csv
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture

warnings.filterwarnings("ignore")

EXPORT_DIR = Path("export")
MAX_K = 6
N_GMM_INITS = 50
N_ARI_RUNS = 100
PCA_DIMS = 5


def anisotropy_baseline():
    path = EXPORT_DIR / "embeddings_random_baseline.npz"
    if not path.exists():
        print("WARNING: no baseline; using 0")
        return 0.0
    embs = np.load(path, allow_pickle=True)["embeddings"]
    n = min(len(embs), 300)
    sims = [1 - cosine(embs[i], embs[j]) for i, j in combinations(range(n), 2)]
    b = float(np.mean(sims))
    print(f"Anisotropy baseline: {b:.4f}")
    return b


def self_similarity(embs, baseline):
    n = len(embs)
    if n < 2:
        return {"raw": None, "corrected": None}
    if n > 300:
        idx = np.random.RandomState(42).choice(n, 300, replace=False)
        embs = embs[idx]
    sims = [1 - cosine(embs[i], embs[j])
            for i, j in combinations(range(len(embs)), 2)]
    raw = float(np.mean(sims))
    return {"raw": raw, "corrected": raw - baseline, "std": float(np.std(sims))}


def mev(embs):
    if len(embs) < 3:
        return {"mev": None}
    pca = PCA(n_components=min(10, len(embs) - 1))
    pca.fit(embs)
    return {"mev": float(pca.explained_variance_ratio_[0]),
            "top_3": [float(v) for v in pca.explained_variance_ratio_[:3]]}


def gmm_clustering(embs):
    if len(embs) < 10:
        return {"best_k": 1, "silhouette": None, "ari_stability_mean": None}
    dims = min(PCA_DIMS, len(embs) - 1, embs.shape[1])
    reduced = PCA(n_components=dims).fit_transform(embs)
    best_k, best_sil, by_k = 1, -1, {}
    for k in range(2, min(MAX_K + 1, len(embs) // 5)):
        try:
            g = GaussianMixture(n_components=k, n_init=N_GMM_INITS, random_state=42)
            labels = g.fit_predict(reduced)
            sil = silhouette_score(reduced, labels)
            by_k[k] = {"silhouette": float(sil), "bic": float(g.bic(reduced))}
            if sil > best_sil:
                best_k, best_sil = k, sil
        except Exception:
            continue
    aris = []
    if best_k > 1:
        ref = GaussianMixture(n_components=best_k, n_init=10,
                              random_state=0).fit_predict(reduced)
        for run in range(N_ARI_RUNS):
            lab = GaussianMixture(n_components=best_k, n_init=1,
                                  random_state=run + 1).fit_predict(reduced)
            aris.append(adjusted_rand_score(ref, lab))
    return {"best_k": best_k,
            "silhouette": float(best_sil) if best_sil > -1 else None,
            "ari_stability_mean": float(np.mean(aris)) if aris else None,
            "ari_stability_std": float(np.std(aris)) if aris else None,
            "results_by_k": by_k, "pca_dims_used": dims}


def css(corrected):
    if corrected is None:
        return None
    return float(max(0, min(100, (corrected + 0.1) / 0.6 * 100)))


def main():
    baseline = anisotropy_baseline()
    all_metrics = []
    for path in sorted(EXPORT_DIR.glob("embeddings_*.npz")):
        slug = path.stem.replace("embeddings_", "")
        if slug == "random_baseline":
            continue
        embs = np.load(path, allow_pickle=True)["embeddings"]
        print(f"\n--- {slug}: {embs.shape[0]} embeddings")
        ss = self_similarity(embs, baseline)
        m = mev(embs)
        g = gmm_clustering(embs)
        rec = {"concept": slug, "n": int(embs.shape[0]),
               "self_similarity": ss, "mev": m, "clustering": g,
               "css_score": css(ss.get("corrected"))}
        all_metrics.append(rec)
        print(f"    self-sim corr={ss.get('corrected')}, MEV={m.get('mev')}, "
              f"K={g['best_k']}, ARI={g.get('ari_stability_mean')}")

    with open(EXPORT_DIR / "metrics_detail.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    with open(EXPORT_DIR / "metrics_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "n", "css", "self_sim_corrected", "mev",
                    "best_k", "silhouette", "ari_stability"])
        for r in all_metrics:
            w.writerow([r["concept"], r["n"], r["css_score"],
                        r["self_similarity"].get("corrected"),
                        r["mev"].get("mev"), r["clustering"]["best_k"],
                        r["clustering"].get("silhouette"),
                        r["clustering"].get("ari_stability_mean")])
    print(f"\nSaved metrics to {EXPORT_DIR}/ — safe to export "
          f"(derived statistics only).")


if __name__ == "__main__":
    main()
