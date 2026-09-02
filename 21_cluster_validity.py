"""
21_cluster_validity.py — Does the within-period structure actually EXIST?

WHY THIS EXISTS. 18 gates clustering with a parametric bootstrap: fit one
Gaussian, sample from it, accept k>1 only if the observed BIC gain beats the
null. 19 then showed multi-cluster structure in 40 of 41 term x period cells
at equalized n — a test that never rejects has no discriminating power. The
diagnosis is that the null is the wrong one: real embedding clouds are
skewed, heavy-tailed and curved, so ANY departure from normality beats a
Gaussian null whether or not there are discrete clusters. Rejecting "these
data are one Gaussian" is not evidence for "these data are two groups".

Two tests that ask the right question, from the cluster-validation
literature:

 1. DIP TEST OF UNIMODALITY (Hartigan & Hartigan 1985, Ann. Statist. 13:70-84)
    on the one-dimensional DISCRIMINANT PROJECTION — the axis through the two
    cluster centroids, i.e. the direction along which the clustering claims
    separation. The dip statistic is the largest gap between the empirical
    CDF and the best-fitting unimodal CDF; the uniform is the asymptotically
    least favourable unimodal distribution, so the reference distribution is
    calibrated against uniform samples. This is a NON-PARAMETRIC test of
    "one mode vs more than one" — it does not care whether the cloud is
    Gaussian, only whether it is unimodal. A cloud that is skewed but
    single-peaked passes (correctly) as one cluster.

 2. BOOTSTRAP CLUSTER STABILITY (Hennig 2007, Comput. Statist. Data Anal.
    52:258-271). Resample the documents with replacement, re-cluster, and
    match each original cluster to its most similar bootstrap cluster by
    Jaccard similarity. A cluster that dissolves under resampling is a
    partition of noise. Hennig's rule of thumb: mean Jaccard < 0.6 =
    unstable/not meaningful; > 0.75 = a real pattern; >= 0.85 = highly
    stable. (Thresholds are heuristics, not p-values — reported as such.)

VERDICT requires BOTH: dip p < 0.05 (multimodal along the separating axis)
AND mean Jaccard >= 0.75 (the partition survives resampling). Anything less
is reported as candidate structure only.

Requires: pip install diptest --break-system-packages

Reads:  data/lineage_emb/*.npz
Writes: results/cluster_validity.csv
Usage:  python3 21_cluster_validity.py
"""

import csv
import importlib.util
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

_spec = importlib.util.spec_from_file_location(
    "lineage", Path(__file__).parent / "18_lineage_analysis.py")
lin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lin)

BOOT = 50
SEED = 42
RESULTS = Path("results")


def main():
    import cluster_core as cc

    data = lin.load_shards()
    if not data:
        raise SystemExit("No shards in data/lineage_emb — run 17 first.")
    rows = []
    print(f"{'term':<24}{'period':<11}{'n':>5}{'k':>3}{'sil':>7}"
          f"{'dip p':>9}{'meanJacc':>10}{'minJacc':>9}   verdict")
    for t in sorted(data):
        if t in lin.CONTROL_TERMS:
            continue
        for p in sorted(data[t]):
            V = data[t][p]["V"].astype(float)
            if len(V) < cc.MIN_N:
                continue
            r = cc.analyze(V, n_boot=BOOT, seed=SEED)
            k = r["k"] if r["labels"] is not None else r.get("proposed_k", 0)
            f = lambda x: "-" if x is None else f"{x:.3f}"
            print(f"{t:<24}{p:<11}{len(V):>5}{k:>3}{f(r['silhouette']):>7}"
                  f"{f(r['dip_p']):>9}{f(r['mean_jaccard']):>10}"
                  f"{f(r['min_jaccard']):>9}   {r['verdict']}")
            rows.append([t, p, len(V), k, r["silhouette"], r["dip_p"],
                         r["mean_jaccard"], r["min_jaccard"], r["verdict"]])

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "cluster_validity.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["term", "period", "n", "k", "silhouette", "dip_p",
                    "mean_jaccard", "min_jaccard", "verdict"])
        w.writerows(rows)
    n_ok = sum(1 for r in rows if r[-1] == "SUPPORTED")
    print(f"\n{n_ok}/{len(rows)} period-clusterings SUPPORTED "
          f"(dip p<{cc.DIP_ALPHA} AND mean Jaccard>={cc.STABLE})")
    print(f"-> {RESULTS / 'cluster_validity.csv'}")
    print("\nOnly branching events whose BOTH periods are SUPPORTED should "
          "be reported as branchings; the rest are candidate structure.")


if __name__ == "__main__":
    main()
