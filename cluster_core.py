"""
cluster_core.py — Propose a partition, then TEST whether it is real.

Shared by 18 (lineage/branching) and 21 (validity report) so both use one
definition of "this period has clades".

WHY THE OLD GATE WAS REPLACED. 18 originally fitted full-covariance
Gaussian mixtures in 5 PCA dimensions and gated on a BIC gain over a
single-Gaussian parametric bootstrap. Synthetic checks showed it failed in
BOTH directions:

  * FALSE NEGATIVES. Two genuinely separated Gaussian clusters (n=240, 5
    dims, centroid separation 1-3 SD) gave NEGATIVE dBIC/n: a full
    covariance in 5 dims costs 20 free parameters per component, so BIC
    charges ~0.46/document per extra component and penalises real moderate
    separation away. Only separation >= 5 SD survived — far beyond anything
    a usage community would show.
  * FALSE POSITIVES. A single skewed, heavy-tailed blob (gamma x
    Student-t, one mode) was declared k=2, because ANY departure from
    normality beats a Gaussian null. 19 then found multi-cluster structure
    in 40 of 41 real cells — a test that never rejects.

So: the proposal step should be cheap and sensitive, and the DECISION
should be made by tests that ask about modality and reproducibility rather
than about Gaussianity.

  PROPOSE   k-means for k = 2..4 in a low-dimensional PCA projection; pick k
            by silhouette. k-means always returns a partition — that is
            fine, it is a hypothesis, not a conclusion.
  TEST 1    Hartigan & Hartigan (1985) dip test of unimodality, on the
            discriminant projection joining the two CLOSEST clusters (the
            claim is only as strong as its weakest separation). Fully
            non-parametric: a skewed but single-peaked cloud passes as one
            cluster, which is exactly the failure mode above.
  TEST 2    Hennig (2007) bootstrap cluster stability: resample documents
            with replacement, re-cluster, match each original cluster to its
            most similar bootstrap cluster by Jaccard. Clusters that
            dissolve under resampling were partitions of noise.

ACCEPT only if dip p < 0.05 AND mean Jaccard >= 0.75 (Hennig's "real
pattern" heuristic; >= 0.85 "highly stable"). Otherwise the period has ONE
usage community as far as we can demonstrate.

References (verified):
  Hartigan & Hartigan 1985, Ann. Statist. 13(1):70-84,
      doi:10.1214/aos/1176346577
  Hennig 2007, Comput. Statist. Data Anal. 52(1):258-271,
      doi:10.1016/j.csda.2006.11.025

Requires: pip install diptest --break-system-packages
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

MIN_N = 60
MAX_K = 4
N_PCA = 5
BOOT = 50
DIP_ALPHA = 0.05
STABLE = 0.75
SEED = 42


def _prep(V, n_pca=N_PCA):
    R = PCA(n_components=min(n_pca, len(V) - 1)).fit_transform(
        np.asarray(V, dtype=np.float64))
    return R / (R.std() + 1e-12)


def _jaccard(s1, s2):
    u = len(s1 | s2)
    return len(s1 & s2) / u if u else 0.0


def _dip_worst_pair(R, labels, k):
    """Dip p-value for the CLOSEST pair of clusters (weakest link)."""
    import diptest
    worst_p, pair = -1.0, None
    for a in range(k):
        for b in range(a + 1, k):
            ca, cb = R[labels == a].mean(0), R[labels == b].mean(0)
            w = cb - ca
            nrm = np.linalg.norm(w)
            if nrm == 0:
                continue
            # project ONLY the two clusters under comparison: points from a
            # third cluster land arbitrarily on this axis and fill in the
            # valley, hiding a real separation
            sub = R[(labels == a) | (labels == b)]
            proj = np.asarray(sub @ (w / nrm), dtype=float)
            _, pv = diptest.diptest(proj)
            if pv > worst_p:
                worst_p, pair = pv, (a, b)
    return worst_p, pair


def _stability(R, labels, k, n_boot=BOOT, seed=SEED):
    rng = np.random.RandomState(seed)
    orig = {c: set(np.where(labels == c)[0]) for c in range(k)}
    scores = {c: [] for c in range(k)}
    for _ in range(n_boot):
        idx = rng.choice(len(R), len(R), replace=True)
        try:
            km = KMeans(k, n_init=5, random_state=seed).fit(R[idx])
        except Exception:
            continue
        # Assign ALL original points to the bootstrap solution (nearest
        # centroid). Comparing against only the resampled points would cap
        # Jaccard at ~0.63 — a bootstrap sample contains ~63% of the
        # originals — and make even perfect clusters look unstable.
        pred = km.predict(R)
        bsets = [set(np.where(pred == c)[0]) for c in range(k)]
        for c in range(k):
            scores[c].append(max(_jaccard(orig[c], bs) for bs in bsets))
    return {c: (float(np.mean(v)) if v else 0.0) for c, v in scores.items()}


def analyze(V, min_n=MIN_N, max_k=MAX_K, n_boot=BOOT, seed=SEED):
    """Returns dict with labels (None if not supported), k, dip_p,
    mean_jaccard, min_jaccard, silhouette, verdict."""
    out = {"labels": None, "k": 1, "dip_p": None, "mean_jaccard": None,
           "min_jaccard": None, "silhouette": None, "verdict": "n too small"}
    if len(V) < min_n:
        return out
    R = _prep(V)
    # Evaluate EVERY k and let the tests choose, rather than letting
    # silhouette pick a k that the tests then reject: with three real
    # clusters, silhouette sometimes prefers k=2, whose merged cluster has no
    # clean valley, and the whole period is lost.
    supported, fallback = [], None
    for k in range(2, max_k + 1):
        if k >= len(R):
            break
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(R)
        if len(set(lab)) < k:
            continue
        sil = float(silhouette_score(R, lab))
        dip_p, _pair = _dip_worst_pair(R, lab, k)
        jac = _stability(R, lab, k, n_boot=n_boot, seed=seed)
        mean_j = float(np.mean(list(jac.values())))
        min_j = float(min(jac.values()))
        rec = (sil, k, lab, dip_p, mean_j, min_j)
        if dip_p is not None and dip_p < DIP_ALPHA and mean_j >= STABLE:
            supported.append(rec)
        if fallback is None or sil > fallback[0]:
            fallback = rec
    if fallback is None:
        out["verdict"] = "no partition"
        return out
    if supported:
        sil, k, lab, dip_p, mean_j, min_j = max(supported, key=lambda r: r[0])
        out.update({"k": k, "labels": lab, "dip_p": dip_p,
                    "mean_jaccard": mean_j, "min_jaccard": min_j,
                    "silhouette": sil, "verdict": "SUPPORTED"})
        return out
    sil, k, lab, dip_p, mean_j, min_j = fallback
    out.update({"k": 1, "labels": None, "dip_p": dip_p,
                "mean_jaccard": mean_j, "min_jaccard": min_j,
                "silhouette": sil, "proposed_k": k,
                "verdict": ("candidate only"
                            if (mean_j >= 0.6 or (dip_p or 1) < DIP_ALPHA)
                            else "NOT SUPPORTED")})
    return out
