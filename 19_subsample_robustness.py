"""
19_subsample_robustness.py — Is late-period branching just corpus growth?

The objection: clustering POWER grows with sample size. A period with 193
documents can resolve structure that a period with 60 documents cannot, so
"branching appears in 2015-2019" may partly mean "2015-2019 is when we
finally had enough documents to see structure that was always there."

The check: for each term, take every period eligible for clustering
(n >= CLUSTER_MIN_N), cap them all at the SAME n (the minimum across those
periods), and rerun the bootstrap-gated clustering on repeated random
subsamples. If a period shows multi-cluster structure at full n but never at
equalized n, its branching is confounded with corpus growth. If structure
survives equalization in most draws, the branching reflects the documents,
not their number.

Reads:  data/lineage_emb/*.npz  (via 18's loader)
Writes: results/subsample_robustness.csv
Usage:  python3 19_subsample_robustness.py
"""

import csv
import importlib.util
from collections import Counter
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "lineage", Path(__file__).parent / "18_lineage_analysis.py")
lin = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lin)

DRAWS = 10
SEED = 42
RESULTS = Path("results")


def main():
    data = lin.load_shards()
    if not data:
        raise SystemExit("No shards in data/lineage_emb — run 17 first.")
    rows = []
    for t in sorted(data):
        if t in lin.CONTROL_TERMS:
            continue
        eligible = {p: sh for p, sh in data[t].items()
                    if len(sh["V"]) >= lin.CLUSTER_MIN_N}
        if len(eligible) < 2:
            continue
        cap = min(len(sh["V"]) for sh in eligible.values())
        print(f"\n{t}: {len(eligible)} eligible periods, equalized n={cap}")
        for p in sorted(eligible):
            V = eligible[p]["V"]
            _, k_full, _ = lin.cluster_period(V)
            ks = []
            for d in range(DRAWS):
                rng = np.random.RandomState(SEED + d)
                idx = rng.choice(len(V), cap, replace=False)
                _, k, _ = lin.cluster_period(V[idx])
                ks.append(k)
            frac = sum(1 for k in ks if k > 1) / len(ks)
            modal = Counter(ks).most_common(1)[0][0]
            print(f"  {p}: n={len(V):>3} k_full={k_full}   "
                  f"equalized n={cap}: multi-cluster in {frac:.0%} "
                  f"of {DRAWS} draws (modal k={modal})")
            rows.append([t, p, len(V), k_full, cap, round(frac, 2), modal])

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "subsample_robustness.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "period", "n_full", "k_full", "n_equalized",
                    "frac_multicluster", "modal_k"])
        w.writerows(rows)
    print(f"\n-> {RESULTS / 'subsample_robustness.csv'}")
    print("\nHOW TO READ: a branching claim for period P survives only if P "
          "shows multi-cluster structure in most equalized draws. k_full>1 "
          "with frac~0 = sample-size artifact; frac high = real structure.")


if __name__ == "__main__":
    main()
