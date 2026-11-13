"""
27b_community_stability.py — Is the trauma-informed / STS / vicarious-trauma
community (C11 in the Session-38 run) an artefact of the Leiden seed or the
resolution? Re-partition the same graph under several seeds and resolutions,
match each run's communities to the baseline by overlap, and report:

  ARI          agreement of the whole partition with the baseline
  C11 Jaccard  overlap between baseline C11 and its best-matching community
  C11 recall   share of baseline-C11 nodes that stay together
  compound     where the three compounds land in each run (top-community share)

The "stability is not evidence" lesson (Session 7) cuts both ways: a community
that survives seeds and resolutions is not thereby real, but one that does not
survive them is not a finding. This script settles only the second half.

USAGE
    python3 27b_community_stability.py --config configs/ptsd_trauma.yaml
    python3 27b_community_stability.py --config ... --seeds 42,7,1234 --resolutions 0.7,1.0,1.3
"""

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

import pipeline_config as pc

CFG, _ARGV = pc.load()


def _opt(name, default=None):
    for i, a in enumerate(_ARGV):
        if a == name and i + 1 < len(_ARGV):
            return _ARGV[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


YEAR_MIN = int(_opt("--year-min", 1995))
SEEDS = [int(x) for x in _opt("--seeds", "42,7,1234").split(",")]
MAX_NEW = int(_opt("--max-new", 0)) or None   # compute at most N uncached partitions this call
N_ITER = int(_opt("--iterations", 2))          # Leiden iterations; -1 = until stable (slow on 1M edges)
GRID = _opt("--grid", "cross")                 # cross = seeds at base res + resolutions at base seed; full = all pairs
RESOLUTIONS = [float(x) for x in _opt("--resolutions", "0.7,1.0,1.3").split(",")]
BASE_SEED, BASE_RES = 42, 1.0
WORKS = Path(CFG["docs_path"]).parent / "citations" / "openalex_works.jsonl"
OUT = CFG.results
MERGE = {"ptsd": "PTSD", "posttraumatic stress disorder": "PTSD",
         "complex ptsd": "complex PTSD", "complex posttraumatic stress disorder": "complex PTSD"}
COMPOUNDS = ["trauma informed", "secondary traumatic stress", "vicarious trauma"]


def load():
    nodes, id2idx = [], {}
    with open(WORKS, encoding="utf-8") as f:
        for line in f:
            w = json.loads(line)
            y = w.get("corpus_year")
            if y is None or y < YEAR_MIN:
                continue
            id2idx[w["openalex_id"]] = len(nodes)
            nodes.append((sorted({MERGE.get(t, t) for t in (w.get("terms") or [])}),
                          w.get("referenced_works") or []))
    src, dst = [], []
    for i, (_, refs) in enumerate(nodes):
        for r in refs:
            j = id2idx.get(r)
            if j is not None and j != i:
                src.append(i); dst.append(j)
    terms = [t for t, _ in nodes]
    return terms, np.array(src), np.array(dst)


_G = None


def partition(n, src, dst, seed, res):
    """Leiden partition, cached to results/<slug>/stability/ so an interrupted
    grid resumes. Cache key matches 27_citation_communities.py
    (leiden_s{seed}_r{res}_i{iterations}_y{year_min}.npz), so the two scripts
    share partitions; iterations=-1 (full convergence) and iterations=2 are
    different partitions and are cached separately."""
    global _G
    cache = OUT / "stability" / f"leiden_s{seed}_r{res}_i{N_ITER}_y{YEAR_MIN}.npz"
    if cache.exists():
        z = np.load(cache)
        if len(z["memb"]) == n and ("n_edges" not in z.files or int(z["n_edges"]) == len(src)):
            return z["memb"], float(z["Q"])
        stale = cache.with_name(cache.stem + f"_stale_n{len(z['memb'])}.npz")
        cache.rename(stale)
        print(f"  cached partition for {len(z['memb']):,} nodes != graph {n:,}: recomputing ({stale.name} kept)")
    import igraph as ig
    import leidenalg as la
    if _G is None:
        _G = ig.Graph(n=n, edges=list(zip(src.tolist(), dst.tolist())), directed=False)
        _G.simplify()
    p = la.find_partition(_G, la.RBConfigurationVertexPartition, resolution_parameter=res,
                          seed=seed, n_iterations=N_ITER)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, memb=np.array(p.membership), Q=p.modularity, n_edges=len(src))
    return np.array(p.membership), p.modularity


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


def main():
    t0 = time.time()
    terms, src, dst = load()
    n = len(terms)
    print(f"[27b] nodes={n:,} edges={len(src):,}")
    has = {c: np.array([c in t for t in terms]) for c in COMPOUNDS}

    base, Qb = partition(n, src, dst, BASE_SEED, BASE_RES)
    # baseline C11 := the community holding the plurality of trauma-informed docs
    c11 = Counter(base[has["trauma informed"]].tolist()).most_common(1)[0][0]
    c11_nodes = set(np.where(base == c11)[0].tolist())
    print(f"[27b] baseline seed={BASE_SEED} res={BASE_RES}: Q={Qb:.3f}, "
          f"{len(set(base))} communities; compound community = C{c11} (n={len(c11_nodes):,})")

    rows = []
    new_done = 0
    pairs = [(r, sd) for r in RESOLUTIONS for sd in SEEDS]
    if GRID == "cross":
        pairs = [(r, sd) for r, sd in pairs if r == BASE_RES or sd == BASE_SEED]
    for res, seed in pairs:
        if True:
            if seed == BASE_SEED and res == BASE_RES:
                memb, Q = base, Qb
            else:
                cpath = OUT / "stability" / f"leiden_s{seed}_r{res}_i{N_ITER}_y{YEAR_MIN}.npz"
                cached = cpath.exists() and len(np.load(cpath)["memb"]) == n
                if not cached:
                    if MAX_NEW is not None and new_done >= MAX_NEW:
                        print(f"  seed={seed:>5} res={res:<4} (deferred; rerun to continue)")
                        continue
                    new_done += 1
                memb, Q = partition(n, src, dst, seed, res)
            # best match to baseline C11 by Jaccard
            cand = Counter(memb[list(c11_nodes)].tolist())
            best, _ = cand.most_common(1)[0]
            best_nodes = set(np.where(memb == best)[0].tolist())
            jac = len(c11_nodes & best_nodes) / len(c11_nodes | best_nodes)
            recall = len(c11_nodes & best_nodes) / len(c11_nodes)
            comp = {}
            for c in COMPOUNDS:
                cc = Counter(memb[has[c]].tolist())
                top, k = cc.most_common(1)[0]
                comp[c] = (round(k / has[c].sum(), 3), top == best)
            rows.append({"seed": seed, "resolution": res, "modularity": round(Q, 3),
                         "n_communities": len(set(memb.tolist())),
                         "ARI_vs_baseline": round(ari(base, memb), 3),
                         "C11_jaccard": round(jac, 3), "C11_recall": round(recall, 3),
                         "C11_match_size": len(best_nodes),
                         "compound_top_share": {c: comp[c][0] for c in COMPOUNDS},
                         "compounds_in_matched_comm": all(comp[c][1] for c in COMPOUNDS)})
            r = rows[-1]
            print(f"  seed={seed:>5} res={res:<4} Q={r['modularity']:.3f} k={r['n_communities']:>4} "
                  f"ARI={r['ARI_vs_baseline']:.3f}  C11: jaccard={r['C11_jaccard']:.3f} "
                  f"recall={r['C11_recall']:.3f} size={r['C11_match_size']:,}  "
                  f"compounds together={r['compounds_in_matched_comm']}  "
                  f"top-share TI/STS/VT={r['compound_top_share']['trauma informed']}/"
                  f"{r['compound_top_share']['secondary traumatic stress']}/"
                  f"{r['compound_top_share']['vicarious trauma']}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "citation_communities_stability.json").write_text(
        json.dumps({"baseline": {"seed": BASE_SEED, "resolution": BASE_RES, "C11": int(c11),
                                 "C11_size": len(c11_nodes)}, "iterations": N_ITER, "runs": rows}, indent=2), encoding="utf-8")
    pc.write_manifest(CFG, __file__, params={"seeds": SEEDS, "resolutions": RESOLUTIONS, "year_min": YEAR_MIN, "iterations": N_ITER, "grid": GRID},
                      outputs=[OUT / "citation_communities_stability.json"])
    print(f"[27b] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
