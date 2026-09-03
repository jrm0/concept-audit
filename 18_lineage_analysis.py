"""
18_lineage_analysis.py — Temporal lineage analysis: drift, divergence,
branching. (No torch needed; runs anywhere sklearn is installed.)

The four analyses, in the order the paper needs them:

 A. DRIFT   Per term: distance of each period's centroid from the term's own
            origin centroid, MINUS the same quantity for control terms
            ("participants", "results"...) — excess drift above what corpus
            change alone produces (Dubossarsky control). Split-half centroid
            reliability gives a noise floor per period.

 B. BIRTH & DIVERGENCE   Cross-term distances per period (masked space is
            commensurable across terms). The speciation prediction: at birth
            (1995-2004) "ego depletion" sits close to its ancestors
            (self-control / self-regulation); afterwards it DIVERGES.
            Also: do the ancestors converge or diverge from each other?

 C. LINEAGES WITHIN TERMS   For high-volume terms, within-period clustering
            (cluster_core: k-means proposal, dip test + bootstrap Jaccard
            gate, document units), then descent links across consecutive
            periods by centroid similarity. A BRANCHING event = two clusters
            in period p+1 whose best ancestor is the same cluster in p and
            whose mutual separation exceeds the noise floor. Only periods
            whose clustering is SUPPORTED produce >1 node.

 D. EVENT ALIGNMENT   Dated events from the config are printed against the
            drift and divergence curves.

Outputs (results/<slug>/):
  lineage_drift.csv        term, period, n, drift_from_origin, control_drift,
                           excess_drift, noise_floor
  term_distances.csv       period, term_a, term_b, cosine_distance
  lineage_tree.json        per-term: clusters per period (with dip p,
                           mean Jaccard, verdict) + descent links +
                           candidate branching events
  lineage_samples_{term}.txt   titles per cluster per period (READ THESE)
  lineage_report.txt       the human-readable summary
  manifest_18_lineage_analysis.json   run provenance

Usage:
    python3 18_lineage_analysis.py [--config configs/<concept>.yaml]
    python3 18_lineage_analysis.py --center    # per-period mean-centering
    python3 18_lineage_analysis.py --sources psyc_family_2000_2015,business
    python3 18_lineage_analysis.py --dbs "PsycInfo,PsycArticles"
    python3 18_lineage_analysis.py --strata journal

SUBSET RE-RUNS (corpus principle 9: corpus dependence is a result). The
filters keep only document vectors whose provenance matches, then run the
identical analysis; outputs get a suffix (_src-<...>, _dbs-<...>,
_strata-<...>) so subset runs never overwrite the full run. --dbs matches
substrings of the RIS DB tags stored per vector by 17 (works on any shard);
--sources / --strata need shards embedded after 2026-09-02 (17 now stores
corpus-spec source ids and register stratum per vector) and fail clearly on
older shards. Control-term shards are filtered the same way so the drift
null stays comparable.
        Anisotropy robustness (Ethayarajh 2019): contextual embeddings bunch
        into a narrow cone, and the cone can shift/tighten as corpus
        composition changes over time. Centering subtracts each period's
        reference centroid before A and B, removing the shared component.

        The reference is the CONTROL terms only ("participants", "results",
        "significant", "measure") — the same four terms with a similar
        document count in every period. A family-pool reference would be
        composition-dependent: 1975-79 contains only self-control /
        self-regulation, 2015-19 contains eleven terms, so the reference
        itself would move for reasons that have nothing to do with the cone,
        contaminating exactly the cross-period drift we want to measure.
        Periods with no control shard are dropped rather than centered
        against a different reference.
        If the curves keep their shape after centering, they're clean.
        C is skipped under --center: PCA is translation-invariant, so the
        clustering is identical to the uncentered run by construction.
        Outputs get a _centered suffix.
"""

import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import pipeline_config as pc

CFG, _ARGV = pc.load()

EMB_DIR = Path(CFG["emb_dir"])
DOCS_PATH = Path(CFG["docs_path"])
RESULTS = CFG.results                      # results/<slug>/
N_SAMPLE_TITLES = int(CFG["analysis"]["n_sample_titles"])
CENTER = "--center" in _ARGV


def _opt(name):
    """Value of --name X or --name=X in the remaining argv, else None."""
    for i, a in enumerate(_ARGV):
        if a == name and i + 1 < len(_ARGV):
            return _ARGV[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


FILTER = {k: [x.strip() for x in (_opt(f"--{k}") or "").split(",") if x.strip()]
          for k in ("sources", "dbs", "strata")}
_fsuf = "".join(f"_{k[:3]}-{'+'.join(v)}" for k, v in FILTER.items() if v)
_fsuf = re.sub(r"[^A-Za-z0-9_+\-]", "", _fsuf)
SUF = ("_centered" if CENTER else "") + _fsuf
MIN_N = int(CFG["analysis"]["min_n"])
CLUSTER_MIN_N = int(CFG["analysis"]["cluster_min_n"])
MAX_K = int(CFG["analysis"]["max_k"])
SEED = int(CFG["analysis"]["seed"])
CONTROL_TERMS = set(CFG.controls)
EVENTS = pc.events(CFG)
KEY_PAIRS = [tuple(p) for p in CFG.get("key_pairs", [])]


def _subset_mask(z, n):
    """Boolean mask over a shard's vectors for the --sources/--dbs/--strata
    filters; None if no filter is active."""
    if not any(FILTER.values()):
        return None
    mask = np.ones(n, dtype=bool)
    if FILTER["dbs"]:
        if "dbs" not in z.files:
            raise SystemExit("--dbs: shards carry no DB tags (re-run 17).")
        tags = [str(x) for x in z["dbs"]]
        mask &= np.array([any(w.lower() in t.lower() for w in FILTER["dbs"])
                          for t in tags])
    if FILTER["sources"]:
        if "sources" not in z.files:
            raise SystemExit("--sources needs shards embedded by 17 after "
                             "2026-09-02 (they store corpus-spec source ids); "
                             "use --dbs on older shards.")
        ids = [set(str(x).split("|")) for x in z["sources"]]
        mask &= np.array([bool(s & set(FILTER["sources"])) for s in ids])
    if FILTER["strata"]:
        if "strata" not in z.files:
            raise SystemExit("--strata needs shards embedded by 17 after "
                             "2026-09-02 (they store the register stratum).")
        mask &= np.isin([str(x) for x in z["strata"]], FILTER["strata"])
    return mask


def load_shards():
    data = defaultdict(dict)   # term -> period -> dict(vectors, keys, ...)
    kept = total = 0
    for f in sorted(EMB_DIR.glob("*.npz")):
        m = re.match(r"(.+)__(\d{4}-\d{4})$", f.stem)
        if not m:
            continue
        term = m.group(1).replace("_", " ")
        z = np.load(f, allow_pickle=True)
        n = len(z["vectors"])
        mask = _subset_mask(z, n)
        total += n
        if mask is not None:
            kept += int(mask.sum())
            if mask.sum() == 0:
                continue
        else:
            kept += n
        sel = (lambda a: a[mask]) if mask is not None else (lambda a: a)
        data[term][m.group(2)] = {
            "V": sel(z["vectors"]), "keys": sel(z["keys"]),
            "years": sel(z["years"]), "journals": sel(z["journals"]),
            "dbs": sel(z["dbs"]) if "dbs" in z.files else None}
    if any(FILTER.values()):
        print(f"SUBSET FILTER {FILTER}: kept {kept}/{total} document vectors")
    return data


def load_titles():
    """doi -> title, so sample files show READABLE titles, not DOIs.
    (17 stored doi-or-title as the key; the qualitative check needs titles.)"""
    titles = {}
    if DOCS_PATH.exists():
        with open(DOCS_PATH, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("doi"):
                    titles[d["doi"]] = d.get("title", "")
    return titles


def split_half_floor(V, reps=40):
    """Distance between centroids of random halves of the SAME group —
    the measurement noise floor for centroid comparisons."""
    if len(V) < 10:
        return None
    rng = np.random.RandomState(SEED)
    ds = []
    for _ in range(reps):
        p = rng.permutation(len(V))
        h = len(V) // 2
        ds.append(cosine(V[p[:h]].mean(0), V[p[h:2 * h]].mean(0)))
    return float(np.mean(ds))


def cluster_period(V, n_null=10):
    """Propose a partition and TEST it (see cluster_core).

    SUPERSEDED IMPLEMENTATION BELOW (kept for the record, unused): a
    full-covariance GMM gated on BIC gain over a single-Gaussian parametric
    bootstrap. Synthetic checks showed it failed in both directions — it
    missed genuinely separated clusters (BIC charges ~0.46/document per extra
    component in 5 dims, so real moderate separation is penalised away) and
    it accepted single skewed/heavy-tailed clouds (any non-Gaussianity beats
    a Gaussian null). Clustering decisions now come from a dip test of
    unimodality plus bootstrap cluster stability.
    """
    import cluster_core
    r = cluster_core.analyze(V, min_n=CLUSTER_MIN_N, max_k=MAX_K, seed=SEED)
    return r["labels"], r["k"], r


def _cluster_period_bic_DEPRECATED(V, n_null=10):
    """GMM on document vectors, gated by a PARAMETRIC BOOTSTRAP null.

    A fixed dBIC threshold does not transfer across dimensionality/spread
    (a genuinely separated 2-cluster case at realistic document spread scores
    ~0.94, under the 1.0 gate calibrated on tighter fixtures). Instead: fit a
    single Gaussian to the data, sample n_null fake datasets from it, and
    accept k>1 only if the observed dBIC exceeds every null dBIC. This asks
    the exact question — "could unimodal data have produced this much BIC
    improvement?" — with no magic constant.
    """
    if len(V) < CLUSTER_MIN_N:
        return None, 1, None
    # float64 + rescale: duplicate documents (reprints, the same abstract in
    # two RIS exports) give IDENTICAL vectors; a GMM component that collapses
    # onto them has singular covariance and Cholesky fails. Rescaling makes
    # the reg_covar ridge meaningful at the data's actual variance scale.
    R = PCA(n_components=min(5, len(V) - 1)).fit_transform(V.astype(np.float64))
    R /= R.std() + 1e-12

    def fit_gmm(k, X, n_init):
        try:
            return GaussianMixture(k, n_init=n_init, reg_covar=1e-4,
                                   random_state=SEED).fit(X)
        except Exception:
            return None   # component collapsed even with ridge — skip this k

    g1 = fit_gmm(1, R, 5)
    if g1 is None:
        return None, 1, None
    bic1 = g1.bic(R)
    best = (1, None, 0.0)
    for k in range(2, MAX_K + 1):
        g = fit_gmm(k, R, 25)
        if g is None:
            continue
        dbic = (bic1 - g.bic(R)) / len(R)
        if dbic > best[2]:
            best = (k, g.predict(R), dbic)
    if best[0] == 1 or best[2] <= 0:
        return None, 1, best[2]
    # Null: how much dBIC does a single Gaussian produce by chance?
    rng = np.random.RandomState(SEED)
    null_max = -np.inf
    for _ in range(n_null):
        S, _lab = g1.sample(len(R))
        S = S[rng.permutation(len(S))]
        gn1 = fit_gmm(1, S, 3)
        if gn1 is None:
            continue
        b1 = gn1.bic(S)
        for k in range(2, MAX_K + 1):
            gk = fit_gmm(k, S, 5)
            if gk is not None:
                null_max = max(null_max, (b1 - gk.bic(S)) / len(S))
    if null_max == -np.inf:
        return None, 1, best[2]   # no usable null — refuse to claim clusters
    if best[2] > null_max:
        return best[1], best[0], best[2]
    return None, 1, best[2]


def main():
    titles_by_doi = load_titles()
    data = load_shards()
    if not data:
        raise SystemExit("No shards in data/lineage_emb — run 17 first.")
    family = sorted(t for t in data if t not in CONTROL_TERMS)
    controls = sorted(t for t in data if t in CONTROL_TERMS)
    periods = sorted({p for t in data for p in data[t]})
    report = []

    def say(s=""):
        print(s)
        report.append(s)

    say(f"Concept: {CFG['concept']}   config: {CFG['_config_path']} "
        f"({CFG['_config_sha1']})")
    say(f"Terms: {family}")
    say(f"Controls: {controls}")
    say(f"Periods: {periods}")
    say(f"MODE: {'PER-PERIOD MEAN-CENTERED (anisotropy robustness)' if CENTER else 'raw (uncentered)'}")
    if any(FILTER.values()):
        say(f"SUBSET: {FILTER}  (outputs suffixed '{_fsuf}')")
    say("")

    # Per-period reference centroid from CONTROL TERMS ONLY (see docstring:
    # a family-pool reference would move with term composition)
    grand = {}
    if CENTER:
        sums, counts = {}, defaultdict(int)
        for t in controls:
            for p, sh in data[t].items():
                sums[p] = sums.get(p, 0) + sh["V"].astype(np.float64).sum(0)
                counts[p] += len(sh["V"])
        grand = {p: sums[p] / counts[p] for p in counts}
        missing = [p for p in periods if p not in grand]
        say(f"Control-reference centroid available for {len(grand)} periods "
            f"(from {len(controls)} control terms)"
            + (f"; DROPPED (no control shard): {missing}" if missing else ""))

    # Centroids + noise floors
    cent, floor, ns = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for t in data:
        for p, sh in data[t].items():
            if CENTER and p not in grand:
                continue          # no composition-stable reference: skip
            if len(sh["V"]) >= MIN_N:
                V = sh["V"] - grand[p] if CENTER else sh["V"]
                cent[t][p] = V.mean(0)
                floor[t][p] = split_half_floor(V)
                ns[t][p] = len(V)

    # --- A. DRIFT with control null ---
    ctrl_drift = {}
    for p in periods:
        vals = []
        for c in controls:
            ps = sorted(cent[c])
            if p in cent[c] and ps and ps[0] != p and ps[0] in cent[c]:
                vals.append(cosine(cent[c][ps[0]], cent[c][p]))
        ctrl_drift[p] = float(np.mean(vals)) if vals else None

    drift_rows = []
    say("=== A. DRIFT FROM ORIGIN (excess over control terms) ===")
    if CENTER:
        say("CAUTION: under --center the control terms DEFINE the reference, "
            "so their residual drift is deflated by construction and 'excess' "
            "is optimistic here. Read centered results for B (cross-term "
            "distances, common reference, unaffected); read A primarily in "
            "the raw run, using centering only as a shape check.")
    for t in family:
        ps = sorted(cent[t])
        if len(ps) < 2:
            continue
        origin = ps[0]
        say(f"\n{t}  (origin {origin})")
        for p in ps:
            d = float(cosine(cent[t][origin], cent[t][p])) if p != origin else 0.0
            cd = ctrl_drift.get(p)
            excess = d - cd if cd is not None else None
            nf = floor[t][p]
            flag = ""
            if excess is not None and nf is not None:
                flag = " *" if excess > 2 * nf else ""
            say(f"  {p}: n={ns[t][p]:>4} drift={d:.4f} "
                f"control={cd if cd is not None else '-'} "
                f"excess={excess if excess is not None else '-'}{flag}")
            drift_rows.append([t, p, ns[t][p], round(d, 4),
                               round(cd, 4) if cd is not None else "",
                               round(excess, 4) if excess is not None else "",
                               round(nf, 4) if nf is not None else ""])
    with open(RESULTS / f"lineage_drift{SUF}.csv", "w", encoding="utf-8") as f:
        f.write("term,period,n,drift_from_origin,control_drift,"
                "excess_drift,noise_floor\n")
        for r in drift_rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # --- B. BIRTH & DIVERGENCE (cross-term distances per period) ---
    say("\n\n=== B. CROSS-TERM DISTANCES (masked space) ===")
    dist_rows = []
    key_pairs = KEY_PAIRS
    for p in periods:
        for a, b in combinations(sorted(cent), 2):
            if p in cent[a] and p in cent[b]:
                dist_rows.append([p, a, b,
                                  round(float(cosine(cent[a][p], cent[b][p])), 4)])
    with open(RESULTS / f"term_distances{SUF}.csv", "w", encoding="utf-8") as f:
        f.write("period,term_a,term_b,cosine_distance\n")
        for r in dist_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    lookup = {(p, a, b): d for p, a, b, d in dist_rows}
    for a, b in key_pairs:
        aa, bb = sorted([a, b])
        series = [(p, lookup.get((p, aa, bb))) for p in periods
                  if lookup.get((p, aa, bb)) is not None]
        if not series:
            say(f"\nd({a}, {b}): no shared periods (term missing from shards?)")
            continue
        say(f"\nd({a}, {b}):")
        for p, d in series:
            fl = [floor[t].get(p) for t in (a, b) if t in floor and floor[t].get(p)]
            nf = max(fl) if fl else None
            sig = " *" if (nf and d > 2 * nf) else ""
            say(f"  {p}: {d:.4f}{sig}")

    # --- C. WITHIN-TERM LINEAGES + branching detection ---
    if CENTER:
        say("\n\n=== C. SKIPPED under --center: PCA is translation-invariant, "
            "so clustering is identical to the uncentered run. ===")
    else:
        say("\n\n=== C. LINEAGES & CANDIDATE BRANCHING EVENTS ===")
    tree = {}
    n_supported = n_tested = 0
    for t in (family if not CENTER else []):
        node_seq = []   # list of (period, [ {id, centroid, n, titles} ], stats)
        for p in sorted(data[t]):
            sh = data[t][p]
            labels, k, cr = cluster_period(sh["V"])
            stats = {"dip_p": cr.get("dip_p"), "mean_jaccard": cr.get("mean_jaccard"),
                     "silhouette": cr.get("silhouette"), "verdict": cr.get("verdict"),
                     "proposed_k": cr.get("proposed_k", k)}
            if cr.get("verdict") != "n too small":
                n_tested += 1
                n_supported += int(cr.get("verdict") == "SUPPORTED")
            nodes = []
            def readable(keys):
                return [titles_by_doi.get(str(kk), str(kk))[:110]
                        for kk in keys[:N_SAMPLE_TITLES]]

            if labels is None:
                if len(sh["V"]) >= MIN_N:
                    nodes.append({"id": f"{p}:0", "n": len(sh["V"]),
                                  "centroid": sh["V"].mean(0),
                                  "titles": readable(sh["keys"]),
                                  "journals": list(sh["journals"][:0])})
            else:
                for c in range(k):
                    idx = np.where(labels == c)[0]
                    jc = defaultdict(int)
                    for j in sh["journals"][idx]:
                        if str(j).strip():
                            jc[str(j).strip()] += 1
                    top_j = sorted(jc.items(), key=lambda x: -x[1])[:4]
                    dbc = defaultdict(int)
                    if sh["dbs"] is not None:
                        for ds in sh["dbs"][idx]:
                            for db in str(ds).split("|"):
                                if db:
                                    dbc[db] += 1
                    yrs = sh["years"][idx]
                    nodes.append({"id": f"{p}:{c}", "n": int(len(idx)),
                                  "centroid": sh["V"][idx].mean(0),
                                  "titles": readable(sh["keys"][idx]),
                                  "top_journals": top_j,
                                  "top_dbs": sorted(dbc.items(),
                                                    key=lambda x: -x[1])[:4],
                                  "median_year": int(np.median(yrs))})
            if nodes:
                node_seq.append((p, nodes, stats))
        # descent links
        links, branchings = [], []
        for i in range(1, len(node_seq)):
            p_prev, prev, _ = node_seq[i - 1]
            p_cur, cur, _ = node_seq[i]
            parent_of = {}
            for nd in cur:
                sims = [(1 - cosine(nd["centroid"], pv["centroid"]), pv["id"])
                        for pv in prev]
                sims.sort(reverse=True)
                parent_of[nd["id"]] = sims[0][1]
                links.append({"child": nd["id"], "parent": sims[0][1],
                              "cos": round(sims[0][0], 4)})
            # branching: >=2 children share a parent AND are mutually separated
            by_parent = defaultdict(list)
            for nd in cur:
                by_parent[parent_of[nd["id"]]].append(nd)
            for par, kids in by_parent.items():
                if len(kids) >= 2:
                    seps = [cosine(a["centroid"], b["centroid"])
                            for a, b in combinations(kids, 2)]
                    nf = floor[t].get(p_cur) or 0.0
                    if min(seps) > 2 * nf:
                        branchings.append(
                            {"parent": par, "period": p_cur,
                             "children": [k_["id"] for k_ in kids],
                             "min_separation": round(min(seps), 4),
                             "noise_floor": round(nf, 4)})
        def _r(x):
            return None if x is None else round(float(x), 4)
        tree[t] = {
            "periods": [{"period": p,
                         "dip_p": _r(st["dip_p"]),
                         "mean_jaccard": _r(st["mean_jaccard"]),
                         "silhouette": _r(st["silhouette"]),
                         "verdict": st["verdict"],
                         "clusters": [{"id": nd["id"], "n": nd["n"]}
                                      for nd in nodes]}
                        for p, nodes, st in node_seq],
            "links": links, "branchings": branchings}
        if branchings:
            say(f"\n{t}: {len(branchings)} candidate branching event(s)")
            for b in branchings:
                say(f"   {b['parent']} -> {b['children']} at {b['period']} "
                    f"(sep {b['min_separation']} vs floor {b['noise_floor']})")
        # qualitative sample file
        with open(RESULTS / f"lineage_samples_{t.replace(' ', '_')}{_fsuf}.txt",
                  "w", encoding="utf-8") as f:
            for p, nodes, st in node_seq:
                dp = st["dip_p"]
                mj = st["mean_jaccard"]
                f.write(f"===== {p} ({st['verdict']}; dip p="
                        f"{'-' if dp is None else f'{dp:.3f}'}, mean Jaccard="
                        f"{'-' if mj is None else f'{mj:.2f}'}) =====\n")
                for nd in nodes:
                    f.write(f"  cluster {nd['id']} (n={nd['n']}"
                            + (f", median year {nd['median_year']}"
                               if "median_year" in nd else "") + "):\n")
                    if nd.get("top_journals"):
                        f.write("    journals: "
                                + "; ".join(f"{j} ({c})"
                                            for j, c in nd["top_journals"])
                                + "\n")
                    if nd.get("top_dbs"):
                        f.write("    databases: "
                                + "; ".join(f"{j} ({c})"
                                            for j, c in nd["top_dbs"])
                                + "\n")
                    for ti in nd["titles"]:
                        f.write(f"    - {str(ti)[:110]}\n")
                f.write("\n")

    if not CENTER:
        say(f"\nWithin-period clustering: {n_supported}/{n_tested} "
            f"(term x period) cells SUPPORTED by cluster_core "
            f"(dip p<0.05 AND mean Jaccard>=0.75); all other periods are "
            f"one node.")
        with open(RESULTS / f"lineage_tree{_fsuf}.json", "w") as f:
            json.dump({t: {k: v for k, v in d.items()}
                       for t, d in tree.items()}, f, indent=2, default=str)

    # --- D. EVENTS ---
    say("\n\n=== D. DATED EVENTS (read curves against these) ===")
    for y, e in sorted(EVENTS.items()):
        say(f"  {y}: {e}")
    say("\nREAD lineage_samples_*.txt before interpreting any branching — "
        "clusters must make sense as usage communities (every prior artifact "
        "was caught by reading, not by metrics).")

    (RESULTS / f"lineage_report{SUF}.txt").write_text("\n".join(report),
                                                      encoding="utf-8")
    outs = [RESULTS / f"lineage_drift{SUF}.csv",
            RESULTS / f"term_distances{SUF}.csv",
            RESULTS / f"lineage_report{SUF}.txt"]
    if not CENTER:
        outs += [RESULTS / f"lineage_tree{_fsuf}.json"] + \
                [RESULTS / f"lineage_samples_{t.replace(' ', '_')}{_fsuf}.txt"
                 for t in family]
    pc.write_manifest(
        CFG, __file__,
        params={"center": CENTER, "subset_filter": FILTER, "suffix": SUF,
                "min_n": MIN_N, "cluster_min_n": CLUSTER_MIN_N,
                "max_k": MAX_K, "seed": SEED, "control_terms": sorted(CONTROL_TERMS),
                "terms": family, "periods": periods, "key_pairs": key_pairs,
                "clustering": "cluster_core (dip + Hennig Jaccard)",
                "cells_tested": n_tested, "cells_supported": n_supported,
                "n_shards": sum(len(v) for v in data.values())},
        outputs=outs)
    print(f"\nAll outputs -> {RESULTS}/ (report: lineage_report{SUF}.txt)")


if __name__ == "__main__":
    main()
