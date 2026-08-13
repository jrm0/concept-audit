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

 C. LINEAGES WITHIN TERMS   For high-volume terms, GMM clustering per period
            (dBIC-gated, document units), then descent links across
            consecutive periods by centroid similarity. A BRANCHING event =
            two clusters in period p+1 whose best ancestor is the same
            cluster in p and whose mutual separation exceeds the noise floor.

 D. EVENT ALIGNMENT   Known dated events are printed against the drift and
            divergence curves: 1998 coinage (Baumeister et al.), 2010
            meta-analysis (Hagger et al.), 2014 re-analysis (Carter &
            McCullough), 2016 RRR (Hagger et al.), 2021 RRR (Vohs et al.).

Outputs (results/):
  lineage_drift.csv        term, period, n, drift_from_origin, control_drift,
                           excess_drift, noise_floor
  term_distances.csv       period, term_a, term_b, cosine_distance
  lineage_tree.json        per-term: clusters per period + descent links +
                           candidate branching events
  lineage_samples_{term}.txt   titles per cluster per period (READ THESE)
  lineage_report.txt       the human-readable summary

Usage:
    python3 18_lineage_analysis.py
"""

import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

EMB_DIR = Path("data/lineage_emb")
DOCS_PATH = Path("data/family/lineage_docs.jsonl")
RESULTS = Path("results")
N_SAMPLE_TITLES = 10       # titles per cluster in the qualitative files
MIN_N = 15                 # min docs for a period centroid
CLUSTER_MIN_N = 60         # min docs to attempt within-period clustering
MAX_K = 4
DBIC_MIN = 1.0             # dBIC/n gate for multi-cluster (session-9 lesson)
SEED = 42
CONTROL_TERMS = {"participants", "results", "significant", "measure"}
EVENTS = {1998: "coinage (Baumeister et al. 1998)",
          2010: "supportive meta-analysis (Hagger et al.)",
          2014: "critical re-analysis (Carter & McCullough)",
          2016: "Registered Replication Report (Hagger et al.)",
          2021: "second RRR (Vohs et al.)"}


def load_shards():
    data = defaultdict(dict)   # term -> period -> dict(vectors, keys, ...)
    for f in sorted(EMB_DIR.glob("*.npz")):
        m = re.match(r"(.+)__(\d{4}-\d{4})$", f.stem)
        if not m:
            continue
        term = m.group(1).replace("_", " ")
        z = np.load(f, allow_pickle=True)
        data[term][m.group(2)] = {
            "V": z["vectors"], "keys": z["keys"],
            "years": z["years"], "journals": z["journals"]}
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
    RESULTS.mkdir(exist_ok=True)
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

    say(f"Terms: {family}")
    say(f"Controls: {controls}")
    say(f"Periods: {periods}\n")

    # Centroids + noise floors
    cent, floor, ns = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    for t in data:
        for p, sh in data[t].items():
            if len(sh["V"]) >= MIN_N:
                cent[t][p] = sh["V"].mean(0)
                floor[t][p] = split_half_floor(sh["V"])
                ns[t][p] = len(sh["V"])

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
    with open(RESULTS / "lineage_drift.csv", "w", encoding="utf-8") as f:
        f.write("term,period,n,drift_from_origin,control_drift,"
                "excess_drift,noise_floor\n")
        for r in drift_rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # --- B. BIRTH & DIVERGENCE (cross-term distances per period) ---
    say("\n\n=== B. CROSS-TERM DISTANCES (masked space) ===")
    dist_rows = []
    key_pairs = [("ego depletion", "self control"),
                 ("ego depletion", "self regulation"),
                 ("ego depletion", "willpower"),
                 ("self control", "self regulation"),
                 ("ego depletion", "executive function"),
                 ("ego depletion", "delay of gratification")]
    for p in periods:
        for a, b in combinations(sorted(cent), 2):
            if p in cent[a] and p in cent[b]:
                dist_rows.append([p, a, b,
                                  round(float(cosine(cent[a][p], cent[b][p])), 4)])
    with open(RESULTS / "term_distances.csv", "w", encoding="utf-8") as f:
        f.write("period,term_a,term_b,cosine_distance\n")
        for r in dist_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    lookup = {(p, a, b): d for p, a, b, d in dist_rows}
    for a, b in key_pairs:
        aa, bb = sorted([a, b])
        series = [(p, lookup.get((p, aa, bb))) for p in periods
                  if lookup.get((p, aa, bb)) is not None]
        if not series:
            continue
        say(f"\nd({a}, {b}):")
        for p, d in series:
            fl = [floor[t].get(p) for t in (a, b) if floor[t].get(p)]
            nf = max(fl) if fl else None
            sig = " *" if (nf and d > 2 * nf) else ""
            say(f"  {p}: {d:.4f}{sig}")

    # --- C. WITHIN-TERM LINEAGES + branching detection ---
    say("\n\n=== C. LINEAGES & CANDIDATE BRANCHING EVENTS ===")
    tree = {}
    for t in family:
        node_seq = []   # list of (period, [ {id, centroid, n, titles} ])
        for p in sorted(data[t]):
            sh = data[t][p]
            labels, k, dbic = cluster_period(sh["V"])
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
                    yrs = sh["years"][idx]
                    nodes.append({"id": f"{p}:{c}", "n": int(len(idx)),
                                  "centroid": sh["V"][idx].mean(0),
                                  "titles": readable(sh["keys"][idx]),
                                  "top_journals": top_j,
                                  "median_year": int(np.median(yrs))})
            if nodes:
                node_seq.append((p, nodes, dbic if labels is not None else 0.0))
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
        tree[t] = {
            "periods": [{"period": p, "dbic": round(db, 3),
                         "clusters": [{"id": nd["id"], "n": nd["n"]}
                                      for nd in nodes]}
                        for p, nodes, db in node_seq],
            "links": links, "branchings": branchings}
        if branchings:
            say(f"\n{t}: {len(branchings)} candidate branching event(s)")
            for b in branchings:
                say(f"   {b['parent']} -> {b['children']} at {b['period']} "
                    f"(sep {b['min_separation']} vs floor {b['noise_floor']})")
        # qualitative sample file
        with open(RESULTS / f"lineage_samples_{t.replace(' ', '_')}.txt",
                  "w", encoding="utf-8") as f:
            for p, nodes, db in node_seq:
                f.write(f"===== {p} (dBIC/n={db:.2f}) =====\n")
                for nd in nodes:
                    f.write(f"  cluster {nd['id']} (n={nd['n']}"
                            + (f", median year {nd['median_year']}"
                               if "median_year" in nd else "") + "):\n")
                    if nd.get("top_journals"):
                        f.write("    journals: "
                                + "; ".join(f"{j} ({c})"
                                            for j, c in nd["top_journals"])
                                + "\n")
                    for ti in nd["titles"]:
                        f.write(f"    - {str(ti)[:110]}\n")
                f.write("\n")

    with open(RESULTS / "lineage_tree.json", "w") as f:
        json.dump({t: {k: v for k, v in d.items()}
                   for t, d in tree.items()}, f, indent=2, default=str)

    # --- D. EVENTS ---
    say("\n\n=== D. DATED EVENTS (read curves against these) ===")
    for y, e in sorted(EVENTS.items()):
        say(f"  {y}: {e}")
    say("\nREAD lineage_samples_*.txt before interpreting any branching — "
        "clusters must make sense as usage communities (every prior artifact "
        "was caught by reading, not by metrics).")

    (RESULTS / "lineage_report.txt").write_text("\n".join(report),
                                                encoding="utf-8")
    print(f"\nAll outputs -> {RESULTS}/ (report: lineage_report.txt)")


if __name__ == "__main__":
    main()
