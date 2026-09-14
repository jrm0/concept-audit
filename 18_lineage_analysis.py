"""
18_lineage_analysis.py — Temporal lineage analysis: drift, divergence,
branching. (No torch needed; runs anywhere sklearn is installed.)

The four analyses, in the order the paper needs them:

 A. DRIFT   Per term: distance of each period's centroid from the term's own
            origin centroid, MINUS the same quantity for control terms
            ("participants", "results"...) — excess drift above what corpus
            change alone produces (Dubossarsky control). Split-half centroid
            reliability gives a noise floor per period.
            MATCHED-n (2026-09-08): every cross-period distance is computed
            at equal sample size (n_eq = min(n_a, n_b, analysis.equalize_n)),
            repeated analysis.equalize_reps times, reported as mean ± sd, with
            the full-data distance alongside. A small period's wobble is
            therefore never read as movement. Control drift and noise floors
            are evaluated at the same n_eq.

 B. BIRTH & DIVERGENCE   Cross-term distances per period (masked space is
            commensurable across terms), matched-n as in A. The speciation
            prediction: at birth "ego depletion" sits close to its ancestors
            (self-control / self-regulation); afterwards it DIVERGES.
            Also: do the ancestors converge or diverge from each other?

 C. LINEAGES WITHIN TERMS   For high-volume terms, within-period clustering
            (cluster_core: k-means proposal, dip test + bootstrap Jaccard
            gate, document units), then descent links across consecutive
            periods by centroid similarity. A BRANCHING event = two clusters
            in period p+1 whose best ancestor is the same cluster in p and
            whose mutual separation exceeds the noise floor. Only periods
            whose clustering is SUPPORTED produce >1 node. Uses EVERY
            document in the period (no cap): minority senses are only
            findable with the full data, and this is where lineages are
            traced.

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
    python3 18_lineage_analysis.py --frame-control   # EXPLORATORY (Session 30):
        partial the grammatical frame (syntax-feature rates) out of every
        vector first; outputs suffixed _framectl. Read with the raw run.

SUBSET RE-RUNS (corpus principle 9: corpus dependence is a result). The
filters keep only document vectors whose provenance matches, then run the
identical analysis; outputs get a suffix (_src-<...>, _dbs-<...>,
_strata-<...>) so subset runs never overwrite the full run. --dbs matches
substrings of the RIS DB tags stored per vector by 17 (works on any shard);
--sources / --strata need shards embedded after 2026-09-08 (17 now stores
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
# --frame-control (Session 30, EXPLORATORY — not in the Session-28
# pre-registration): partial the grammatical frame out of every vector before
# any analysis. Per term, pooled over periods, OLS of the vectors on the 7
# syntax-feature rates; the residual (plus the term's grand mean) replaces
# the vector. Removes the part of the geometry that follows the term's
# syntactic role (modifier / premodified / attributive / coordinated / title /
# definitional / parenthetical). Caveat stated in the report: frame and sense
# are correlated ("trauma exposure" = event sense, "trauma symptoms" =
# condition sense), so this also removes real sense variance that happens to
# ride on frame. Read raw and frame-controlled together.
FRAME_CONTROL = "--frame-control" in _ARGV
SUF = ("_centered" if CENTER else "") + ("_framectl" if FRAME_CONTROL else "") + _fsuf
MIN_N = int(CFG["analysis"]["min_n"])
CLUSTER_MIN_N = int(CFG["analysis"]["cluster_min_n"])
MAX_K = int(CFG["analysis"]["max_k"])
SEED = int(CFG["analysis"]["seed"])
CONTROL_TERMS = set(CFG.controls)
SYNTAX_FLAG = float(CFG["analysis"].get("syntax_flag_diff", 0.5))   # Session 29 guard
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
                             "2026-09-08 (they store corpus-spec source ids); "
                             "use --dbs on older shards.")
        ids = [set(str(x).split("|")) for x in z["sources"]]
        mask &= np.array([bool(s & set(FILTER["sources"])) for s in ids])
    if FILTER["strata"]:
        if "strata" not in z.files:
            raise SystemExit("--strata needs shards embedded by 17 after "
                             "2026-09-08 (they store the register stratum).")
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
            "dbs": sel(z["dbs"]) if "dbs" in z.files else None,
            # (n,4) per-doc syntax rates from 17 (Session 29); None on old shards
            "syntax": sel(z["syntax"]) if "syntax" in z.files else None}
    if any(FILTER.values()):
        print(f"SUBSET FILTER {FILTER}: kept {kept}/{total} document vectors")
    return data


def frame_residualize(data):
    """--frame-control: replace V by V - (S - S̄) @ B per term, B from OLS of
    the pooled (all periods) vectors on the syntax-feature rates S. Returns
    {term: {"r2": variance explained by frame, "n": pooled n}}. Refuses shards
    without a syntax array (re-run 17)."""
    report = {}
    for t, per in data.items():
        missing = [p for p, sh in per.items() if sh.get("syntax") is None]
        if missing:
            raise SystemExit(f"--frame-control needs the syntax array on every shard; "
                             f"{t} lacks it for {missing} — re-run 17 (--refresh-changed).")
        ps = sorted(per)
        V = np.vstack([per[p]["V"] for p in ps])
        S = np.vstack([per[p]["syntax"] for p in ps]).astype(float)
        if len(V) < 3 * S.shape[1] + 5:          # too few docs to fit 8 parameters sensibly
            report[t] = {"r2": None, "n": int(len(V))}
            continue
        Sc = S - S.mean(0)
        B, *_ = np.linalg.lstsq(Sc, V - V.mean(0), rcond=None)
        fitted = Sc @ B
        ss_tot = float(((V - V.mean(0)) ** 2).sum())
        r2 = float((fitted ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0
        i = 0
        for p in ps:
            n = len(per[p]["V"])
            per[p]["V"] = per[p]["V"] - fitted[i:i + n]
            i += n
        report[t] = {"r2": round(r2, 4), "n": int(len(V))}
    return report


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
    frame_r2 = frame_residualize(data) if FRAME_CONTROL else None
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
    say(f"MODE: {'PER-PERIOD MEAN-CENTERED (anisotropy robustness)' if CENTER else 'raw (uncentered)'}"
        + ("  + FRAME-CONTROLLED (EXPLORATORY, outside the pre-registration)" if FRAME_CONTROL else ""))
    if any(FILTER.values()):
        say(f"SUBSET: {FILTER}  (outputs suffixed '{SUF}')")
    if FRAME_CONTROL:
        say("\nFRAME CONTROL: per term, pooled over periods, vectors regressed on the 7 "
            "syntax-feature rates (" + "/".join(pc.SYNTAX_FEATURE_NAMES) + "); residuals "
            "replace vectors in every section below. R2 = share of vector variance the "
            "frame explains. Caveat: frame and sense are correlated (\"trauma exposure\" "
            "= event, \"trauma symptoms\" = condition), so real sense variance that rides "
            "on frame is removed too. Read together with the raw run.")
        for t in sorted(frame_r2, key=lambda x: -(frame_r2[x]["r2"] or 0)):
            r = frame_r2[t]
            r2_txt = "-" if r["r2"] is None else f"{r['r2']:.3f}"
            say(f"  {t:36s} n={r['n']:6d}  frame R2={r2_txt}")
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

    # Full-data centroids (used for C, for descriptive output, and as the
    # "full" point estimate); vectors kept for matched-n resampling.
    cent, floor, ns, vecs = (defaultdict(dict), defaultdict(dict),
                             defaultdict(dict), defaultdict(dict))
    for t in data:
        for p, sh in data[t].items():
            if CENTER and p not in grand:
                continue          # no composition-stable reference: skip
            if len(sh["V"]) >= MIN_N:
                V = (sh["V"] - grand[p] if CENTER else sh["V"]).astype(np.float64)
                vecs[t][p] = V
                cent[t][p] = V.mean(0)
                floor[t][p] = split_half_floor(V)
                ns[t][p] = len(V)

    # ---------------------------------------------------------------------
    # MATCHED-n RESAMPLING (2026-09-08). Cross-period comparisons are made at
    # equal sample size so that the wobble of a small period is not read as
    # movement: for each pair of document sets (A, B) draw n_eq = min(|A|,
    # |B|, equalize_n) documents from each, compute the centroid distance,
    # repeat EQ_REPS times, report mean and sd. The full-data distance is
    # reported alongside. Within-period description (section C) uses every
    # document — that is the point of removing the embedding cap.
    # Note: when n_eq equals a set's size there is nothing to resample and
    # the sd reads 0.0; the matched noise floor carries the uncertainty then.
    # ---------------------------------------------------------------------
    EQ_N = int(CFG["analysis"].get("equalize_n") or 250)
    EQ_REPS = int(CFG["analysis"].get("equalize_reps") or 100)
    rng_eq = np.random.RandomState(SEED)

    def matched_distance(A, B):
        """(full_d, eq_mean, eq_sd, n_eq) for centroid cosine distance."""
        full = float(cosine(A.mean(0), B.mean(0)))
        n_eq = min(len(A), len(B), EQ_N)
        if n_eq < MIN_N:
            return full, None, None, n_eq
        ds = []
        for _ in range(EQ_REPS):
            ia = rng_eq.choice(len(A), n_eq, replace=False)
            ib = rng_eq.choice(len(B), n_eq, replace=False)
            ds.append(cosine(A[ia].mean(0), B[ib].mean(0)))
        return full, float(np.mean(ds)), float(np.std(ds)), n_eq

    def matched_floor(V, n_eq):
        """Split-half noise floor evaluated on samples of size n_eq (so the
        floor is on the same footing as the matched distance)."""
        if n_eq is None or n_eq < 10:
            return None
        ds = []
        for _ in range(max(10, EQ_REPS // 4)):
            idx = rng_eq.choice(len(V), min(len(V), n_eq), replace=False)
            h = len(idx) // 2
            ds.append(cosine(V[idx[:h]].mean(0), V[idx[h:2 * h]].mean(0)))
        return float(np.mean(ds))

    say(f"Matched-n resampling: n_eq = min(n_a, n_b, {EQ_N}); {EQ_REPS} draws; "
        f"full-data estimate reported alongside.")

    # --- A. DRIFT with control null ---
    # control drift per period, matched to the SAME n_eq used for the term
    # (controls are large; sub-sampling them to n_eq keeps the null honest)
    def ctrl_drift_at(p, n_eq):
        vals = []
        for c in controls:
            ps = sorted(vecs[c])
            if p in vecs[c] and ps and ps[0] != p:
                A, B = vecs[c][ps[0]], vecs[c][p]
                k = min(len(A), len(B), n_eq)
                if k < MIN_N:
                    continue
                ds = []
                for _ in range(max(10, EQ_REPS // 4)):
                    ia = rng_eq.choice(len(A), k, replace=False)
                    ib = rng_eq.choice(len(B), k, replace=False)
                    ds.append(cosine(A[ia].mean(0), B[ib].mean(0)))
                vals.append(float(np.mean(ds)))
        return float(np.mean(vals)) if vals else None

    drift_rows = []
    say("\n=== A. DRIFT FROM ORIGIN (matched-n; excess over control terms) ===")
    if CENTER:
        say("CAUTION: under --center the control terms DEFINE the reference, "
            "so their residual drift is deflated by construction and 'excess' "
            "is optimistic here. Read centered results for B (cross-term "
            "distances, common reference, unaffected); read A primarily in "
            "the raw run, using centering only as a shape check.")
    for t in family:
        ps = sorted(vecs[t])
        if len(ps) < 2:
            continue
        origin = ps[0]
        say(f"\n{t}  (origin {origin}, n={ns[t][origin]})")
        for p in ps:
            if p == origin:
                full, d, sd, n_eq = 0.0, 0.0, 0.0, min(ns[t][p], EQ_N)
            else:
                full, d, sd, n_eq = matched_distance(vecs[t][origin], vecs[t][p])
            cd = ctrl_drift_at(p, n_eq) if d is not None else None
            excess = (d - cd) if (d is not None and cd is not None) else None
            nf = matched_floor(vecs[t][p], n_eq)
            flag = " *" if (excess is not None and nf is not None
                            and excess > 2 * nf) else ""
            say(f"  {p}: n={ns[t][p]:>5} n_eq={n_eq:>4} "
                f"drift={'-' if d is None else f'{d:.4f}'}"
                f"±{'-' if sd is None else f'{sd:.4f}'} (full {full:.4f}) "
                f"control={'-' if cd is None else f'{cd:.4f}'} "
                f"excess={'-' if excess is None else f'{excess:.4f}'}{flag}")
            drift_rows.append([t, p, ns[t][p], n_eq,
                               "" if d is None else round(d, 4),
                               "" if sd is None else round(sd, 4),
                               round(full, 4),
                               "" if cd is None else round(cd, 4),
                               "" if excess is None else round(excess, 4),
                               "" if nf is None else round(nf, 4)])
    with open(RESULTS / f"lineage_drift{SUF}.csv", "w", encoding="utf-8") as f:
        # drift_from_origin / control_drift / excess_drift are the MATCHED-n
        # estimates (primary); drift_full is the all-documents estimate.
        f.write("term,period,n,n_eq,drift_from_origin,drift_eq_sd,drift_full,"
                "control_drift,excess_drift,noise_floor\n")
        for r in drift_rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # --- B. BIRTH & DIVERGENCE (cross-term distances per period) ---
    say("\n\n=== B. CROSS-TERM DISTANCES (masked space; matched-n) ===")
    dist_rows = []
    key_pairs = KEY_PAIRS
    for p in periods:
        for a, b in combinations(sorted(vecs), 2):
            if p in vecs[a] and p in vecs[b]:
                full, d, sd, n_eq = matched_distance(vecs[a][p], vecs[b][p])
                dist_rows.append([p, a, b,
                                  round(full, 4) if d is None else round(d, 4),
                                  "" if sd is None else round(sd, 4),
                                  round(full, 4), n_eq])
    with open(RESULTS / f"term_distances{SUF}.csv", "w", encoding="utf-8") as f:
        # cosine_distance = matched-n estimate (falls back to full when a
        # side is below MIN_N); distance_full = all documents.
        f.write("period,term_a,term_b,cosine_distance,distance_eq_sd,"
                "distance_full,n_eq\n")
        for r in dist_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    lookup = {(p, a, b): (d, sd, n_eq) for p, a, b, d, sd, _f, n_eq in dist_rows}
    for a, b in key_pairs:
        aa, bb = sorted([a, b])
        series = [(p, lookup.get((p, aa, bb))) for p in periods
                  if lookup.get((p, aa, bb)) is not None]
        if not series:
            say(f"\nd({a}, {b}): no shared periods (term missing from shards?)")
            continue
        say(f"\nd({a}, {b}):")
        for p, (d, sd, n_eq) in series:
            fl = [matched_floor(vecs[t][p], n_eq) for t in (a, b)
                  if t in vecs and p in vecs[t]]
            fl = [x for x in fl if x]
            nf = max(fl) if fl else None
            sig = " *" if (nf and d > 2 * nf) else ""
            say(f"  {p}: {d:.4f}±{'-' if sd in (None, '') else f'{sd:.4f}'} "
                f"(n_eq={n_eq}){sig}")

    # --- C. WITHIN-TERM LINEAGES + branching detection ---
    if CENTER:
        say("\n\n=== C. SKIPPED under --center: PCA is translation-invariant, "
            "so clustering is identical to the uncentered run. ===")
    else:
        say("\n\n=== C. LINEAGES & CANDIDATE BRANCHING EVENTS ===")
    tree = {}
    n_supported = n_tested = 0
    syntax_flags = []          # (term, period, feature, diff)
    n_guard_checked = n_guard_missing = 0
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
            # SYNTAX GUARD (Session 29): a supported split whose clusters differ
            # by > SYNTAX_FLAG on any syntax-feature rate is a register split
            # (title vs abstract sentence, definition vs use, parenthetical,
            # attributive), not a sense split — flagged, never silently passed.
            stats["syntax_guard"] = None
            if cr.get("verdict") == "SUPPORTED" and labels is not None:
                if sh["syntax"] is None:
                    n_guard_missing += 1
                    stats["syntax_guard"] = "no syntax array (shard predates Session 29)"
                else:
                    n_guard_checked += 1
                    rates = {c: sh["syntax"][labels == c].mean(0)
                             for c in range(k)}
                    n_feat = sh["syntax"].shape[1]        # 4 (Session 29) or 7 (Session 30)
                    fnames = pc.SYNTAX_FEATURE_NAMES[:n_feat]
                    worst = (None, 0.0)
                    for fi, fname in enumerate(fnames):
                        vals = [float(rates[c][fi]) for c in range(k)]
                        diff = max(vals) - min(vals)
                        if diff > worst[1]:
                            worst = (fname, diff)
                    flagged = worst[1] > SYNTAX_FLAG
                    kind = ("frame" if worst[0] in pc.FRAME_FEATURES else "register") if flagged else None
                    stats["syntax_guard"] = {
                        "features": fnames,
                        "rates": {f"{p}:{c}": [round(float(x), 3) for x in rates[c]]
                                  for c in range(k)},
                        "worst_feature": worst[0], "max_diff": round(worst[1], 3),
                        "flagged": bool(flagged), "kind": kind}
                    if flagged:
                        syntax_flags.append((t, p, worst[0], worst[1], kind))
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
                         "syntax_guard": st.get("syntax_guard"),
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
        with open(RESULTS / f"lineage_samples_{t.replace(' ', '_')}{SUF}.txt",
                  "w", encoding="utf-8") as f:
            for p, nodes, st in node_seq:
                dp = st["dip_p"]
                mj = st["mean_jaccard"]
                sg = st.get("syntax_guard")
                flag_txt = ""
                if isinstance(sg, dict):
                    flag_txt = (f"; {sg.get('kind', 'syntax').upper()} SPLIT: "
                                f"{sg['worst_feature']} differs by {sg['max_diff']:.2f}"
                                if sg["flagged"]
                                else f"; syntax guard ok (max diff {sg['max_diff']:.2f} on {sg['worst_feature']})")
                f.write(f"===== {p} ({st['verdict']}; dip p="
                        f"{'-' if dp is None else f'{dp:.3f}'}, mean Jaccard="
                        f"{'-' if mj is None else f'{mj:.2f}'}{flag_txt}) =====\n")
                for nd in nodes:
                    f.write(f"  cluster {nd['id']} (n={nd['n']}"
                            + (f", median year {nd['median_year']}"
                               if "median_year" in nd else "") + "):\n")
                    if isinstance(sg, dict) and nd["id"] in sg["rates"]:
                        f.write("    syntax rates ("
                                + "/".join(sg.get("features", pc.SYNTAX_FEATURE_NAMES))
                                + "): "
                                + "/".join(f"{x:.2f}" for x in sg["rates"][nd["id"]])
                                + "\n")
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
        say(f"\nSYNTAX GUARD (threshold {SYNTAX_FLAG} on any of "
            f"{'/'.join(pc.SYNTAX_FEATURE_NAMES)}): "
            f"{len(syntax_flags)} of {n_guard_checked} supported splits FLAGGED "
            f"as register splits"
            + (f"; {n_guard_missing} supported splits could not be checked "
               f"(shards predate the syntax array — re-run 17)"
               if n_guard_missing else "") + ".")
        for t, p, fname, diff, kind in syntax_flags:
            say(f"   {kind.upper():8s} {t} | {p}: {fname} rate differs by {diff:.2f} between clusters")
        if syntax_flags:
            say("   REGISTER = how the term is presented (title vs abstract sentence, "
                "definition vs use, parenthetical): not a sense split.\n"
                "   FRAME    = the term's grammatical role (modifier 'PTSD symptoms', "
                "premodified 'childhood trauma', attributive 'trauma-focused', "
                "coordinated list). Frame productivity is itself a form of creep — "
                "report it as such, not as a sense split; test sense structure with "
                "--frame-control.")
        with open(RESULTS / f"lineage_tree{SUF}.json", "w") as f:
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
        outs += [RESULTS / f"lineage_tree{SUF}.json"] + \
                [RESULTS / f"lineage_samples_{t.replace(' ', '_')}{SUF}.txt"
                 for t in family]
    pc.write_manifest(
        CFG, __file__,
        params={"center": CENTER, "subset_filter": FILTER, "suffix": SUF,
                "frame_control": FRAME_CONTROL, "frame_r2": frame_r2,
                "equalize_n": EQ_N, "equalize_reps": EQ_REPS,
                "min_n": MIN_N, "cluster_min_n": CLUSTER_MIN_N,
                "max_k": MAX_K, "seed": SEED, "control_terms": sorted(CONTROL_TERMS),
                "terms": family, "periods": periods, "key_pairs": key_pairs,
                "clustering": "cluster_core (dip + Hennig Jaccard)",
                "cells_tested": n_tested, "cells_supported": n_supported,
                "syntax_guard": {"threshold": SYNTAX_FLAG,
                                 "checked": n_guard_checked,
                                 "unchecked_old_shards": n_guard_missing,
                                 "flagged": [[t, p, f, round(d, 3), kind]
                                             for t, p, f, d, kind in syntax_flags]},
                "n_shards": sum(len(v) for v in data.values())},
        outputs=outs)
    print(f"\nAll outputs -> {RESULTS}/ (report: lineage_report{SUF}.txt)")


if __name__ == "__main__":
    main()
