"""
30_correspondence_drift.py — correspondence metrics beside the global drift.

The global metrics the lineage pipeline reports (18: centroid distance = PRT;
pairwise distance = APD) register ANY redistribution of usages, including a
purely topical one: if 2020 abstracts talk about the same senses of a word in
different proportions, the centroid moves. Goworek & Dubossarsky (2026)
propose correspondence metrics that ask instead whether each usage in one
period has a close counterpart in the other:

  PRT(A,B)  = d(mu(A), mu(B))                       prototype (centroid) distance
  APD(A,B)  = mean over a in A, b in B of d(a,b)      average pairwise distance
  AMD(A->B) = mean over a in A of min_b d(a,b)        average minimum distance
  AMD(A,B)  = (AMD(A->B) + AMD(B->A)) / 2            symmetric
  SAMD(A,B) = mean matched-pair distance under greedy one-to-one matching
              of equal-size samples (smallest remaining distance first)
  d = cosine distance.

Reading: these distances characterize the selected representation. Neither
AMD near a reference floor nor AMD above it establishes semantic stability
or novelty without independent validation. Every metric is reported with
its own noise floor — the same metric between two random halves of the SAME
period — and for the control terms (the drift null), so a number is only
ever read against what the method produces on nothing.

SOURCE: Goworek & Dubossarsky (2026), LChange proceedings, pp. 147-161,
https://aclanthology.org/2026.lchange-1.13/ .

Inputs: stored document vectors with identities resolved against the corpus.
Each document receives one vector per set (mean across its member vectors).
All between-set and within-set samples use n=min(cap, floor(N_A/2), floor(N_B/2)).
Reported SDs describe subsampling variability, not confidence intervals.

Centering: --center controls (default) subtracts each period's control-term
centroid before cosine (18's anisotropy reference); --center none for raw.

USAGE
    python3 30_correspondence_drift.py --config configs/ptsd_trauma.yaml
    python3 30_correspondence_drift.py --config configs/ego_depletion.yaml --terms "ego depletion,self control"
    options: --equalize 250 --reps 20 --min-n 15 --center controls|none --pairs consecutive|origin|both
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import pipeline_config as pc
from revision_core import document_id, consolidate_vectors, matched_size

CFG, _ARGV = pc.load()


def _opt(name, default=None):
    for i, a in enumerate(_ARGV):
        if a == name and i + 1 < len(_ARGV):
            return _ARGV[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


ONLY = [t.strip() for t in (_opt("--terms") or "").split(",") if t.strip()]
EQUALIZE = int(_opt("--equalize", CFG.get("analysis", {}).get("equalize_n", 250)))
REPS = int(_opt("--reps", 20))
MIN_N = int(_opt("--min-n", 15))
CENTER = _opt("--center", "controls")
PAIRS = _opt("--pairs", "both")
SEED = int(_opt("--seed", CFG.get("analysis", {}).get("seed", 42)))
CROSS = _opt("--cross")
COMPARE = [c.strip() for c in (_opt("--compare") or "").split(";") if c.strip()]
CROSS_NAME = _opt("--cross-name", "correspondence_cross")
EMB = CFG.path("emb_dir")
OUT = CFG.results
SUF = "" if CENTER == "controls" else "_raw"


SHARD_KEYS = defaultdict(dict)
IDENTITY_AUDIT = {}


def load_shards():
    """Resolve legacy truncated keys through full corpus metadata.

    Exclude ambiguous/unmatched keys instead of merging distinct titles.
    Duplicate records with the same DOI/full-title identity are averaged.
    """
    wanted = None
    if ONLY and not CROSS:
        wanted = {pc.slug(t) for t in ONLY} | {pc.slug(c) for c in CFG.controls}
    if CROSS:
        terms = CROSS.replace('PTSD', 'ptsd,posttraumatic stress disorder')
        wanted = {pc.slug(t.strip()) for part in terms.split(';')
                  for t in part.split('=', 1)[1].split('@')[0].split(',')}
        wanted |= {pc.slug(c) for c in CFG.controls}
    index = defaultdict(set)
    with CFG.path('docs_path').open(encoding='utf-8') as f:
        for line in f:
            d = json.loads(line)
            if d['year'] > CFG.get('revision_analysis_year_max', 9999) or document_id(d) is None:
                continue
            old = d.get('doi') or d.get('title', '')[:60]
            for t in d.get('mention_sentences', {}):
                slug = pc.slug(t)
                if wanted is None or slug in wanted:
                    key = (slug, d.get('period'), old, int(d['year']), d.get('journal', ''))
                    index[key].add(document_id(d))
    shards = defaultdict(dict)
    for f in sorted(EMB.glob('*__*.npz')):
        slug, period = f.stem.rsplit('__', 1)
        if wanted is not None and slug not in wanted:
            continue
        if int(period.split('-')[0]) > CFG.get('revision_analysis_year_max', 9999):
            continue
        with np.load(f, allow_pickle=True) as z:
            X = np.asarray(z['vectors'], dtype=np.float64)
            resolved, keep = [], []
            for i, (key, year, journal) in enumerate(zip(z['keys'], z['years'], z['journals'])):
                candidates = index.get((slug, period, str(key), int(year), str(journal)), set())
                if len(candidates) == 1:
                    keep.append(i); resolved.append(next(iter(candidates)))
            Y, keys = consolidate_vectors(X[keep], resolved)
            IDENTITY_AUDIT[f.stem] = {'input_vectors': len(X), 'unresolved_or_ambiguous': len(X)-len(keep),
                                     'duplicate_rows_consolidated': len(keep)-len(Y), 'documents': len(Y)}
            if len(Y):
                shards[slug][period] = Y
                SHARD_KEYS[slug][period] = keys
    return shards


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def prt(A, B):
    a, b = A.mean(0), B.mean(0)
    return float(1 - a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def cross(A, B):
    return 1.0 - unit(A) @ unit(B).T          # cosine distance matrix |A| x |B|


def apd(D):
    return float(D.mean())


def amd(D):
    return float(0.5 * (D.min(1).mean() + D.min(0).mean()))


def amd_dir(D):
    """(AMD(A->B), AMD(B->A)): mean over rows of the row-min, mean over cols of the col-min."""
    return float(D.min(1).mean()), float(D.min(0).mean())


def samd(D):
    """Greedy one-to-one matching, smallest remaining distance first (equal sizes)."""
    D = D.copy()
    n = min(D.shape)
    order = np.argsort(D, axis=None)
    used_r, used_c, tot, k = np.zeros(D.shape[0], bool), np.zeros(D.shape[1], bool), 0.0, 0
    for flat in order:
        i, j = divmod(int(flat), D.shape[1])
        if used_r[i] or used_c[j]:
            continue
        used_r[i] = used_c[j] = True
        tot += D[i, j]; k += 1
        if k == n:
            break
    return float(tot / max(k, 1))


def metrics(A, B):
    D = cross(A, B)
    return {"PRT": prt(A, B), "APD": apd(D), "AMD": amd(D), "SAMD": samd(D)}


def matched(A, B, rng, n):
    """Metrics at matched n, REPS draws: mean and sd per metric."""
    acc = defaultdict(list)
    for _ in range(REPS):
        a = A[rng.choice(len(A), n, replace=False)]
        b = B[rng.choice(len(B), n, replace=False)]
        for k, v in metrics(a, b).items():
            acc[k].append(v)
    return {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for k, v in acc.items()}


def floor(A, rng, n):
    """Reference comparison with exactly n documents in each disjoint half."""
    if n < 5 or 2 * n > len(A):
        raise ValueError("Floor requires two disjoint samples at the comparison size")
    m = n
    acc = defaultdict(list)
    for _ in range(REPS):
        idx = rng.permutation(len(A))
        for k, v in metrics(A[idx[:m]], A[idx[m:2 * m]]).items():
            acc[k].append(v)
    return {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for k, v in acc.items()}


def parse_sets(spec, shards):
    """'NAME=term1,term2@Y1-Y2;...' -> {NAME: [(slug, period), ...]}"""
    sets = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, _, rest = part.partition("=")
        terms, _, yrs = rest.partition("@")
        y1, y2 = (int(x) for x in yrs.split("-"))
        slugs = []
        for t in terms.split(","):
            t = t.strip()
            if t.upper() == "PTSD":
                slugs += ["ptsd", "posttraumatic_stress_disorder"]
            else:
                slugs.append(pc.slug(t))
        members = []
        for sl in dict.fromkeys(slugs):
            for per in shards.get(sl, {}):
                start = int(per.split("-")[0])
                if y1 <= start <= y2:
                    members.append((sl, per))
        sets[name.strip()] = members
    return sets


def cross_mode(shards, controls, get):
    """Addendum A1: correspondence between pooled, period-centred sets."""
    rng = np.random.default_rng(SEED)
    sets = parse_sets(CROSS, shards)
    X, set_keys = {}, {}
    for name, members in sets.items():
        parts, keys = [], []
        for sl, per in members:
            v = get(sl, per)
            if v is not None and len(v):
                parts.append(v)
                keys.extend(SHARD_KEYS[sl][per])
        raw = np.vstack(parts) if parts else np.zeros((0, 1))
        X[name], set_keys[name] = consolidate_vectors(raw, keys)
        print(f"[30] set {name:<10} n={len(X[name]):>6}  from " + ", ".join(f"{sl}@{per[:4]}({len(shards[sl][per])})" for sl, per in members))
    small = min(len(v) for v in X.values())
    if small < MIN_N:
        raise SystemExit(f"[30] a set has n={small} < --min-n {MIN_N}: NOT TESTABLE at this corpus")
    n = matched_size([len(v) for v in X.values()], EQUALIZE)
    if n < 5:
        raise SystemExit("Insufficient documents for disjoint matched reference samples")
    overlaps = {f"{a}:{b}": len(set(set_keys[a]) & set(set_keys[b]))
                for i, a in enumerate(X) for b in list(X)[i+1:]}
    if any(overlaps.values()):
        raise SystemExit(f"Cross sets share documents; specify disjoint sets before comparing: {overlaps}")
    names = list(X)
    print(f"[30] matched n = {n} per set, {REPS} draws (one sample per set per draw; pairs are paired by draw)")
    pair_draws = {}   # (A,B) -> {metric: [per draw]}
    floor_draws = {name: defaultdict(list) for name in names}
    for _ in range(REPS):
        samp = {name: X[name][rng.choice(len(X[name]), n, replace=False)] for name in names}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                D = cross(samp[a], samp[b])
                m = metrics(samp[a], samp[b])
                ab, ba = amd_dir(D)
                m["AMD_A->B"], m["AMD_B->A"] = ab, ba
                acc = pair_draws.setdefault((a, b), defaultdict(list))
                for k, v in m.items():
                    acc[k].append(v)
        # Floors use exactly the same n as the between-set comparisons.
        for name in names:
            m_ = n
            if m_ < 5:
                continue
            idx = rng.permutation(len(X[name]))
            for k, v in metrics(X[name][idx[:m_]], X[name][idx[m_:2 * m_]]).items():
                floor_draws[name][k].append(v)
    def summ(d):
        return {k: (round(float(np.mean(v)), 4), round(float(np.std(v)), 4)) for k, v in d.items()}
    pairs = {f"{a}:{b}": {"n": n, "metrics": summ(acc), "draws": {k: [round(x, 5) for x in v] for k, v in acc.items()}}
             for (a, b), acc in pair_draws.items()}
    floors = {name: summ(d) for name, d in floor_draws.items()}
    # paired comparisons of symmetric AMD: "A:B<C:D"
    comps = []
    for c in COMPARE:
        lhs, _, rhs = c.partition("<")
        def draws_of(key):
            a, b = key.split(":")
            acc = pair_draws.get((a, b)) or pair_draws.get((b, a))
            return np.array(acc["AMD"]) if acc else None
        L_, R_ = draws_of(lhs.strip()), draws_of(rhs.strip())
        if L_ is None or R_ is None:
            comps.append({"comparison": c, "result": "NA", "detail": "pair not found"}); continue
        diff = R_ - L_
        md, sd = float(diff.mean()), float(diff.std())
        res = "PASS" if (md > 0 and md > 2 * sd) else "FAIL"
        comps.append({"comparison": c, "result": res, "mean_diff": round(md, 4), "sd_diff": round(sd, 4),
                      "share_draws_lhs_smaller": round(float((diff > 0).mean()), 3),
                      "lhs_AMD": round(float(L_.mean()), 4), "rhs_AMD": round(float(R_.mean()), 4)})
    out = OUT / f"{CROSS_NAME}{SUF}.json"
    out.write_text(json.dumps({"settings": {"cross": CROSS, "compare": COMPARE, "equalize": EQUALIZE, "reps": REPS,
                                            "min_n": MIN_N, "center": CENTER, "seed": SEED, "matched_n": n,
                                            "sets": {k: [list(x) for x in v] for k, v in sets.items()},
                                            "sampling_rule": "same n for between and both disjoint within-set comparisons",
                                            "unit": "unique corpus document; pooled vectors averaged within document",
                                            "uncertainty": "draw SD, not a confidence interval", "identity_audit": IDENTITY_AUDIT,
                                            "set_sizes": {k: int(len(v)) for k, v in X.items()}},
                               "pairs": pairs, "floors": floors, "comparisons": comps}, indent=1), encoding="utf-8")
    L = [f"CORRESPONDENCE — CROSS SETS — {CFG['concept']}  (center={CENTER}, matched n={n}, {REPS} paired draws)", "",
         "sets: " + "; ".join(f"{k} n={len(X[k])}" for k in names), "",
         f"  {'pair':22}{'PRT':>14}{'APD':>14}{'AMD':>14}{'AMD A->B':>14}{'AMD B->A':>14}{'SAMD':>14}"]
    for key, r in pairs.items():
        m = r["metrics"]
        L.append(f"  {key:22}" + "".join(f"{m[k][0]:>9.3f}±{m[k][1]:<4.3f}" for k in ("PRT", "APD", "AMD", "AMD_A->B", "AMD_B->A", "SAMD")))
    L.append("")
    L.append(f"  {'floor (set vs itself)':22}{'PRT':>14}{'APD':>14}{'AMD':>14}{'SAMD':>14}")
    for name, m in floors.items():
        if m:
            L.append(f"  {name:22}" + "".join(f"{m[k][0]:>9.3f}±{m[k][1]:<4.3f}" for k in ("PRT", "APD", "AMD", "SAMD")))
    if comps:
        L.append("")
        L.append("  comparisons (symmetric AMD, paired by draw; PASS = mean diff > 0 and > 2 sd):")
        for c in comps:
            L.append(f"  {c['comparison']:<26} {c['result']:<6} " + ", ".join(f"{k}={v}" for k, v in c.items() if k not in ("comparison", "result")))
    rep = OUT / f"{CROSS_NAME}{SUF}_report.txt"
    rep.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    pc.write_manifest(CFG, __file__, params={"cross": CROSS, "compare": COMPARE, "equalize": EQUALIZE, "reps": REPS,
                                             "min_n": MIN_N, "center": CENTER, "seed": SEED},
                      outputs=[out, rep], notes="Addendum A1 cross-set correspondence (pooled, period-centred sets).")


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    shards = load_shards()
    controls = [pc.slug(c) for c in CFG.controls]
    periods_all = sorted({p for s in shards.values() for p in s})
    # per-period control centroid (anisotropy reference), as in 18 --center
    def reference(p, exclude=None):
        """Control centroid for period p; a control term is centred on the OTHER
        controls (leave-one-out), otherwise it would be centred on itself and
        its own metrics would be degenerate."""
        Xs = [shards[c][p] for c in controls if p in shards.get(c, {}) and c != exclude]
        return np.vstack(Xs).mean(0) if Xs else None
    ref_cache = {}
    def get(slug, p):
        X = shards[slug][p]
        if CENTER == "controls":
            key = (p, slug if slug in controls else None)
            if key not in ref_cache:
                ref_cache[key] = reference(p, exclude=key[1])
            r = ref_cache[key]
            if r is None:
                return None if slug not in controls else X   # single control: leave it raw
            X = X - r
        return X

    if CROSS:
        cross_mode(shards, controls, get)
        print(f"[30] done in {time.time() - t0:.0f}s")
        return

    terms = [s for s in shards if s not in controls]
    if ONLY:
        want = {pc.slug(t) for t in ONLY}
        terms = [s for s in terms if s in want]
    order = terms + controls
    print(f"[30] {CFG['concept']}  shards for {len(terms)} terms + {len(controls)} controls; "
          f"center={CENTER}; matched n <= {EQUALIZE} x {REPS} draws")

    results = {}
    for slug in order:
        per = {p: get(slug, p) for p in sorted(shards[slug]) if len(shards[slug][p]) >= MIN_N}
        per = {p: X for p, X in per.items() if X is not None}
        ps = list(per)
        if len(ps) < 2:
            continue
        rows = []
        pairs = []
        if PAIRS in ("consecutive", "both"):
            pairs += [(ps[i], ps[i + 1], "consecutive") for i in range(len(ps) - 1)]
        if PAIRS in ("origin", "both"):
            pairs += [(ps[0], p, "origin") for p in ps[1:]]
        for a, b, kind in pairs:
            n = matched_size([len(per[a]), len(per[b])], EQUALIZE)
            if n < 5:
                continue
            m = matched(per[a], per[b], rng, n)
            fa = floor(per[a], rng, n)
            fb = floor(per[b], rng, n)
            rows.append({"from": a, "to": b, "kind": kind, "n": int(n), "metrics": m,
                         "floor_from": fa, "floor_to": fb, "floor_n": int(n)})
        results[slug] = {"periods": ps, "n": {p: int(len(per[p])) for p in ps}, "pairs": rows,
                         "is_control": slug in controls}
        print(f"  {slug:34} periods={len(ps):>2}  pairs={len(rows):>3}  {time.time() - t0:5.0f}s")

    out = OUT / f"correspondence_drift{SUF}.json"
    out.write_text(json.dumps({"settings": {"equalize": EQUALIZE, "reps": REPS, "min_n": MIN_N, "center": CENTER,
                                            "pairs": PAIRS, "seed": SEED, "emb_dir": str(EMB), "controls": controls,
                                            "sampling_rule": "same n for between and both disjoint within-set comparisons",
                                            "uncertainty": "draw SD, not a confidence interval", "identity_audit": IDENTITY_AUDIT},
                               "results": results}, indent=1), encoding="utf-8")

    # --- report: origin->latest and mean consecutive step, each metric vs its floor and vs controls
    L = [f"CORRESPONDENCE DRIFT — {CFG['concept']}  (center={CENTER}, matched n<={EQUALIZE}, {REPS} draws)", "",
         "Each cell: metric value / within-period floor (same metric between two halves of the origin period).",
         "Floors use the same n as the between-period comparison; SDs are subsampling variability.",
         "Distances describe representations, not validated semantic stability or novelty. Both floors are in JSON.", ""]
    hdr = f"  {'term':30}{'origin->last':>14}{'n':>5}" + "".join(f"{k:>16}" for k in ("PRT", "APD", "AMD", "SAMD")) + "   mean consecutive AMD/APD"
    L.append(hdr)
    for slug in order:
        r = results.get(slug)
        if not r:
            continue
        orig = [x for x in r["pairs"] if x["kind"] == "origin"]
        cons = [x for x in r["pairs"] if x["kind"] == "consecutive"]
        if orig:
            x = orig[-1]
            cells = "".join(f"{x['metrics'][k][0]:>8.3f}/{(x['floor_from'] or {}).get(k, (float('nan'),))[0]:<7.3f}"
                            for k in ("PRT", "APD", "AMD", "SAMD"))
            ca = np.mean([c["metrics"]["AMD"][0] for c in cons]) if cons else float("nan")
            cp = np.mean([c["metrics"]["APD"][0] for c in cons]) if cons else float("nan")
            tag = " (control)" if r["is_control"] else ""
            L.append(f"  {slug + tag:30}{x['from'][:4] + '->' + x['to'][:4]:>14}{x['n']:>5}{cells}   {ca:.3f}/{cp:.3f}")
    L.append("")
    for slug in order:
        r = results.get(slug)
        if not r or r["is_control"]:
            continue
        L.append(f"== {slug} — consecutive steps (value / floor_from) ==")
        L.append(f"  {'step':24}{'n':>5}{'PRT':>16}{'APD':>16}{'AMD':>16}{'SAMD':>16}")
        for x in r["pairs"]:
            if x["kind"] != "consecutive":
                continue
            cells = "".join(f"{x['metrics'][k][0]:>8.3f}/{(x['floor_from'] or {}).get(k, (float('nan'),))[0]:<7.3f}"
                            for k in ("PRT", "APD", "AMD", "SAMD"))
            L.append(f"  {x['from'] + ' -> ' + x['to']:24}{x['n']:>5}{cells}")
        L.append("")
    rep = OUT / f"correspondence_drift{SUF}_report.txt"
    rep.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    pc.write_manifest(CFG, __file__, params={"equalize": EQUALIZE, "reps": REPS, "min_n": MIN_N, "center": CENTER,
                                             "pairs": PAIRS, "seed": SEED}, outputs=[out, rep],
                      notes="AMD/SAMD after Goworek & Dubossarsky (2026, LChange proceedings); "
                            "APD/PRT as in the LSCD literature. Floors = same metric on two halves of one period.")
    print(f"[30] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
