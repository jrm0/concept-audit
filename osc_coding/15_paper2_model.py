"""
15_paper2_model.py — The four pre-specified attenuation checks on the
Paper-2 null, plus the declared set-level secondary. (sklearn/scipy only.)

Checks, agreed with Jonathan (session 21) BEFORE running, with the
decision rule: if the covariate-adjusted, clean-subsample estimate is
still ~0 AND the predictor is reliable, the null is a finding.

 1. PROPER MODEL. Logistic replicated_sig ~ pos_post_pct + power_r +
    surprising_o + discipline. Likelihood-ratio test of the semantic
    term: does it add ANY deviance over the known predictors?
 2. CONTINUOUS DV. Effect-size decline (es_o - es_r) and es_r, Spearman
    with pos_post_pct (permutation p).
 3. CLEAN-MEASUREMENT SUBSAMPLE. label_class == ATTESTED-COMMON,
    n_pre >= 30, original vector from >= 2 sentences. Noise attenuates;
    this is the fair test.
 4. RELIABILITY OF THE PREDICTOR. Split each original's mention-sentence
    vectors in half, compute pos_post_pct from each half, Spearman across
    studies. If unreliable, the null is UNINFORMATIVE and the upgrade is
    full text, not a different model.

 SECONDARY (declared): set-level mismatch = MAX pos_post_pct across the
 study's construct set (one drifted construct is enough to break a
 replication; IV-vs-DV cases like study 008 motivate this). Requires
 13 --all-slots shards; skipped with a message otherwise.

Usage (from code/osc_coding/):
    python3 15_paper2_model.py
Output: results/paper2_model_report.txt
"""

import csv
import re
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import chi2, spearmanr

EMB = Path("data/construct_emb")
DATASET = Path("../results/paper2_dataset.csv")
FINAL = Path("data/final_constructs.csv")
RESULTS = Path("../results")
SEED = 42
MIN_N_PRE_CLEAN = 30


def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def logistic_lr(X_red, X_full, y):
    """LR test: does the extra column in X_full add deviance? Returns
    (beta_of_added_term_standardized, lr_stat, p)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def ll(X):
        m = LogisticRegression(C=np.inf, max_iter=2000).fit(X, y)
        p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
        return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), m
    Xr = StandardScaler().fit_transform(X_red)
    Xf = StandardScaler().fit_transform(X_full)
    llr, _ = ll(Xr)
    llf, mf = ll(Xf)
    lr = 2 * (llf - llr)
    return float(mf.coef_[0][-1]), lr, float(chi2.sf(max(lr, 0), df=1))


def perm_spearman(x, y, n=20000, seed=SEED):
    rho, _ = spearmanr(x, y)
    rng = np.random.RandomState(seed)
    y = np.array(y, dtype=float)
    cnt = 0
    for _ in range(n):
        rng.shuffle(y)
        r2, _ = spearmanr(x, y)
        if abs(r2) >= abs(rho):
            cnt += 1
    return float(rho), cnt / n


def pct_of(dist, ref):
    return float((np.asarray(ref) < dist).mean() * 100)


def main():
    report = []

    def say(s=""):
        print(s)
        report.append(s)

    rows = list(csv.DictReader(open(DATASET, encoding="utf-8")))
    for r in rows:
        for k in ("pos_post_pct", "pos_pre_pct", "power_r", "surprising_o",
                  "es_o", "es_r", "n_pre"):
            r[k] = fnum(r.get(k))
        r["replicated_sig"] = (int(r["replicated_sig"])
                               if r.get("replicated_sig") in ("0", "1")
                               else None)

    # original n_sentences (for the clean filter + reliability)
    orig = np.load(EMB / "osc_originals.npz", allow_pickle=True)
    nsent = {str(p): int(n) for p, n in zip(orig["pilot_ids"],
                                            orig["n_sentences"])}

    usable = [r for r in rows if r["replicated_sig"] is not None
              and r["pos_post_pct"] == r["pos_post_pct"]]

    # Sentence-count-bias diagnostic: a vector averaged over more sentences
    # sits mechanically closer to the centroid. Field docs are capped at 3
    # sentences and originals' means now use the same cap, but the residual
    # 1-3 range still varies — if this correlates with pos_post_pct, add
    # n_sentences as a covariate in the logistic model.
    diag = [(nsent.get(r["pilot_id"], 0), r["pos_post_pct"])
            for r in usable if nsent.get(r["pilot_id"], 0) > 0]
    if len(diag) >= 15:
        rho_ns, p_ns = spearmanr([d[0] for d in diag], [d[1] for d in diag])
        say(f"sentence-count bias check: Spearman(n_sentences, "
            f"pos_post_pct) = {rho_ns:+.3f} (p={p_ns:.3f}) "
            f"{'— ADD n_sentences AS COVARIATE' if p_ns < .10 else '— ok'}")
    clean = [r for r in usable if r.get("label_class") == "ATTESTED-COMMON"
             and r["n_pre"] >= MIN_N_PRE_CLEAN
             and nsent.get(r["pilot_id"], 0) >= 2]
    say(f"usable (DV + pos_post_pct): {len(usable)}; "
        f"clean subsample: {len(clean)}")

    # ---- 1 + 3: logistic LR, full sample then clean
    say("\n=== 1. LOGISTIC MODEL (LR test of pos_post_pct over "
        "power_r + surprising_o + discipline + n_sentences) ===")
    say("  (n_sentences is in BOTH models: the sentence-count diagnostic "
        "fired, so the LR isolates pos_post_pct beyond it)")
    for name, grp in [("all usable", usable), ("CLEAN subsample", clean)]:
        g = [r for r in grp if r["power_r"] == r["power_r"]
             and r["surprising_o"] == r["surprising_o"]
             and nsent.get(r["pilot_id"], 0) > 0]
        if len(g) < 20:
            say(f"  {name}: n={len(g)} too small for the model")
            continue
        disc = [1.0 if "social" in (r.get("discipline_o") or "").lower()
                else 0.0 for r in g]
        ns = [float(nsent[r["pilot_id"]]) for r in g]
        Xr = np.column_stack([[r["power_r"] for r in g],
                              [r["surprising_o"] for r in g], disc, ns])
        Xf = np.column_stack([Xr, [r["pos_post_pct"] for r in g]])
        y = np.array([r["replicated_sig"] for r in g])
        beta, lr, p = logistic_lr(Xr, Xf, y)
        say(f"  {name} (n={len(g)}, {int(y.sum())} replicated): "
            f"std beta(pos_post_pct) = {beta:+.3f}, LR = {lr:.2f}, "
            f"p = {p:.4f}")

    # ---- 2: continuous DV
    say("\n=== 2. CONTINUOUS DV (Spearman, permutation p) ===")
    for name, grp in [("all usable", usable), ("CLEAN subsample", clean)]:
        g = [r for r in grp if r["es_o"] == r["es_o"]
             and r["es_r"] == r["es_r"]]
        if len(g) < 15:
            say(f"  {name}: n={len(g)} too small")
            continue
        x = [r["pos_post_pct"] for r in g]
        for dv, vals in [("es_decline (es_o - es_r)",
                          [r["es_o"] - r["es_r"] for r in g]),
                         ("es_r", [r["es_r"] for r in g])]:
            rho, p = perm_spearman(x, vals)
            say(f"  {name}, {dv}: rho = {rho:+.3f}, p = {p:.4f} "
                f"(n={len(g)})")

    # ---- 4: reliability of pos_post_pct
    say("\n=== 4. SPLIT-HALF RELIABILITY of pos_post_pct ===")
    sv = np.load(EMB / "osc_original_sentvecs.npz", allow_pickle=True)
    by_pid = {}
    for p, s, v in zip(sv["pilot_ids"], sv["slots"], sv["vectors"]):
        if int(s) == 1:
            by_pid.setdefault(str(p), []).append(np.asarray(v, float))
    finals = {r["pilot_id"].strip().zfill(3): r
              for r in csv.DictReader(open(FINAL, encoding="utf-8"))}
    rng = np.random.RandomState(SEED)
    h1, h2 = [], []
    for pid, vecs in sorted(by_pid.items()):
        if len(vecs) < 2:
            continue
        c = (finals.get(pid, {}).get("construct_1") or "").strip().lower()
        f = EMB / f"{slug(c)}__post.npz"
        if not c or not f.exists():
            continue
        V = np.load(f, allow_pickle=True)["vectors"].astype(np.float64)
        if len(V) < 10:
            continue
        cpost = V.mean(0)
        ref = [cosine(v, cpost) for v in V]
        idx = rng.permutation(len(vecs))
        a = np.mean([vecs[i] for i in idx[:len(vecs) // 2]], axis=0)
        b = np.mean([vecs[i] for i in idx[len(vecs) // 2:]], axis=0)
        h1.append(pct_of(cosine(a, cpost), ref))
        h2.append(pct_of(cosine(b, cpost), ref))
    if len(h1) >= 10:
        rho, pv = spearmanr(h1, h2)
        # Spearman-Brown step-up for the full-length (both-halves) measure
        sb = 2 * rho / (1 + rho) if rho > -1 else float("nan")
        say(f"  n={len(h1)} studies with >=2 sentences: split-half "
            f"rho = {rho:.3f} (p={pv:.4f}); Spearman-Brown full-length "
            f"reliability = {sb:.3f}")
        say("  READ: low reliability means the null is UNINFORMATIVE about "
            "the hypothesis — the upgrade is full-text usage, not a "
            "different model.")
    else:
        say(f"  only {len(h1)} studies with >=2 sentences — run 13 first "
            f"(it now saves per-sentence vectors).")

    # ---- SECONDARY: set-level mismatch (max over construct set)
    say("\n=== SECONDARY (declared): set-level mismatch = MAX pos_post_pct "
        "over the construct set ===")
    allo_path = EMB / "osc_originals_all.npz"
    if not allo_path.exists():
        say("  osc_originals_all.npz missing — run: "
            "caffeinate -i python3 13_embed_constructs.py --all-slots")
    else:
        allo = np.load(allo_path, allow_pickle=True)
        setpos = {}
        for pid, slot, c, v in zip(allo["pilot_ids"], allo["slots"],
                                   allo["constructs"], allo["vectors"]):
            f = EMB / f"{slug(str(c))}__post.npz"
            if not f.exists():
                continue
            V = np.load(f, allow_pickle=True)["vectors"].astype(np.float64)
            if len(V) < 10:
                continue
            cpost = V.mean(0)
            ref = [cosine(x, cpost) for x in V]
            pct = pct_of(cosine(np.asarray(v, float), cpost), ref)
            setpos.setdefault(str(pid), []).append(pct)
        n_multi = sum(1 for v in setpos.values() if len(v) >= 2)
        say(f"  studies with set positions: {len(setpos)} "
            f"({n_multi} with >=2 constructs)")
        g = [r for r in usable if r["pilot_id"] in setpos]
        if len(g) >= 20:
            x_max = [max(setpos[r["pilot_id"]]) for r in g]
            y = [r["replicated_sig"] for r in g]
            t = perm_spearman(x_max, y)
            say(f"  set_max_pos_post_pct vs replicated: rho = {t[0]:+.3f}, "
                f"p = {t[1]:.4f} (n={len(g)})")
            gm = [r for r in g if r["power_r"] == r["power_r"]
                  and r["surprising_o"] == r["surprising_o"]
                  and nsent.get(r["pilot_id"], 0) > 0]
            if len(gm) >= 20:
                disc = [1.0 if "social" in (r.get("discipline_o")
                                            or "").lower() else 0.0
                        for r in gm]
                ns = [float(nsent[r["pilot_id"]]) for r in gm]
                Xr = np.column_stack([[r["power_r"] for r in gm],
                                      [r["surprising_o"] for r in gm],
                                      disc, ns])
                Xf = np.column_stack([Xr, [max(setpos[r["pilot_id"]])
                                           for r in gm]])
                yv = np.array([r["replicated_sig"] for r in gm])
                beta, lr, p = logistic_lr(Xr, Xf, yv)
                say(f"  logistic LR (covariate-adjusted): std beta = "
                    f"{beta:+.3f}, LR = {lr:.2f}, p = {p:.4f} (n={len(gm)})")

    say("\nDECISION RULE (stated in advance): covariate-adjusted clean "
        "estimate ~0 AND reliability adequate -> the null is a finding. "
        "Reliability low -> 'abstracts insufficient', full text is the "
        "upgrade path.")
    (RESULTS / "paper2_model_report.txt").write_text("\n".join(report),
                                                     encoding="utf-8")
    print(f"\n-> {RESULTS / 'paper2_model_report.txt'}")


if __name__ == "__main__":
    main()
