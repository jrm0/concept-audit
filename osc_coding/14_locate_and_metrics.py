"""
14_locate_and_metrics.py — Paper 2's dataset: locate every OSC original in
its construct's usage landscape and join the RP:P outcomes. (sklearn only.)

Per study (primary construct):
  ORIGINAL-PAPER POSITION (the clade-mismatch machinery at n=1):
    pos_pre_pct   percentile of the original paper's distance-to-centroid
                  among PRE-2010 documents (its own era): was the original
                  a TYPICAL usage of its time, or peripheral?
    pos_post_pct  same against POST-2010 (replication-era) documents: how
                  typical is the original's usage of the era that tested
                  it? HIGH percentile = the field's usage had moved away
                  from the original by the time it was replicated.
  FIELD DYNAMICS:
    drift         cosine(pre centroid, post centroid), with split-half
                  noise floors of both windows (drift_over_floor ratio)
    disp_pre/post mean distance to own-window centroid (usage diversity)
  SPARSITY STRATUM:
    n_pre/n_post  document counts; isolation = 1 if total < 30 (metrics
                  set to NA — the isolation FLAG is the predictor there)

DV join (RP:P, Open Science Collaboration 2015 data):
  key = Replicated.study.number.R (zfill3) -> our pilot_id, VERIFIED by
  fuzzy title match against our full_studies.csv (mismatches printed).
  DVs: replicated_sig (T.pval.USE.R < .05), o_within_ci (O.within.CI.R),
  meta_sig (Meta.analysis.significant), es_o (T.r.O), es_r (T.r.R).
  Moderators: power_r, n_o, n_r, discipline_o, surprising_o.

Output: results/paper2_dataset.csv + a PRELIMINARY group comparison
(replicated vs not, permutation tests) printed and saved. Preliminary =
descriptive; the registered analysis comes after we lock the model.

Usage (from code/osc_coding/):
    python3 14_locate_and_metrics.py
"""

import csv
import re
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

EMB = Path("data/construct_emb")
FINAL = Path("data/final_constructs.csv")
STUDIES = Path("data/full_studies.csv")
RPP = Path("data/rpp/rpp_data_raw.csv")
AUDIT = Path("data/construct_label_audit.csv")
RESULTS = Path("../results")
SEED = 42
MIN_DOCS = 30          # below this total -> isolation stratum


def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def split_half_floor(V, reps=40):
    if len(V) < 10:
        return None
    rng = np.random.RandomState(SEED)
    ds = []
    for _ in range(reps):
        p = rng.permutation(len(V))
        h = len(V) // 2
        ds.append(cosine(V[p[:h]].mean(0), V[p[h:2 * h]].mean(0)))
    return float(np.mean(ds))


def pct_of(dist, ref):
    return float((np.asarray(ref) < dist).mean() * 100)


def perm_test(a, b, n=20000, seed=SEED):
    a = [x for x in a if x == x]
    b = [x for x in b if x == x]
    if len(a) < 5 or len(b) < 5:
        return None
    rng = np.random.RandomState(seed)
    obs = np.mean(b) - np.mean(a)
    pool = np.array(a + b)
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[len(a):].mean() - pool[:len(a)].mean()) >= abs(obs):
            cnt += 1
    return obs, cnt / n, len(a), len(b)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main():
    finals = {r["pilot_id"].strip().zfill(3): r
              for r in csv.DictReader(open(FINAL, encoding="utf-8"))}
    our_titles = {r["study_id"].strip().zfill(3): r["title"]
                  for r in csv.DictReader(open(STUDIES, encoding="utf-8"))}
    audit = {(r["pilot_id"].strip().zfill(3), int(r["slot"])): r
             for r in csv.DictReader(open(AUDIT, encoding="utf-8"))}

    orig = np.load(EMB / "osc_originals.npz", allow_pickle=True)
    ovec = {str(p): v for p, v in zip(orig["pilot_ids"], orig["vectors"])}

    # RP:P join — BY NORMALIZED TITLE. Neither Local.ID (non-numeric in the
    # published file) nor Replicated.study.number.R (populated for only 8
    # rows) matches our numbering; titles match 99/99 and are
    # self-verifying, so the title IS the key.
    by_title = {}
    with open(RPP, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = norm_title(r.get("Study.Title.O"))[:60]
            if t:
                by_title[t] = r
    rpp = {}
    for pid, title in our_titles.items():
        r = by_title.get(norm_title(title)[:60])
        if r is not None:
            rpp[pid] = r
    print(f"RP:P title-join: {len(rpp)}/{len(our_titles)} studies matched")
    for pid, title in sorted(our_titles.items()):
        if pid not in rpp:
            print(f"  UNMATCHED {pid}: {title[:60]}")

    rows_out = []
    for pid, fr in sorted(finals.items()):
        c = (fr.get("construct_1") or "").strip().lower()
        if not c:
            continue
        rec = {"pilot_id": pid, "construct": c}
        # field shards
        Vs = {}
        for w in ("pre", "post"):
            f = EMB / f"{slug(c)}__{w}.npz"
            if f.exists():
                Vs[w] = np.load(f, allow_pickle=True)["vectors"] \
                    .astype(np.float64)
        n_pre = len(Vs.get("pre", []))
        n_post = len(Vs.get("post", []))
        rec.update(n_pre=n_pre, n_post=n_post,
                   isolation=int(n_pre + n_post < MIN_DOCS))
        a = audit.get((pid, 1))
        rec["label_class"] = a["class"] if a else ""
        rec["field_docs_pool"] = a["field_docs"] if a else ""

        v0 = ovec.get(pid)
        rec["has_original_vec"] = int(v0 is not None)
        nan = float("nan")
        rec.update(pos_pre_pct=nan, pos_post_pct=nan, drift=nan,
                   drift_over_floor=nan, disp_pre=nan, disp_post=nan)
        if not rec["isolation"]:
            if "pre" in Vs and n_pre >= 10:
                cpre = Vs["pre"].mean(0)
                ref = [cosine(v, cpre) for v in Vs["pre"]]
                rec["disp_pre"] = float(np.mean(ref))
                if v0 is not None:
                    rec["pos_pre_pct"] = pct_of(cosine(v0, cpre), ref)
            if "post" in Vs and n_post >= 10:
                cpost = Vs["post"].mean(0)
                ref = [cosine(v, cpost) for v in Vs["post"]]
                rec["disp_post"] = float(np.mean(ref))
                if v0 is not None:
                    rec["pos_post_pct"] = pct_of(cosine(v0, cpost), ref)
            if "pre" in Vs and "post" in Vs and n_pre >= 10 and n_post >= 10:
                d = float(cosine(Vs["pre"].mean(0), Vs["post"].mean(0)))
                fl = [x for x in (split_half_floor(Vs["pre"]),
                                  split_half_floor(Vs["post"]))
                      if x is not None]
                rec["drift"] = d
                if fl:
                    rec["drift_over_floor"] = d / max(np.mean(fl), 1e-9)

        # RP:P outcomes (join already title-verified by construction)
        rp = rpp.get(pid)
        if rp:
            pval_r = fnum(rp.get("T.pval.USE.R"))
            rec.update(
                pval_r=pval_r,
                replicated_sig=(int(pval_r < .05) if pval_r == pval_r
                                else ""),
                o_within_ci=rp.get("O.within.CI.R", ""),
                meta_sig=rp.get("Meta.analysis.significant", ""),
                es_o=fnum(rp.get("T.r.O")), es_r=fnum(rp.get("T.r.R")),
                power_r=fnum(rp.get("Power.R")), n_o=fnum(rp.get("N.O")),
                n_r=fnum(rp.get("N.R")),
                discipline_o=rp.get("Discipline.O", ""),
                surprising_o=fnum(rp.get("Surprising.result.O")))
        rows_out.append(rec)

    fields = list(rows_out[0].keys())
    for r in rows_out:
        for k in r:
            if k not in fields:
                fields.append(k)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "paper2_dataset.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows_out)
    print(f"\n{len(rows_out)} studies -> {RESULTS / 'paper2_dataset.csv'}")

    # ---- preliminary comparisons (descriptive!)
    have = [r for r in rows_out if r.get("replicated_sig") in (0, 1)]
    rep = [r for r in have if r["replicated_sig"] == 1]
    non = [r for r in have if r["replicated_sig"] == 0]
    print(f"\nPRELIMINARY (descriptive) — replicated {len(rep)} vs "
          f"failed {len(non)}:")
    for m in ("pos_post_pct", "pos_pre_pct", "drift_over_floor",
              "disp_post", "n_pre", "isolation"):
        t = perm_test([r[m] for r in non if isinstance(r[m], (int, float))],
                      [r[m] for r in rep if isinstance(r[m], (int, float))])
        if t is None:
            print(f"  {m}: insufficient n")
        else:
            obs, pv, na, nb = t
            print(f"  {m}: replicated mean - failed mean = {obs:+.3f} "
                  f"(n={nb} vs {na}, perm p={pv:.4f})")
    print("\nHypothesis reminder: clade-mismatch predicts FAILED replications"
          " have HIGHER pos_post_pct (original usage atypical of the "
          "replication era). Treat today's numbers as descriptive — model "
          "specification gets locked with Jonathan before any claim.")


if __name__ == "__main__":
    main()
