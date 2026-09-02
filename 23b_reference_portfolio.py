"""
23b_reference_portfolio.py — Where do the applied papers actually point?

Extends 23's propagation result three ways, using the ALREADY-CACHED
reference lists (data/citations/openalex_refs.jsonl — no bulk re-fetch):

 1. FAIRNESS CHECK — broader crisis-engagement anchor set. Citing an RRR
    is not the only way to acknowledge the debate. Adds (resolved at
    runtime, printed for verification, like 23):
      hagger_2010_meta  supportive meta-analysis — a MILESTONE CONTROL:
                        if applied papers cite this but not the critiques,
                        the gap is correction-specific, not
                        milestone-in-general
      inzlicht_2012     mechanistic-revision paper (debate-adjacent)
      friese_2019       "Is ego depletion real?" review (debate-central)
      dang_2018         updated meta-analysis (debate-adjacent)
    Recomputes propagation with crisis_any = RRR16|RRR21|Carter14|
    Friese19|Dang18|Inzlicht12.

 2. HUB TABLE — the intermediaries. Among work/org ego-depletion papers,
    the most-referenced works (overall, and split citers vs non-citers of
    any crisis source), with IDs resolved to titles/years in a few batched
    calls. If non-citers share a small set of hub references, THAT is the
    insulation mechanism: the concept arrives via translators, not via the
    primary literature.

 3. WITHIN-VENUE SEMANTIC SLICE — 23's citers-vs-non-citers drift-axis
    difference, recomputed WITHIN core psych only and WITHIN work/org
    only. 23's pooled version partly restates the migration finding
    (venue confound); the within-venue version is the honest test.

Usage (from code/):
    python3 23b_reference_portfolio.py                  # show anchors
    python3 23b_reference_portfolio.py --confirm-anchors
Outputs: results/reference_portfolio_report.txt
         results/hub_references_workorg.csv
"""

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

API = "https://api.openalex.org/works"
MAILTO = "jrmorgan3@gmail.com"
CACHE = Path("data/citations/openalex_refs.jsonl")
SEM = Path("results/citation_semantics.csv")
RESULTS = Path("results")
SEED = 42

ANCHORS = [
    ("rrr_2016", "multilab preregistered replication of the ego-depletion "
     "effect", 2016),
    ("rrr_2021", "multisite preregistered paradigmatic test of the ego "
     "depletion effect", 2021),
    ("carter_2014", "publication bias and the limited strength model of "
     "self-control", 2014),
    ("hagger_2010_meta", "ego depletion and the strength model of "
     "self-control a meta-analysis", 2010),
    ("inzlicht_2012", "what is ego depletion toward a mechanistic revision "
     "of the resource model", 2012),
    # These two are resolved by DOI: online-first vs issue-date year
    # mismatches break title+year search. DOIs verified by web search
    # against the journal records (PSPR 23(2):107-131; Psych Research
    # 82(4):645-651).
    ("friese_2019", "doi:10.1177/1088868318762183", None),
    ("dang_2018", "doi:10.1007/s00426-017-0862-x", None),
]
CRISIS = {"rrr_2016", "rrr_2021", "carter_2014", "inzlicht_2012",
          "friese_2019", "dang_2018"}


def get(params, retries=6):
    import requests
    for i in range(retries):
        r = requests.get(API, params=dict(params, mailto=MAILTO), timeout=60)
        if r.status_code == 200:
            return r.json()
        ra = r.headers.get("Retry-After")
        wait = int(ra) if ra and ra.isdigit() else min(60, 3 * 2 ** i)
        print(f"    HTTP {r.status_code}; wait {wait}s")
        time.sleep(min(wait, 120))
    raise SystemExit("OpenAlex quota/persistent error — try later; "
                     "everything here is cheap and resumable.")


def resolve_anchors():
    print("Resolving anchors (VERIFY):")
    out = {}
    for label, q, year in ANCHORS:
        flt = (q if q.startswith("doi:")
               else f"title.search:{q},publication_year:{year}")
        js = get({"filter": flt,
                  "per-page": 2,
                  "select": "id,doi,title,publication_year,authorships,"
                            "cited_by_count"})
        res = js.get("results", [])
        if not res:
            print(f"  !! {label}: NOT FOUND")
            continue
        w = res[0]
        first = (w.get("authorships") or [{}])[0].get("author", {}) \
            .get("display_name", "?")
        print(f"  {label}: '{w['title'][:65]}' ({w['publication_year']}) "
              f"{first}; {w.get('doi','')} [{w['cited_by_count']}]")
        out[label] = w["id"]
    return out


def perm_test(a, b, n=20000, seed=SEED):
    if len(a) < 5 or len(b) < 5:
        return None
    rng = np.random.RandomState(seed)
    obs = np.mean(b) - np.mean(a)
    pool = np.array(list(a) + list(b))
    cnt = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[len(a):].mean() - pool[:len(a)].mean()) >= abs(obs):
            cnt += 1
    return obs, cnt / n


def main():
    confirm = "--confirm-anchors" in sys.argv
    anchors = resolve_anchors()
    if not confirm:
        print("\nRe-run with --confirm-anchors if correct.")
        return

    refs = {}
    for line in open(CACHE, encoding="utf-8"):
        try:
            j = json.loads(line)
            refs[j["doi"]] = j["refs"]
        except (json.JSONDecodeError, KeyError):
            pass
    sem = list(csv.DictReader(open(SEM, encoding="utf-8")))
    report = []

    def say(s=""):
        print(s)
        report.append(s)

    # flags per doc against the wider anchor set
    docs = []
    for r in sem:
        rf = refs.get(r["doi"])
        if rf is None:
            continue
        r = dict(r)
        r["cites_1998"] = r.get("cites_1998") == "1"   # CSV strings: "0" is truthy!
        flags = {lab: (aid in rf) for lab, aid in anchors.items()}
        docs.append({**r, **flags,
                     "crisis_any": any(flags.get(l) for l in CRISIS),
                     "refs": rf})

    say("=== 1. PROPAGATION, broader anchor set (2017+ papers) ===")
    say(f"{'venue':<20}{'n':>5}{'1998':>7}{'2010meta':>10}{'anyRRR':>8}"
        f"{'crisisANY':>11}")
    for cat in sorted({d["venue_cat"] for d in docs}):
        sel = [d for d in docs if d["venue_cat"] == cat]
        n = len(sel)

        def pct(k):
            return f"{100*sum(1 for d in sel if d[k])/max(n,1):.0f}%"
        rrr = sum(1 for d in sel if d["rrr_2016"] or d["rrr_2021"])
        cri = sum(1 for d in sel if d["crisis_any"])
        say(f"{cat:<20}{n:>5}"
            f"{pct('cites_1998') if 'cites_1998' in sel[0] else '-':>7}"
            f"{pct('hagger_2010_meta'):>10}"
            f"{100*rrr/max(n,1):>7.0f}%"
            f"{100*cri/max(n,1):>10.0f}%")
    say("\nREAD: '2010meta' is the supportive-milestone control. If a venue "
        "cites the 2010 meta but not the critiques, the gap is "
        "correction-specific.")

    # Conditional propagation: among papers that DEMONSTRABLY engage the
    # primary methodological literature (they cite the 2010 meta), how many
    # also cite any critique? This defuses the objection that applied papers
    # simply cite fewer methodology papers of every kind.
    say("\n--- Conditional: P(cites a crisis source | cites 2010 meta) ---")
    for cat in sorted({d["venue_cat"] for d in docs}):
        sel = [d for d in docs if d["venue_cat"] == cat
               and d["hagger_2010_meta"]]
        if len(sel) < 5:
            say(f"  {cat}: n={len(sel)} (too few meta-citers)")
            continue
        k = sum(1 for d in sel if d["crisis_any"])
        say(f"  {cat}: {k}/{len(sel)} ({100*k/len(sel):.0f}%) of "
            f"meta-citing papers also cite a critique")

    # 2. hub references among work/org papers
    say("\n=== 2. HUB REFERENCES in work/org ego-depletion papers ===")
    wo = [d for d in docs if d["venue_cat"] == "work/org"]
    eng = [d for d in wo if d["crisis_any"]]
    non = [d for d in wo if not d["crisis_any"]]
    say(f"work/org papers with refs: {len(wo)} "
        f"(crisis-engaged {len(eng)}, non-engaged {len(non)})")
    top_non = Counter(w for d in non for w in d["refs"]).most_common(25)
    ids = [w for w, _ in top_non]
    titles = {}
    for i in range(0, len(ids), 50):
        js = get({"filter": "openalex_id:" + "|".join(ids[i:i + 50]),
                  "per-page": 50,
                  "select": "id,title,publication_year"})
        for w in js.get("results", []):
            titles[w["id"]] = f"{w.get('title','')[:70]} " \
                              f"({w.get('publication_year','?')})"
    say("\nTop references of the NON-ENGAGED work/org papers "
        "(the import channel):")
    rows_out = []
    for wid, cnt in top_non:
        t = titles.get(wid, wid)
        say(f"  {cnt:>3}x  {t}")
        rows_out.append([cnt, t, wid])
    with open(RESULTS / "hub_references_workorg.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n_citing", "work", "openalex_id"])
        w.writerows(rows_out)

    # 3. within-venue semantic slice
    say("\n=== 3. DRIFT-AXIS: crisis-engaged vs non, WITHIN venue ===")
    for name, grp in [("core/other psych",
                       [d for d in docs
                        if d["venue_cat"] == "core/other psych"]),
                      ("work/org", wo)]:
        a = [float(d["drift_axis_pos"]) for d in grp
             if not d["crisis_any"] and d["drift_axis_pos"]]
        b = [float(d["drift_axis_pos"]) for d in grp
             if d["crisis_any"] and d["drift_axis_pos"]]
        t = perm_test(a, b)
        if t is None:
            say(f"  {name}: insufficient n ({len(b)} vs {len(a)})")
        else:
            obs, pv = t
            say(f"  {name}: engaged n={len(b)} non n={len(a)}  "
                f"diff={obs:+.3f}  perm p={pv:.4f}")
    say("\n(A difference that survives WITHIN venues is a real usage "
        "difference; one that vanishes was the migration confound.)")

    (RESULTS / "reference_portfolio_report.txt").write_text(
        "\n".join(report), encoding="utf-8")
    print(f"\n-> {RESULTS}/reference_portfolio_report.txt and "
          f"hub_references_workorg.csv")


if __name__ == "__main__":
    main()
