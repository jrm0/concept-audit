"""
23_citation_semantics.py — Wed the semantic landscape to the citation graph.

THE QUESTION. The migration finding (12%->25% of ego-depletion papers in
work/organizational venues after the 2016 RRR) shows the concept moved; it
does not show whether the CORRECTION moved with it. This script measures
correction propagation directly, then annotates citation edges semantically:

  1. PROPAGATION. For every post-2016 ego-depletion document with a DOI,
     fetch its reference list from OpenAlex and ask: does it cite the 2016
     RRR (Hagger et al.) or the 2021 RRR (Vohs et al.)? Does it cite the
     1998 coining paper (Baumeister et al.)? Report citation rates by venue
     category (core psych / work-org / health / education / other) and
     period. The hypothesis on the table: applied venues cite the coinage
     but not the correction.
  2. SEMANTIC ANNOTATION. Each citing document already has a position in
     our masked-embedding usage space. Compare RRR-citers to non-citers
     WITHIN the same period on the drift axis (earliest->latest usage,
     control-centered): do papers that acknowledge the correction use the
     term differently from papers that don't? Permutation test.

ANCHOR PAPERS ARE RESOLVED AT RUNTIME, NOT HARDCODED. The script queries
OpenAlex by title + year, prints exactly what it resolved (title, year,
first author, DOI, citation count), and requires --confirm-anchors on the
real run. Reason: a wrong hardcoded DOI would silently corrupt every count
(citation-integrity rule applied to code).

Network: OpenAlex REST API, polite pool (mailto), batched DOI lookups,
checkpointed to data/citations/openalex_refs.jsonl — resumable, safe to
re-run. Runs LOCALLY (needs internet).

Usage (from code/):
    python3 23_citation_semantics.py                  # resolve + show anchors
    python3 23_citation_semantics.py --confirm-anchors  # full run
Outputs:
    data/citations/openalex_refs.jsonl   raw reference lists (cache)
    results/citation_semantics.csv       per-document flags + semantics
    results/citation_semantics_report.txt
"""

import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

API = "https://api.openalex.org/works"
MAILTO = "jrmorgan3@gmail.com"
DOCS = Path("data/family/lineage_docs.jsonl")
EMB = Path("data/lineage_emb")
CACHE = Path("data/citations/openalex_refs.jsonl")
RESULTS = Path("results")
FOCAL = "ego depletion"
CONTROLS = {"participants", "results", "significant", "measure"}
BATCH = 25            # DOIs per OpenAlex filter call (max 50; be polite)
SEED = 42

# Anchors are (label, title-search string, publication year). Resolution is
# printed for human verification — titles below are used as SEARCH QUERIES,
# not asserted citations.
ANCHORS = [
    ("coinage_1998", "Ego depletion is the active self a limited resource",
     1998),
    ("rrr_2016", "multilab preregistered replication of the ego-depletion "
     "effect", 2016),
    ("rrr_2021", "multisite preregistered paradigmatic test of the ego "
     "depletion effect", 2021),
    ("carter_2014", "publication bias and the limited strength model of "
     "self-control", 2014),
]

CATS = [
    # NOTE: includes applied-psych journal names (JAP etc.), which the first
    # version missed. Sensitivity check on the migration result:
    # original 12.0%->24.8% (+12.7pp, p=0.0001); this version
    # 15.5%->29.6% (+14.1pp, p=0.0004); corpus baseline flat under both.
    ("work/org", r"organi[sz]ation|management|business|occupational"
     r"|vocational|work|employ|leader|market|consumer|account|tourism"
     r"|hospitality|industrial|labor|labour|applied psycholog|personnel"
     r"|human resource|human performance|ergonomic"),
    ("health/med", r"health|medic|nurs|clinical|addict|obes|diet|patient"
     r"|psychosomat|prevent"),
    ("education", r"educat|school|teach|learn|student|academic|curricul"),
    ("sport", r"sport|exercise|athlet|kinesio|physical activity"),
    ("crime/social", r"crimin|delinq|justice|offend|social work|deviant"
     r"|violence"),
]


def venue_cat(j):
    j = (j or "").lower()
    for name, pat in CATS:
        if re.search(pat, j):
            return name
    return "core/other psych"


def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def get(url, params, retries=6):
    import requests
    params = dict(params, mailto=MAILTO)
    for i in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        ra = r.headers.get("Retry-After")
        wait = int(ra) if ra and ra.isdigit() else min(60, 2 ** i * 3)
        print(f"    HTTP {r.status_code}; retrying in {wait}s ...")
        time.sleep(min(wait, 120))
    if r.status_code == 429:
        raise SystemExit(
            "OpenAlex daily quota exhausted (resets midnight UTC). "
            "Progress is cached in data/citations/ — re-run the same "
            "command later and it resumes.")
    raise RuntimeError(f"OpenAlex gave up: {url}")


def resolve_anchors():
    print("Resolving anchor papers on OpenAlex (VERIFY these lines):")
    out = {}
    for label, q, year in ANCHORS:
        js = get(API, {"filter": f"title.search:{q},publication_year:{year}",
                       "per-page": 3,
                       "select": "id,doi,title,publication_year,"
                                 "authorships,cited_by_count"})
        results = js.get("results", [])
        if not results:
            print(f"  !! {label}: NOT FOUND — fix the search string")
            continue
        w = results[0]
        first = (w.get("authorships") or [{}])[0].get("author", {}) \
            .get("display_name", "?")
        print(f"  {label}: '{w['title'][:70]}' ({w['publication_year']}) "
              f"{first}; {w.get('doi','')} [{w['cited_by_count']} citations]")
        out[label] = w["id"]
    return out


def load_focal_docs():
    docs = []
    for line in open(DOCS, encoding="utf-8"):
        d = json.loads(line)
        if FOCAL in d["terms"] and d.get("doi") and d.get("year"):
            docs.append(d)
    return docs


def fetch_references(dois):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if CACHE.exists():
        for line in open(CACHE, encoding="utf-8"):
            try:
                have.add(json.loads(line)["doi"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [d for d in dois if d not in have]
    print(f"{len(dois)} DOIs; {len(have)} cached; fetching {len(todo)}")
    with open(CACHE, "a", encoding="utf-8") as out:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            flt = "doi:" + "|".join(chunk)
            js = get(API, {"filter": flt, "per-page": BATCH,
                           "select": "id,doi,referenced_works"})
            found = set()
            for w in js.get("results", []):
                doi = (w.get("doi") or "").replace("https://doi.org/", "")
                found.add(doi.lower())
                out.write(json.dumps(
                    {"doi": doi.lower(),
                     "refs": w.get("referenced_works", [])}) + "\n")
            for d in chunk:                       # negative cache
                if d not in found:
                    out.write(json.dumps({"doi": d, "refs": None}) + "\n")
            out.flush()
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)}")
            time.sleep(0.4)
    refs = {}
    for line in open(CACHE, encoding="utf-8"):
        try:
            j = json.loads(line)
            refs[j["doi"]] = j["refs"]
        except (json.JSONDecodeError, KeyError):
            pass
    return refs


def drift_axis_positions():
    """doi -> (period, position on the control-centered drift axis)."""
    ref = {}
    for c in CONTROLS:
        for f in EMB.glob(f"{slug(c)}__*.npz"):
            p = f.stem.split("__")[1]
            z = np.load(f, allow_pickle=True)
            s, n = ref.get(p, (0, 0))
            ref[p] = (s + z["vectors"].astype(np.float64).sum(0),
                      n + len(z["vectors"]))
    ref = {p: s / n for p, (s, n) in ref.items() if n}
    shards = {}
    for f in sorted(EMB.glob(f"{slug(FOCAL)}__*.npz")):
        p = f.stem.split("__")[1]
        if p in ref:
            z = np.load(f, allow_pickle=True)
            shards[p] = (z["vectors"].astype(np.float64) - ref[p], z["keys"])
    if len(shards) < 2:
        return {}
    ps = sorted(shards)
    axis = shards[ps[-1]][0].mean(0) - shards[ps[0]][0].mean(0)
    axis /= np.linalg.norm(axis) + 1e-12
    pos = {}
    for p, (V, keys) in shards.items():
        proj = V @ axis
        for k, x in zip(keys, proj):
            pos[str(k).lower()] = (p, float(x))
    return pos


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
        print("\nIf the four lines above are the right papers, re-run with "
              "--confirm-anchors to do the full fetch + analysis.")
        return
    if len(anchors) < len(ANCHORS):
        raise SystemExit("Not all anchors resolved — fix before running.")

    docs = load_focal_docs()
    post = [d for d in docs if d["year"] >= 2017]
    print(f"\n{len(docs)} ego-depletion docs with DOI; "
          f"{len(post)} from 2017+ (the propagation window)")
    refs = fetch_references(sorted({d["doi"] for d in post}))
    pos = drift_axis_positions()

    report, rows = [], []

    def say(s=""):
        print(s)
        report.append(s)

    stats = defaultdict(lambda: Counter())
    for d in post:
        r = refs.get(d["doi"])
        if r is None:
            continue                    # not indexed / no refs available
        flags = {lab: (aid in r) for lab, aid in anchors.items()}
        cat = venue_cat(d["journal"])
        key = (cat, d["period"])
        stats[key]["n"] += 1
        for lab, v in flags.items():
            stats[key][lab] += int(v)
        pp = pos.get(d["doi"], (None, None))
        rows.append([d["doi"], d["year"], d["period"], cat, d["journal"][:60],
                     len(r), *[int(flags[l]) for l, _a, _y in
                               [(x[0], 0, 0) for x in ANCHORS]],
                     pp[1] if pp[1] is not None else ""])

    say("=== CORRECTION PROPAGATION (2017+ ego-depletion papers) ===")
    say(f"{'venue':<18}{'period':<11}{'n':>4}{'cites 1998':>12}"
        f"{'cites RRR16':>13}{'cites RRR21':>13}{'either RRR':>12}")
    for (cat, p), c in sorted(stats.items()):
        n = c["n"]
        either = "-"        # per-doc 'either' needs the row level:
        say(f"{cat:<18}{p:<11}{n:>4}{c['coinage_1998']:>9} "
            f"({100*c['coinage_1998']//max(n,1):>2}%)"
            f"{c['rrr_2016']:>9} ({100*c['rrr_2016']//max(n,1):>2}%)"
            f"{c['rrr_2021']:>9} ({100*c['rrr_2021']//max(n,1):>2}%)")

    # aggregate: applied vs core, either RRR
    def either_rrr(row):
        # columns: ... flags start at index 6 in ANCHORS order
        return row[7] or row[8]
    applied = [r for r in rows if r[3] != "core/other psych"]
    core = [r for r in rows if r[3] == "core/other psych"]
    say("")
    for name, grp in [("core/other psych", core), ("applied venues", applied)]:
        if grp:
            k = sum(either_rrr(r) for r in grp)
            say(f"{name}: {k}/{len(grp)} ({100*k/len(grp):.0f}%) cite an RRR")

    # semantic annotation: RRR-citers vs non-citers on the drift axis,
    # within-period to control for time
    say("\n=== SEMANTIC POSITION: RRR-citers vs non-citers (drift axis) ===")
    for period in sorted({r[2] for r in rows}):
        cit = [r[-1] for r in rows if r[2] == period and either_rrr(r)
               and r[-1] != ""]
        non = [r[-1] for r in rows if r[2] == period and not either_rrr(r)
               and r[-1] != ""]
        t = perm_test(non, cit)
        if t is None:
            say(f"  {period}: insufficient n (citers={len(cit)}, "
                f"non={len(non)})")
        else:
            obs, pv = t
            say(f"  {period}: citers n={len(cit)} non n={len(non)}  "
                f"mean diff (citer - non) = {obs:+.3f}  perm p = {pv:.4f}")
    say("\nReading: a positive diff = RRR-citers sit FURTHER along the "
        "drift axis (more 'recent-style' usage); near-zero = acknowledging "
        "the correction does not distinguish how the term is used.")

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "citation_semantics.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doi", "year", "period", "venue_cat", "journal", "n_refs",
                    "cites_1998", "cites_rrr2016", "cites_rrr2021",
                    "cites_carter2014", "drift_axis_pos"])
        w.writerows(rows)
    (RESULTS / "citation_semantics_report.txt").write_text(
        "\n".join(report), encoding="utf-8")
    print(f"\n-> {RESULTS / 'citation_semantics.csv'} and _report.txt")


if __name__ == "__main__":
    main()
