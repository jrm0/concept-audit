"""
13_build_family_corpus.py — Concept-FIRST corpus for a single construct family.

Design shift: instead of "all papers in 20 journals" (journal-first), retrieve
"all papers using any term in this concept family" (concept-first), across all
of OpenAlex. For a single-construct case study this is the correct population
and carries no circularity — we are not asking which concepts a field has, we
are asking how a known family is used.

Advantages over the journal-first corpus for this purpose:
  - reaches the PERIPHERY (specialty journals) where daughter lineages surface
    first — invisible to a 20-journal core
  - spans the full timeline, including ANCESTOR terms that predate the focal
    construct (self-control/self-regulation long precede ego depletion, 1998),
    so the construct's emergence can be dated as a branching event
  - free, targeted API queries — no bulk downloading, no licensed databases

Retrieval: OpenAlex `title_and_abstract.search` per term, unioned by DOI.
Records field/topic so cross-domain homonyms (e.g. tryptophan depletion,
resource depletion in ecology) can be filtered or studied deliberately.

Outputs:
  data/family/{family}_works.jsonl   doi, year, journal, title, abstract,
                                     matched_terms, field, cited_by_count
  data/family/{family}_report.csv    per-term and per-decade counts

Usage (from code/, venv active):
    python3 13_build_family_corpus.py families/ego_depletion_family.csv
"""

import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

CONTACT_EMAIL = "jrmorgan3@gmail.com"
OUT_DIR = Path("data/family")
YEAR_MIN, YEAR_MAX = 1970, 2025
PER_PAGE = 200

# STRATIFIED TEMPORAL SAMPLING (added after the first run hit two problems):
#  - ancestor terms are ~50x larger than the focal term (self-regulation 62k vs
#    ego depletion 1.1k); an exhaustive pull swamps the family and blows quota
#  - an exhaustive pull is also temporally unbalanced (50 works in 1990, 5000 in
#    2020), which confounds drift with sample size
# So: sample up to PERIOD_CAP works per term per period, using OpenAlex's
# `sample` + `seed` for reproducibility. Balanced timeline, small quota cost.
PERIOD_YEARS = 5
PERIOD_CAP = 200          # per term, per period (<=200 keeps it single-page)
SAMPLE_SEED = 42


def get(url, params, tries=90):
    """PATIENT backoff: OpenAlex's daily quota resets at midnight UTC, so this
    escalates to 20-minute waits and will sit through a multi-hour block rather
    than dying. Safe to launch in the evening with caffeinate and leave; it is
    also resumable per term-period, so nothing is lost either way."""
    for attempt in range(tries):
        r = requests.get(url, params=dict(params, mailto=CONTACT_EMAIL),
                         timeout=60)
        if r.status_code == 429:
            wait = min(60 * (attempt + 1), 1200)
            print(f"    rate-limited; waiting {wait}s "
                  f"(attempt {attempt + 1}/{tries} — daily quota resets "
                  f"midnight UTC)")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


def reconstruct(inv):
    if not inv:
        return ""
    pos = {}
    for w, ps in inv.items():
        for p in ps:
            pos[p] = w
    return " ".join(pos[i] for i in sorted(pos))


def load_family(path):
    """CSV: term, role (focal|synonym|ancestor|control), field (optional
    OpenAlex field id, e.g. 32 = Psychology, to exclude homonyms), note"""
    terms = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("term") or "").strip()
            if t:
                terms.append({"term": t,
                              "role": (row.get("role") or "").strip(),
                              "field": (row.get("field") or "").strip(),
                              "note": (row.get("note") or "").strip()})
    return terms


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 13_build_family_corpus.py <family_csv>")
    fam_path = Path(sys.argv[1])
    family = fam_path.stem.replace("_family", "")
    terms = load_family(fam_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{family}_works.jsonl"

    print(f"Family '{family}': {[t['term'] for t in terms]}\n")
    works = {}          # doi -> record
    per_term = Counter()

    # Resume support: completed (term, period) pairs are logged
    done_log = OUT_DIR / f"{family}_done.log"
    done = set()
    if done_log.exists():
        done = set(x.strip() for x in open(done_log) if x.strip())
        print(f"Resuming: {len(done)} term-periods already fetched\n")
    # Reload any works already saved so a resumed run keeps them
    if out_path.exists() and done:
        for line in open(out_path, encoding="utf-8"):
            try:
                rec = json.loads(line)
                works[rec["doi"]] = rec
            except json.JSONDecodeError:
                continue
        print(f"  reloaded {len(works)} works from previous run\n")

    periods = [(y, min(y + PERIOD_YEARS - 1, YEAR_MAX))
               for y in range(YEAR_MIN, YEAR_MAX + 1, PERIOD_YEARS)]

    for spec in terms:
        term = spec["term"]
        field_filter = (spec.get("field") or "").strip()
        print(f"=== '{term}' ({spec['role']})"
              + (f" [field: {field_filter}]" if field_filter else ""))
        got = 0
        for (y0, y1) in periods:
            key = f"{term}|{y0}"
            if key in done:
                continue
            flt = (f'title_and_abstract.search:"{term}",'
                   f"from_publication_date:{y0}-01-01,"
                   f"to_publication_date:{y1}-12-31,"
                   f"type:article,language:en")
            if field_filter:
                flt += f",topics.field.id:fields/{field_filter}"
            # First ask how many exist, then sample up to the cap
            head = get("https://api.openalex.org/works",
                       {"filter": flt, "per-page": 1})
            total = head.get("meta", {}).get("count", 0)
            time.sleep(0.4)
            if total == 0:
                with open(done_log, "a") as dl:
                    dl.write(key + "\n")
                continue
            take = min(PERIOD_CAP, total)
            data = get("https://api.openalex.org/works",
                       {"filter": flt, "per-page": take,
                        "sample": take, "seed": SAMPLE_SEED,
                        "select": "doi,publication_year,title,"
                                  "abstract_inverted_index,primary_location,"
                                  "cited_by_count,topics"})
            for w in data.get("results", []):
                doi = (w.get("doi") or "").replace("https://doi.org/", "").lower()
                if not doi:
                    continue
                rec = works.get(doi)
                if rec is None:
                    src = (w.get("primary_location") or {}).get("source") or {}
                    topics = w.get("topics") or []
                    field = ""
                    if topics:
                        field = ((topics[0].get("field") or {}).get("display_name")
                                 or "")
                    rec = {
                        "doi": doi,
                        "year": w.get("publication_year"),
                        "journal": src.get("display_name") or "",
                        "title": w.get("title") or "",
                        "abstract": reconstruct(w.get("abstract_inverted_index")),
                        "cited_by_count": w.get("cited_by_count", 0),
                        "field": field,
                        "matched_terms": [],
                        "period_total": {},
                    }
                    works[doi] = rec
                if term not in rec["matched_terms"]:
                    rec["matched_terms"].append(term)
                # record the population size this sample came from (for weighting)
                rec.setdefault("period_total", {})[f"{term}|{y0}"] = total
                got += 1
            with open(done_log, "a") as dl:
                dl.write(key + "\n")
            print(f"    {y0}-{y1}: {total} exist, sampled {take}")
            time.sleep(0.4)
        per_term[term] = got
        print(f"    total sampled: {got}; unique works so far: {len(works)}")

    with_abs = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in works.values():
            if rec["abstract"]:
                with_abs += 1
            f.write(json.dumps(rec) + "\n")

    # Report: per-term, per-decade, per-field
    decades, fields = Counter(), Counter()
    term_decade = defaultdict(Counter)
    for rec in works.values():
        y = rec["year"]
        d = f"{int(y)//10*10}s" if y else "unknown"
        decades[d] += 1
        fields[rec["field"] or "unknown"] += 1
        for t in rec["matched_terms"]:
            term_decade[t][d] += 1

    print(f"\nUnique works: {len(works)} ({with_abs} with abstracts)")
    print("\nBy decade:")
    for d in sorted(decades):
        print(f"  {d}: {decades[d]}")
    print("\nTop fields (homonym check — non-psychology = candidate homonyms):")
    for fl, c in fields.most_common(8):
        print(f"  {c:>6}  {fl}")
    print("\nPer-term by decade (temporal depth per lineage):")
    ds = sorted(decades)
    print(f"{'term':<26}" + "".join(f"{d:>8}" for d in ds))
    for spec in terms:
        t = spec["term"]
        print(f"{t:<26}" + "".join(f"{term_decade[t].get(d,0):>8}" for d in ds))

    with open(OUT_DIR / f"{family}_report.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "decade", "n_works"])
        for t, dd in term_decade.items():
            for d, c in sorted(dd.items()):
                w.writerow([t, d, c])

    print(f"\nWorks -> {out_path}")
    print(f"Report -> {OUT_DIR / f'{family}_report.csv'}")
    print("\nNext: check the field breakdown for homonyms (e.g. 'resource "
          "depletion' in ecology, 'tryptophan depletion' in neuroscience), "
          "then run the temporal lineage analysis.")


if __name__ == "__main__":
    main()
