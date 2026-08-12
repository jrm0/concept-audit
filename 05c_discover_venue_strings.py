"""
05c_discover_venue_strings.py — For LOW-ratio journals, discover the venue
strings Semantic Scholar actually uses, by decade.

Method: for each flagged journal, pull a sample of papers per decade from
OpenAlex (highly-cited first — most likely to be in S2), then look them up in
S2 by DOI (batch endpoint) and tally the venue strings S2 returns. Also counts
papers S2 doesn't have at all — that part of the coverage gap can't be fixed
by string variants and must be filled by Routes 2-4.

Output: printed tally + data/s2orc/venue_strings_discovered.csv.
Action: add every discovered variant (not already present) to s2_names in
venues_config.py, then re-run 05 phase_a and 05b.

Usage (from code/, venv active; S2_API_KEY set):
    python3 05c_discover_venue_strings.py                       # all LOW from venue_audit.csv
    python3 05c_discover_venue_strings.py "Memory & Cognition"  # specific journal(s)
"""

import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path

import requests

CONTACT_EMAIL = "jrmorgan3@gmail.com"
S2_API_KEY = os.environ.get("S2_API_KEY", "")
S2_HEADERS = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}

SOURCES_URL = "https://api.openalex.org/sources"
WORKS_URL = "https://api.openalex.org/works"
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

AUDIT_PATH = Path("data/s2orc/venue_audit.csv")
OUT_PATH = Path("data/s2orc/venue_strings_discovered.csv")
DECADES = [(1980, 1989), (1990, 1999), (2000, 2009), (2010, 2019), (2020, 2025)]
SAMPLE_PER_DECADE = 10


def oa_get(url, params):
    params = dict(params, mailto=CONTACT_EMAIL)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def s2_batch(dois):
    """Look up papers in S2 by DOI; returns list aligned with input (None = not in S2)."""
    r = requests.post(
        S2_BATCH_URL,
        params={"fields": "venue,year,externalIds"},
        json={"ids": [f"DOI:{d}" for d in dois]},
        headers=S2_HEADERS, timeout=60)
    if r.status_code == 429:
        time.sleep(5)
        return s2_batch(dois)
    r.raise_for_status()
    return r.json()


def journals_to_check():
    if len(sys.argv) > 1:
        return sys.argv[1:]
    if not AUDIT_PATH.exists():
        sys.exit("Run 05b_venue_audit.py first, or pass journal names as arguments.")
    with open(AUDIT_PATH, encoding="utf-8") as f:
        return [row["journal"] for row in csv.DictReader(f)
                if row.get("flag", "").strip() == "LOW"]


def main():
    targets = journals_to_check()
    print(f"Checking {len(targets)} journal(s): {targets}\n")
    rows = []

    for journal in targets:
        print(f"=== {journal} ===")
        src = oa_get(SOURCES_URL, {"search": journal, "per-page": 1})["results"]
        if not src:
            print("  OpenAlex source not found — skipping")
            continue
        src_id = src[0]["id"].rsplit("/", 1)[-1]
        time.sleep(0.3)

        venue_tally = Counter()   # (decade, venue_string) -> n
        missing = Counter()       # decade -> n not in S2
        sampled = Counter()       # decade -> n sampled with DOI
        for (y0, y1) in DECADES:
            flt = (f"primary_location.source.id:{src_id},"
                   f"from_publication_date:{y0}-01-01,"
                   f"to_publication_date:{y1}-12-31,type:article")
            works = oa_get(WORKS_URL, {"filter": flt, "per-page": SAMPLE_PER_DECADE,
                                       "sort": "cited_by_count:desc"}).get("results", [])
            dois = [w["doi"].replace("https://doi.org/", "")
                    for w in works if w.get("doi")]
            time.sleep(0.3)
            if not dois:
                continue
            sampled[f"{y0}s"] = len(dois)
            results = s2_batch(dois)
            time.sleep(1.0)
            for res in results:
                if res is None:
                    missing[f"{y0}s"] += 1
                else:
                    venue_tally[(f"{y0}s", res.get("venue") or "(empty)")] += 1

        print("  S2 venue strings found (by decade):")
        seen_strings = set()
        for (decade, vs), n in sorted(venue_tally.items()):
            print(f"    {decade}: '{vs}' x{n}")
            seen_strings.add(vs)
            rows.append((journal, decade, vs, n, ""))
        for decade, n in sorted(missing.items()):
            tot = sampled[decade]
            print(f"    {decade}: NOT IN S2: {n}/{tot}")
            rows.append((journal, decade, "(NOT IN S2)", n, f"of {tot} sampled"))
        print(f"  -> distinct strings: {sorted(seen_strings)}\n")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["journal", "decade", "s2_venue_string", "n", "note"])
        w.writerows(rows)
    print(f"Saved -> {OUT_PATH}")
    print("Next: add missing variants to s2_names in venues_config.py, "
          "re-run 05 phase_a, then 05b to confirm ratios improved. "
          "'NOT IN S2' gaps are unfixable via strings — Routes 2-4 fill those.")


if __name__ == "__main__":
    main()
