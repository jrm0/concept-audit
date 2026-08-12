"""
05b_venue_audit.py — Audit Phase A venue coverage against OpenAlex.

Why: Semantic Scholar's `venue` filter matches name strings, which can silently
drop a journal entirely (Memory & Cognition: 0) or shave off eras where the
venue string differed. OpenAlex resolves journals as entities (with ISSNs), so
its per-journal counts are a solid external benchmark.

For each Stratum 1 journal:
  1. Resolve the journal on OpenAlex (sources search) — prints the matched
     display_name and ISSN so YOU can verify it's the right journal
  2. Count OpenAlex works in the 1980-2025 window
  3. Compare to the Phase A count (from data/s2orc/target_papers_meta.csv)

Output: data/s2orc/venue_audit.csv + a summary table. Journals where Phase A
found < 70% of the OpenAlex count get a LOW flag — fix s2_names variants in
venues_config.py and re-run phase_a before starting phase_b.

Usage (from code/, venv active):
    python3 05b_venue_audit.py
"""

import csv
import time
from collections import Counter
from pathlib import Path

import requests

from venues_config import STRATUM1_JOURNALS, YEAR_START, YEAR_END

CONTACT_EMAIL = "jrmorgan3@gmail.com"
SOURCES_URL = "https://api.openalex.org/sources"
WORKS_URL = "https://api.openalex.org/works"
META_PATH = Path("data/s2orc/target_papers_meta.csv")
OUT_PATH = Path("data/s2orc/venue_audit.csv")
LOW_THRESHOLD = 0.70


def get(url, params):
    params = dict(params, mailto=CONTACT_EMAIL)
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def phase_a_counts():
    counts = Counter()
    if not META_PATH.exists():
        print(f"WARNING: {META_PATH} not found — run phase_a first. "
              f"Proceeding with OpenAlex counts only.")
        return counts
    # Plain tab-split (not csv module): titles can contain quote characters
    # that confuse quote-aware parsing. Field 2 is the journal.
    with open(META_PATH, encoding="utf-8") as f:
        next(f, None)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                counts[parts[1]] += 1
    return counts


def main():
    pa = phase_a_counts()
    rows = []
    print(f"{'journal':<58} {'phaseA':>7} {'openalex':>9} {'ratio':>6}  flag")
    for canonical in STRATUM1_JOURNALS:
        # 1. Resolve journal entity on OpenAlex
        data = get(SOURCES_URL, {"search": canonical, "per-page": 3})
        results = data.get("results", [])
        if not results:
            print(f"{canonical:<58} {'—':>7} {'NOT FOUND':>9}")
            rows.append((canonical, pa.get(canonical, 0), "", "", "NOT FOUND", ""))
            continue
        src = results[0]
        src_id = src["id"].rsplit("/", 1)[-1]
        issn = ",".join(src.get("issn") or []) or (src.get("issn_l") or "")
        time.sleep(0.3)

        # 2. Count works in window
        flt = (f"primary_location.source.id:{src_id},"
               f"from_publication_date:{YEAR_START}-01-01,"
               f"to_publication_date:{YEAR_END}-12-31,"
               f"type:article")
        wdata = get(WORKS_URL, {"filter": flt, "per-page": 1})
        oa_count = wdata.get("meta", {}).get("count", 0)
        time.sleep(0.3)

        # 3. Compare
        pa_count = pa.get(canonical, 0)
        ratio = (pa_count / oa_count) if oa_count else 0.0
        flag = "LOW" if ratio < LOW_THRESHOLD else ""
        print(f"{canonical:<58} {pa_count:>7} {oa_count:>9} {ratio:>6.2f}  {flag}")
        print(f"    matched: {src.get('display_name')} (ISSN {issn}) — verify!")
        rows.append((canonical, pa_count, oa_count, f"{ratio:.2f}",
                     flag, f"{src.get('display_name')} | {issn}"))

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["journal", "phase_a_count", "openalex_count", "ratio",
                    "flag", "openalex_match_and_issn"])
        w.writerows(rows)
    print(f"\nSaved -> {OUT_PATH}")
    print("For LOW journals: find the S2 venue string via a known-paper DOI "
          "lookup, add variants to venues_config.py s2_names, re-run phase_a.")
    print("Note: ratios can legitimately sit below 1.0 (OpenAlex counts "
          "editorials/corrections as 'article' sometimes) — worry at < 0.70.")


if __name__ == "__main__":
    main()
