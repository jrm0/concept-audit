"""
07_pull_abstracts_openalex.py — The abstracts backbone (all ~86k roster papers).

Why: Phase B showed OA full text is 8% overall and ~0% pre-2000. Abstracts are
the only text stratum with uniform coverage across 1980-2025, so they become
the corpus backbone; full text (S2ORC/PMC) is the methods-section enhancement
where available. See coverage_threeway.csv.

Walks the same OpenAlex per-journal rosters as 05 phase_a, this time pulling
abstract_inverted_index, and writes one record per paper:
  data/abstracts/abstracts_stratum1.jsonl
    {"doi", "journal", "year", "title", "abstract"}
  data/abstracts/abstract_coverage.csv  (per journal x decade: roster vs with-abstract)

Resumable per journal (finished journals logged in journals_done.log).

Usage (from code/, venv active):
    python3 07_pull_abstracts_openalex.py
"""

import csv
import json
import time
from collections import Counter
from pathlib import Path

import requests

from venues_config import STRATUM1_JOURNALS, YEAR_START, YEAR_END

CONTACT_EMAIL = "jrmorgan3@gmail.com"
OUT_DIR = Path("data/abstracts")
OUT_PATH = OUT_DIR / "abstracts_stratum1.jsonl"
DONE_LOG = OUT_DIR / "journals_done.log"
COV_PATH = OUT_DIR / "abstract_coverage.csv"


def get_with_backoff(url, params, tries=60):
    """PATIENT backoff: escalates to 20-minute waits so the script can sit
    through a daily-quota block (resets midnight UTC) and resume on its own.
    Leave it running with caffeinate; it will get there."""
    for attempt in range(tries):
        r = requests.get(url, params=dict(params, mailto=CONTACT_EMAIL), timeout=60)
        if r.status_code == 429:
            wait = min(30 * (attempt + 1), 1200)
            print(f"    rate-limited; waiting {wait}s "
                  f"(attempt {attempt + 1}/{tries} — quota blocks clear on their own)")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()


def reconstruct(inv):
    positions = {}
    for word, poss in inv.items():
        for p in poss:
            positions[p] = word
    return " ".join(positions[i] for i in sorted(positions))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set(DONE_LOG.read_text().splitlines()) if DONE_LOG.exists() else set()
    cov = []  # (journal, decade, roster_n, with_abstract_n)

    for canonical in STRATUM1_JOURNALS:
        if canonical in done:
            print(f"=== {canonical} — already done, skipping")
            continue
        print(f"\n=== {canonical} ===")
        src = get_with_backoff("https://api.openalex.org/sources",
                               {"search": canonical, "per-page": 1}).json()["results"]
        if not src:
            print("  !! source not found — skipping")
            continue
        src_id = src[0]["id"].rsplit("/", 1)[-1]
        print(f"  OpenAlex: {src[0].get('display_name')}")
        time.sleep(0.3)

        totals, withab = Counter(), Counter()
        cursor = "*"
        flt = (f"primary_location.source.id:{src_id},"
               f"from_publication_date:{YEAR_START}-01-01,"
               f"to_publication_date:{YEAR_END}-12-31,type:article")
        n = 0
        with open(OUT_PATH, "a", encoding="utf-8") as out:
            while cursor:
                data = get_with_backoff(
                    "https://api.openalex.org/works",
                    {"filter": flt, "per-page": 200, "cursor": cursor,
                     "select": "doi,publication_year,title,abstract_inverted_index"}
                ).json()
                for w in data.get("results", []):
                    year = w.get("publication_year")
                    try:
                        dec = f"{int(year)//10*10}s"
                    except (TypeError, ValueError):
                        dec = "unknown"
                    totals[dec] += 1
                    inv = w.get("abstract_inverted_index")
                    if not inv:
                        continue
                    withab[dec] += 1
                    n += 1
                    out.write(json.dumps({
                        "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                        "journal": canonical,
                        "year": year,
                        "title": w.get("title") or "",
                        "abstract": reconstruct(inv),
                    }) + "\n")
                cursor = data.get("meta", {}).get("next_cursor")
                time.sleep(0.3)
        tot = sum(totals.values())
        print(f"  abstracts: {n}/{tot} ({n/tot:.0%})" if tot else "  no papers?")
        for dec in sorted(totals):
            cov.append((canonical, dec, totals[dec], withab[dec]))
        with open(DONE_LOG, "a", encoding="utf-8") as dl:
            dl.write(canonical + "\n")

    # Append coverage rows (header once)
    new_file = not COV_PATH.exists()
    with open(COV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["journal", "decade", "roster_n", "with_abstract_n"])
        w.writerows(cov)
    print(f"\nDone. Abstracts -> {OUT_PATH}; coverage by decade -> {COV_PATH}")
    print("Check the per-decade abstract coverage — if pre-1990 abstracts are "
          "also thin, that's a PsycInfo/library item, and analyses must "
          "condition on it either way.")


if __name__ == "__main__":
    main()
