"""
08_psych_venue_checklist.py — External checklist of psychology venues, for
auditing the TDM Studio dataset selection (Stratum 2).

Pulls the top psychology journals from OpenAlex (entity-resolved, so names
don't matter) ranked by works count. Output is a checklist to diff against
the TDM Studio title list: journals on this list but NOT in the TDM dataset
are either (a) findable under a different name in TDM's title search, or
(b) genuinely unlicensed — record which, and the union becomes the documented
Stratum 2 definition with known exclusions.

NOTE: OpenAlex's field/topic filter syntax has changed before. The script
tries several filter variants and tells you which worked. Run AFTER the
OpenAlex daily quota resets (same quota as 07).

Usage (from code/, venv active):
    python3 08_psych_venue_checklist.py
Output:
    data/psych_venues_checklist.csv  (name, ISSN, works_count, is_oa, publisher)
"""

import csv
import time
from pathlib import Path

import requests

CONTACT_EMAIL = "jrmorgan3@gmail.com"
OUT_PATH = Path("data/psych_venues_checklist.csv")
TOP_N = 400

# Filter variants to try, most-preferred first. Each selects journal-type
# sources whose dominant topics sit in the psychology field.
FILTER_VARIANTS = [
    "type:journal,topics.field.id:fields/32",        # 32 = Psychology field id
    "type:journal,x_concepts.id:C15744967",          # legacy concept: Psychology
]


def get(params):
    r = requests.get("https://api.openalex.org/sources",
                     params=dict(params, mailto=CONTACT_EMAIL), timeout=30)
    r.raise_for_status()
    return r.json()


def main():
    rows = []
    used = None
    for flt in FILTER_VARIANTS:
        try:
            probe = get({"filter": flt, "per-page": 1})
            if probe.get("meta", {}).get("count", 0) > 100:
                used = flt
                print(f"Filter works: {flt} "
                      f"({probe['meta']['count']} sources)")
                break
            print(f"Filter returned too few results, trying next: {flt}")
        except requests.RequestException as e:
            print(f"Filter failed ({e}), trying next: {flt}")
    if not used:
        raise SystemExit("No filter variant worked — check OpenAlex docs "
                         "(api.openalex.org) for the current field filter "
                         "syntax and update FILTER_VARIANTS.")

    cursor = "*"
    while cursor and len(rows) < TOP_N:
        data = get({"filter": used, "per-page": 100, "cursor": cursor,
                    "sort": "works_count:desc",
                    "select": "display_name,issn,works_count,is_oa,"
                              "host_organization_name"})
        for s in data.get("results", []):
            rows.append((s.get("display_name", ""),
                         ",".join(s.get("issn") or []),
                         s.get("works_count", 0),
                         s.get("is_oa", ""),
                         s.get("host_organization_name", "")))
        cursor = data.get("meta", {}).get("next_cursor")
        time.sleep(0.3)

    rows = rows[:TOP_N]
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["journal", "issn", "openalex_works_count", "is_oa",
                    "publisher", "in_tdm_studio (fill in: yes/no/other-name)"])
        for r in rows:
            w.writerow(list(r) + [""])
    print(f"\nWrote top {len(rows)} psychology venues -> {OUT_PATH}")
    print("Diff against the TDM Studio title list; fill the last column. "
          "Journals marked 'no' are the documented exclusions of Stratum 2.")


if __name__ == "__main__":
    main()
