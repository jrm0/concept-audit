"""
09_ebsco_worklist.py — Generate the prioritized EBSCO search worklist for
enriching the OSC-99 construct corpora.

Strategy (locked with Jonathan): OpenAlex (08) unblocks Paper 2's first
pass; EBSCO enriches. EBSCO's abstract coverage is decisive for the
publisher-withheld portion (Elsevier/Springer withhold abstracts from
OpenAlex; PsycInfo has them). Daily export budget: 100k records.

This script reads 08's coverage table and emits a worklist CSV:
  * priority 1 = constructs still thin after the OpenAlex pull
    (total docs < THIN) — these NEED EBSCO
  * priority 2 = mid-coverage constructs (THIN..RICH) — enrichment
  * skipped    = generic/rich constructs (>= RICH docs) — already covered,
    and generic one-word labels would return huge noisy sets
  * for each: a ready-to-paste boolean phrase (quoted, AB OR TI fields),
    hyphen/space variants included, and a batch number that keeps each
    day's expected volume under the export budget (rough estimate: capped
    at EST_CAP per search for budgeting).

Export convention (same as the family pulls): RIS with abstracts, one file
per search or batched, dropped into ../data/ebsco exports/constructs/ —
the ingest script will parse DB tags and dedupe into the construct corpora.

Usage (from code/osc_coding/):
    python3 09_ebsco_worklist.py
Output: data/ebsco_worklist.csv + printed summary
"""

import csv
import re
from pathlib import Path

COVERAGE = Path("../data/construct_corpus/_coverage.csv")
OUT = Path("data/ebsco_worklist.csv")
THIN = 60        # below this after OpenAlex -> priority 1
RICH = 250       # above this -> skip (well covered / too generic)
EST_CAP = 8000   # budget estimate per search (PsycInfo phrase searches)
DAILY_BUDGET = 100_000


def variants(term):
    """Quoted phrase variants: spaces <-> hyphens (the session-8 lesson —
    surface forms differ across journals)."""
    t = term.strip().lower()
    vs = {t}
    if " " in t:
        vs.add(t.replace(" ", "-"))
    if "-" in t:
        vs.add(t.replace("-", " "))
    return sorted(vs)


def boolean_phrase(term):
    parts = [f'"{v}"' for v in variants(term)]
    q = " OR ".join(parts)
    return f"AB ( {q} ) OR TI ( {q} )"


def main():
    if not COVERAGE.exists():
        raise SystemExit("Run 08 first — need _coverage.csv to know what is "
                         "still thin.")
    rows = list(csv.DictReader(open(COVERAGE, encoding="utf-8")))
    work = []
    skipped = []
    for r in rows:
        c = r["construct"]
        total = int(r["pool_docs"] or 0) + int(r["pulled_docs"] or 0)
        n_words = len(re.findall(r"\w+", c))
        if total >= RICH or (n_words == 1 and total >= THIN):
            skipped.append((c, total))
            continue
        pr = 1 if total < THIN else 2
        work.append((pr, total, c))
    work.sort()

    batch, day, spent = [], 1, 0
    out_rows = []
    for pr, total, c in work:
        est = EST_CAP
        if spent + est > DAILY_BUDGET:
            day += 1
            spent = 0
        spent += est
        out_rows.append({"priority": pr, "day_batch": day, "construct": c,
                         "current_docs": total,
                         "search": boolean_phrase(c),
                         "databases": "APA PsycInfo; APA PsycArticles",
                         "note": ("NEEDS docs" if pr == 1 else "enrichment")})

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    n1 = sum(1 for r in out_rows if r["priority"] == 1)
    n2 = len(out_rows) - n1
    days = max(r["day_batch"] for r in out_rows) if out_rows else 0
    print(f"worklist -> {OUT}")
    print(f"  priority 1 (needs docs): {n1}")
    print(f"  priority 2 (enrichment): {n2}")
    print(f"  budget-batched into ~{days} day(s) at {DAILY_BUDGET:,}/day")
    print(f"  skipped as covered/generic: {len(skipped)}")
    print("\nExport RIS (with abstracts) into "
          "'../data/ebsco exports/constructs/' — one search at a time or "
          "batched; the DB tag in each record is provenance ground truth.")
    print("First few searches:")
    for r in out_rows[:5]:
        print(f"  [day {r['day_batch']}] {r['construct']} "
              f"({r['current_docs']} docs): {r['search']}")


if __name__ == "__main__":
    main()
