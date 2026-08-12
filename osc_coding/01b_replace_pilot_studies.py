"""
01b_replace_pilot_studies.py — Replace pilot studies that have no abstract
(e.g., short reports) with seeded, same-journal draws from the RP:P frame.

Why a script and not a hand-pick: "abstract available" is effectively an
inclusion criterion for the pilot, and short reports cluster by journal.
Documented, seeded replacement from the same journal stratum keeps the
sample defensible. Every replacement is recorded in the CSV (`note` column).

For the FULL 100-study coding later: short reports must NOT be dropped —
their opening paragraphs (from full text) will stand in for abstracts.
This is logged as a known selection issue in the corpus protocol.

Usage (from code/osc_coding/, venv active):
    python3 01b_replace_pilot_studies.py 04 09

Reads:  data/pilot_10_studies.csv, data/rpp_studies_all.csv
Writes: updated data/pilot_10_studies.csv (backup saved as .bak),
        new data/pilot_texts/study_NN.txt and study_NN_masked.txt
Then:   hand-code the replacement studies before running 02.
"""

import csv
import importlib.util
import random
import sys
import time
from pathlib import Path

REPLACEMENT_SEED = 43  # distinct from the original SEED=42; do not reuse
DATA_DIR = Path("data")
TEXT_DIR = DATA_DIR / "pilot_texts"
PILOT_PATH = DATA_DIR / "pilot_10_studies.csv"
FRAME_PATH = DATA_DIR / "rpp_studies_all.csv"

# Load fetch_abstract/clean_title from 01 (filename starts with a digit,
# so a normal import won't work)
_spec = importlib.util.spec_from_file_location(
    "fetch_pilot", Path(__file__).parent / "01_fetch_pilot_studies.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_abstract, clean_title = _mod.fetch_abstract, _mod.clean_title


def main():
    targets = [t.zfill(2) for t in sys.argv[1:]]
    if not targets:
        sys.exit("Usage: python3 01b_replace_pilot_studies.py <pilot_id> [<pilot_id> ...]  "
                 "e.g.: python3 01b_replace_pilot_studies.py 04 09")

    with open(PILOT_PATH, encoding="utf-8") as f:
        pilot_rows = list(csv.DictReader(f))
    with open(FRAME_PATH, encoding="utf-8") as f:
        frame = list(csv.DictReader(f))

    used_titles = {r["title"] for r in pilot_rows}
    random.seed(REPLACEMENT_SEED)

    for row in pilot_rows:
        row.setdefault("note", "")
        if row["pilot_id"] not in targets:
            continue
        journal = row["journal"]
        old_title = row["title"]
        candidates = [r for r in frame
                      if r["journal"] == journal and r["title"] not in used_titles]
        random.shuffle(candidates)

        replaced = False
        for cand in candidates:
            title = clean_title(cand["title"])
            print(f"[{row['pilot_id']}] trying: {title[:70]}")
            abstract, doi, source = fetch_abstract(title)
            time.sleep(1.0)
            if not abstract:
                print("    no abstract — next candidate")
                used_titles.add(cand["title"])
                continue
            print(f"    -> abstract from {source}")
            pid = row["pilot_id"]
            (TEXT_DIR / f"study_{pid}.txt").write_text(
                f"TITLE: {title}\nAUTHORS: {cand.get('authors','')}\n"
                f"JOURNAL: {journal}\n\nABSTRACT:\n{abstract}\n", encoding="utf-8")
            (TEXT_DIR / f"study_{pid}_masked.txt").write_text(
                f"ABSTRACT:\n{abstract}\n", encoding="utf-8")
            row.update({
                "study_num": cand.get("study_num", ""),
                "title": title,
                "authors": cand.get("authors", ""),
                "doi": doi,
                "abstract_source": source,
                "abstract_found": "yes",
                "note": (f"replacement (seed {REPLACEMENT_SEED}) for abstract-less "
                         f"short report: '{old_title[:60]}'"),
            })
            used_titles.add(cand["title"])
            replaced = True
            break
        if not replaced:
            print(f"[{row['pilot_id']}] NO candidate with an abstract found in "
                  f"{journal} — investigate before proceeding.")

    PILOT_PATH.rename(PILOT_PATH.with_suffix(".csv.bak"))
    fieldnames = list(pilot_rows[0].keys())
    if "note" not in fieldnames:
        fieldnames.append("note")
    with open(PILOT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(pilot_rows)

    print(f"\nUpdated {PILOT_PATH} (previous version saved as .bak).")
    print("IMPORTANT: hand-code the replacement studies (coding_protocol.md) "
          "before running 02 — and don't peek at replication outcomes.")


if __name__ == "__main__":
    main()
