"""
04_fetch_all_studies.py — Fetch coding texts for ALL completed RP:P studies.

Scales the pilot's Step 1 to the full frame (~99 studies):
  1. Reads data/rpp_studies_all.csv (written by 01_fetch_pilot_studies.py)
  2. Fetches each study's abstract via the pilot's 3-source chain
     (OpenAlex -> Crossref -> Semantic Scholar)
  3. Writes:
       data/full_texts/study_{NNN}.txt         — UNMASKED (title+authors+abstract)
       data/full_texts/study_{NNN}_masked.txt  — MASKED (abstract only)
       data/full_studies.csv                   — manifest with abstract source/DOI
       data/coding_sheet_full.csv              — empty human coding sheet
     where NNN = zero-padded RP:P Study Num.

Resumable: studies whose text files already exist are skipped, so re-running
only fetches the gaps (and won't clobber manually pasted texts).

Abstract-less studies (short reports) are listed at the end. Per protocol §2,
these are NOT dropped: paste the paper's opening paragraph(s) into both text
files by hand, and note "text_source: opening paragraphs" in the manifest.

Usage (from code/osc_coding/, venv active):
    python3 04_fetch_all_studies.py
"""

import csv
import importlib.util
import time
from pathlib import Path

DATA_DIR = Path("data")
FRAME_PATH = DATA_DIR / "rpp_studies_all.csv"
TEXT_DIR = DATA_DIR / "full_texts"
MANIFEST_PATH = DATA_DIR / "full_studies.csv"
SHEET_PATH = DATA_DIR / "coding_sheet_full.csv"

# Reuse the pilot's fetch chain (filename starts with a digit -> importlib)
_spec = importlib.util.spec_from_file_location(
    "fetch_pilot", Path(__file__).parent / "01_fetch_pilot_studies.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fetch_abstract, clean_title = _mod.fetch_abstract, _mod.clean_title


def main():
    if not FRAME_PATH.exists():
        raise SystemExit("Missing data/rpp_studies_all.csv — run "
                         "01_fetch_pilot_studies.py first.")
    TEXT_DIR.mkdir(exist_ok=True)
    with open(FRAME_PATH, encoding="utf-8") as f:
        frame = list(csv.DictReader(f))
    print(f"Frame: {len(frame)} completed replication studies")

    manifest = []
    missing = []
    for i, row in enumerate(frame, 1):
        num = (row.get("study_num") or str(i)).strip() or str(i)
        sid = num.zfill(3)
        title = clean_title(row["title"])
        masked_path = TEXT_DIR / f"study_{sid}_masked.txt"

        if masked_path.exists():
            text = masked_path.read_text(encoding="utf-8")
            if "ABSTRACT NOT FOUND" not in text:
                # Real text present (fetched or manually pasted) — never clobber
                print(f"[{i}/{len(frame)}] study {sid}: exists — skipping fetch")
                manifest.append((sid, title, row["journal"], "", "existing", "ok"))
                continue
            # Placeholder only (e.g., an earlier rate-limited run) — RE-FETCH
            print(f"[{i}/{len(frame)}] study {sid}: placeholder found — re-fetching")

        print(f"[{i}/{len(frame)}] study {sid}: {title[:65]}")
        abstract, doi, source = fetch_abstract(title)
        print(f"    -> {source}")
        (TEXT_DIR / f"study_{sid}.txt").write_text(
            f"TITLE: {title}\nAUTHORS: {row.get('authors','')}\n"
            f"JOURNAL: {row['journal']}\n\n"
            f"ABSTRACT:\n{abstract or '[ABSTRACT NOT FOUND — paste opening paragraphs manually]'}\n",
            encoding="utf-8")
        masked_path.write_text(
            f"ABSTRACT:\n{abstract or '[ABSTRACT NOT FOUND — paste opening paragraphs manually]'}\n",
            encoding="utf-8")
        manifest.append((sid, title, row["journal"], doi, source,
                         "ok" if abstract else "MISSING"))
        if not abstract:
            missing.append((sid, title))
        time.sleep(1.0)

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["study_id", "title", "journal", "doi",
                    "abstract_source", "status"])
        w.writerows(manifest)
    if not SHEET_PATH.exists():
        with open(SHEET_PATH, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pilot_id", "coder", "run", "masked",
                        "construct_1", "construct_2", "construct_3", "notes"])

    ok = sum(1 for m in manifest if m[5] != "MISSING")
    print(f"\nDone: {ok}/{len(manifest)} studies have coding text.")
    if missing:
        print(f"\n{len(missing)} studies need manual text (short reports — "
              f"paste opening paragraphs into BOTH files, per protocol §2):")
        for sid, title in missing:
            print(f"  study {sid}: {title[:70]}")
        print("Then re-run this script to refresh the manifest.")
    print(f"\nManifest -> {MANIFEST_PATH}")
    print("Next: python3 02_run_llm_coding.py full")


if __name__ == "__main__":
    main()
