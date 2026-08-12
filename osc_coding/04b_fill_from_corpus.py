"""
04b_fill_from_corpus.py — Fill missing coding texts from OUR OWN Track 2 corpus.

For studies still lacking an abstract after 04 (short reports / rate-limit
casualties), look them up by DOI in the S2ORC full-text corpus
(../data/s2orc/fulltext_stratum1.jsonl) and extract:
  - the abstract span if S2ORC has one, else
  - the first two paragraph spans (opening paragraphs, per protocol §2).

Writes into data/full_texts/study_{NNN}(.txt|_masked.txt), replacing
placeholders only — never touches files with real text. Records the text
source per study in data/full_studies_fill_log.csv.

Run AFTER phase_b completes (re-runnable; safe to run mid-download too — it
just finds fewer matches).

Usage (from code/osc_coding/):
    python3 04b_fill_from_corpus.py
"""

import csv
import json
from pathlib import Path

DATA_DIR = Path("data")
TEXT_DIR = DATA_DIR / "full_texts"
MANIFEST_PATH = DATA_DIR / "full_studies.csv"
CORPUS_PATH = Path("../data/s2orc/fulltext_stratum1.jsonl")
FILL_LOG = DATA_DIR / "full_studies_fill_log.csv"


def spans_text(fulltext, annotation_json, max_spans=None):
    """Extract text for annotation spans (S2ORC stores spans as a JSON string)."""
    if not annotation_json:
        return None
    try:
        spans = json.loads(annotation_json) if isinstance(annotation_json, str) \
            else annotation_json
    except json.JSONDecodeError:
        return None
    seen = set()
    chunks = []
    for s in spans:
        key = (s.get("start"), s.get("end"))
        if key in seen or key[0] is None:
            continue
        seen.add(key)
        chunks.append(fulltext[s["start"]:s["end"]])
        if max_spans and len(chunks) >= max_spans:
            break
    return "\n\n".join(chunks).strip() or None


def main():
    if not MANIFEST_PATH.exists():
        raise SystemExit("Run 04_fetch_all_studies.py first.")
    if not CORPUS_PATH.exists():
        raise SystemExit(f"Corpus not found at {CORPUS_PATH} — "
                         f"has phase_b produced output yet?")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))

    # Studies that still need text AND have a known DOI
    wanted = {}
    no_doi = []
    for row in manifest:
        masked = TEXT_DIR / f"study_{row['study_id']}_masked.txt"
        if masked.exists() and "ABSTRACT NOT FOUND" in masked.read_text(encoding="utf-8"):
            doi = (row.get("doi") or "").lower().strip()
            if doi:
                wanted[doi] = row
            else:
                no_doi.append(row["study_id"])
    print(f"Studies needing text with a known DOI: {len(wanted)}"
          + (f"; without DOI (manual): {no_doi}" if no_doi else ""))
    if not wanted:
        print("Nothing to fill — all set.")
        return

    filled = []
    scanned = 0
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            scanned += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = ((rec.get("externalids") or {}).get("doi") or "").lower().strip()
            doi = doi.replace("https://doi.org/", "")
            if doi not in wanted:
                continue
            row = wanted[doi]
            content = rec.get("content") or {}
            text = content.get("text") or ""
            ann = content.get("annotations") or {}
            extract = spans_text(text, ann.get("abstract"))
            source = "s2orc-abstract"
            if not extract:
                extract = spans_text(text, ann.get("paragraph"), max_spans=2)
                source = "s2orc-opening-paragraphs"
            if not extract:
                continue
            sid = row["study_id"]
            (TEXT_DIR / f"study_{sid}.txt").write_text(
                f"TITLE: {row['title']}\nJOURNAL: {row['journal']}\n\n"
                f"ABSTRACT:\n{extract}\n", encoding="utf-8")
            (TEXT_DIR / f"study_{sid}_masked.txt").write_text(
                f"ABSTRACT:\n{extract}\n", encoding="utf-8")
            filled.append((sid, doi, source))
            print(f"  filled study {sid} ({source})")
            del wanted[doi]
            if not wanted:
                break

    with open(FILL_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["study_id", "doi", "text_source"])
        w.writerows(filled)

    print(f"\nScanned {scanned} corpus records; filled {len(filled)} studies.")
    if wanted:
        print(f"Still missing ({len(wanted)}) — not in the full-text corpus; "
              f"manual retrieval via DOI:")
        for doi, row in wanted.items():
            print(f"  study {row['study_id']}: https://doi.org/{doi}")
    print("\nIMPORTANT for the methods record: studies filled here use "
          "'s2orc-opening-paragraphs' or 's2orc-abstract' as text_source — "
          "keep the distinction in analyses (text-source effect).")


if __name__ == "__main__":
    main()
