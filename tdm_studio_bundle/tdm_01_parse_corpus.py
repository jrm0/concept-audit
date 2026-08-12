"""
tdm_01_parse_corpus.py — Parse TDM Studio XML docs into per-concept sentence files.

Reads every XML document in the dataset, extracts (id, title, venue, year,
full text), splits into sentences, and writes sentences containing each
concept's variant terms to workspace/sentences_{concept_slug}.csv.

These sentence files STAY INSIDE the enclave (they are licensed text).
Only derived artifacts from tdm_02/tdm_03 go to the export folder.

Before running: confirm field names with tdm_00_probe_format.py and adjust
FIELDS below if needed.

Usage:
    python3 tdm_01_parse_corpus.py            # first dataset under ~/data
    python3 tdm_01_parse_corpus.py MyDataset
"""

import csv
import html
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_ROOT = Path.home() / "SageMaker" / "data"
if not DATA_ROOT.exists():
    DATA_ROOT = Path.home() / "data"  # fallback for other deployments
WORK_DIR = Path("workspace")
CONCEPTS_CSV = Path("concepts.csv")

# Confirmed against real ProQuest TDM Studio XML (July 2026). First matching
# tag in DOCUMENT ORDER wins. Notes:
#   - <Title> occurs twice: article title (under Obj/TitleAtt) comes FIRST,
#     journal title (under CurrentTitle) comes later — so "Title" gets the article.
#   - <SortTitle> is the normalized JOURNAL name (venue) — used for per-venue
#     stratification and register controls.
#   - <Text> is HTML-wrapped full text (clean_text strips the markup).
#     <AbsText> (abstract) is separate; we take full text for the enclave.
FIELDS = {
    "docid": ["GOID", "RecordID", "DOCID"],
    "title": ["Title", "SortTitle"],
    "venue": ["SortTitle", "JournalCode", "CurrentTitle"],
    "date": ["NumericDate", "NumericPubDate", "AlphaDate"],
    "text": ["Text", "FullText", "HiddenText"],
}

TAG_RE = re.compile(r"<[^>]+>")
# Simple sentence splitter (identical policy to the local pipeline):
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def first_tag_text(root, names):
    for name in names:
        for el in root.iter(name):
            if el.text and el.text.strip():
                return el.text.strip()
            # Some ProQuest text fields nest markup — serialize children
            inner = "".join(ET.tostring(c, encoding="unicode") for c in el)
            if inner.strip():
                return inner
    return ""


def clean_text(raw):
    txt = html.unescape(raw)
    txt = TAG_RE.sub(" ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def year_from(datestr):
    m = re.search(r"(19|20)\d{2}", datestr or "")
    return m.group(0) if m else ""


def load_concepts():
    if not CONCEPTS_CSV.exists():
        sys.exit("concepts.csv not found — copy concepts_template.csv and fill in.")
    concepts = []
    with open(CONCEPTS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variants = [v.strip().lower() for v in
                        (row.get("variants") or "").split("|") if v.strip()]
            anchor = (row.get("retrieval_anchor") or "").strip().lower()
            terms = sorted(set([anchor] + variants) - {""})
            if not terms:
                continue
            patterns = [re.compile(r"\b" + re.escape(t) + r"\b", re.I)
                        for t in terms]
            concepts.append((row["concept_slug"].strip(), patterns))
    return concepts


def main():
    ds_name = sys.argv[1] if len(sys.argv) > 1 else None
    ds = (DATA_ROOT / ds_name) if ds_name else \
        next((d for d in sorted(DATA_ROOT.iterdir()) if d.is_dir()), None)
    if ds is None or not ds.exists():
        sys.exit(f"Dataset folder not found under {DATA_ROOT}")
    WORK_DIR.mkdir(exist_ok=True)

    concepts = load_concepts()
    print(f"Dataset: {ds.name}; concepts: {[c for c, _ in concepts]}")

    writers, files = {}, {}
    for slug, _ in concepts:
        path = WORK_DIR / f"sentences_{slug}.csv"
        fh = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(fh)
        w.writerow(["docid", "venue", "year", "sentence"])
        writers[slug], files[slug] = w, fh

    counts = {slug: 0 for slug, _ in concepts}
    venue_year = {}  # (venue, year) -> doc count, for the coverage export
    n_docs, n_failed = 0, 0

    xml_files = sorted(ds.glob("*.xml"))
    for i, f in enumerate(xml_files, 1):
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            n_failed += 1
            continue
        docid = first_tag_text(root, FIELDS["docid"]) or f.stem
        venue = clean_text(first_tag_text(root, FIELDS["venue"]))
        year = year_from(first_tag_text(root, FIELDS["date"]))
        text = clean_text(first_tag_text(root, FIELDS["text"]))
        if not text:
            n_failed += 1
            continue
        n_docs += 1
        venue_year[(venue, year)] = venue_year.get((venue, year), 0) + 1

        for sent in SENT_RE.split(text):
            if not (30 <= len(sent) <= 1000):
                continue
            for slug, patterns in concepts:
                if any(p.search(sent) for p in patterns):
                    writers[slug].writerow([docid, venue, year, sent.strip()])
                    counts[slug] += 1
        if i % 500 == 0:
            print(f"  {i}/{len(xml_files)} docs; sentence counts so far: {counts}")

    for fh in files.values():
        fh.close()

    # Derived coverage table (exportable)
    Path("export").mkdir(exist_ok=True)
    with open("export/sentence_counts_by_venue_year.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["venue", "year", "n_docs"])
        for (v, y), n in sorted(venue_year.items()):
            w.writerow([v, y, n])
    with open("export/concept_sentence_counts.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept_slug", "n_sentences"])
        for slug, n in counts.items():
            w.writerow([slug, n])

    print(f"\nParsed {n_docs} docs ({n_failed} unreadable/empty).")
    print(f"Sentence counts: {counts}")
    print("Sentence files -> workspace/ (KEEP INSIDE). "
          "Counts -> export/ (safe to export).")
    print("Next: python3 tdm_02_embed.py")


if __name__ == "__main__":
    main()
