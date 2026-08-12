"""
tdm_00b_check_fulltext.py — Does this dataset contain FULL TEXT or only abstracts?

The make-or-break question for an APA/PsycArticles dataset. Scans a few
documents and reports EVERY element carrying substantial text (>100 chars),
with tag name and character length. Full text shows up as one element with
thousands of characters (the body); abstract-only datasets top out at a few
hundred chars (the abstract).

Usage (from ~/SageMaker):
    python3 tdm_00b_check_fulltext.py test-set-psych-JPSP-PB
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_ROOT = Path.home() / "SageMaker" / "data"
if not DATA_ROOT.exists():
    DATA_ROOT = Path.home() / "data"
N_DOCS = 5


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 tdm_00b_check_fulltext.py <dataset-folder-name>")
    ds = DATA_ROOT / sys.argv[1]
    files = sorted(ds.glob("*.xml"))[:N_DOCS]
    if not files:
        sys.exit(f"No .xml in {ds}")

    print(f"Dataset: {ds.name} — scanning {len(files)} docs for text-bearing "
          f"elements (>100 chars)\n")
    max_seen_overall = 0
    for f in files:
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError as e:
            print(f"{f.name}: parse error {e}")
            continue
        print(f"===== {f.name} =====")
        rows = []
        for el in root.iter():
            t = (el.text or "").strip()
            if len(t) > 100:
                rows.append((el.tag, len(t), t[:60].replace("\n", " ")))
        rows.sort(key=lambda r: -r[1])
        if not rows:
            print("  (no element with >100 chars of text — likely metadata only)")
        for tag, n, preview in rows[:8]:
            print(f"  <{tag}>  {n} chars   '{preview}...'")
            max_seen_overall = max(max_seen_overall, n)
        print()

    print("VERDICT:")
    if max_seen_overall > 3000:
        print(f"  Largest text block = {max_seen_overall} chars → FULL TEXT present. "
              f"Note which tag carries it and set FIELDS['text'] to that tag.")
    elif max_seen_overall > 0:
        print(f"  Largest text block = {max_seen_overall} chars → looks like "
              f"ABSTRACT-ONLY (no article body). This dataset would feed the "
              f"abstract stratum, not the full-text stratum.")
    else:
        print("  No substantial text found — metadata/indexing only.")


if __name__ == "__main__":
    main()
