"""
tdm_00c_fulltext_by_journal.py — Which journals in a dataset have FULL TEXT?

ProQuest full-text availability varies by publisher (open-access sources like
PLoS = full text; APA = abstract-only). For the broad psychology dataset,
this samples documents across the whole folder, groups by journal (SortTitle),
and reports the median largest-text-block per journal — so you can see which
journals give minable full text and which are abstract-only.

Decision input: if few/no non-open-access journals have full text here, the
enclave adds little over the free OpenAlex abstracts route.

Usage (from ~/SageMaker):
    python3 tdm_00c_fulltext_by_journal.py <dataset-folder>   e.g. the broad 274k set
Optional 2nd arg = how many docs to sample (default 1500).
"""

import statistics
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

DATA_ROOT = Path.home() / "SageMaker" / "data"
if not DATA_ROOT.exists():
    DATA_ROOT = Path.home() / "data"


def biggest_text(root):
    return max((len((el.text or "").strip()) for el in root.iter()), default=0)


def journal_of(root):
    for el in root.iter("SortTitle"):
        t = (el.text or "").strip()
        if t:
            return t
    return "(unknown)"


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python3 tdm_00c_fulltext_by_journal.py <dataset-folder> [sample_n]")
    ds = DATA_ROOT / sys.argv[1]
    sample_n = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    files = sorted(ds.glob("*.xml"))
    if not files:
        sys.exit(f"No .xml in {ds}")

    # Even sampling across the folder
    step = max(1, len(files) // sample_n)
    sampled = files[::step][:sample_n]
    print(f"Dataset: {ds.name}; {len(files)} docs, sampling {len(sampled)}\n")

    by_journal = defaultdict(list)
    for f in sampled:
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError:
            continue
        by_journal[journal_of(root)].append(biggest_text(root))

    # Report: journals sorted by median text size (full-text first)
    rows = []
    for j, sizes in by_journal.items():
        rows.append((statistics.median(sizes), len(sizes), j))
    rows.sort(reverse=True)

    ft_journals = abs_journals = 0
    print(f"{'median_text':>11}  {'n':>4}  journal")
    for med, n, j in rows:
        kind = "FULLTEXT" if med > 3000 else ("abstract" if med > 200 else "meta")
        if med > 3000:
            ft_journals += 1
        elif med > 200:
            abs_journals += 1
        print(f"{med:>11.0f}  {n:>4}  [{kind}] {j[:55]}")

    print(f"\nSummary: {ft_journals} journals look FULL-TEXT, "
          f"{abs_journals} abstract-only, across {len(rows)} journals sampled.")
    print("Full-text journals that are NOT open-access (cross-check against "
          "your S2ORC coverage) are the enclave's unique value. If that set is "
          "small, prefer the free OpenAlex abstracts route (07) and treat TDM "
          "Studio as optional.")


if __name__ == "__main__":
    main()
