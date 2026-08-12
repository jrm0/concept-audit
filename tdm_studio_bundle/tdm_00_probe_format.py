"""
tdm_00_probe_format.py — Inspect the TDM Studio dataset format before parsing.

TDM Studio delivers datasets as one XML file per document inside
data/<DATASET_NAME>/. Field names vary by content source, so LOOK FIRST:
this prints the tag structure and text-bearing fields of a few documents.

Usage (inside TDM Studio Jupyter):
    python3 tdm_00_probe_format.py            # auto-detects first dataset
    python3 tdm_00_probe_format.py MyDataset  # specific dataset folder
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# TDM Studio's Jupyter root is ~/SageMaker (verified July 2026 at BU);
# datasets land in ~/SageMaker/data/<DatasetName>/
DATA_ROOT = Path.home() / "SageMaker" / "data"
if not DATA_ROOT.exists():
    DATA_ROOT = Path.home() / "data"  # fallback for other deployments
N_DOCS = 3


def find_dataset():
    if len(sys.argv) > 1:
        return DATA_ROOT / sys.argv[1]
    candidates = [d for d in sorted(DATA_ROOT.iterdir()) if d.is_dir()] \
        if DATA_ROOT.exists() else []
    if not candidates:
        sys.exit(f"No dataset folders under {DATA_ROOT} — check the path "
                 f"(ls ~/data) and pass the folder name as an argument.")
    print(f"Datasets found: {[d.name for d in candidates]}")
    return candidates[0]


def show_structure(elem, depth=0, max_depth=4, seen=None):
    if seen is None:
        seen = set()
    key = (depth, elem.tag)
    if key in seen or depth > max_depth:
        return
    seen.add(key)
    text = (elem.text or "").strip()
    preview = f"  -> '{text[:60]}'" if text else ""
    print("  " * depth + f"<{elem.tag}>{preview}")
    for child in elem:
        show_structure(child, depth + 1, max_depth, seen)


def main():
    ds = find_dataset()
    files = sorted(ds.glob("*.xml"))[:N_DOCS]
    if not files:
        sys.exit(f"No .xml files in {ds} — list the folder and check.")
    print(f"Dataset: {ds.name} — probing {len(files)} of "
          f"{len(list(ds.glob('*.xml')))} documents\n")
    for f in files:
        print(f"===== {f.name} =====")
        try:
            root = ET.parse(f).getroot()
        except ET.ParseError as e:
            print(f"  XML parse error: {e}")
            continue
        show_structure(root)
        # Flag the tags most likely to matter
        print("\n  Likely fields:")
        for tag in ("Title", "SortTitle", "PubTitle", "PublicationTitle",
                    "AlphaPubDate", "NumericPubDate", "NumericDate",
                    "Text", "FullText", "HiddenText", "GOID"):
            hits = root.iter(tag)
            for h in hits:
                t = (h.text or "").strip()
                print(f"    <{tag}>: '{t[:80]}'")
                break
        print()
    print("Note the tags carrying: document id, title, publication title, "
          "date, and full text. If they differ from the FIELDS defaults in "
          "tdm_01_parse_corpus.py, edit that dict before parsing.")


if __name__ == "__main__":
    main()
