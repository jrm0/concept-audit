"""
12_fetch_rpp_outcomes.py — Download the RP:P outcome data (Paper 2's DV).

Source: the Reproducibility Project: Psychology data file, from the
Center for Open Science's public GitHub mirror (RPPdataConverted.csv in
CenterForOpenScience/rpp), falling back to the OSF record (osf.io/fgjvw).
Both are the published data behind Open Science Collaboration (2015).

What it does:
  1. Downloads the raw file -> data/rpp/rpp_data_raw.csv (kept verbatim).
  2. Prints the column names it finds (the file has ~150 columns with
     idiosyncratic names; we DO NOT guess silently).
  3. Extracts a tidy candidates file -> data/rpp/rpp_outcomes.csv with the
     study number plus every column whose name fuzzy-matches known outcome
     measures (replication p-value, effect sizes O/R, subjective
     "did it replicate", meta-analytic estimate). Final column choice is
     made together with Fable AFTER seeing what's actually in the file.

Usage (from code/osc_coding/):
    python3 12_fetch_rpp_outcomes.py
"""

import csv
import io
import re
from pathlib import Path

URLS = [
    ("github mirror",
     "https://raw.githubusercontent.com/CenterForOpenScience/rpp/master/"
     "data_allformats/RPPdataConverted.csv"),
    ("OSF", "https://osf.io/fgjvw/download"),
]
OUT_DIR = Path("data/rpp")

# fuzzy patterns for candidate outcome/moderator columns
WANT = [r"study.*num|^ID\b", r"replicat", r"p.?val", r"effect.?size|_es|\bes\b",
        r"meta.?analytic", r"power", r"direction", r"significan",
        r"discipline|journal", r"surprising|importance"]


def main():
    import requests
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = None
    for name, url in URLS:
        try:
            print(f"trying {name}: {url}")
            r = requests.get(url, timeout=120, allow_redirects=True)
            if r.status_code == 200 and len(r.text) > 10000:
                text = r.text
                print(f"  ok ({len(r.text):,} bytes)")
                break
            print(f"  status {r.status_code}, {len(r.text)} bytes — next")
        except Exception as e:
            print(f"  {type(e).__name__}: {e} — next")
    if text is None:
        raise SystemExit("Could not download from either source.")

    raw_path = OUT_DIR / "rpp_data_raw.csv"
    raw_path.write_text(text, encoding="utf-8")
    print(f"raw file -> {raw_path}")

    rows = list(csv.reader(io.StringIO(text)))
    header = rows[0]
    print(f"\n{len(rows)-1} data rows, {len(header)} columns. Columns:")
    for i, h in enumerate(header):
        print(f"  [{i:>3}] {h}")

    keep = [i for i, h in enumerate(header)
            if any(re.search(p, h, re.I) for p in WANT)]
    print(f"\nkeeping {len(keep)} candidate columns (fuzzy match)")
    with open(OUT_DIR / "rpp_outcomes.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([header[i] for i in keep])
        for r in rows[1:]:
            if len(r) == len(header):
                w.writerow([r[i] for i in keep])
    print(f"tidy candidates -> {OUT_DIR / 'rpp_outcomes.csv'}")
    print("\nSend Fable the column list above — the DV definition (and the "
          "join key to our pilot_ids) gets decided on the actual columns, "
          "not from memory.")


if __name__ == "__main__":
    main()
