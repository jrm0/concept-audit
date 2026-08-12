"""
14_parse_ris_exports.py — Parse EBSCO/PsycInfo RIS exports into the corpus schema.

Why this route matters: RIS citation exports carry FULL ABSTRACTS at ~100%
coverage, versus OpenAlex's publisher-dependent coverage (JPSP 56%, Cognition
14%). They also carry APA Index Terms (controlled vocabulary), which enables
the keyword-triangulation check in the corpus protocol — comparing the field's
*normative* ontology against *descriptive* usage measured from embeddings.

(These are citation exports produced by the platform's own export function on
a bounded search result — standard research practice, unlike bulk full-text
retrieval, which remains off the table.)

Reads:  data/ebsco exports/*.ris
Writes: data/family/{name}_works.jsonl   same schema as 13_build_family_corpus
          + index_terms field (APA controlled vocabulary)
        data/family/{name}_indexterms.csv   term frequency, for triangulation

Usage (from code/, venv active):
    python3 14_parse_ris_exports.py                       # all .ris in the folder
    python3 14_parse_ris_exports.py ego_depletion_psycinfo  # output name
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

IN_DIR = Path("data/ebsco exports")
OUT_DIR = Path("data/family")
TAG_RE = re.compile(r"^([A-Z][A-Z0-9])  - (.*)$")


def parse_ris(path):
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    records, cur = [], {}
    for line in text.splitlines():
        m = TAG_RE.match(line)
        if not m:
            # continuation line — append to the last tag seen
            if cur and cur.get("_last"):
                cur[cur["_last"]][-1] += " " + line.strip()
            continue
        tag, val = m.group(1), m.group(2).strip()
        if tag == "ER":
            cur.pop("_last", None)
            if cur.get("TI"):
                records.append(cur)
            cur = {}
            continue
        cur.setdefault(tag, []).append(val)
        cur["_last"] = tag
    if cur.get("TI"):
        cur.pop("_last", None)
        records.append(cur)
    return records


def year_of(rec):
    for tag in ("Y1", "PY", "DA"):
        for v in rec.get(tag, []):
            m = re.search(r"(19|20)\d{2}", v)
            if m:
                return int(m.group(0))
    return None


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "psycinfo_export"
    files = sorted(IN_DIR.glob("*.ris"))
    if not files:
        raise SystemExit(f"No .ris files in {IN_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    works, kw_counter, db_counter = {}, Counter(), Counter()
    dupes = 0
    for f in files:
        recs = parse_ris(f)
        print(f"{f.name}: {len(recs)} records")
        for r in recs:
            doi = (r.get("DO", [""])[0] or "").lower().strip()
            doi = doi.replace("https://doi.org/", "")
            an = (r.get("AN", [""])[0] or "").strip()
            key = doi or f"AN:{an}" or (r.get("TI", [""])[0][:80])
            if key in works:
                dupes += 1
                continue
            terms = [k.strip() for k in r.get("KW", []) if k.strip()]
            for t in terms:
                kw_counter[t] += 1
            db = (r.get("DB", [""])[0] or "")
            db_counter[db] += 1
            works[key] = {
                "doi": doi,
                "accession": an,
                "year": year_of(r),
                "journal": (r.get("JO", [""])[0] or r.get("T2", [""])[0] or ""),
                "title": r.get("TI", [""])[0],
                "abstract": (r.get("AB", [""])[0] or ""),
                "index_terms": terms,
                "authors": r.get("AU", []),
                "publisher": (r.get("PB", [""])[0] or ""),
                "source_db": db,
                "source": "ris_export",
            }

    out_path = OUT_DIR / f"{name}_works.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for w in works.values():
            f.write(json.dumps(w) + "\n")

    with_abs = sum(1 for w in works.values() if w["abstract"])
    with_doi = sum(1 for w in works.values() if w["doi"])
    years = [w["year"] for w in works.values() if w["year"]]
    print(f"\nUnique works: {len(works)} ({dupes} duplicates dropped)")
    print(f"  with abstract: {with_abs} ({with_abs/len(works):.0%})")
    print(f"  with DOI:      {with_doi} ({with_doi/len(works):.0%})")
    if years:
        dec = Counter(f"{y//10*10}s" for y in years)
        print("  by decade: " + ", ".join(f"{d}:{dec[d]}" for d in sorted(dec)))
    print("  source dbs: " + ", ".join(f"{k}:{v}" for k, v in db_counter.most_common()))

    kw_path = OUT_DIR / f"{name}_indexterms.csv"
    with open(kw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index_term", "n_works", "share"])
        for t, c in kw_counter.most_common():
            w.writerow([t, c, round(c / len(works), 4)])
    print(f"\nWorks -> {out_path}")
    print(f"Index terms ({len(kw_counter)} unique) -> {kw_path}")
    print("\nTriangulation check: compare the top index terms against the "
          "concept family. A focal construct ABSENT from the controlled "
          "vocabulary (e.g. 'Ego Depletion') is evidence that the normative "
          "ontology collapses a distinction the literature makes.")


if __name__ == "__main__":
    main()
