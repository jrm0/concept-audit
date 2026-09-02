"""
11_ingest_construct_ris.py — Merge EBSCO RIS exports into the per-construct
corpora (Paper 2), complementing 08's OpenAlex pulls.

Drop RIS files (exported with abstracts) into:
    ../data/ebsco exports/constructs/
File names don't matter; a record's DB tag is provenance ground truth, and
records are matched to constructs by pattern against title+abstract (so one
export can feed several constructs, and misfiled records land correctly).

Merging: appends to data/construct_corpus/{slug}.jsonl, deduping against
whatever is already there (OpenAlex pulls) by DOI then normalized title.
Every record carries source="ebsco" + its DB tags, so source-composition
checks per construct are possible later.

Usage (from code/osc_coding/):
    python3 11_ingest_construct_ris.py
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

RIS_DIR = Path("../data/ebsco exports/constructs")
OUT_DIR = Path("../data/construct_corpus")
FINAL = Path("data/final_constructs.csv")
TAG_RE = re.compile(r"^([A-Z][A-Z0-9])  - (.*)$")


def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def pat_of(term):
    body = re.escape(term.lower()).replace(r"\ ", "[-\\s]+") \
        .replace(r"\-", "[-\\s]+")
    return re.compile(r"\b" + body + r"\b", re.I)


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()[:120]


def parse_ris(path):
    recs, cur, last = [], {}, None
    for line in path.read_text(encoding="utf-8-sig",
                               errors="replace").splitlines():
        m = TAG_RE.match(line)
        if not m:
            if cur and last:
                cur[last][-1] += " " + line.strip()
            continue
        tag, val = m.group(1), m.group(2).strip()
        if tag == "ER":
            if cur.get("TI"):
                recs.append(cur)
            cur, last = {}, None
            continue
        cur.setdefault(tag, []).append(val)
        last = tag
    if cur.get("TI"):
        recs.append(cur)
    return recs


def main():
    constructs = set()
    for r in csv.DictReader(open(FINAL, encoding="utf-8")):
        for i in (1, 2, 3, 4):
            c = (r.get(f"construct_{i}") or "").strip().lower()
            if c:
                constructs.add(c)
    pats = {c: pat_of(c) for c in constructs}
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # existing keys per construct (dedupe against OpenAlex pulls)
    have = {}
    for c in constructs:
        keys = set()
        f = OUT_DIR / f"{slug(c)}.jsonl"
        if f.exists():
            for line in open(f, encoding="utf-8"):
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get("doi"):
                    keys.add(d["doi"].lower())
                keys.add(norm_title(d.get("title")))
        have[c] = keys

    ris_files = sorted(RIS_DIR.glob("*.ris")) if RIS_DIR.exists() else []
    if not ris_files:
        raise SystemExit(f"No .ris files in {RIS_DIR} — export there first.")
    added, matched_any, total = Counter(), 0, 0
    handles = {}
    try:
        for f in ris_files:
            for r in parse_ris(f):
                total += 1
                title = r.get("TI", [""])[0]
                ab = (r.get("AB", [""])[0] or "")
                if len(ab) < 200:
                    continue
                doi = (r.get("DO", [""])[0] or "").lower() \
                    .replace("https://doi.org/", "").rstrip(".")
                year = None
                for tag in ("Y1", "PY", "DA"):
                    for v in r.get(tag, []):
                        m = re.search(r"(19|20)\d{2}", v)
                        if m:
                            year = int(m.group(0))
                            break
                    if year:
                        break
                text = f"{title}. {ab}".lower()
                hit = False
                for c, p in pats.items():
                    if c[:4] not in text or not p.search(text):
                        continue
                    key = doi or norm_title(title)
                    if key in have[c] or norm_title(title) in have[c]:
                        continue
                    hit = True
                    have[c].add(key)
                    have[c].add(norm_title(title))
                    if c not in handles:
                        handles[c] = open(OUT_DIR / f"{slug(c)}.jsonl", "a",
                                          encoding="utf-8")
                    handles[c].write(json.dumps({
                        "doi": doi, "year": year, "title": title,
                        "journal": (r.get("JO", [""])[0]
                                    or r.get("T2", [""])[0] or "").strip(),
                        "abstract": ab, "construct": c, "source": "ebsco",
                        "dbs": [v.strip() for v in r.get("DB", [])
                                if v.strip()]}) + "\n")
                    added[c] += 1
                if hit:
                    matched_any += 1
            print(f"{f.name}: done (running total {sum(added.values())} "
                  f"additions)")
    finally:
        for h in handles.values():
            h.close()

    print(f"\n{total} RIS records read; {matched_any} matched >=1 construct; "
          f"{sum(added.values())} new (construct, doc) additions")
    for c, n in added.most_common(15):
        print(f"  +{n:>4}  {c}")
    print("\nRe-run any time — dedupe makes it idempotent. "
          "Then coverage: python3 08_build_construct_corpus.py (it counts "
          "cached files) or just proceed to the embedding step.")


if __name__ == "__main__":
    main()
