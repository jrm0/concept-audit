"""
16_build_lineage_corpus.py — Assemble the document-level lineage corpus.

Merges every family source (RIS exports incl. subfolders + any family
works.jsonl) into ONE document-level dataset for the lineage analysis:

  data/family/lineage_docs.jsonl   one row per unique document:
      doi, year, period, journal, title, abstract, index_terms,
      terms (which family terms actually appear in title+abstract),
      mention_sentences {term: [sentences]}   <- the embedding input

Design notes:
- DOCUMENT is the unit (session-10 lesson: sentences are not independent).
- Term presence is recomputed from the text itself, so ancestor exports
  (retrieved by query, no per-record match info) get correct term labels.
- Control-term mentions ("participants", "results", "significant", "measure")
  are extracted from the same abstracts — the drift null costs nothing.
- 5-year periods from 1975.

Usage (from code/):
    python3 16_build_lineage_corpus.py
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RIS_ROOT = Path("data/ebsco exports")
FAMILY_DIR = Path("data/family")
OUT_PATH = FAMILY_DIR / "lineage_docs.jsonl"
FAMILY_CSV = Path("families/ego_depletion_family.csv")

PERIOD_YEARS = 5
YEAR_MIN, YEAR_MAX = 1975, 2025

CONTROL_TERMS = ["participants", "results", "significant", "measure"]

TAG_RE = re.compile(r"^([A-Z][A-Z0-9])  - (.*)$")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def load_family_terms():
    terms = {}
    with open(FAMILY_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = (row.get("term") or "").strip()
            if t:
                terms[t.lower()] = (row.get("role") or "").strip()
    return terms


def compile_patterns(term_names):
    pats = {}
    for t in term_names:
        pats[t] = re.compile(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b",
                             re.I)
    return pats


def parse_ris(path):
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    recs, cur, last = [], {}, None
    for line in text.splitlines():
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


def year_of(rec):
    for tag in ("Y1", "PY", "DA"):
        for v in rec.get(tag, []):
            m = re.search(r"(19|20)\d{2}", v)
            if m:
                return int(m.group(0))
    return None


def period_of(year):
    if year is None or year < YEAR_MIN:
        return None
    y0 = YEAR_MIN + ((year - YEAR_MIN) // PERIOD_YEARS) * PERIOD_YEARS
    return f"{y0}-{min(y0 + PERIOD_YEARS - 1, YEAR_MAX)}"


def main():
    family = load_family_terms()
    pats = compile_patterns(list(family) + CONTROL_TERMS)
    docs = {}

    def add_doc(key, year, journal, title, abstract, index_terms):
        period = period_of(year)
        if period is None or not abstract:
            return
        text = f"{title}. {abstract}"
        sentences = [s.strip() for s in SENT_RE.split(text)
                     if 30 <= len(s.strip()) <= 1200]
        mentions = defaultdict(list)
        for t, pat in pats.items():
            for s in sentences:
                if pat.search(s):
                    mentions[t].append(s)
        fam_terms = [t for t in mentions if t in family]
        if not fam_terms:
            return  # no family term actually present
        docs[key] = {
            "doi": key if not key.startswith("AN:") else "",
            "year": year, "period": period,
            "journal": journal, "title": title,
            "index_terms": index_terms,
            "terms": sorted(fam_terms),
            "roles": sorted({family[t] for t in fam_terms}),
            "mention_sentences": {t: v[:6] for t, v in mentions.items()},
        }

    # 1. All RIS files, recursively
    ris_files = sorted(RIS_ROOT.rglob("*.ris"))
    for f in ris_files:
        recs = parse_ris(f)
        n_before = len(docs)
        for r in recs:
            doi = (r.get("DO", [""])[0] or "").lower().strip()
            doi = doi.replace("https://doi.org/", "")
            an = (r.get("AN", [""])[0] or "").strip()
            key = doi or (f"AN:{an}" if an else r.get("TI", [""])[0][:80].lower())
            if key in docs:
                continue
            add_doc(key, year_of(r),
                    (r.get("JO", [""])[0] or r.get("T2", [""])[0] or "").strip(),
                    r.get("TI", [""])[0],
                    (r.get("AB", [""])[0] or ""),
                    [k.strip() for k in r.get("KW", []) if k.strip()])
        print(f"{f.name}: +{len(docs) - n_before} docs (total {len(docs)})")

    # 2. OpenAlex-derived family works, if present (adds field metadata later)
    for jf in sorted(FAMILY_DIR.glob("ego_depletion*works.jsonl")):
        n_before = len(docs)
        for line in open(jf, encoding="utf-8"):
            try:
                w = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (w.get("doi") or "").lower() or \
                f"AN:{w.get('accession','')}" if w.get("accession") else None
            if not key or key in docs:
                continue
            add_doc(key, w.get("year"), w.get("journal", ""),
                    w.get("title", ""), w.get("abstract", ""),
                    w.get("index_terms", []))
        print(f"{jf.name}: +{len(docs) - n_before} docs (total {len(docs)})")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for d in docs.values():
            f.write(json.dumps(d) + "\n")

    # Report: term x period document counts (the power table for the analysis)
    tp = defaultdict(Counter)
    for d in docs.values():
        for t in d["terms"]:
            tp[t][d["period"]] += 1
    periods = sorted({d["period"] for d in docs.values()})
    print(f"\n{len(docs)} documents -> {OUT_PATH}\n")
    print("DOCUMENTS PER TERM PER PERIOD (lineage power table):")
    print(f"{'term':<26}" + "".join(f"{p[2:4]}-{p[7:9]:>3}" for p in periods))
    for t in sorted(tp, key=lambda x: -sum(tp[x].values())):
        role = load_family_terms().get(t, "")
        print(f"{t:<26}" + "".join(f"{tp[t].get(p, 0):>7}" for p in periods)
              + f"   [{role}]")

    with open(FAMILY_DIR / "lineage_power_table.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term"] + periods)
        for t in sorted(tp):
            w.writerow([t] + [tp[t].get(p, 0) for p in periods])
    print(f"\nPower table -> {FAMILY_DIR / 'lineage_power_table.csv'}")
    print("Next: python3 17_embed_documents.py   (local, needs torch)")


if __name__ == "__main__":
    main()
