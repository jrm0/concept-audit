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


def canon(term):
    """Canonical form: hyphens and spaces are the SAME separator.

    'ego-depletion' and 'ego depletion' are one term, not two. Keeping them
    apart split the focal term's documents across two rows AND collided in
    17, whose filename slug maps both to 'ego_depletion' — so the second
    variant's documents were silently skipped as an existing checkpoint.
    This is the session-8 surface-form lesson at the level of term identity.
    """
    return re.sub(r"[\s\-]+", " ", (term or "").strip().lower())


def load_family_terms():
    terms = {}
    with open(FAMILY_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = canon(row.get("term"))
            if t and t not in terms:
                terms[t] = (row.get("role") or "").strip()
    return terms


def compile_patterns(term_names):
    """Match either separator, so one canonical term captures all variants."""
    pats = {}
    for t in term_names:
        body = re.escape(t).replace(r"\ ", "[-\\s]+").replace(r"\-", "[-\\s]+")
        pats[t] = re.compile(r"\b" + body + r"\b", re.I)
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


ERRATUM_RE = re.compile(
    r"^\W*(erratum(\s+to)?|correction(\s+to)?|corrigendum(\s+to)?)\b[:\s]*",
    re.I)
NONALNUM_RE = re.compile(r"[^a-z0-9]+")


def norm_title(title):
    """Normalized title for CROSS-DATABASE dedupe.

    The same article indexed in PsycInfo, Business Source and SocINDEX gets a
    DIFFERENT accession number in each and often a DOI in only some of them,
    so accession/DOI matching alone leaves triplicates. Titles match. Errata
    and corrections are folded into the original (they are the same document
    for usage purposes, and previously inflated cluster sizes).
    Returns "" when the title is too short to be a safe key.
    """
    t = ERRATUM_RE.sub("", (title or "").strip().strip("'\""))
    t = NONALNUM_RE.sub(" ", t.lower()).strip()
    return t[:120] if len(t) >= 25 else ""


def main():
    family = load_family_terms()
    pats = compile_patterns(list(family) + CONTROL_TERMS)
    docs = {}
    by_title = {}      # normalized title -> canonical key
    stats = Counter()

    def merge(key, dbs):
        """Same document from another database: keep one record, UNION the
        provenance tags (a doc indexed in 3 databases carries all 3)."""
        if dbs:
            docs[key]["dbs"] = sorted(set(docs[key].get("dbs", [])) | set(dbs))
        stats["merged_duplicates"] += 1

    def add_doc(key, year, journal, title, abstract, index_terms, dbs):
        period = period_of(year)
        if period is None or not abstract:
            stats["skipped_no_abstract_or_year"] += 1
            return None
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
            stats["skipped_no_family_term"] += 1
            return None  # no family term actually present
        stats["kept"] += 1
        docs[key] = {
            "doi": key if not key.startswith("AN:") else "",
            "year": year, "period": period,
            "journal": journal, "title": title,
            "index_terms": index_terms,
            "dbs": sorted(set(dbs)),   # source database(s) — migration tracking
            "terms": sorted(fam_terms),
            "roles": sorted({family[t] for t in fam_terms}),
            "mention_sentences": {t: v[:6] for t, v in mentions.items()},
        }
        return key

    # 1. All RIS files, recursively.
    #    DB tags are ground truth for provenance; the file a record came in
    #    is only a convenience. Dedupe order: DOI, then normalized title.
    ris_files = sorted(RIS_ROOT.rglob("*.ris"))
    for f in ris_files:
        recs = parse_ris(f)
        n_before, merged_before = len(docs), stats["merged_duplicates"]
        for r in recs:
            doi = (r.get("DO", [""])[0] or "").lower().strip()
            doi = doi.replace("https://doi.org/", "").rstrip(".")
            title = r.get("TI", [""])[0]
            nt = norm_title(title)
            dbs = [v.strip() for v in r.get("DB", []) if v.strip()]

            if doi and doi in docs:               # same DOI
                merge(doi, dbs)
                continue
            if nt and nt in by_title:             # same title, other database
                merge(by_title[nt], dbs)
                continue

            an = (r.get("AN", [""])[0] or "").strip()
            key = doi or (f"AN:{an}" if an else f"T:{nt or title[:80].lower()}")
            if key in docs:
                merge(key, dbs)
                continue
            made = add_doc(key, year_of(r),
                           (r.get("JO", [""])[0] or r.get("T2", [""])[0]
                            or "").strip(),
                           title, (r.get("AB", [""])[0] or ""),
                           [k.strip() for k in r.get("KW", []) if k.strip()],
                           dbs)
            if made and nt:
                by_title[nt] = made
        print(f"{f.name}: +{len(docs) - n_before} docs, "
              f"{stats['merged_duplicates'] - merged_before} merged "
              f"(total {len(docs)})")

    # 2. OpenAlex-derived family works, if present (adds field metadata later)
    for jf in sorted(FAMILY_DIR.glob("ego_depletion*works.jsonl")):
        n_before = len(docs)
        for line in open(jf, encoding="utf-8"):
            try:
                w = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = (w.get("doi") or "").lower().replace(
                "https://doi.org/", "").rstrip(".")
            nt = norm_title(w.get("title", ""))
            if doi and doi in docs:
                merge(doi, ["OpenAlex"])
                continue
            if nt and nt in by_title:
                merge(by_title[nt], ["OpenAlex"])
                continue
            key = doi or (f"T:{nt}" if nt else None)
            if not key or key in docs:
                continue
            made = add_doc(key, w.get("year"), w.get("journal", ""),
                           w.get("title", ""), w.get("abstract", ""),
                           w.get("index_terms", []), ["OpenAlex"])
            if made and nt:
                by_title[nt] = made
        print(f"{jf.name}: +{len(docs) - n_before} docs (total {len(docs)})")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for d in docs.values():
            f.write(json.dumps(d) + "\n")

    print(f"\nRECORD ACCOUNTING: kept {stats['kept']}, "
          f"merged duplicates {stats['merged_duplicates']}, "
          f"no family term {stats['skipped_no_family_term']}, "
          f"no abstract/year {stats['skipped_no_abstract_or_year']}")

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

    dbc = Counter()
    for d in docs.values():
        for db in d.get("dbs", []):
            dbc[db] += 1
    if dbc:
        print("\nDATABASE PROVENANCE (a record may carry several):")
        for db, c in dbc.most_common():
            print(f"  {db}: {c}")

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
