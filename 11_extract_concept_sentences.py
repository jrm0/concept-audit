"""
11_extract_concept_sentences.py — Pull every sentence mentioning each target
concept from the unified corpus (abstracts + full text).

This is the bridge from corpus to clade profiles. For each concept in the
config CSV, scans:
  - data/abstracts/abstracts_stratum1.jsonl        (OpenAlex abstracts)
  - data/abstracts/abstracts_s2_backfill.jsonl     (S2 backfill, if present)
  - data/s2orc/fulltext_stratum1.jsonl             (S2ORC full text)
and writes one CSV per concept with per-sentence metadata:
    doi, journal, year, source_type, section, sentence

Why the metadata matters (protocol §5.8 and the pilot's text-source finding):
  source_type  abstract vs fulltext — register differs; analyses condition on it
  journal/year — venue and temporal stratification (clade birthdates, prestige)
  section      full-text sentences carry their section header when detectable

Full text is read from PARAGRAPH spans only (excludes references, figure
captions, tables) using S2ORC's annotations.

Outputs:
  data/concepts/sentences_{slug}.csv
  data/concepts/extraction_report.csv   per-concept yields by source/decade

Usage (from code/, venv active):
    python3 11_extract_concept_sentences.py                 # uses concepts_pilot.csv
    python3 11_extract_concept_sentences.py my_concepts.csv
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("data")
OUT_DIR = DATA / "concepts"
DEFAULT_CONFIG = "concepts_pilot.csv"

SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
MIN_LEN, MAX_LEN = 30, 1000


def load_concepts(path):
    concepts = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            terms = [t.strip().lower() for t in
                     (row.get("variants") or "").split("|") if t.strip()]
            anchor = (row.get("retrieval_anchor") or "").strip().lower()
            if anchor:
                terms.append(anchor)
            terms = sorted(set(terms), key=len, reverse=True)
            if not terms:
                continue
            pats = [re.compile(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b",
                               re.I) for t in terms]
            concepts.append({"slug": row["concept_slug"].strip(),
                             "patterns": pats, "terms": terms})
    return concepts


def split_sentences(text):
    for s in SENT_RE.split(text or ""):
        s = s.strip()
        if MIN_LEN <= len(s) <= MAX_LEN:
            yield s


def paragraph_spans(content):
    """Yield (text, section_hint) for paragraph spans of an S2ORC record."""
    text = content.get("text") or ""
    ann = content.get("annotations") or {}
    try:
        paras = json.loads(ann.get("paragraph") or "[]")
    except (json.JSONDecodeError, TypeError):
        paras = []
    try:
        heads = json.loads(ann.get("sectionheader") or "[]")
    except (json.JSONDecodeError, TypeError):
        heads = []
    def as_int(v):
        # S2ORC span offsets are usually ints but occasionally strings/None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    head_points = []
    for h in heads:
        hs, he = as_int(h.get("start")), as_int(h.get("end"))
        if hs is None or he is None:
            continue
        head_points.append((hs, text[hs:he]))
    head_points.sort(key=lambda x: x[0])

    for p in paras:
        s, e = as_int(p.get("start")), as_int(p.get("end"))
        if s is None or e is None:
            continue
        section = ""
        for hs, htext in head_points:
            if hs <= s:
                section = htext.strip()[:40]
            else:
                break
        yield text[s:e], section


def main():
    config = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    concepts = load_concepts(config)
    if not concepts:
        raise SystemExit(f"No concepts loaded from {config}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Concepts: {[c['slug'] for c in concepts]}\n")

    writers, files, seen = {}, {}, {}
    counts = defaultdict(Counter)   # slug -> Counter(source_type)
    decades = defaultdict(Counter)  # slug -> Counter(decade)
    for c in concepts:
        path = OUT_DIR / f"sentences_{c['slug']}.csv"
        fh = open(path, "w", newline="", encoding="utf-8")
        w = csv.writer(fh)
        w.writerow(["doi", "journal", "year", "source_type", "section", "sentence"])
        writers[c["slug"]], files[c["slug"]] = w, fh
        seen[c["slug"]] = set()

    def emit(slug, doi, journal, year, source, section, sentence):
        key = hash(sentence)
        if key in seen[slug]:
            return
        seen[slug].add(key)
        writers[slug].writerow([doi, journal, year, source, section, sentence])
        counts[slug][source] += 1
        try:
            decades[slug][f"{int(year)//10*10}s"] += 1
        except (TypeError, ValueError):
            decades[slug]["unknown"] += 1

    # --- Abstracts ---
    for name in ("abstracts_stratum1.jsonl", "abstracts_s2_backfill.jsonl"):
        path = DATA / "abstracts" / name
        if not path.exists():
            continue
        print(f"Scanning {name} ...")
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                doi = (rec.get("doi") or "").lower()
                journal, year = rec.get("journal", ""), rec.get("year", "")
                for sent in split_sentences(rec.get("abstract")):
                    for c in concepts:
                        if any(p.search(sent) for p in c["patterns"]):
                            emit(c["slug"], doi, journal, year, "abstract", "", sent)
                if n % 20000 == 0:
                    print(f"  {n} abstracts scanned")
        print(f"  done ({n} abstracts)")

    # --- Full text (S2ORC) ---
    ft_path = DATA / "s2orc/fulltext_stratum1.jsonl"
    if ft_path.exists():
        # corpusid -> (journal, year, doi) from the meta file
        meta = {}
        with open(DATA / "s2orc/target_papers_meta.csv", encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 5:
                    meta[p[0]] = (p[1], p[2],
                                  (p[4] or "").replace("https://doi.org/", "").lower())
        print(f"\nScanning full text ({ft_path.name}) ...")
        n = 0
        with open(ft_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                cid = str(rec.get("corpusid", ""))
                journal, year, doi = meta.get(cid, ("", "", ""))
                content = rec.get("content") or {}
                for para, section in paragraph_spans(content):
                    for sent in split_sentences(para):
                        for c in concepts:
                            if any(p.search(sent) for p in c["patterns"]):
                                emit(c["slug"], doi, journal, year,
                                     "fulltext", section, sent)
                if n % 1000 == 0:
                    print(f"  {n} full-text papers scanned")
        print(f"  done ({n} papers)")

    for fh in files.values():
        fh.close()

    # --- Report ---
    print(f"\n{'concept':<22}{'abstract':>10}{'fulltext':>10}{'total':>8}")
    rows = []
    for c in concepts:
        s = c["slug"]
        a, ft = counts[s]["abstract"], counts[s]["fulltext"]
        print(f"{s:<22}{a:>10}{ft:>10}{a+ft:>8}")
        rows.append([s, a, ft, a + ft,
                     ";".join(f"{d}:{n}" for d, n in sorted(decades[s].items()))])
    with open(OUT_DIR / "extraction_report.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept", "n_abstract", "n_fulltext", "n_total", "by_decade"])
        w.writerows(rows)
    print(f"\nSentence files -> {OUT_DIR}/; report -> "
          f"{OUT_DIR / 'extraction_report.csv'}")
    print("Sanity check: concepts below ~200 sentences are underpowered for "
          "clustering (protocol §3 floor). Next: python3 12_clade_profiles.py")


if __name__ == "__main__":
    main()
