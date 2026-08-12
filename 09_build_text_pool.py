"""
09_build_text_pool.py — Unify all text sources into one corpus index + coverage report.

The corpus is FEDERATED by design: OpenAlex abstracts where publishers deposit
them (SAGE, APA), S2ORC/PMC full text where they don't (Elsevier, Springer).
This script merges the sources, deduplicates by DOI (preferring full text over
abstract), and reports the COMBINED per-journal coverage — which is the real
answer to "how do we fill the abstract gaps."

Sources:
  data/abstracts/abstracts_stratum1.jsonl   (OpenAlex abstracts; doi-keyed)
  data/s2orc/fulltext_ids.txt + target_papers_meta.csv  (S2ORC full text)
  data/s2orc/s2_coverage_by_journal.csv     (per-journal OpenAlex roster = denominator)

Outputs:
  data/corpus_index.jsonl        one row per unique paper: doi, journal, year,
                                 has_fulltext, has_abstract, best_source
  data/corpus_coverage.csv       per-journal: roster, full-text, abstract-only,
                                 any-text, combined %

Usage (from code/, venv active):
    python3 09_build_text_pool.py
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

DATA = Path("data")


def norm_doi(d):
    return (d or "").replace("https://doi.org/", "").strip().lower()


def main():
    # 1. S2ORC meta: corpusid -> (journal, year, doi)
    meta = {}
    meta_path = DATA / "s2orc/target_papers_meta.csv"
    with open(meta_path, encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 5:
                meta[p[0]] = {"journal": p[1], "year": p[2], "doi": norm_doi(p[4])}

    # 2. Which corpusids have full text
    ft_ids = set(x.strip() for x in open(DATA / "s2orc/fulltext_ids.txt")
                 if x.strip())

    # 3. Roster denominator per journal (OpenAlex roster)
    roster = {}
    with open(DATA / "s2orc/s2_coverage_by_journal.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            roster[row["journal"]] = int(row["openalex_roster"])

    # Build the unified index keyed by DOI
    index = {}  # doi -> {journal, year, has_fulltext, has_abstract}

    # 3a. Full-text papers (S2ORC)
    ft_no_doi = 0
    for cid in ft_ids:
        m = meta.get(cid)
        if not m:
            continue
        doi = m["doi"]
        if not doi:
            ft_no_doi += 1
            continue
        rec = index.setdefault(doi, {"journal": m["journal"], "year": m["year"],
                                     "has_fulltext": False, "has_abstract": False})
        rec["has_fulltext"] = True

    # 3b. Abstracts (OpenAlex primary + Semantic Scholar backfill, if present)
    abstract_files = [DATA / "abstracts/abstracts_stratum1.jsonl",
                      DATA / "abstracts/abstracts_s2_backfill.jsonl"]
    for path in abstract_files:
        if not path.exists():
            continue
        n = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    a = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doi = norm_doi(a.get("doi"))
                if not doi:
                    continue
                rec = index.setdefault(doi, {"journal": a.get("journal", ""),
                                             "year": str(a.get("year") or ""),
                                             "has_fulltext": False,
                                             "has_abstract": False})
                rec["has_abstract"] = True
                n += 1
        print(f"  read {n} abstracts from {path.name}")

    # Write index
    with open(DATA / "corpus_index.jsonl", "w", encoding="utf-8") as out:
        for doi, r in index.items():
            best = "fulltext" if r["has_fulltext"] else "abstract"
            out.write(json.dumps({"doi": doi, "journal": r["journal"],
                                  "year": r["year"],
                                  "has_fulltext": r["has_fulltext"],
                                  "has_abstract": r["has_abstract"],
                                  "best_source": best}) + "\n")

    # Per-journal combined coverage
    per = defaultdict(lambda: {"ft": 0, "ab_only": 0})
    for r in index.values():
        j = r["journal"]
        if r["has_fulltext"]:
            per[j]["ft"] += 1
        elif r["has_abstract"]:
            per[j]["ab_only"] += 1

    rows = []
    print(f"{'journal':<52}{'roster':>7}{'fulltxt':>8}{'abs-only':>9}"
          f"{'any':>7}{'%':>6}")
    for j in sorted(per, key=lambda x: -(roster.get(x, 0))):
        ft, ab = per[j]["ft"], per[j]["ab_only"]
        any_text = ft + ab
        r = roster.get(j, 0)
        pct = any_text / r if r else 0
        print(f"{j:<52}{r:>7}{ft:>8}{ab:>9}{any_text:>7}{pct:>6.0%}")
        rows.append((j, r, ft, ab, any_text, f"{pct:.3f}"))
    tot_r = sum(roster.get(j, 0) for j in per)
    tot_any = sum(v["ft"] + v["ab_only"] for v in per.values())
    print(f"{'TOTAL':<52}{tot_r:>7}{'':>8}{'':>9}{tot_any:>7}"
          f"{tot_any/tot_r if tot_r else 0:>6.0%}")

    with open(DATA / "corpus_coverage.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["journal", "roster", "fulltext", "abstract_only",
                    "any_text", "combined_share"])
        w.writerows(rows)

    print(f"\nIndex -> data/corpus_index.jsonl ({len(index)} unique papers)")
    print(f"Coverage -> data/corpus_coverage.csv")
    if ft_no_doi:
        print(f"Note: {ft_no_doi} full-text papers had no DOI in meta "
              f"(counted nowhere above; recoverable via title match if needed).")
    print("PMC full text (data/pmc/, ~2.4k papers) not yet merged — it lacks "
          "DOIs in the stored records; fold in later via a pmcid->doi map if "
          "its coverage adds meaningfully over S2ORC.")


if __name__ == "__main__":
    main()
