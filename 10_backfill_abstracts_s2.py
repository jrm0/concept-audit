"""
10_backfill_abstracts_s2.py — Recover abstracts Semantic Scholar has but OpenAlex lacks.

Why: OpenAlex abstract coverage is a publisher signature — SAGE/APA deposit
(90-100%), Elsevier/Springer withhold (Cognition 14%, JESP 8%, Memory &
Cognition 9%, Cognitive Psychology 10%, Psychonomic B&R 15%). Semantic Scholar
has its own publisher agreements and often holds abstracts OpenAlex doesn't.
This tries the automated route before any manual (Zotero) work.

Candidates = papers in the S2ORC meta (so they have DOIs and are in-scope) that
currently have NEITHER full text NOR an OpenAlex abstract, per corpus_index.jsonl.
Fetches in batches of 500 via the S2 batch endpoint (same one used in 05 phase_a).

Outputs:
  data/abstracts/abstracts_s2_backfill.jsonl   same schema as the OpenAlex file
  prints per-journal recovery so you can see which gaps closed

Then re-run 09_build_text_pool.py to fold these into the index.

Usage (from code/, venv active; S2_API_KEY set):
    python3 10_backfill_abstracts_s2.py
"""

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import requests

DATA = Path("data")
OUT_PATH = DATA / "abstracts/abstracts_s2_backfill.jsonl"
DONE_LOG = DATA / "abstracts/s2_backfill_done.log"
API_KEY = os.environ.get("S2_API_KEY", "")
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}
BATCH = 500


def norm_doi(d):
    return (d or "").replace("https://doi.org/", "").strip().lower()


def main():
    if not API_KEY:
        raise SystemExit("Set S2_API_KEY first (the batch endpoint needs it).")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1. Which DOIs already have text?
    have_text = set()
    idx_path = DATA / "corpus_index.jsonl"
    if not idx_path.exists():
        raise SystemExit("Run 09_build_text_pool.py first.")
    with open(idx_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("has_fulltext") or r.get("has_abstract"):
                have_text.add(r["doi"])

    # 2. Candidates from the S2ORC meta (doi + journal + year known)
    candidates = {}
    with open(DATA / "s2orc/target_papers_meta.csv", encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            doi = norm_doi(p[4])
            if not doi or doi in have_text:
                continue
            candidates[doi] = {"journal": p[1], "year": p[2], "title": p[5]}

    done = set()
    if DONE_LOG.exists():
        done = set(x.strip() for x in open(DONE_LOG) if x.strip())
    todo = sorted(d for d in candidates if d not in done)
    print(f"Candidates missing text: {len(candidates)}; "
          f"already attempted: {len(done)}; to fetch: {len(todo)}")
    by_journal_try = Counter(candidates[d]["journal"] for d in todo)

    recovered = defaultdict(int)
    n_new = 0
    with open(OUT_PATH, "a", encoding="utf-8") as out, \
         open(DONE_LOG, "a", encoding="utf-8") as dl:
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            for attempt in range(6):
                r = requests.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": "abstract,title,year"},
                    json={"ids": [f"DOI:{d}" for d in chunk]},
                    headers=HEADERS, timeout=120)
                if r.status_code == 429 or r.status_code >= 500:
                    wait = 2 ** (attempt + 1)
                    print(f"  transient {r.status_code}; waiting {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                break
            else:
                raise RuntimeError(f"S2 batch failed at offset {i}")

            for doi, rec in zip(chunk, r.json()):
                dl.write(doi + "\n")
                if not rec or not rec.get("abstract"):
                    continue
                c = candidates[doi]
                out.write(json.dumps({
                    "doi": doi, "journal": c["journal"], "year": c["year"],
                    "title": rec.get("title") or c["title"],
                    "abstract": rec["abstract"], "source": "semanticscholar",
                }) + "\n")
                recovered[c["journal"]] += 1
                n_new += 1
            out.flush()
            print(f"  {min(i + BATCH, len(todo))}/{len(todo)} attempted; "
                  f"{n_new} abstracts recovered")
            time.sleep(1.2)  # S2 key limit is 1 req/sec cumulative — stay under

    print(f"\nRecovered {n_new} abstracts -> {OUT_PATH}")
    print(f"{'journal':<52}{'tried':>7}{'got':>6}{'rate':>7}")
    for j, tried in by_journal_try.most_common():
        got = recovered.get(j, 0)
        print(f"{j:<52}{tried:>7}{got:>6}{got/tried if tried else 0:>7.0%}")
    print("\nNext: re-run 09_build_text_pool.py to fold these into the index "
          "(it reads only the OpenAlex file today — see the note there).")


if __name__ == "__main__":
    main()
