"""
08_build_construct_corpus.py — Fill the usage corpora for the OSC-99
constructs so each has an embeddable usage distribution.

WHY. Coverage check against our existing 53k-abstract pool: of the 92
unique primary constructs, 32 have >=100 documents, 20 have 30-99, and 40
have <30 — the thin ones are hyper-specific labels ("intermixed-blocked
effect", "value heuristic"). A usage DISTRIBUTION needs documents. This
script pulls additional abstracts per thin construct from OpenAlex
(title+abstract search, Psychology field filter, 5-year periods 1995-2024,
capped per period), in the style of 13's stratified sampler.

NOTE ON SPARSITY AS DATA. Constructs that stay thin even after this pull
are not failures: "the field barely uses this term outside the original
paper" is itself a measurable property (construct isolation), and a
candidate predictor for Paper 2. We record the post-pull document count
for every construct rather than dropping thin ones.

Reads:  data/final_constructs.csv (from 07)
        ../data/abstracts/*.jsonl (existing pool, counted first)
Writes: ../data/construct_corpus/{slug}.jsonl   (one per construct)
        ../data/construct_corpus/_coverage.csv  (docs per construct/source)
Resumable: constructs with an existing .jsonl are skipped.
Runs LOCALLY (needs internet). ~1-2 s/request with backoff; the full pull
is a few hundred requests.

Usage (from code/osc_coding/):
    python3 08_build_construct_corpus.py            # thin constructs only
    python3 08_build_construct_corpus.py --all      # every construct
"""

import csv
import glob
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

API = "https://api.openalex.org/works"
MAILTO = "jrmorgan3@gmail.com"
# OpenAlex now uses CREDIT-based limits: list requests cost 10 credits, free
# budget 100k credits/day (and in practice exhaustion arrives much sooner).
# Mitigations: TWO windows per construct instead of six periods (3x cheaper;
# year recorded per doc, so periods are reconstructed locally), and API-key
# support — academics can request raised limits free: support@openalex.org.
API_KEY = os.environ.get("OPENALEX_API_KEY", "")
FIELD_PSYCH = "fields/32"          # OpenAlex Psychology field id (as in 13)
OUT_DIR = Path("../data/construct_corpus")
POOL_GLOB = "../data/abstracts/*.jsonl"
FINAL = Path("data/final_constructs.csv")
PERIODS = [(1995, 2009), (2010, 2024)]   # two windows, per-page 200 each
CAP_PER_PERIOD = 200
THIN_THRESHOLD = 100               # pool docs below this -> pull


def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def pat_of(term):
    body = re.escape(term.lower()).replace(r"\ ", "[-\\s]+") \
        .replace(r"\-", "[-\\s]+")
    return re.compile(r"\b" + body + r"\b", re.I)


def inverted_to_text(inv):
    if not inv:
        return ""
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))


class QuotaExhausted(Exception):
    """Persistent 429: the OpenAlex daily quota (resets midnight UTC) is
    spent — the same wall 13 hit. Not worth spinning; save and exit."""


def get(params, retries=7):
    import requests
    params = dict(params, mailto=MAILTO)
    if API_KEY:
        params["api_key"] = API_KEY
    for i in range(retries):
        r = requests.get(API, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        retry_after = r.headers.get("Retry-After")
        wait = (int(retry_after) if retry_after and retry_after.isdigit()
                else min(90, 3 * 2 ** i))
        print(f"    HTTP {r.status_code}; wait {wait}s")
        time.sleep(min(wait, 120))
    if r.status_code == 429:
        raise QuotaExhausted
    raise RuntimeError(f"OpenAlex gave up (last status {r.status_code})")


def main():
    pull_all = "--all" in sys.argv
    constructs = set()
    for r in csv.DictReader(open(FINAL, encoding="utf-8")):
        for i in (1, 2, 3, 4):
            c = (r.get(f"construct_{i}") or "").strip().lower()
            if c:
                constructs.add(c)
    print(f"{len(constructs)} unique constructs across all sets")

    # count existing pool coverage
    pats = {c: pat_of(c) for c in constructs}
    pool_count = Counter()
    for f in glob.glob(POOL_GLOB):
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (d.get("title") or "") + ". " + (d.get("abstract") or "")
            for c, p in pats.items():
                if p.search(text):
                    pool_count[c] += 1

    todo = sorted(c for c in constructs
                  if pull_all or pool_count[c] < THIN_THRESHOLD)
    print(f"pulling for {len(todo)} constructs "
          f"({'all' if pull_all else f'pool < {THIN_THRESHOLD} docs'})")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cov_rows = []
    quota_hit = False
    for ci, c in enumerate(todo):
        if quota_hit:
            cov_rows.append([c, pool_count[c], 0, "PENDING (quota)"])
            continue
        out = OUT_DIR / f"{slug(c)}.jsonl"
        if out.exists():
            n = sum(1 for _ in open(out, encoding="utf-8"))
            cov_rows.append([c, pool_count[c], n, "cached"])
            continue
        print(f"[{ci+1}/{len(todo)}] {c} (pool={pool_count[c]})")
        got, seen = [], set()
        for y0, y1 in PERIODS:
            try:
                js = get({"filter": f"title_and_abstract.search:{c},"
                                    f"primary_topic.field.id:{FIELD_PSYCH},"
                                    f"publication_year:{y0}-{y1}",
                          "per-page": CAP_PER_PERIOD,
                          "sort": "cited_by_count:desc",
                          "select": "id,doi,title,publication_year,"
                                    "abstract_inverted_index,primary_location"})
            except QuotaExhausted:
                print("    DAILY QUOTA EXHAUSTED — saving progress and "
                      "stopping. Re-run after midnight UTC; cached "
                      "constructs are skipped automatically.")
                quota_hit = True
                got = None          # do not write a partial file
                break
            for w in js.get("results", []):
                doi = (w.get("doi") or "").replace("https://doi.org/", "")
                key = doi or (w.get("title") or "")[:80].lower()
                if key in seen:
                    continue
                seen.add(key)
                ab = inverted_to_text(w.get("abstract_inverted_index"))
                if len(ab) < 200:
                    continue
                src = (w.get("primary_location") or {}).get("source") or {}
                got.append({"doi": doi, "year": w.get("publication_year"),
                            "title": w.get("title") or "",
                            "journal": src.get("display_name", ""),
                            "abstract": ab, "construct": c})
            time.sleep(0.3)
        if got is None:                     # quota hit mid-construct
            cov_rows.append([c, pool_count[c], 0, "PENDING (quota)"])
            continue
        with open(out, "w", encoding="utf-8") as f:
            for d in got:
                f.write(json.dumps(d) + "\n")
        print(f"    -> {len(got)} docs")
        cov_rows.append([c, pool_count[c], len(got), "pulled"])

    for c in sorted(constructs):
        if c not in todo:
            cov_rows.append([c, pool_count[c], 0, "pool sufficient"])
    with open(OUT_DIR / "_coverage.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["construct", "pool_docs", "pulled_docs", "status"])
        w.writerows(sorted(cov_rows))
    print(f"\ncoverage table -> {OUT_DIR / '_coverage.csv'}")
    print("Next: embedding script (per-construct masked-target document "
          "vectors, 17-style) — ask Fable once corpora are in.")


if __name__ == "__main__":
    main()
