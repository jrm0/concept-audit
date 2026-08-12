"""
05_pull_fulltext_s2orc.py — Route 1: full text from S2ORC via the Semantic Scholar
Datasets API, filtered to the Stratum 1 journal list.

Two phases (run the same script twice, or let it run both in sequence):

  PHASE A (fast, ~minutes): use the Graph API bulk search to collect the corpusId
  of every paper published in the Stratum 1 journals, 1980-2025.
    -> data/s2orc/target_corpusids.txt   (one id per line)
    -> data/s2orc/target_papers_meta.csv (id, title, year, venue — for the audit)

  PHASE B (slow, disk/bandwidth heavy): download the S2ORC bulk dataset shard by
  shard, keep only records whose corpusid is in the target set, delete each shard
  after filtering. Resumable — completed shards are logged and skipped on re-run.
    -> data/s2orc/fulltext_stratum1.jsonl (one JSON record per matched paper)
    -> data/s2orc/shards_done.log

BANDWIDTH WARNING: Phase B streams the *entire* S2ORC dataset (tens of shards;
total transfer is large — plan for an overnight run on a good connection).
Only ~a few thousand records will be kept; everything else is discarded on the
fly, so disk use stays at ~one shard at a time plus the filtered output.

Usage:
    export S2_API_KEY=...           # required — Datasets API needs a key
    python3 05_pull_fulltext_s2orc.py phase_a
    python3 05_pull_fulltext_s2orc.py phase_b
    python3 05_pull_fulltext_s2orc.py            # both phases in sequence

Endpoints used (validate against https://api.semanticscholar.org/api-docs/ if
either 404s — the Datasets API has changed shape before):
  Graph bulk search: GET /graph/v1/paper/search/bulk
  Datasets:          GET /datasets/v1/release/latest/dataset/s2orc
"""

import csv
import gzip
import json
import os
import sys
import time
from pathlib import Path

import requests

from venues_config import STRATUM1_JOURNALS, YEAR_START, YEAR_END

CONTACT_EMAIL = "jrmorgan3@gmail.com"  # OpenAlex polite pool

# --- Configuration ---

API_KEY = os.environ.get("S2_API_KEY", "")
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}
BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
DATASETS_URL = "https://api.semanticscholar.org/datasets/v1/release/latest/dataset/s2orc"

OUT_DIR = Path("data/s2orc")
IDS_PATH = OUT_DIR / "target_corpusids.txt"
META_PATH = OUT_DIR / "target_papers_meta.csv"
FULLTEXT_PATH = OUT_DIR / "fulltext_stratum1.jsonl"
DONE_LOG = OUT_DIR / "shards_done.log"


def get_with_retry(url, params=None, stream=False, max_tries=6):
    for attempt in range(max_tries):
        try:
            r = requests.get(url, params=params, headers=HEADERS,
                             stream=stream, timeout=120)
            if r.status_code == 429:
                wait = 2 ** attempt
                print(f"  rate-limited; sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  request error ({e}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_tries} tries: {url}")


def phase_a():
    """v2 (July 2026): OpenAlex roster -> S2 DOI-batch mapping.

    The original venue-string search proved unreliable: 05c discovery showed
    many S2 records carry an EMPTY venue string (JESP 1980s-2000s, PSPB,
    Developmental Psychology, SPPS), which no name variant can retrieve.
    New design: OpenAlex (entity-resolved journals) provides the complete
    per-journal DOI roster; S2's batch endpoint maps DOIs -> corpusIds.
    No venue matching anywhere. Also writes per-journal S2 coverage stats.
    """
    if not API_KEY:
        sys.exit("ERROR: set S2_API_KEY (batch/Datasets endpoints need a key).")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ids = set()
    meta_rows = []
    coverage = []

    for canonical in STRATUM1_JOURNALS:
        print(f"\n=== {canonical} ===")
        # 1. Resolve the journal entity on OpenAlex (verify the printed match!)
        src_resp = requests.get("https://api.openalex.org/sources",
                                params={"search": canonical, "per-page": 1,
                                        "mailto": CONTACT_EMAIL}, timeout=30)
        src_resp.raise_for_status()
        results = src_resp.json().get("results", [])
        if not results:
            print("  !! OpenAlex source not found — skipping")
            continue
        src = results[0]
        src_id = src["id"].rsplit("/", 1)[-1]
        print(f"  OpenAlex match: {src.get('display_name')} "
              f"(ISSN {','.join(src.get('issn') or [])})")
        time.sleep(0.3)

        # 2. Full DOI roster via cursor paging
        dois = {}  # doi -> (year, title)
        cursor = "*"
        flt = (f"primary_location.source.id:{src_id},"
               f"from_publication_date:{YEAR_START}-01-01,"
               f"to_publication_date:{YEAR_END}-12-31,type:article")
        while cursor:
            w = requests.get("https://api.openalex.org/works",
                             params={"filter": flt, "per-page": 200,
                                     "cursor": cursor,
                                     "select": "doi,publication_year,title",
                                     "mailto": CONTACT_EMAIL}, timeout=30)
            w.raise_for_status()
            data = w.json()
            for work in data.get("results", []):
                if work.get("doi"):
                    doi = work["doi"].replace("https://doi.org/", "").lower()
                    dois[doi] = (work.get("publication_year"),
                                 (work.get("title") or "").replace("\t", " "))
            cursor = data.get("meta", {}).get("next_cursor")
            time.sleep(0.25)
        print(f"  OpenAlex roster: {len(dois)} articles with DOIs")

        # 3. Map DOIs -> S2 corpusIds via batch endpoint (500 per call)
        doi_list = sorted(dois)
        found = 0
        for i in range(0, len(doi_list), 500):
            chunk = doi_list[i:i + 500]
            for attempt in range(6):
                br = requests.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": "corpusId,isOpenAccess"},
                    json={"ids": [f"DOI:{d}" for d in chunk]},
                    headers=HEADERS, timeout=120)
                if br.status_code == 429 or br.status_code >= 500:
                    time.sleep(2 ** (attempt + 1))
                    continue
                br.raise_for_status()
                break
            else:
                raise RuntimeError(f"S2 batch failed repeatedly at chunk {i}")
            for doi, rec in zip(chunk, br.json()):
                if rec is None or rec.get("corpusId") is None:
                    continue
                cid = rec["corpusId"]
                if cid in ids:
                    continue
                ids.add(cid)
                found += 1
                year, title = dois[doi]
                meta_rows.append((cid, canonical, year,
                                  rec.get("isOpenAccess"), doi, title))
            time.sleep(1.0)
            if (i // 500) % 10 == 0 and i > 0:
                print(f"    mapped {i + len(chunk)}/{len(doi_list)} DOIs...")
        pct = found / len(dois) if dois else 0
        print(f"  in S2: {found}/{len(dois)} ({pct:.0%})")
        coverage.append((canonical, len(dois), found, f"{pct:.2f}"))

    IDS_PATH.write_text("\n".join(str(i) for i in sorted(ids)), encoding="utf-8")
    with open(META_PATH, "w", encoding="utf-8") as f:
        f.write("corpusid\tjournal\tyear\tis_open_access\tdoi\ttitle\n")
        for row in meta_rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    cov_path = OUT_DIR / "s2_coverage_by_journal.csv"
    with open(cov_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["journal", "openalex_roster", "in_s2", "s2_share"])
        w.writerows(coverage)
    print(f"\nPhase A done: {len(ids)} unique corpusIds -> {IDS_PATH}")
    print(f"Metadata -> {META_PATH}; S2 coverage table -> {cov_path}")
    print("The (roster - in_s2) gap per journal is the precise TDM ask.")


def phase_b():
    """Stream S2ORC shards; keep records whose corpusid is in the target set."""
    if not IDS_PATH.exists():
        sys.exit("ERROR: run phase_a first (no target_corpusids.txt found).")
    targets = set(int(line) for line in IDS_PATH.read_text().splitlines() if line.strip())
    print(f"Loaded {len(targets)} target corpusIds")

    done = set()
    if DONE_LOG.exists():
        done = set(DONE_LOG.read_text().splitlines())
        print(f"Resuming: {len(done)} shards already processed")

    def fetch_shard_urls():
        """Shard URLs are PRE-SIGNED and expire (~1h). Re-fetch for fresh ones.
        Shard IDs (filename stems) are stable across refreshes."""
        info = get_with_retry(DATASETS_URL).json()
        release = info.get("releaseId") or info.get("release_id") or "unknown"
        urls = {u.split("?")[0].rsplit("/", 1)[-1]: u
                for u in info.get("files", [])}
        return release, urls

    release, shard_urls = fetch_shard_urls()
    print(f"S2ORC release {release}: {len(shard_urls)} shards")
    print("(Record this release id in the protocol retrieval log.)")

    kept_total = 0
    with open(FULLTEXT_PATH, "a", encoding="utf-8") as out:
        todo = [s for s in sorted(shard_urls) if s not in done]
        for i, shard_id in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] {shard_id}")
            tmp = OUT_DIR / "shard.tmp.gz"
            # Download with expired-signature handling: a 400/403 means the
            # pre-signed URL aged out — refresh the list and retry, don't
            # hammer the dead URL.
            for attempt in range(5):
                url = shard_urls[shard_id]
                try:
                    r = requests.get(url, stream=True, timeout=300)
                    if r.status_code in (400, 403):
                        print("  pre-signed URL expired — refreshing shard list")
                        _, shard_urls = fetch_shard_urls()
                        continue
                    r.raise_for_status()
                    with r, open(tmp, "wb") as fh:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                    break
                except requests.RequestException as e:
                    wait = 2 ** (attempt + 1)
                    print(f"  download error ({e}); retrying in {wait}s")
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Could not download shard {shard_id} "
                                   f"after repeated refreshes.")
            kept = 0
            with gzip.open(tmp, "rt", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("corpusid") in targets:
                        out.write(line if line.endswith("\n") else line + "\n")
                        kept += 1
            tmp.unlink()
            out.flush()
            with open(DONE_LOG, "a", encoding="utf-8") as dl:
                dl.write(shard_id + "\n")
            kept_total += kept
            print(f"  kept {kept} (running total {kept_total})")

    print(f"\nPhase B done: {kept_total} full-text records -> {FULLTEXT_PATH}")
    print("Note for the coverage audit: kept/target ratio = S2ORC full-text "
          "coverage of Stratum 1; the gap is what Routes 2-4 must fill.")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "both"
    if arg in ("phase_a", "both"):
        phase_a()
    if arg in ("phase_b", "both"):
        phase_b()
