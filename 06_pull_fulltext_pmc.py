"""
06_pull_fulltext_pmc.py — Route 2: full text from the PubMed Central Open Access
subset for the Stratum 1 journals, via NCBI E-utilities + the BioC API.

Per journal:
  1. esearch (db=pmc) for articles in the date window with the open-access filter
  2. fetch each article as BioC JSON (full text with section labels)
  3. append to data/pmc/fulltext_pmc.jsonl (one record per article, with
     journal, pmcid, and per-passage section types preserved)

Resumable: fetched PMCIDs are logged in data/pmc/pmcids_done.log and skipped.

Expect modest yields: PMC's psychology coverage is partial (NIH-funded and
OA-deposited work; skews health/clinical and post-2008). That is fine — this
route fills gaps, the coverage audit measures what it adds over Route 1.

Usage:
    export NCBI_API_KEY=...   # optional but recommended (10 req/s vs 3 req/s)
    export NCBI_EMAIL=you@example.com   # NCBI asks for a contact email
    python3 06_pull_fulltext_pmc.py

Endpoints (documented at https://www.ncbi.nlm.nih.gov/books/NBK25501/ and the
BioC API page; PMC permits bulk retrieval ONLY via approved services like
E-utilities/BioC/FTP — do not scrape article pages):
  esearch: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
  BioC:    https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{PMCID}/unicode
"""

import json
import os
import time
from pathlib import Path

import requests

from venues_config import STRATUM1_JOURNALS, YEAR_START, YEAR_END

# --- Configuration ---

API_KEY = os.environ.get("NCBI_API_KEY", "")
EMAIL = os.environ.get("NCBI_EMAIL", "")
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
BIOC_URL = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmcid}/unicode"
SLEEP = 0.11 if API_KEY else 0.34  # stay under 10/s with key, 3/s without

OUT_DIR = Path("data/pmc")
OUT_PATH = OUT_DIR / "fulltext_pmc.jsonl"
DONE_LOG = OUT_DIR / "pmcids_done.log"
COUNTS_PATH = OUT_DIR / "counts_per_journal.csv"


def eutils_params(extra):
    p = {"retmode": "json"}
    if API_KEY:
        p["api_key"] = API_KEY
    if EMAIL:
        p["email"] = EMAIL
    p.update(extra)
    return p


def get_with_retry(url, params=None, max_tries=5):
    for attempt in range(max_tries):
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  request error ({e}); retry {attempt + 1}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed after {max_tries} tries: {url}")


def search_journal(pmc_name):
    """Return all PMCIDs for a journal in the window, OA subset only."""
    term = (f'"{pmc_name}"[Journal] AND ("{YEAR_START}/01/01"[PubDate] : '
            f'"{YEAR_END}/12/31"[PubDate]) AND open access[filter]')
    ids, retstart, retmax = [], 0, 500
    while True:
        params = eutils_params({"db": "pmc", "term": term,
                                "retstart": retstart, "retmax": retmax})
        data = get_with_retry(ESEARCH_URL, params).json()
        result = data.get("esearchresult", {})
        batch = result.get("idlist", [])
        ids.extend(batch)
        total = int(result.get("count", 0))
        retstart += retmax
        time.sleep(SLEEP)
        if retstart >= total or not batch:
            break
    return ids


def fetch_bioc(pmcid):
    """Fetch one article as BioC JSON; return a slim record or None."""
    url = BIOC_URL.format(pmcid=f"PMC{pmcid}" if not str(pmcid).startswith("PMC") else pmcid)
    r = get_with_retry(url)
    try:
        doc = r.json()
    except json.JSONDecodeError:
        return None
    # BioC JSON: list of collections -> documents -> passages (with section_type infons)
    passages = []
    collections = doc if isinstance(doc, list) else [doc]
    for coll in collections:
        for d in coll.get("documents", []):
            for p in d.get("passages", []):
                text = p.get("text", "")
                if not text:
                    continue
                infons = p.get("infons", {})
                passages.append({
                    "section_type": infons.get("section_type") or infons.get("type", ""),
                    "text": text,
                })
    if not passages:
        return None
    return {"pmcid": str(pmcid), "passages": passages}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = set()
    if DONE_LOG.exists():
        done = set(DONE_LOG.read_text().splitlines())
        print(f"Resuming: {len(done)} PMCIDs already fetched")

    counts = []
    with open(OUT_PATH, "a", encoding="utf-8") as out:
        for canonical, spec in STRATUM1_JOURNALS.items():
            journal_ids = []
            for pmc_name in spec["pmc_names"]:
                found = search_journal(pmc_name)
                print(f"{canonical} [{pmc_name}]: {len(found)} OA articles")
                journal_ids.extend(found)
            journal_ids = sorted(set(journal_ids))
            counts.append((canonical, len(journal_ids)))

            fetched = 0
            for pmcid in journal_ids:
                if pmcid in done:
                    continue
                rec = fetch_bioc(pmcid)
                time.sleep(SLEEP)
                with open(DONE_LOG, "a", encoding="utf-8") as dl:
                    dl.write(pmcid + "\n")
                done.add(pmcid)
                if rec is None:
                    continue
                rec["journal"] = canonical
                out.write(json.dumps(rec) + "\n")
                fetched += 1
                if fetched % 100 == 0:
                    out.flush()
                    print(f"  {canonical}: {fetched} fetched")
            print(f"  {canonical}: done ({fetched} new records)")

    with open(COUNTS_PATH, "w", encoding="utf-8") as f:
        f.write("journal,oa_articles_found\n")
        for j, n in counts:
            f.write(f'"{j}",{n}\n')
    print(f"\nDone. Full text -> {OUT_PATH}; per-journal counts -> {COUNTS_PATH}")
    print("De-duplication against the S2ORC pull (by DOI/title) happens in the "
          "next pipeline step, not here — keep both raw files intact.")


if __name__ == "__main__":
    main()
