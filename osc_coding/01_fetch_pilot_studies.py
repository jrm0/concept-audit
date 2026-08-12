"""
01_fetch_pilot_studies.py (v2) — Set up the OSC RP:P construct-coding pilot.

v2 fixes (July 2026):
  1. FRAME: v1 sampled from all 167 rows with a title. The official RP:P analyses
     select completed replications as rows where BOTH T_pval_USE (O) and (R) are
     non-missing ("We have 99 studies for which p-values and effect sizes could
     be calculated" — RPP_figures.R, CenterForOpenScience/rpp). v2 applies that
     filter before sampling.
  2. ABSTRACTS: Semantic Scholar returns null abstracts for many paywalled
     psychology journals (publisher licensing). v2 tries three sources in order:
     OpenAlex (abstract_inverted_index) -> Crossref -> Semantic Scholar, and
     reports per study which source worked.

What it does:
  1. Downloads rpp_data.csv (RP:P master file) from OSF
  2. Filters to the ~99 completed replication studies (official frame)
  3. Draws a seeded, journal-stratified sample of 10 studies
  4. Fetches each study's abstract (OpenAlex -> Crossref -> Semantic Scholar)
  5. Writes:
       data/rpp_studies_all.csv        — the completed-replication frame
       data/pilot_10_studies.csv       — the pilot sample (with abstract source)
       data/pilot_texts/study_NN.txt          — UNMASKED (title + authors + abstract)
       data/pilot_texts/study_NN_masked.txt   — MASKED (abstract only)
       data/coding_sheet_pilot.csv     — empty coding sheet

Usage (from code/osc_coding/, venv active):
    python3 01_fetch_pilot_studies.py

NOTE: re-running overwrites previous outputs. The v1 sample came from the wrong
frame — discard any coding done on it.
"""

import csv
import difflib
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

# --- Configuration ---

RPP_DATA_URL = "https://osf.io/fgjvw/download"  # rpp_data.csv (OSF project ezcuj)
SEED = 42
PILOT_N = 10
DATA_DIR = Path("data")
TEXT_DIR = DATA_DIR / "pilot_texts"

CONTACT_EMAIL = "jrmorgan3@gmail.com"  # polite-pool identifier for OpenAlex/Crossref
S2_API_KEY = os.environ.get("S2_API_KEY", "")

OPENALEX_URL = "https://api.openalex.org/works"
CROSSREF_URL = "https://api.crossref.org/works"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

COLUMN_HINTS = {
    "study_num": re.compile(r"study\s*num", re.I),
    "title": re.compile(r"study\s*title.*\(o\)", re.I),
    "authors": re.compile(r"^authors?\s*\(o\)", re.I),
    "journal": re.compile(r"^journal\s*\(o\)", re.I),
    # Official completed-replication frame columns (headers appear R-mangled
    # in the CSV, e.g. "T_pval_USE..O.")
    "pval_use_o": re.compile(r"T_pval_USE.*O", re.I),
    "pval_use_r": re.compile(r"T_pval_USE.*R", re.I),
}


def find_column(fieldnames, pattern):
    for name in fieldnames:
        if pattern.search(name or ""):
            return name
    return None


def clean_title(t):
    """Normalize known CSV quirks (e.g. underscores standing in for hyphens)."""
    return re.sub(r"\s+", " ", t.replace("_", "-")).strip().rstrip(".")


def norm_for_match(t):
    # Strip spaces too: rpp_data.csv titles have spacing quirks
    # ("Single-and", "(un) accomplished") that must not block a match
    return re.sub(r"[^a-z0-9]", "", t.lower())


def titles_match(a, b):
    na, nb = norm_for_match(a), norm_for_match(b)
    if not na or not nb:
        return False
    if na[:60] == nb[:60] or na in nb or nb in na:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.92


# --- Abstract sources (tried in order) ---

def get_with_backoff(url, params, headers=None, tries=5):
    """GET with escalating waits on 429 — rate limits are transient, not fatal."""
    r = None
    for attempt in range(tries):
        r = requests.get(url, params=params, headers=headers or {}, timeout=30)
        if r.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"    rate-limited; waiting {wait}s")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def from_openalex(title):
    params = {"filter": f"title.search:{title}", "per-page": 3,
              "mailto": CONTACT_EMAIL}
    r = get_with_backoff(OPENALEX_URL, params)
    for work in r.json().get("results", []):
        if not titles_match(title, work.get("title") or ""):
            continue
        inv = work.get("abstract_inverted_index")
        if not inv:
            continue
        return reconstruct_inverted(inv), work.get("doi") or ""
    return None, None


def from_crossref(title):
    params = {"query.bibliographic": title, "rows": 3, "mailto": CONTACT_EMAIL}
    r = get_with_backoff(CROSSREF_URL, params)
    for item in r.json().get("message", {}).get("items", []):
        cand = (item.get("title") or [""])[0]
        if not titles_match(title, cand):
            continue
        abstract = item.get("abstract")
        if not abstract:
            continue
        abstract = re.sub(r"<[^>]+>", " ", abstract)  # strip JATS tags
        abstract = re.sub(r"\s+", " ", abstract).strip()
        return abstract, item.get("DOI") or ""
    return None, None


def from_s2(title):
    headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
    params = {"query": title, "fields": "title,abstract,externalIds", "limit": 3}
    r = get_with_backoff(S2_SEARCH_URL, params, headers=headers)
    for p in r.json().get("data", []) or []:
        if not titles_match(title, p.get("title") or ""):
            continue
        if p.get("abstract"):
            doi = (p.get("externalIds") or {}).get("DOI", "")
            return p["abstract"], doi
    return None, None


def reconstruct_inverted(inv):
    positions = {}
    for word, poss in inv.items():
        for p in poss:
            positions[p] = word
    return " ".join(positions[i] for i in sorted(positions))


def resolve_doi(title):
    """Crossref bibliographic search tolerates mangled titles well —
    use it to recover the DOI even when direct abstract lookups fail."""
    params = {"query.bibliographic": title, "rows": 3, "mailto": CONTACT_EMAIL}
    r = get_with_backoff(CROSSREF_URL, params)
    for item in r.json().get("message", {}).get("items", []):
        cand = (item.get("title") or [""])[0]
        if titles_match(title, cand):
            return item.get("DOI")
    return None


def abstract_by_doi(doi):
    """Fetch abstract for a known DOI: OpenAlex first, then S2."""
    try:
        r = get_with_backoff(f"https://api.openalex.org/works/https://doi.org/{doi}",
                             {"mailto": CONTACT_EMAIL})
        inv = r.json().get("abstract_inverted_index")
        if inv:
            return reconstruct_inverted(inv), "openalex-by-doi"
    except requests.RequestException:
        pass
    try:
        headers = {"x-api-key": S2_API_KEY} if S2_API_KEY else {}
        r = get_with_backoff(
            f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
            {"fields": "abstract"}, headers=headers)
        ab = r.json().get("abstract")
        if ab:
            return ab, "s2-by-doi"
    except requests.RequestException:
        pass
    return None, None


def fetch_abstract(title):
    """Stage 1: title search (OpenAlex -> Crossref -> S2).
    Stage 2: resolve DOI via Crossref fuzzy match, then fetch abstract by DOI.
    Returns (abstract, doi, source)."""
    for source, fn in (("openalex", from_openalex),
                       ("crossref", from_crossref),
                       ("semanticscholar", from_s2)):
        try:
            abstract, doi = fn(title)
        except requests.RequestException as e:
            print(f"    {source}: request failed ({e})")
            abstract, doi = None, None
        if abstract:
            return abstract, doi, source
        time.sleep(1.0)

    # Stage 2: DOI route
    try:
        doi = resolve_doi(title)
    except requests.RequestException as e:
        print(f"    doi-resolve: request failed ({e})")
        doi = None
    if doi:
        print(f"    resolved DOI: {doi}")
        abstract, source = abstract_by_doi(doi)
        if abstract:
            return abstract, doi, source
        return None, doi, "NONE (doi known)"
    return None, "", "NONE"


def main():
    DATA_DIR.mkdir(exist_ok=True)
    TEXT_DIR.mkdir(exist_ok=True)

    print(f"Downloading RP:P master data from {RPP_DATA_URL} ...")
    raw = requests.get(RPP_DATA_URL, timeout=60).text
    reader = csv.DictReader(io.StringIO(raw))
    fieldnames = reader.fieldnames or []

    cols = {}
    for key, pattern in COLUMN_HINTS.items():
        cols[key] = find_column(fieldnames, pattern)
        print(f"[column map] {key}: {cols[key] or 'NOT FOUND'}")
    for required in ("title", "journal", "pval_use_o", "pval_use_r"):
        if not cols[required]:
            sys.exit(f"ERROR: could not locate '{required}' column — "
                     f"inspect headers and adjust COLUMN_HINTS.")

    def has_value(row, col):
        v = (row.get(col) or "").strip()
        return v not in ("", "NA", "N/A", "NaN")

    all_rows = [r for r in reader if (r.get(cols["title"]) or "").strip()]
    rows = [r for r in all_rows
            if has_value(r, cols["pval_use_o"]) and has_value(r, cols["pval_use_r"])]
    print(f"\nRows with a title: {len(all_rows)}")
    print(f"Completed-replication frame (both T_pval_USE non-missing): {len(rows)} "
          f"(official analyses report 99)")

    # Write the frame
    all_path = DATA_DIR / "rpp_studies_all.csv"
    with open(all_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["study_num", "title", "authors", "journal",
                    "pval_use_o", "pval_use_r"])
        for r in rows:
            w.writerow([r.get(cols["study_num"], ""), clean_title(r[cols["title"]]),
                        r.get(cols["authors"], ""), r[cols["journal"]],
                        r.get(cols["pval_use_o"], ""), r.get(cols["pval_use_r"], "")])
    print(f"Wrote {all_path}")

    # Journal-stratified pilot sample
    random.seed(SEED)
    by_journal = {}
    for r in rows:
        by_journal.setdefault(r[cols["journal"]].strip(), []).append(r)
    print("Frame per journal:", {j: len(v) for j, v in sorted(by_journal.items())})

    total = len(rows)
    quotas = {j: max(1, round(PILOT_N * len(v) / total)) for j, v in by_journal.items()}
    while sum(quotas.values()) > PILOT_N:
        quotas[max(quotas, key=lambda j: quotas[j])] -= 1
    while sum(quotas.values()) < PILOT_N:
        quotas[max(by_journal, key=lambda j: len(by_journal[j]))] += 1
    pilot = []
    for j in sorted(by_journal):
        pilot.extend(random.sample(by_journal[j], quotas[j]))
    print(f"Pilot sample: {len(pilot)} studies, quotas {quotas} (seed={SEED})")

    # Fetch abstracts and write outputs
    pilot_path = DATA_DIR / "pilot_10_studies.csv"
    sheet_path = DATA_DIR / "coding_sheet_pilot.csv"
    found_count = 0
    with open(pilot_path, "w", newline="", encoding="utf-8") as pf, \
         open(sheet_path, "w", newline="", encoding="utf-8") as sf:
        pw = csv.writer(pf)
        sw = csv.writer(sf)
        pw.writerow(["pilot_id", "study_num", "title", "authors", "journal",
                     "doi", "abstract_source", "abstract_found"])
        sw.writerow(["pilot_id", "coder", "run", "masked",
                     "construct_1", "construct_2", "construct_3", "notes"])
        for i, r in enumerate(pilot, 1):
            title = clean_title(r[cols["title"]])
            authors = r.get(cols["authors"], "")
            print(f"\n[{i}/{len(pilot)}] {title[:70]}")
            abstract, doi, source = fetch_abstract(title)
            print(f"    -> abstract from: {source}")
            if abstract:
                found_count += 1

            (TEXT_DIR / f"study_{i:02d}.txt").write_text(
                f"TITLE: {title}\nAUTHORS: {authors}\nJOURNAL: {r[cols['journal']]}\n\n"
                f"ABSTRACT:\n{abstract or '[ABSTRACT NOT FOUND — paste manually]'}\n",
                encoding="utf-8")
            (TEXT_DIR / f"study_{i:02d}_masked.txt").write_text(
                f"ABSTRACT:\n{abstract or '[ABSTRACT NOT FOUND — paste manually]'}\n",
                encoding="utf-8")
            pw.writerow([f"{i:02d}", r.get(cols["study_num"], ""), title, authors,
                         r[cols["journal"]], doi, source,
                         "yes" if abstract else "NO"])
            time.sleep(1.0)

    print(f"\nDone: {found_count}/{len(pilot)} abstracts found.")
    print(f"Outputs: {pilot_path}, {sheet_path}, {TEXT_DIR}/")
    if found_count < len(pilot):
        print("For rows marked NO: paste the abstract into both study_NN.txt and "
              "study_NN_masked.txt by hand (use the DOI to find the paper).")
    print("\nNext: hand-code the 10 studies (coding_protocol.md) BEFORE running "
          "the LLM queries (llm_prompt.md).")


if __name__ == "__main__":
    main()
