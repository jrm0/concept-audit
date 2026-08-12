"""
01_pull_corpus.py — Pull abstracts from Semantic Scholar for target concepts.

Usage:
    python3 01_pull_corpus.py

Outputs:
    data/corpus_{concept}.csv for each concept (columns: paper_id, title, abstract, sentence)
    data/corpus_summary.csv — counts per concept

Each CSV contains one row per sentence that mentions the target concept,
extracted from paper abstracts.
"""

import requests
import csv
import os
import time
import re
from pathlib import Path

# --- Configuration ---

# Target concepts for the feasibility test
CONCEPTS = {
    # Tier 1: Expected stable
    "reaction time": {"tier": "stable", "field": "cognitive psychology"},
    "working memory": {"tier": "stable", "field": "cognitive psychology"},
    # Tier 2: Expected unstable
    "well-being": {"tier": "unstable", "field": "psychology"},
    "resilience": {"tier": "unstable", "field": "psychology"},
    # Synonym candidate
    "self-esteem": {"tier": "synonym", "field": "psychology"},
}

# How many papers to pull per concept (Semantic Scholar allows up to 100 per request)
PAPERS_PER_CONCEPT = 500

# Minimum abstract length (characters) to include
MIN_ABSTRACT_LENGTH = 100

# Output directory
DATA_DIR = Path(__file__).parent.parent / "data"

# Semantic Scholar API base
SS_API = "https://api.semanticscholar.org/graph/v1/paper/search"

# Rate limit: Semantic Scholar allows ~1 request/sec without API key.
# With a free API key (recommended): 10 requests/sec.
# Get one at: https://www.semanticscholar.org/product/api#api-key
DELAY_BETWEEN_REQUESTS = 3.0  # seconds between paginated requests

# Set this to your API key to get better rate limits (optional but recommended).
# You can also set the environment variable S2_API_KEY instead.
API_KEY = None

# Retry settings for 429 (rate limit) errors
MAX_RETRIES = 5
INITIAL_BACKOFF = 5  # seconds; doubles each retry


def search_papers(query, limit=100, offset=0):
    """Search Semantic Scholar for papers matching a query, with retry on rate limit."""
    fields = "paperId,title,abstract,year,venue,citationCount,fieldsOfStudy"
    params = {
        "query": query,
        "limit": min(limit, 100),  # API max is 100 per request
        "offset": offset,
        "fields": fields,
    }

    # Use API key if available
    headers = {}
    api_key = API_KEY or os.environ.get("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(MAX_RETRIES):
        response = requests.get(SS_API, params=params, headers=headers, timeout=30)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            wait_time = INITIAL_BACKOFF * (2 ** attempt)
            print(f"    Rate limited (429). Waiting {wait_time}s before retry {attempt + 1}/{MAX_RETRIES}...")
            time.sleep(wait_time)
        else:
            response.raise_for_status()

    # If we exhausted retries, raise the last error
    response.raise_for_status()
    return {}


def extract_sentences_with_context(abstract, concept):
    """
    Split abstract into sentences and return those containing the concept,
    along with surrounding context (sentence before + after).
    Case-insensitive matching.

    Returns list of dicts with 'sentence' (the matching sentence) and
    'context' (sentence before + matching sentence + sentence after).
    """
    if not abstract:
        return []

    # Simple sentence splitting (handles common abbreviations)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', abstract)

    results = []
    concept_lower = concept.lower()
    for i, sent in enumerate(sentences):
        if concept_lower in sent.lower():
            sent = sent.strip()
            if len(sent) > 30:  # Skip very short fragments
                # Build context window: sentence before + target + sentence after
                context_parts = []
                if i > 0:
                    context_parts.append(sentences[i - 1].strip())
                context_parts.append(sent)
                if i < len(sentences) - 1:
                    context_parts.append(sentences[i + 1].strip())
                context = " ".join(context_parts)

                results.append({
                    "sentence": sent,
                    "context": context,
                })
    return results


def pull_corpus_for_concept(concept, metadata, target_sentences=500):
    """
    Pull abstracts from Semantic Scholar and extract sentences mentioning the concept.
    Returns list of dicts with paper metadata, sentence, and surrounding context.
    """
    print(f"\n--- Pulling corpus for: '{concept}' ---")

    all_rows = []
    offset = 0
    papers_seen = 0

    while len(all_rows) < target_sentences and offset < 2000:
        try:
            results = search_papers(
                query=f"{concept} {metadata['field']}",
                limit=100,
                offset=offset,
            )
        except requests.exceptions.RequestException as e:
            print(f"  API error at offset {offset}: {e}")
            break

        papers = results.get("data", [])
        if not papers:
            print(f"  No more results at offset {offset}")
            break

        for paper in papers:
            abstract = paper.get("abstract", "")
            if not abstract or len(abstract) < MIN_ABSTRACT_LENGTH:
                continue

            extracted = extract_sentences_with_context(abstract, concept)
            for item in extracted:
                # Format fields of study as semicolon-separated string
                fos = paper.get("fieldsOfStudy") or []
                fos_str = "; ".join(fos) if fos else ""

                all_rows.append({
                    "paper_id": paper.get("paperId", ""),
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "venue": paper.get("venue", ""),
                    "citation_count": paper.get("citationCount", ""),
                    "fields_of_study": fos_str,
                    "sentence": item["sentence"],
                    "context": item["context"],
                    "concept": concept,
                    "tier": metadata["tier"],
                    "field": metadata["field"],
                })

            papers_seen += 1

        offset += 100
        print(f"  Offset {offset}: {len(all_rows)} sentences from {papers_seen} papers so far")
        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"  Final: {len(all_rows)} sentences from {papers_seen} papers")
    return all_rows


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    all_rows = []  # Combined across all concepts

    for concept, metadata in CONCEPTS.items():
        rows = pull_corpus_for_concept(concept, metadata)
        all_rows.extend(rows)

        # Save per-concept CSV
        slug = concept.replace(" ", "_").replace("-", "_")
        out_path = DATA_DIR / f"corpus_{slug}.csv"

        if rows:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Saved to {out_path}")
        else:
            print(f"  WARNING: No sentences found for '{concept}'")

        summary.append({
            "concept": concept,
            "tier": metadata["tier"],
            "field": metadata["field"],
            "num_sentences": len(rows),
            "num_unique_papers": len(set(r["paper_id"] for r in rows)) if rows else 0,
        })

    # Save combined CSV (all concepts in one file)
    if all_rows:
        combined_path = DATA_DIR / "corpus_all.csv"
        with open(combined_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCombined corpus saved to {combined_path} ({len(all_rows)} total sentences)")

    # Save summary
    summary_path = DATA_DIR / "corpus_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n=== Summary saved to {summary_path} ===")
    for row in summary:
        print(f"  {row['concept']:20s}  {row['num_sentences']:4d} sentences  {row['num_unique_papers']:4d} papers")


if __name__ == "__main__":
    main()
