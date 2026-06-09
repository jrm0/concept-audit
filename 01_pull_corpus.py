"""
01_pull_corpus.py — Pull abstracts from Semantic Scholar for target concepts.

Usage:
    python 01_pull_corpus.py

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

# Rate limit: Semantic Scholar allows ~100 requests/5 min without API key
DELAY_BETWEEN_REQUESTS = 1.5  # seconds


def search_papers(query, limit=100, offset=0, fields="paperId,title,abstract"):
    """Search Semantic Scholar for papers matching a query."""
    params = {
        "query": query,
        "limit": min(limit, 100),  # API max is 100 per request
        "offset": offset,
        "fields": fields,
    }
    response = requests.get(SS_API, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def extract_sentences(abstract, concept):
    """
    Split abstract into sentences and return those containing the concept.
    Case-insensitive matching.
    """
    if not abstract:
        return []

    # Simple sentence splitting (handles common abbreviations)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', abstract)

    matching = []
    concept_lower = concept.lower()
    for sent in sentences:
        if concept_lower in sent.lower():
            sent = sent.strip()
            if len(sent) > 30:  # Skip very short fragments
                matching.append(sent)
    return matching


def pull_corpus_for_concept(concept, metadata, target_sentences=500):
    """
    Pull abstracts from Semantic Scholar and extract sentences mentioning the concept.
    Returns list of dicts with paper_id, title, abstract, sentence.
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

            sentences = extract_sentences(abstract, concept)
            for sent in sentences:
                all_rows.append({
                    "paper_id": paper.get("paperId", ""),
                    "title": paper.get("title", ""),
                    "abstract": abstract,
                    "sentence": sent,
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

    for concept, metadata in CONCEPTS.items():
        rows = pull_corpus_for_concept(concept, metadata)

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
