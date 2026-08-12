# Conceptual Coherence — Analysis Pipeline

Computational pipeline for measuring the semantic stability of scientific concepts using contextual embeddings (SciBERT).

## Quick Start

```bash
# 1. Clone and set up environment
cd code/
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Pull corpus data from Semantic Scholar
python3 01_pull_corpus.py

# 3. Extract SciBERT embeddings (this is the slow step — ~30-60 min on CPU)
python3 02_extract_embeddings.py

# 4. Compute metrics (self-similarity, MEV, clustering)
python3 03_compute_metrics.py

# 5. Generate visualizations
python3 04_visualize.py
```

## Pipeline Overview

| Script | What it does | Input | Output |
|--------|-------------|-------|--------|
| `01_pull_corpus.py` | Pulls abstracts from Semantic Scholar API | None (uses API) | `data/corpus_{concept}.csv` |
| `02_extract_embeddings.py` | Extracts SciBERT contextual embeddings | Corpus CSVs | `data/embeddings_{concept}.npz` |
| `03_compute_metrics.py` | Computes CSS, MEV, GMM clustering | Embedding arrays | `results/metrics_summary.csv`, `results/metrics_detail.json` |
| `04_visualize.py` | Generates comparison figures | Embeddings + metrics | `results/fig_*.png` |
| `05_pull_fulltext_s2orc.py` | Route 1: full text from S2ORC (Datasets API), filtered to Stratum 1 journals. Two phases; phase B is bandwidth-heavy (streams the whole S2ORC dataset) but resumable | `venues_config.py`; `S2_API_KEY` env var | `data/s2orc/fulltext_stratum1.jsonl` + corpusId/meta files |
| `06_pull_fulltext_pmc.py` | Route 2: full text from PMC Open Access subset (E-utilities + BioC), per journal, resumable | `venues_config.py`; optional `NCBI_API_KEY`/`NCBI_EMAIL` env vars | `data/pmc/fulltext_pmc.jsonl` + per-journal counts |
| `venues_config.py` | Shared Stratum 1 journal list (DRAFT — edit here, both pull scripts import it) | — | — |
| `osc_coding/01_fetch_pilot_studies.py` | Downloads RP:P master data from OSF, draws seeded journal-stratified 10-study pilot, fetches abstracts, writes masked/unmasked texts + coding sheet | Optional `S2_API_KEY` | `data/pilot_10_studies.csv`, `data/pilot_texts/`, `data/coding_sheet_pilot.csv` |

See `osc_coding/coding_protocol.md` and `osc_coding/llm_prompt.md` for the construct-coding procedure. Corpus construction rules live in `../Conceptual_Coherence_Corpus_Protocol_v1.md` (fill in its retrieval log after each pull).

## Configuration

Each script has a configuration section at the top. Key settings:

**01_pull_corpus.py:**
- `CONCEPTS` — dict of target concepts with tier and field metadata
- `PAPERS_PER_CONCEPT` — how many papers to search (default: 500)

**02_extract_embeddings.py:**
- `MODEL_NAME` — which model to use (default: `allenai/scibert_scivocab_uncased`)
- `EXTRACTION_LAYER` — which layer(s) to extract from (default: `last_four_mean`)
- `MAX_SENTENCES_PER_CONCEPT` — cap per concept for speed (default: 400)

**03_compute_metrics.py:**
- `MAX_K` — maximum clusters for GMM (default: 6)
- `N_ARI_RUNS` — number of runs for clustering stability check (default: 100)

## Requirements

- Python 3.9+
- ~2GB disk space for SciBERT model (downloaded automatically on first run)
- Internet access for step 01 (Semantic Scholar API) and first run of step 02 (model download)
- No GPU required (CPU works, just slower — expect ~30-60 min for step 02 with 5 concepts)

## Feasibility Test Concepts

| Concept | Tier | Expected |
|---------|------|----------|
| Reaction time | Stable control | High CSS, high MEV, 1 cluster |
| Working memory | Stable control | High CSS, high MEV, 1 cluster |
| Well-being | Unstable target | Low CSS, low MEV, multiple clusters |
| Resilience | Unstable target | Low CSS, low MEV, multiple clusters |
| Self-esteem | Synonym candidate | Mid-range; to be compared with future synonym cluster |

## After Running

Results will be in `results/`:
- `metrics_summary.csv` — one row per concept with CSS, MEV, cluster count
- `metrics_detail.json` — full details including cluster info and sensitivity data
- `fig_css_comparison.png` — bar chart ranking concepts by stability
- `fig_mev_comparison.png` — bar chart ranking by MEV
- `fig_semantic_clouds.png` — PCA overlay of all concept clouds
- `fig_cloud_detail_{concept}.png` — per-concept cloud with cluster coloring

Sync these back to the project folder for analysis in subsequent sessions.
