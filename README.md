# Conceptual Coherence — Analysis Pipeline

Computational pipeline for measuring the semantic stability of scientific concepts using contextual embeddings (SciBERT).

## Quick Start

```bash
# 1. Clone and set up environment
cd code/
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Pull corpus data from Semantic Scholar
python 01_pull_corpus.py

# 3. Extract SciBERT embeddings (this is the slow step — ~30-60 min on CPU)
python 02_extract_embeddings.py

# 4. Compute metrics (self-similarity, MEV, clustering)
python 03_compute_metrics.py

# 5. Generate visualizations
python 04_visualize.py
```

## Pipeline Overview

| Script | What it does | Input | Output |
|--------|-------------|-------|--------|
| `01_pull_corpus.py` | Pulls abstracts from Semantic Scholar API | None (uses API) | `data/corpus_{concept}.csv` |
| `02_extract_embeddings.py` | Extracts SciBERT contextual embeddings | Corpus CSVs | `data/embeddings_{concept}.npz` |
| `03_compute_metrics.py` | Computes CSS, MEV, GMM clustering | Embedding arrays | `results/metrics_summary.csv`, `results/metrics_detail.json` |
| `04_visualize.py` | Generates comparison figures | Embeddings + metrics | `results/fig_*.png` |

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
