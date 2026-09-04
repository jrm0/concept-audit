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

## Lineage pipeline (16–23b) — one config per concept

Since 2026-09-02 the lineage scripts take **no per-concept literals**. Everything that identifies a concept — family file, control terms, period bounds, event dates, term pairs, figure lists, venue-category regexes, citation anchors — lives in one YAML under `configs/`, loaded by `pipeline_config.py`.

```bash
# default config is configs/ego_depletion.yaml, so these are unchanged:
python3 16_build_lineage_corpus.py
python3 17_embed_documents.py            # local, torch
python3 18_lineage_analysis.py [--center]
python3 21_cluster_validity.py
python3 22_make_figures.py
python3 23_citation_semantics.py --confirm-anchors
python3 23b_reference_portfolio.py --confirm-anchors
python3 22b_fig_propagation.py

# a second concept = a second YAML, nothing else:
python3 18_lineage_analysis.py --config configs/ptsd.yaml
```

Outputs go to `results/<slug>/` (one directory per concept). Every script writes `results/<slug>/manifest_<script>.json` (timestamp, git commit, config hash, parameters, library versions) and appends to `results/<slug>/run_log.jsonl`. **An output without a manifest is unversioned and should not be cited** — the stale `lineage_tree.json` of Aug 13 (produced under a clustering gate the code itself later retired) is why this exists.

To add a concept: copy `configs/ego_depletion.yaml`, change `concept`, `slug`, `family_csv`, `docs_path`, `emb_dir`, `results_dir`, the events/pairs/anchors, and (if the concept's habitat differs) the venue-category regexes. The `control_terms` block should stay the same across concepts so drift nulls are comparable.

Requires `pyyaml` and `diptest` (in `requirements.txt`).

**Corpus spec.** The config's `corpus:` block is the retrieval log made machine-readable: every export under `ris_root` must match a `sources[].files` or `ignore[].files` glob or 16 refuses to run. Each document carries `sources` and a register `stratum` (journal / thesis / book); 16 prints a per-source coverage audit and writes it to the manifest. Guiding principles for building a corpus are in `../TLoN-lab_notebook.md` (Track A → Corpus construction).

**No embedding cap; matched-n at analysis time (2026-09-02).** 17 embeds every document by default (`embedding.cap_per_term_period: null`); each shard records `n_pool` and `cap`. 18 makes every cross-period comparison (drift, cross-term distances, their noise floors) at equal sample size — `n_eq = min(n_a, n_b, analysis.equalize_n)`, repeated `analysis.equalize_reps` times, mean ± sd, full-data estimate alongside — while within-period clustering uses all documents. Use a new `emb_dir` per cap setting (shards are checkpointed); `data/lineage_emb` is the Aug-13 250-cap run, `data/lineage_emb_full` the uncapped one. Subset re-runs: `18 --dbs`, `--sources`, `--strata` (outputs suffixed).

| Script | Reads (from config) | Writes (results/<slug>/) |
|---|---|---|
| `16_build_lineage_corpus.py` | `ris_root`, `family_csv`, `family_works_glob`, `periods`, `control_terms` | `docs_path`, `lineage_power_table.csv` (next to it) |
| `17_embed_documents.py` | `docs_path`, `embedding.*` | `emb_dir/*.npz` |
| `18_lineage_analysis.py` | `emb_dir`, `analysis.*`, `events`, `key_pairs` | `lineage_drift.csv`, `term_distances.csv`, `lineage_tree.json` (with dip p / Jaccard / verdict per period), `lineage_samples_*.txt`, `lineage_report.txt` |
| `21_cluster_validity.py` | same as 18 | `cluster_validity.csv` |
| `20_locate_paper.py` | `emb_dir`, `docs_path`, `embedding.*`; `--term` defaults to `concept` | stdout |
| `22_make_figures.py` | `figures.*`, `venue_categories`, `event_short_labels` | `figures/fig1–4` |
| `23_citation_semantics.py` | `citation.anchors`, `citation.year_min`, `correction_set`, `venue_categories` | `citation_semantics.csv` (flag columns `cites_<anchor_label>`), `_report.txt` |
| `23b_reference_portfolio.py` | `citation.anchors + anchors_extra`, `crisis_set`, `milestone_control`, `coinage`, `hub_category` | `reference_portfolio_report.txt`, `hub_references_<cat>.csv` |
| `22b_fig_propagation.py` | outputs of 23 + 23b | `figures/fig5` |

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
