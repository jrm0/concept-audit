# TDM Studio Bundle — Conceptual Coherence Pipeline (enclave edition)

Runs the same measurement pipeline as the local repo (SciBERT last-four-mean
embeddings → self-similarity/MEV/GMM metrics) inside ProQuest TDM Studio,
exporting ONLY derived artifacts (embeddings, metrics, counts). Methodology is
kept identical to `02_extract_embeddings.py` / `03_compute_metrics.py` so
results are comparable across strata.

## One-time setup

1. **Create the dataset(s)** in TDM Studio (Create New Dataset → select
   publication titles). Start with JPSP + Psychological Bulletin (~8.3k docs);
   scale to the psychology title list later. **Export/save the exact title
   list and filters used — that is the formal corpus definition (protocol §3).**
2. **Download SciBERT locally, then upload it** (the enclave has no internet
   to HuggingFace). On your Mac:
   ```bash
   pip install huggingface_hub
   python3 -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/scibert_scivocab_uncased', local_dir='scibert_scivocab_uncased')"
   zip -r scibert.zip scibert_scivocab_uncased
   ```
   Upload `scibert.zip` via Jupyter (see Upload Instructions.ipynb in their
   Manual folder), then in a terminal there: `unzip scibert.zip`.
3. **Upload this bundle** (all `tdm_*.py` files + your `concepts.csv`).
4. **Install packages** in the Jupyter terminal (Anaconda repos are reachable
   from inside):
   ```bash
   conda install -y pytorch cpuonly -c pytorch
   conda install -y -c conda-forge transformers scikit-learn scipy numpy
   ```
5. Fill in `concepts.csv` (copy `concepts_template.csv`): one row per
   construct — `concept_slug, retrieval_anchor, variants` (variants are
   pipe-separated surface forms to match).

## Run order (Jupyter terminal or notebook cells)

```bash
python3 tdm_00_probe_format.py      # inspect dataset XML; confirm field names
python3 tdm_01_parse_corpus.py      # XML -> per-concept sentence files (stays inside)
python3 tdm_02_embed.py             # SciBERT embeddings, checkpointed
python3 tdm_03_metrics.py           # metrics -> export/ folder
```

- `tdm_00` prints the structure of a few documents. If title/text/date field
  names differ from the defaults, edit the `FIELDS` dict at the top of
  `tdm_01_parse_corpus.py` (the probe output tells you what to use).
- `tdm_02` checkpoints after every concept AND every 200 sentences — safe to
  stop/restart. Log in at least every 48h or the VM pauses.
- Embedding pace on their 4-core VM: very roughly 2-4 sentences/sec →
  ~an hour per 10k sentences. Plan batches accordingly.

## Export discipline (read before exporting anything)

Move files to the export folder ONLY from `export/`:
- `metrics_summary.csv`, `metrics_detail.json` — derived statistics: fine.
- `embeddings_{concept}.npz` — derived vectors (no verbatim text is included;
  tdm_02 strips the sentences array in the enclave build): fine.
- `sentence_counts_by_venue_year.csv` — counts: fine.
- Representative sentences for qualitative cluster reading: DO NOT bulk-export
  verbatim text. Read them inside the platform; if short excerpts are needed
  in writeups, check TDM Studio's Supplemental Terms of Use first.

## Provenance to record in the lab notebook

Dataset name(s), title list, filters, creation date, doc counts; SciBERT
model hash (printed by tdm_02); package versions (printed by each script).
