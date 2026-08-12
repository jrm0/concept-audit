"""
tdm_02_embed.py — SciBERT embeddings inside TDM Studio.

METHODOLOGY IS IDENTICAL to the local 02_extract_embeddings.py (comparability
across strata depends on this): SciBERT scivocab_uncased, mean of last 4
hidden layers, mean-pooled over the concept's token span, MAX_LENGTH 256,
random-[CLS] anisotropy baseline (n=500), sampling seed 42.

Enclave adaptations:
- Model loads from a LOCAL uploaded folder (no internet): ./scibert_scivocab_uncased
- Checkpoints every CHECKPOINT_EVERY sentences per concept (48h-login rule) —
  re-running resumes where it left off.
- Output .npz contains embeddings + venue/year metadata but NO sentence text
  (keeps exports derived-only). Sentence row indices align with the
  workspace/sentences_{slug}.csv files if inside-enclave reading is needed.

Usage:
    python3 tdm_02_embed.py
"""

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_PATH = "./scibert_scivocab_uncased"
EXTRACTION_LAYER = "last_four_mean"
MAX_LENGTH = 256
MAX_SENTENCES_PER_CONCEPT = 2000   # raise vs. local 400: full text is richer
NUM_RANDOM_BASELINE = 500
CHECKPOINT_EVERY = 200
SEED = 42

WORK_DIR = Path("workspace")
EXPORT_DIR = Path("export")


def load_rows(slug):
    path = WORK_DIR / f"sentences_{slug}.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def concept_positions(input_ids, slug, tokenizer):
    words = slug.replace("_", " ").lower().split()
    target = []
    for w in words:
        target.extend(tokenizer.encode(w, add_special_tokens=False))
    ids = input_ids[0].tolist()
    for i in range(len(ids) - len(target) + 1):
        if ids[i:i + len(target)] == target:
            return list(range(i, i + len(target)))
    return []


def embed_sentence(sent, slug, tokenizer, model):
    tokens = tokenizer(sent, return_tensors="pt", truncation=True,
                       max_length=MAX_LENGTH, padding=False)
    with torch.no_grad():
        out = model(**tokens, output_hidden_states=True)
    hs = out.hidden_states
    layer = torch.stack(hs[-4:]).mean(dim=0) if EXTRACTION_LAYER == "last_four_mean" \
        else hs[-1]
    pos = concept_positions(tokens["input_ids"], slug, tokenizer) if slug else []
    vec = layer[0, pos, :].mean(dim=0) if pos else layer[0, 0, :]
    return vec.numpy()


def main():
    EXPORT_DIR.mkdir(exist_ok=True)
    if not Path(MODEL_PATH).exists():
        sys.exit(f"Model folder {MODEL_PATH} not found — upload and unzip "
                 f"SciBERT first (see README_TDM.md).")
    print("Loading SciBERT from local path...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()
    import transformers
    print(f"transformers={transformers.__version__} torch={torch.__version__}")

    slugs = sorted(p.stem.replace("sentences_", "")
                   for p in WORK_DIR.glob("sentences_*.csv"))
    print(f"Concepts with sentence files: {slugs}")

    pool = []  # for the random baseline
    for slug in slugs:
        out_path = EXPORT_DIR / f"embeddings_{slug}.npz"
        ckpt_path = WORK_DIR / f"ckpt_{slug}.npz"
        if out_path.exists():
            print(f"--- {slug}: already complete, skipping")
            continue
        rows = load_rows(slug)
        if len(rows) > MAX_SENTENCES_PER_CONCEPT:
            random.seed(SEED)
            rows = random.sample(rows, MAX_SENTENCES_PER_CONCEPT)
        pool.extend(r["sentence"] for r in rows[:200])

        start, embs = 0, []
        if ckpt_path.exists():
            ck = np.load(ckpt_path, allow_pickle=True)
            embs = list(ck["embeddings"])
            start = len(embs)
            print(f"--- {slug}: resuming at {start}/{len(rows)}")
        else:
            print(f"--- {slug}: {len(rows)} sentences")

        for i in range(start, len(rows)):
            embs.append(embed_sentence(rows[i]["sentence"], slug, tokenizer, model))
            if (i + 1) % CHECKPOINT_EVERY == 0:
                np.savez_compressed(ckpt_path, embeddings=np.array(embs))
                print(f"    {i + 1}/{len(rows)} (checkpointed)")

        arr = np.array(embs)
        # DERIVED-ONLY export: vectors + metadata, NO sentence text
        np.savez_compressed(
            out_path,
            embeddings=arr,
            venues=np.array([r["venue"] for r in rows], dtype=object),
            years=np.array([r["year"] for r in rows], dtype=object),
            docids=np.array([r["docid"] for r in rows], dtype=object),
            layer=str(EXTRACTION_LAYER),
        )
        if ckpt_path.exists():
            ckpt_path.unlink()
        print(f"    saved {arr.shape[0]} x {arr.shape[1]} -> {out_path}")

    base_path = EXPORT_DIR / "embeddings_random_baseline.npz"
    if not base_path.exists() and pool:
        print(f"--- random baseline ({NUM_RANDOM_BASELINE} sentences)")
        random.seed(SEED)
        sample = random.sample(pool, min(NUM_RANDOM_BASELINE, len(pool)))
        base = np.array([embed_sentence(s, None, tokenizer, model) for s in sample])
        np.savez_compressed(base_path, embeddings=base)
        print(f"    saved baseline -> {base_path}")

    print("\nDone. Next: python3 tdm_03_metrics.py")


if __name__ == "__main__":
    main()
