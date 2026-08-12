"""
02_extract_embeddings.py — Extract SciBERT contextual embeddings for concept mentions.

Usage:
    python3 02_extract_embeddings.py

Reads:
    data/corpus_{concept}.csv files from step 01

Outputs:
    data/embeddings_{concept}.npz — numpy archive with:
        - embeddings: (N, hidden_dim) array of contextual embeddings
        - sentences: array of source sentences
        - layer: which layer was extracted
    data/embeddings_random_baseline.npz — random sentence embeddings for anisotropy correction
"""

import csv
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm

# --- Configuration ---

MODEL_NAME = "allenai/scibert_scivocab_uncased"

# Which layer to extract. Options:
#   - "last": final hidden layer (most context-specific)
#   - "second_to_last": layer -2
#   - "last_four_mean": mean of last 4 layers (common in NLP)
#   - integer (e.g., 6): specific layer index
EXTRACTION_LAYER = "last_four_mean"

# Max tokens per sentence (SciBERT max is 512)
MAX_LENGTH = 256

# How many sentences to sample per concept (for speed; set to None for all)
MAX_SENTENCES_PER_CONCEPT = 400

# How many random sentences for anisotropy baseline
NUM_RANDOM_BASELINE = 500

DATA_DIR = Path(__file__).parent.parent / "data"

# Concepts (must match 01_pull_corpus.py)
CONCEPTS = ["reaction_time", "working_memory", "well_being", "resilience", "self_esteem"]


def load_sentences(concept_slug):
    """Load sentences from corpus CSV."""
    path = DATA_DIR / f"corpus_{concept_slug}.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return []

    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentences.append(row["sentence"])

    return sentences


def get_concept_token_positions(tokens, concept_slug, tokenizer):
    """
    Find the token positions corresponding to the concept in the tokenized input.
    Returns list of token indices. For multi-word concepts, returns all constituent tokens.
    """
    # Convert slug back to concept words
    concept_words = concept_slug.replace("_", " ").lower().split()

    # Get the token IDs for each concept word
    concept_token_ids = []
    for word in concept_words:
        word_tokens = tokenizer.encode(word, add_special_tokens=False)
        concept_token_ids.extend(word_tokens)

    # Find positions in the full token sequence
    input_ids = tokens["input_ids"][0].tolist()

    positions = []
    for i in range(len(input_ids) - len(concept_token_ids) + 1):
        if input_ids[i:i + len(concept_token_ids)] == concept_token_ids:
            positions.extend(range(i, i + len(concept_token_ids)))
            break  # Take first occurrence

    return positions


def extract_embeddings_for_sentences(sentences, concept_slug, tokenizer, model, device):
    """
    Extract contextual embeddings for concept mentions in each sentence.
    Returns (N, hidden_dim) numpy array.
    """
    embeddings = []

    for sent in tqdm(sentences, desc=f"  Embedding {concept_slug}", leave=False):
        tokens = tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)

        hidden_states = outputs.hidden_states  # tuple of (batch, seq_len, hidden_dim)

        # Select layer(s)
        if EXTRACTION_LAYER == "last":
            layer_output = hidden_states[-1]
        elif EXTRACTION_LAYER == "second_to_last":
            layer_output = hidden_states[-2]
        elif EXTRACTION_LAYER == "last_four_mean":
            layer_output = torch.stack(hidden_states[-4:]).mean(dim=0)
        elif isinstance(EXTRACTION_LAYER, int):
            layer_output = hidden_states[EXTRACTION_LAYER]
        else:
            raise ValueError(f"Unknown EXTRACTION_LAYER: {EXTRACTION_LAYER}")

        # Find concept token positions
        positions = get_concept_token_positions(tokens, concept_slug, tokenizer)

        if positions:
            # Mean of concept token embeddings
            concept_emb = layer_output[0, positions, :].mean(dim=0)
        else:
            # Fallback: use [CLS] token
            concept_emb = layer_output[0, 0, :]

        embeddings.append(concept_emb.cpu().numpy())

    return np.array(embeddings)


def extract_random_baseline(all_sentences, tokenizer, model, device, n=500):
    """
    Extract [CLS] embeddings from random sentences for anisotropy correction.
    """
    import random
    sampled = random.sample(all_sentences, min(n, len(all_sentences)))

    embeddings = []
    for sent in tqdm(sampled, desc="  Random baseline", leave=False):
        tokens = tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        if EXTRACTION_LAYER == "last_four_mean":
            layer_output = torch.stack(hidden_states[-4:]).mean(dim=0)
        else:
            layer_output = hidden_states[-1]

        cls_emb = layer_output[0, 0, :]
        embeddings.append(cls_emb.cpu().numpy())

    return np.array(embeddings)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Extraction layer: {EXTRACTION_LAYER}")

    print(f"\nLoading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print("  Model loaded.")

    all_sentences_pool = []

    for concept_slug in CONCEPTS:
        print(f"\n--- Processing: {concept_slug} ---")

        sentences = load_sentences(concept_slug)
        if not sentences:
            continue

        # Sample if needed
        if MAX_SENTENCES_PER_CONCEPT and len(sentences) > MAX_SENTENCES_PER_CONCEPT:
            import random
            random.seed(42)
            sentences = random.sample(sentences, MAX_SENTENCES_PER_CONCEPT)
            print(f"  Sampled {MAX_SENTENCES_PER_CONCEPT} of {len(sentences)} sentences")

        all_sentences_pool.extend(sentences)

        # Extract embeddings
        embs = extract_embeddings_for_sentences(sentences, concept_slug, tokenizer, model, device)

        # Save
        out_path = DATA_DIR / f"embeddings_{concept_slug}.npz"
        np.savez_compressed(
            out_path,
            embeddings=embs,
            sentences=np.array(sentences, dtype=object),
            layer=str(EXTRACTION_LAYER),
        )
        print(f"  Saved {embs.shape[0]} embeddings ({embs.shape[1]}d) to {out_path}")

    # Random baseline for anisotropy correction
    print(f"\n--- Extracting random baseline ({NUM_RANDOM_BASELINE} sentences) ---")
    baseline_embs = extract_random_baseline(
        all_sentences_pool, tokenizer, model, device, n=NUM_RANDOM_BASELINE
    )
    baseline_path = DATA_DIR / "embeddings_random_baseline.npz"
    np.savez_compressed(baseline_path, embeddings=baseline_embs)
    print(f"  Saved {baseline_embs.shape[0]} baseline embeddings to {baseline_path}")

    print("\n=== Embedding extraction complete ===")


if __name__ == "__main__":
    main()
