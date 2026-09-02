"""
17_embed_documents.py — Document-level MASKED-target embeddings for the
lineage analysis. (Runs LOCALLY: torch + SciBERT.)

Two artifact lessons are baked in:
 1. DOCUMENT units (one vector per document per term = mean over its mention
    sentences) — sentences are not independent observations (session 10).
 2. MASKED-target embedding: the matched term is replaced by [MASK] and the
    vector is taken at the mask position. The representation therefore
    reflects pure CONTEXT. This is required for CROSS-term comparisons
    (ego depletion vs self-control): with visible targets, term-to-term
    distances partly measure the target tokens themselves — the session-8
    orthography artifact at the term level. Masking removes it exactly.

Control terms are embedded the same way from the same abstracts — the drift
null (all embeddings move as corpus composition changes; Dubossarsky).

Reads:  data/family/lineage_docs.jsonl   (from 16)
Writes: data/lineage_emb/{term_slug}__{period}.npz
            vectors (n,768), keys (doi), years, journals
        checkpointed per (term, period): re-running skips finished shards.

Usage:
    caffeinate -i python3 17_embed_documents.py
    python3 17_embed_documents.py "ego depletion" "self-control"   # subset
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

MODEL_NAME = "allenai/scibert_scivocab_uncased"
MAX_LENGTH = 256
SEED = 42
CAP_PER_TERM_PERIOD = 250      # doc cap per (term, period) — CPU budget
MAX_SENT_PER_DOC = 3           # mention sentences averaged per document
CONTROL_TERMS = ["participants", "results", "significant", "measure"]
CONTROL_CAP = 120              # docs per period for the control null

IN_PATH = Path("data/family/lineage_docs.jsonl")
OUT_DIR = Path("data/lineage_emb")


def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    only = set(sys.argv[1:])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    mask_tok = tokenizer.mask_token
    import transformers
    print(f"transformers={transformers.__version__} torch={torch.__version__} "
          f"mask='{mask_tok}'")

    # Group docs by (term, period)
    groups = defaultdict(list)   # (term, period) -> [doc]
    n_docs = 0
    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            n_docs += 1
            for t, sents in d["mention_sentences"].items():
                if only and t not in only and t not in CONTROL_TERMS:
                    continue
                if t in CONTROL_TERMS:
                    continue   # controls sampled separately below
                groups[(t, d["period"])].append(d)
    print(f"{n_docs} documents; {len(groups)} (term, period) shards to embed")

    # Control shards: sample docs per period regardless of family term
    ctrl = defaultdict(list)     # period -> [doc]
    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if any(c in d["mention_sentences"] for c in CONTROL_TERMS):
                ctrl[d["period"]].append(d)

    def masked_vector(sentence, pattern):
        m = pattern.search(sentence)
        if not m:
            return None
        text = sentence[:m.start()] + mask_tok + sentence[m.end():]
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=MAX_LENGTH)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        layer = torch.stack(out.hidden_states[-4:]).mean(dim=0)
        ids = enc["input_ids"][0].tolist()
        try:
            pos = ids.index(tokenizer.mask_token_id)
        except ValueError:
            return None      # mask truncated away
        return layer[0, pos, :].numpy()

    def embed_shard(term, period, docs, cap):
        out_path = OUT_DIR / f"{slug(term)}__{period}.npz"
        if out_path.exists():
            return 0
        # hyphen/space are the same separator (see canon() in 16)
        body = re.escape(term).replace(r"\ ", "[-\\s]+").replace(r"\-",
                                                                "[-\\s]+")
        pat = re.compile(r"\b" + body + r"\b", re.I)
        if len(docs) > cap:
            random.seed(SEED)
            docs = random.sample(docs, cap)
        vecs, keys, years, journals, dbs = [], [], [], [], []
        for d in docs:
            sents = d["mention_sentences"].get(term, [])[:MAX_SENT_PER_DOC]
            sv = [v for v in (masked_vector(s, pat) for s in sents)
                  if v is not None]
            if not sv:
                continue
            vecs.append(np.mean(sv, axis=0))
            keys.append(d["doi"] or d["title"][:60])
            years.append(d["year"])
            journals.append(d["journal"])
            dbs.append("|".join(d.get("dbs", [])))
        if vecs:
            np.savez_compressed(out_path, vectors=np.array(vecs),
                                keys=np.array(keys, dtype=object),
                                years=np.array(years),
                                journals=np.array(journals, dtype=object),
                                dbs=np.array(dbs, dtype=object))
        return len(vecs)

    total = 0
    for (t, p) in sorted(groups):
        n = embed_shard(t, p, groups[(t, p)], CAP_PER_TERM_PERIOD)
        if n:
            total += n
            print(f"  {t} | {p}: {n} docs embedded")
    for c in CONTROL_TERMS:
        if only and c not in only and only:
            pass
        for p, docs in sorted(ctrl.items()):
            docs_c = [d for d in docs if c in d["mention_sentences"]]
            n = embed_shard(c, p, docs_c, CONTROL_CAP)
            if n:
                total += n
                print(f"  [control] {c} | {p}: {n} docs")

    print(f"\nDone: {total} new document vectors -> {OUT_DIR}/")
    print("Next: python3 18_lineage_analysis.py")


if __name__ == "__main__":
    main()
