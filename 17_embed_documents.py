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

NO CAP BY DEFAULT (2026-09-02). Until now each (term, period) was capped at
250 documents, sampled from the pooled corpus. That equalized sample size
across periods — which matters for CROSS-period comparisons (a centroid from
50 documents wobbles more than one from 5,000; a split test with 5,000
documents can see what it never could with 60) — but it did so by throwing
information away, and it made minority sub-senses invisible (8% of usage
under a 250 cap = 20 documents). Equalization now happens at ANALYSIS time in
18 (repeated matched-n subsampling for drift and distances); within-period
description (sense discovery, sizing, citation join) uses every document.
Set embedding.cap_per_term_period to a number to restore a cap (e.g. for a
sensitivity check). Control terms keep a cap (embedding.control_cap): they
only need a stable reference centroid, and they occur in nearly every
abstract.

Each shard records n_pool (documents available for that term-period before
any cap) and cap, so prevalence is reconstructible and a capped shard is
recognisable. Re-running with a different cap into the same emb_dir would
silently keep the old shards (checkpointing) — use a NEW emb_dir per cap
setting; the config's emb_dir names the run.

Reads:  data/family/lineage_docs.jsonl   (from 16)
Writes: data/lineage_emb/{term_slug}__{period}.npz
            vectors (n,768), keys (doi), years, journals
        checkpointed per (term, period): re-running skips finished shards.

Usage:
    caffeinate -i python3 17_embed_documents.py [--config configs/<concept>.yaml]
    python3 17_embed_documents.py "ego depletion" "self-control"   # subset

Model, caps, seed, paths and control terms come from the config.
"""

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import pipeline_config as pc
from pipeline_config import slug, term_pattern

CFG, _ARGV = pc.load()

MODEL_NAME = CFG["embedding"]["model"]
MAX_LENGTH = int(CFG["embedding"]["max_length"])
SEED = int(CFG["embedding"]["seed"])
_cap = CFG["embedding"].get("cap_per_term_period")
CAP_PER_TERM_PERIOD = int(_cap) if _cap not in (None, "", "none", "null") else None
MAX_SENT_PER_DOC = int(CFG["embedding"]["max_sent_per_doc"])
CONTROL_TERMS = CFG.controls
CONTROL_CAP = int(CFG["embedding"].get("control_cap") or 300)

IN_PATH = Path(CFG["docs_path"])
OUT_DIR = Path(CFG["emb_dir"])


def main():
    import torch
    from transformers import AutoModel, AutoTokenizer

    only = set(_ARGV)
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
    n_to_embed = sum(len(v) if CAP_PER_TERM_PERIOD is None
                     else min(len(v), CAP_PER_TERM_PERIOD) for v in groups.values())
    print(f"{n_docs} documents; {len(groups)} (term, period) shards; "
          f"cap={'none (embed all)' if CAP_PER_TERM_PERIOD is None else CAP_PER_TERM_PERIOD}; "
          f"~{n_to_embed} document vectors to compute (skips existing shards)")

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
        # hyphen/space are the same separator (see canon() in pipeline_config)
        pat = term_pattern(term)
        n_pool = len(docs)
        if cap is not None and len(docs) > cap:
            random.seed(SEED)
            docs = random.sample(docs, cap)
        vecs, keys, years, journals, dbs = [], [], [], [], []
        sources, strata = [], []
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
            sources.append("|".join(d.get("sources", [])))   # corpus-spec ids
            strata.append(d.get("stratum", ""))              # register
        if vecs:
            np.savez_compressed(out_path, vectors=np.array(vecs),
                                keys=np.array(keys, dtype=object),
                                years=np.array(years),
                                journals=np.array(journals, dtype=object),
                                dbs=np.array(dbs, dtype=object),
                                sources=np.array(sources, dtype=object),
                                strata=np.array(strata, dtype=object),
                                n_pool=np.array(n_pool),
                                cap=np.array(-1 if cap is None else cap))
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
    pc.write_manifest(
        CFG, __file__,
        params={"model": MODEL_NAME, "max_length": MAX_LENGTH, "seed": SEED,
                "cap_per_term_period": CAP_PER_TERM_PERIOD,   # None = all
                "emb_dir": str(OUT_DIR),
                "control_cap": CONTROL_CAP, "max_sent_per_doc": MAX_SENT_PER_DOC,
                "control_terms": CONTROL_TERMS, "subset": sorted(only),
                "n_docs_in": n_docs, "n_new_vectors": total,
                "n_shards_total": len(list(OUT_DIR.glob("*.npz")))},
        outputs=[OUT_DIR],
        notes="Shards are checkpointed: existing .npz files are skipped, so "
              "n_new_vectors=0 means nothing was recomputed this run.")
    print("Next: python3 18_lineage_analysis.py")


if __name__ == "__main__":
    main()
