"""
20_locate_paper.py — Situate a SINGLE paper in the lineage landscape.

Why this works (the question was whether single-paper location and the
lineage analysis are commensurable): they are the same measurement. Static
word embeddings collapse a word to one vector per corpus and genuinely
cannot recover an individual usage. Our pipeline never does that — every
clade centroid is an average of per-document masked-target vectors. So a
single paper is a point in the same space, and locating it is a lookup, not
a new method. It is embedded here EXACTLY as 17 embeds corpus documents
(same model, same masking, same layers, same document-level averaging);
any deviation would make the comparison meaningless.

What it reports, for a chosen term:
  * cosine distance from the paper to each period's clade centroid
  * the PERCENTILE of that distance among same-period documents — the honest
    statistic, because one paper contributes only a few mention sentences and
    its raw distance means little without the reference distribution
  * nearest individual documents (titles), the qualitative check
  * which clade it is closest to in each period, and the trajectory

Usage (from code/):
    python3 20_locate_paper.py --doi 10.1037/0022-3514.74.5.1252
    python3 20_locate_paper.py --title "Ego depletion: Is the active self a limited resource?"
    python3 20_locate_paper.py --text "abstract text ..." --term "ego depletion"
    python3 20_locate_paper.py --file some_abstract.txt --term "self control"

--doi / --title look the paper up in the corpus (<docs_path> from the config)
and, if found, reuse its stored text; otherwise supply --text or --file.
Default term is the config's focal concept; --term picks another term's space.
Add --config configs/<concept>.yaml to use a different concept's landscape.

NOTE: distances are reported in RAW space by default and additionally under
control-centering (--center) when control shards exist, because raw cosines
in a contextual-embedding space are compressed by anisotropy (Ethayarajh
2019) and should not be read as absolute similarity.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

import pipeline_config as pc
from pipeline_config import slug, term_pattern

CFG, _ARGV = pc.load()

MODEL_NAME = CFG["embedding"]["model"]
MAX_LENGTH = int(CFG["embedding"]["max_length"])
MAX_SENT_PER_DOC = int(CFG["embedding"]["max_sent_per_doc"])
EMB_DIR = Path(CFG["emb_dir"])
DOCS_PATH = Path(CFG["docs_path"])
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
CONTROL_TERMS = set(CFG.controls)


def load_corpus_doc(doi=None, title=None):
    if not DOCS_PATH.exists():
        return None
    want_t = (title or "").lower().strip()
    with open(DOCS_PATH, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if doi and d.get("doi", "").lower() == doi.lower().strip():
                return d
            if want_t and want_t in (d.get("title") or "").lower():
                return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi")
    ap.add_argument("--title")
    ap.add_argument("--text")
    ap.add_argument("--file")
    ap.add_argument("--term", default=CFG["concept"])
    ap.add_argument("--center", action="store_true",
                    help="also report under control-term centering")
    args = ap.parse_args(_ARGV)

    term = pc.canon(args.term)
    label = args.doi or args.title or (args.file or "pasted text")

    # 1. Get the text
    text = None
    doc = load_corpus_doc(args.doi, args.title)
    if doc:
        label = doc.get("title") or label
        text = f"{doc.get('title','')}. {doc.get('abstract','') or ''}"
        if not (doc.get("abstract") or "").strip():
            # 16 keeps mention sentences even when it does not retain the
            # full abstract — they are what 17 would have embedded anyway
            sents = doc.get("mention_sentences", {}).get(term, [])
            text = " ".join(sents) if sents else text
        print(f"Found in corpus: {label} ({doc.get('year')}, "
              f"{doc.get('journal','')})")
    if args.text:
        text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    if not text or not text.strip():
        raise SystemExit(
            "No text. The paper is not in the corpus (or has no abstract) — "
            "supply --text or --file with its abstract.")

    # 2. Find the term's mention sentences
    pat = term_pattern(term)
    sents = [s.strip() for s in SENT_RE.split(text)
             if 30 <= len(s.strip()) <= 1200 and pat.search(s)]
    if not sents:
        raise SystemExit(
            f"'{term}' does not appear in this text. Use --term to pick a "
            f"term the paper actually uses (e.g. --term 'self control').")
    print(f"{len(sents)} mention sentence(s) of '{term}'; "
          f"embedding {min(len(sents), MAX_SENT_PER_DOC)} (as 17 does)")

    # 3. Embed IDENTICALLY to 17 (masked target, last-4-layer mean)
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    def masked_vector(sentence):
        m = pat.search(sentence)
        if not m:
            return None
        s = sentence[:m.start()] + tok.mask_token + sentence[m.end():]
        enc = tok(s, return_tensors="pt", truncation=True,
                  max_length=MAX_LENGTH)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        layer = torch.stack(out.hidden_states[-4:]).mean(dim=0)
        ids = enc["input_ids"][0].tolist()
        try:
            pos = ids.index(tok.mask_token_id)
        except ValueError:
            return None
        return layer[0, pos, :].numpy()

    vs = [v for v in (masked_vector(s) for s in sents[:MAX_SENT_PER_DOC])
          if v is not None]
    if not vs:
        raise SystemExit("Could not embed (mask truncated away).")
    q = np.mean(vs, axis=0)

    # 4. Compare against every period shard of that term
    shards = sorted(EMB_DIR.glob(f"{slug(term)}__*.npz"))
    if not shards:
        raise SystemExit(f"No embedding shards for '{term}' — run 17 first.")
    titles = {}
    if DOCS_PATH.exists():
        with open(DOCS_PATH, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("doi"):
                    titles[d["doi"]] = d.get("title", "")

    ctrl_ref = {}
    if args.center:
        for c in CONTROL_TERMS:
            for f_ in EMB_DIR.glob(f"{slug(c)}__*.npz"):
                p = f_.stem.split("__")[1]
                z = np.load(f_, allow_pickle=True)
                acc, n = ctrl_ref.get(p, (0, 0))
                ctrl_ref[p] = (acc + z["vectors"].astype(np.float64).sum(0),
                               n + len(z["vectors"]))
        ctrl_ref = {p: s / n for p, (s, n) in ctrl_ref.items() if n}

    print(f"\nPOSITION OF THIS PAPER IN THE '{term}' LANDSCAPE")
    print(f"{'period':<12}{'n':>5}{'d(paper,centroid)':>20}"
          f"{'percentile':>12}   interpretation")
    rows = []
    for f_ in shards:
        p = f_.stem.split("__")[1]
        z = np.load(f_, allow_pickle=True)
        V = z["vectors"].astype(np.float64)
        qq = q.astype(np.float64)
        if args.center:
            if p not in ctrl_ref:
                continue
            V = V - ctrl_ref[p]
            qq = qq - ctrl_ref[p]
        c = V.mean(0)
        d = float(cosine(qq, c))
        # reference distribution: how far is a TYPICAL same-period document
        # from that same centroid?
        ref = np.array([cosine(v, c) for v in V])
        pct = float((ref < d).mean() * 100)
        note = ("typical member" if pct < 50 else
                "outer half" if pct < 90 else
                "PERIPHERAL — outside 90% of same-period usage")
        print(f"{p:<12}{len(V):>5}{d:>20.4f}{pct:>11.0f}%   {note}")
        rows.append((p, d, pct))
        # nearest individual documents
        sims = sorted(((float(cosine(qq, v)), k) for v, k in zip(V, z["keys"])))
        near = "; ".join(titles.get(str(k), str(k))[:60] for _, k in sims[:3])
        print(f"{'':<12}nearest: {near}")

    if len(rows) >= 2:
        first, last = rows[0], rows[-1]
        print(f"\nTRAJECTORY: this usage sits at the {first[2]:.0f}th "
              f"percentile of {first[0]} usage and the {last[2]:.0f}th "
              f"percentile of {last[0]} usage.")
        print("Rising percentile = the field moved away from this paper's "
              "usage; falling = the field moved toward it.")
    print("\nCAUTION: one paper is a handful of sentences — read the nearest "
          "titles before interpreting, and prefer percentiles to raw "
          "distances (raw cosines are anisotropy-compressed).")


if __name__ == "__main__":
    main()
