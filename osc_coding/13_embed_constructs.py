"""
13_embed_constructs.py — Masked-target document embeddings for every OSC-99
PRIMARY construct, plus each original study's own usage. (LOCAL: torch.)

Identical embedding procedure to the lineage pipeline (code/17): the
construct term is replaced by [MASK] in each mention sentence, the vector
is the last-4-layer mean at the mask position, one vector per document
(mean over <=3 mention sentences). Identity of procedure is what makes the
original paper's usage commensurable with the field distribution.

Sources per construct (deduped by DOI then normalized title):
  * data/construct_corpus/{slug}.jsonl   (EBSCO + OpenAlex pulls)
  * ../data/abstracts/*.jsonl            (the 53k pool — rich constructs
                                          were never copied into
                                          construct_corpus; scanned once)

Two windows: PRE (year < 2010; the original studies are 2008) and POST
(2010-2025; the replication era — RP:P replications ran 2011-2014,
published 2015). Cap 250 docs per (construct, window), seeded.

Also embeds each study's OWN usage of its primary construct from
data/full_texts/study_XXX.txt -> osc_originals.npz (pilot_id-keyed).

Checkpointed per shard; re-runs skip finished files.
Usage (from code/osc_coding/):
    caffeinate -i python3 13_embed_constructs.py
    python3 13_embed_constructs.py 015 049        # subset of studies
"""

import csv
import glob
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
CAP_PER_WINDOW = 250
MAX_SENT_PER_DOC = 3
MIN_ABS = 200

FINAL = Path("data/final_constructs.csv")
CORPUS_DIR = Path("../data/construct_corpus")
POOL_GLOB = "../data/abstracts/*.jsonl"
TEXTS = Path("data/full_texts")
OUT = Path("data/construct_emb")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def pat_of(term):
    body = re.escape(term.lower()).replace(r"\ ", "[-\\s]+") \
        .replace(r"\-", "[-\\s]+")
    return re.compile(r"\b" + body + r"\b", re.I)


def norm_title(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()[:120]


def year_int(y):
    """Years arrive as int, str ('2008'), or garbage depending on source."""
    try:
        y = int(str(y)[:4])
        return y if 1900 <= y <= 2030 else None
    except (TypeError, ValueError):
        return None


def window_of(year):
    y = year_int(year)
    if y is None:
        return None
    return "pre" if y < 2010 else "post"


def mention_sentences(title, abstract, pat):
    text = f"{title}. {abstract}"
    return [s.strip() for s in SENT_RE.split(text)
            if 30 <= len(s.strip()) <= 1200 and pat.search(s)]


def main():
    only = {a.zfill(3) for a in sys.argv[1:] if a.isdigit()}
    # --all-slots: also embed constructs 2-4 (the study's full construct
    # set) for the pre-specified SECONDARY analysis (set-level mismatch,
    # aggregator = max across constructs). Default remains primary-only.
    all_slots = "--all-slots" in sys.argv
    n_slots = 4 if all_slots else 1
    rows = list(csv.DictReader(open(FINAL, encoding="utf-8")))
    studies = {}                     # pilot_id -> [(slot, construct)]
    for r in rows:
        pid = r["pilot_id"].strip().zfill(3)
        if only and pid not in only:
            continue
        lst = []
        for i in range(1, n_slots + 1):
            c = (r.get(f"construct_{i}") or "").strip().lower()
            if c:
                lst.append((i, c))
        if lst:
            studies[pid] = lst
    constructs = sorted({c for lst in studies.values() for _, c in lst})
    pats = {c: pat_of(c) for c in constructs}
    print(f"{len(studies)} studies, {len(constructs)} constructs "
          f"({'all slots' if all_slots else 'primaries only'})")
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- gather documents per construct (corpus files + one pool scan)
    docs = defaultdict(dict)         # construct -> key -> doc
    for c in constructs:
        f = CORPUS_DIR / f"{slug(c)}.jsonl"
        if not f.exists():
            continue
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if len(d.get("abstract") or "") < MIN_ABS:
                continue
            key = (d.get("doi") or "").lower() or norm_title(d.get("title"))
            docs[c].setdefault(key, d)
    need_pool = [c for c in constructs]
    print("scanning abstract pool once ...")
    for pf in glob.glob(POOL_GLOB):
        for line in open(pf, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ab = d.get("abstract") or ""
            if len(ab) < MIN_ABS:
                continue
            text = ((d.get("title") or "") + ". " + ab).lower()
            for c in need_pool:
                if c[:4] in text and pats[c].search(text):
                    key = (d.get("doi") or "").lower() \
                        or norm_title(d.get("title"))
                    docs[c].setdefault(key, d)
    for c in constructs:
        n = len(docs[c])
        pre = sum(1 for d in docs[c].values()
                  if window_of(d.get("year")) == "pre")
        print(f"  {c}: {n} docs (pre {pre} / post {n - pre})")

    # ---- model
    import torch
    from transformers import AutoModel, AutoTokenizer
    print(f"loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    def masked_vector(sentence, pat):
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

    def embed_docs(dlist, pat):
        vecs, keys, years = [], [], []
        for d in dlist:
            sents = mention_sentences(d.get("title") or "",
                                      d.get("abstract") or "",
                                      pat)[:MAX_SENT_PER_DOC]
            sv = [v for v in (masked_vector(s, pat) for s in sents)
                  if v is not None]
            if not sv:
                continue
            vecs.append(np.mean(sv, axis=0))
            keys.append((d.get("doi") or norm_title(d.get("title")))[:80])
            years.append(year_int(d.get("year")) or 0)
        return vecs, keys, years

    # ---- field shards
    total = 0
    for ci, c in enumerate(constructs):
        pat = pats[c]
        by_w = defaultdict(list)
        for d in docs[c].values():
            w = window_of(d.get("year"))
            if w:
                by_w[w].append(d)
        for w, dlist in sorted(by_w.items()):
            out_path = OUT / f"{slug(c)}__{w}.npz"
            if out_path.exists():
                continue
            if len(dlist) > CAP_PER_WINDOW:
                random.seed(SEED)
                dlist = random.sample(dlist, CAP_PER_WINDOW)
            vecs, keys, years = embed_docs(dlist, pat)
            if vecs:
                np.savez_compressed(out_path, vectors=np.array(vecs),
                                    keys=np.array(keys, dtype=object),
                                    years=np.array(years))
                total += len(vecs)
                print(f"[{ci+1}/{len(constructs)}] {c} | {w}: "
                      f"{len(vecs)} vectors")

    # ---- each study's own usage (per slot), plus PER-SENTENCE vectors for
    # the split-half reliability check of the position metric itself
    prim = {"v": [], "p": [], "n": []}
    allo = {"p": [], "s": [], "c": [], "v": [], "n": []}
    sv_store = {"p": [], "s": [], "v": []}
    for pid, lst in sorted(studies.items()):
        f = TEXTS / f"study_{pid}.txt"
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        title = ""
        m = re.search(r"^TITLE: (.*)$", text, re.M)
        if m:
            title = m.group(1)
        body = re.sub(r"^(TITLE|AUTHORS|JOURNAL|DOI|SOURCE):.*$", "", text,
                      flags=re.M)
        for slot, c in lst:
            # up to 6 sentence-vectors are SAVED (split-half reliability
            # needs >=2), but the original's MEAN vector uses only the
            # first MAX_SENT_PER_DOC — the same cap as field documents.
            # Averaging more sentences shrinks a vector toward the
            # centroid, which would deflate percentiles for long-abstract
            # originals (sentence-count bias).
            sents = mention_sentences(title, body,
                                      pats[c])[:MAX_SENT_PER_DOC + 3]
            sv = [v for v in (masked_vector(s, pats[c]) for s in sents)
                  if v is not None]
            if not sv:
                if slot == 1:
                    print(f"  !! study {pid}: no embeddable mention of "
                          f"'{c}' in its own text")
                continue
            mv = np.mean(sv[:MAX_SENT_PER_DOC], axis=0)
            if slot == 1:
                prim["v"].append(mv)
                prim["p"].append(pid)
                prim["n"].append(len(sv))
            allo["p"].append(pid)
            allo["s"].append(slot)
            allo["c"].append(c)
            allo["v"].append(mv)
            allo["n"].append(len(sv))
            for v in sv:
                sv_store["p"].append(pid)
                sv_store["s"].append(slot)
                sv_store["v"].append(v)
    np.savez_compressed(OUT / "osc_originals.npz",
                        vectors=np.array(prim["v"]),
                        pilot_ids=np.array(prim["p"], dtype=object),
                        n_sentences=np.array(prim["n"]))
    np.savez_compressed(OUT / "osc_originals_all.npz",
                        vectors=np.array(allo["v"]),
                        pilot_ids=np.array(allo["p"], dtype=object),
                        slots=np.array(allo["s"]),
                        constructs=np.array(allo["c"], dtype=object),
                        n_sentences=np.array(allo["n"]))
    np.savez_compressed(OUT / "osc_original_sentvecs.npz",
                        vectors=np.array(sv_store["v"]),
                        pilot_ids=np.array(sv_store["p"], dtype=object),
                        slots=np.array(sv_store["s"]))
    print(f"\n{total} field vectors; {len(prim['p'])}/{len(studies)} "
          f"primary originals embedded -> {OUT}/")
    print("Next: python3 15_paper2_model.py")


if __name__ == "__main__":
    main()
