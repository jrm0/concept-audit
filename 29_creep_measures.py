"""
29_creep_measures.py — the concept-creep instrument (Baes / Haslam / Vylomova),
re-implemented on OUR corpus so Paper 1 can report their measures beside ours.

The positioning memo's substitution test asks: could the concept-creep group
produce our result with their instrument? The only honest way to answer is to
run their instrument on the same documents. This script computes, per family
term x period (and per year for the bare parent term):

  SENTIMENT (valence)   repetition-weighted mean Warriner valence of collocates
                        within +/-5 cleaned tokens of the target (1-9, 9 = happy)
  AROUSAL               same, Warriner arousal (1-9, 9 = agitated)
  DOMINANCE             same, Warriner dominance (reported, not in their papers)
  SEVERITY              (9 - valence) + arousal, range 2-18 (Baes et al. 2023)
  PATHOLOGIZATION       share of collocate tokens in a 17-word disease lexicon
  INTENSIFIER INDEX     share of target mentions carrying one of 11 adjectival
                        intensifiers as an `amod` dependent (Baes et al. 2024)
  SALIENCE (family share)  documents mentioning the term / documents in the
                        corpus, per period. NOTE: our corpus is term-bounded,
                        so this is share-of-family, not relative frequency in
                        a general psychology corpus. Reported as such.
  BREADTH (theirs)      mean pairwise cosine DISTANCE between all-mpnet-base-v2
                        sentence embeddings of up to 50 sampled mention
                        sentences per period, averaged over 10 draws
  BREADTH (ours)        same sampling rule on 17's masked-target SciBERT
                        document vectors, so the two breadths are matched-n

Implementation follows the authors' public code (naomibaes/SemanticSeverity:
spaCy lemmatisation, punctuation/symbol/particle/number removal, stop-word
removal, +/-5 window on the cleaned stream, repetition-weighted means) and the
ACL 2024 description of breadth, the intensifier index and pathologization.
Multiword targets are collapsed to one slot, as in their preprocessing.

SOURCES (verified 2026-10-01 against the ACL Anthology / PLOS ONE / GitHub):
  Baes, N., Vylomova, E., Zyphur, M., & Haslam, N. (2023). The semantic
    inflation of "trauma" in psychology. Psychology of Language and
    Communication, 27(1), 23-45. doi:10.58734/plc-2023-0002
  Baes, N., Haslam, N., & Vylomova, E. (2024). A multidimensional framework
    for evaluating lexical semantic change with social science applications.
    Proc. ACL 2024 (Long Papers), 1390-1415.
  Xiao, Y., Baes, N., Vylomova, E., & Haslam, N. (2023). Have the concepts of
    'anxiety' and 'depression' been normalized or pathologized? PLOS ONE,
    18(6), e0288027.
  Warriner, A. B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence,
    arousal, and dominance for 13,915 English lemmas. Behavior Research
    Methods, 45(4), 1191-1207. doi:10.3758/s13428-012-0314-x
  Code: https://github.com/naomibaes/SemanticSeverity (warriner_rat.csv and
    the published trauma severity series data/output/df.csv are copied to
    data/external/ with provenance notes).

PHASES
  affect   parse mention sentences, write per-document collocate statistics to
           results/<slug>/creep_doc_stats.jsonl and per-chunk collocate counts
           to creep_collocates.jsonl (checkpointed at document level; rerun to
           continue, --max-seconds N to slice). Needs spaCy + en_core_web_sm.
           ~300 sentences/s on CPU; the trauma corpus (546k sentences) ~30 min.
  breadth  needs sentence-transformers (all-mpnet-base-v2) and 17's shards;
           writes results/<slug>/creep_breadth.json. Run on the machine that
           holds data/<slug>/lineage_emb*/.
  report   joins everything: tables per term x period with document-bootstrap
           CIs, OLS trends (HC3) on standardised series, the bare-parent
           per-year series against Baes et al. (2023)'s published trauma
           series, and the substitution-test summary.

USAGE
    python3 29_creep_measures.py --config configs/ptsd_trauma.yaml --phase affect
    python3 29_creep_measures.py --config configs/ptsd_trauma.yaml --phase breadth
    python3 29_creep_measures.py --config configs/ptsd_trauma.yaml --phase report
    options: --terms "trauma,ptsd"   --max-seconds N   --boot 500   --seed 42
             --min-n 15 (cell floor)   --n-sample 50 --reps 10 (breadth rule)
             --parent trauma (default: config citation.parent or 'trauma')
"""

import csv
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import pipeline_config as pc

CFG, _ARGV = pc.load()


def _opt(name, default=None):
    for i, a in enumerate(_ARGV):
        if a == name and i + 1 < len(_ARGV):
            return _ARGV[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


PHASE = re.sub(r"[^a-z]", "", (_opt("--phase", "report") or "").lower())   # strips stray paste characters
ONLY_TERMS = [t.strip() for t in (_opt("--terms") or "").split(",") if t.strip()]
MAX_SECONDS = float(_opt("--max-seconds", 0)) or None
BOOT = int(_opt("--boot", 500))
SEED = int(_opt("--seed", CFG.get("analysis", {}).get("seed", 42)))
MIN_N = int(_opt("--min-n", 15))
N_SAMPLE = int(_opt("--n-sample", 50))
REPS = int(_opt("--reps", 10))
_CIT = CFG.get("citation", {}) or {}
PARENT = pc.canon(_opt("--parent") or _CIT.get("parent") or "trauma")
# Label merges for the report (same convention as 27): CLI "a=b;c=d", or the
# config's citation.merge; default = the trauma case's PTSD spellings.
_m = _opt("--merge") or _CIT.get("merge")
if _m is None:
    MERGE = {"ptsd": "PTSD", "posttraumatic stress disorder": "PTSD",
             "complex ptsd": "complex PTSD", "complex posttraumatic stress disorder": "complex PTSD"}
elif isinstance(_m, str):
    MERGE = dict(kv.split("=", 1) for kv in _m.split(";") if "=" in kv)
else:
    MERGE = dict(_m)

DOCS = CFG.path("docs_path")
OUT = CFG.results
EXTERNAL = Path("data/external")
WARRINER = EXTERNAL / "warriner_rat.csv"
BAES_TRAUMA = EXTERNAL / "baes2023_trauma_severity_psych.csv"
DOC_STATS = OUT / "creep_doc_stats.jsonl"
COLLOCATES = OUT / "creep_collocates.jsonl"   # per (term, period, chunk) collocate counts, top 300
BREADTH_JSON = OUT / "creep_breadth.json"
WINDOW = 5

# Baes, Haslam & Vylomova (2024): 11 adjectival intensifiers; 17 disease words.
INTENSIFIERS = {"great", "intense", "severe", "harsh", "major", "extreme", "powerful",
                "serious", "devastating", "destructive", "debilitating"}
PATHOLOGY = {"ailment", "clinical", "clinic", "cure", "diagnosis", "disease", "disorder",
             "ill", "illness", "medical", "medicine", "pathology", "prognosis", "sick",
             "sickness", "symptom", "treatment"}
EXTRA_STOPS = {"%", "em", "<", ">", "β", "à", "×", "+", "@"}   # from preprocess_data_spacy.py
DROP_POS = {"PUNCT", "SYM", "PART", "SPACE", "NUM"}


# ----------------------------------------------------------------- loading

def load_docs():
    docs = []
    with open(DOCS, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def family_terms(docs):
    """Canonical family terms = every mention_sentences key seen (minus the
    generic control words); patterns from pipeline_config so surface variants
    (hyphen/space/prefix) match exactly as 16 matched them."""
    seen = Counter()
    for d in docs:
        for t in d.get("mention_sentences", {}):
            seen[t] += 1
    terms = [t for t in seen if t not in set(CFG.controls)]
    return {t: pc.term_pattern(t) for t in terms}


def doc_key(d):
    return d.get("doi") or ("title:" + pc.normalize_text(d.get("title", "")).lower()[:120])


def load_warriner():
    if not WARRINER.exists():
        raise SystemExit(f"{WARRINER} not found. Copy warriner_rat.csv from "
                         "github.com/naomibaes/SemanticSeverity/data/input/ (Warriner et al. 2013).")
    norms = {}
    with open(WARRINER, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            w = (row.get("word") or "").strip().lower()
            if not w or w in ("true", "false"):
                continue
            try:
                norms[w] = (float(row["V.Mean.Sum"]), float(row["A.Mean.Sum"]), float(row["D.Mean.Sum"]))
            except (KeyError, ValueError):
                continue
    return norms


# ------------------------------------------------------------ phase: affect

def phase_affect():
    import spacy
    from spacy.lang.en.stop_words import STOP_WORDS
    stops = set(STOP_WORDS) | EXTRA_STOPS
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    norms = load_warriner()
    docs = load_docs()
    pats = family_terms(docs)
    rules = pc.mention_rules(CFG)
    terms = [t for t in pats if (not ONLY_TERMS or t in ONLY_TERMS)]
    done = defaultdict(set)   # term -> doc keys already written (resume at document level)
    if DOC_STATS.exists():
        with open(DOC_STATS, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["term"]].add(r["key"])
    print(f"[29] affect: {len(terms)} terms; {sum(len(v) for v in done.values()):,} document rows already in {DOC_STATS.name}")
    t0 = time.time()
    CHUNK = 4000   # sentences per flush (cut at document boundaries)

    def new_stat():
        return {"n_mention": 0, "n_intens": 0, "n_coll": 0, "n_match": 0,
                "sumV": 0.0, "sumA": 0.0, "sumD": 0.0, "n_path": 0, "coll": Counter()}

    def flush(term, per_doc):
        with open(DOC_STATS, "a", encoding="utf-8") as out:
            for i, st in per_doc.items():
                d = docs[i]
                for c, n in st["coll"].items():
                    st["n_coll"] += n
                    if c in PATHOLOGY:
                        st["n_path"] += n
                    r = norms.get(c)
                    if r:
                        st["n_match"] += n
                        st["sumV"] += r[0] * n; st["sumA"] += r[1] * n; st["sumD"] += r[2] * n
                out.write(json.dumps({"term": term, "key": doc_key(d), "year": d.get("year"),
                                      "period": d.get("period"), "journal": d.get("journal"),
                                      "n_mention": st["n_mention"], "n_intens": st["n_intens"],
                                      "n_coll": st["n_coll"], "n_match": st["n_match"],
                                      "sumV": round(st["sumV"], 3), "sumA": round(st["sumA"], 3),
                                      "sumD": round(st["sumD"], 3), "n_path": st["n_path"]}) + "\n")
        by_period = defaultdict(Counter)
        for i, st in per_doc.items():
            by_period[docs[i].get("period")].update(st["coll"])
        with open(COLLOCATES, "a", encoding="utf-8") as out:
            for p, cnt in by_period.items():
                out.write(json.dumps({"term": term, "period": p, "n_docs": sum(1 for i in per_doc if docs[i].get("period") == p),
                                      "counts": cnt.most_common(300)}) + "\n")

    stopped = False
    for term in terms:
        tt = time.time()
        items = []   # (doc_idx, sentence), ordered by document
        for i, d in enumerate(docs):
            ss = d.get("mention_sentences", {}).get(term)
            if ss and doc_key(d) not in done[term]:
                for s in ss:
                    items.append((i, s))
        if not items:
            continue
        # cut into chunks at document boundaries
        chunks, cur = [], []
        for it in items:
            if len(cur) >= CHUNK and it[0] != cur[-1][0]:
                chunks.append(cur); cur = []
            cur.append(it)
        if cur:
            chunks.append(cur)
        n_done = 0
        for chunk in chunks:
            per_doc = defaultdict(new_stat)
            for (i, sent), sp in zip(chunk, nlp.pipe([s for _, s in chunk], batch_size=256)):
                spans = pc.valid_spans(term, pc.all_spans(sent, pats), sent, rules)
                if not spans:
                    per_doc[i]   # a document with no valid span still gets a (zero) row
                    continue
                st = per_doc[i]
                tok_in_target = [False] * len(sp)
                heads = []
                for (a, b) in spans:
                    idx = [t.i for t in sp if t.idx < b and (t.idx + len(t.text)) > a]
                    if not idx:
                        continue
                    for k in idx:
                        tok_in_target[k] = True
                    inside = set(idx)
                    heads.append(next((sp[k] for k in idx if sp[k].head.i not in inside), sp[idx[-1]]))
                for h in heads:
                    st["n_mention"] += 1
                    if any(c.dep_ == "amod" and c.lemma_.lower() in INTENSIFIERS for c in h.children):
                        st["n_intens"] += 1
                stream, prev_target = [], False
                for k, t in enumerate(sp):
                    if tok_in_target[k]:
                        if not prev_target:
                            stream.append(None)
                        prev_target = True
                        continue
                    prev_target = False
                    if t.pos_ in DROP_POS:
                        continue
                    lem = t.lemma_.lower().strip("-.")
                    if not lem or lem in stops or not re.search(r"[a-z]", lem):
                        continue
                    stream.append(lem)
                for k, tok in enumerate(stream):
                    if tok is not None:
                        continue
                    for c in stream[max(0, k - WINDOW):k] + stream[k + 1:k + 1 + WINDOW]:
                        if c is not None:
                            st["coll"][c] += 1
            flush(term, per_doc)
            n_done += len(per_doc)
            if MAX_SECONDS and time.time() - t0 > MAX_SECONDS:
                stopped = True
                break
        print(f"  {term:34} docs={n_done:>6} sentences={len(items):>7}  {time.time() - tt:5.0f}s"
              + ("  (partial — rerun to continue)" if stopped else ""))
        if stopped:
            print("[29] --max-seconds reached; rerun to continue")
            break
    print(f"[29] affect done in {time.time() - t0:.0f}s")


# ----------------------------------------------------------- phase: breadth

def _pairwise_mean_cos_dist(X):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = X @ X.T
    n = len(X)
    iu = np.triu_indices(n, 1)
    return float(1.0 - S[iu].mean())


def _sampled_breadth(X, rng):
    """Baes et al. (2024) rule: up to N_SAMPLE vectors per interval, REPS draws,
    mean pairwise cosine distance averaged over draws."""
    n = len(X)
    if n < MIN_N:
        return None
    vals = []
    for _ in range(REPS):
        idx = rng.choice(n, size=min(N_SAMPLE, n), replace=False)
        vals.append(_pairwise_mean_cos_dist(X[idx]))
    return {"mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals)), 4), "n_pool": int(n)}


def phase_breadth():
    rng = np.random.default_rng(SEED)
    docs = load_docs()
    pats = family_terms(docs)
    terms = [t for t in pats if (not ONLY_TERMS or t in ONLY_TERMS)]
    # the generic method words (config control_terms) are embedded by 17 from the
    # same abstracts: their breadth series is the null for the family series.
    # If "participants" narrows over the decades as much as "trauma" does, the
    # narrowing is the encoder's or the register's, not the concept's (the
    # SciBERT pretraining-cutoff control owed since Session 37).
    terms = terms + [c for c in CFG.controls if c not in terms]
    out = {"settings": {"n_sample": N_SAMPLE, "reps": REPS, "min_n": MIN_N, "seed": SEED, "controls": CFG.controls,
                        "sentence_model": "sentence-transformers/all-mpnet-base-v2",
                        "masked_model": CFG.get("embedding", {}).get("model"), "emb_dir": CFG["emb_dir"]},
           "cells": {}}
    # --- theirs: sentence embeddings of mention sentences (one sentence per document)
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    except Exception as e:   # noqa
        st = None
        print(f"[29] sentence-transformers unavailable ({e!r}); skipping mpnet breadth")
    pool = defaultdict(list)   # (term, period) -> sentences
    for d in docs:
        for t in terms:
            ss = d.get("mention_sentences", {}).get(t)
            if ss and d.get("period"):
                pool[(t, d["period"])].append(ss[rng.integers(len(ss))])
    if st is not None:
        for (t, p), ss in sorted(pool.items()):
            if len(ss) < MIN_N:
                continue
            keep = ss if len(ss) <= N_SAMPLE * REPS else [ss[i] for i in rng.choice(len(ss), N_SAMPLE * REPS, replace=False)]
            X = st.encode(keep, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
            cell = out["cells"].setdefault(f"{t}|{p}", {})
            cell["breadth_mpnet"] = _sampled_breadth(np.asarray(X, dtype=float), rng)
            cell["breadth_mpnet"]["n_pool"] = len(ss)
    # --- ours: masked-target document vectors from 17's shards, same rule
    emb = CFG.path("emb_dir")
    n_sh = 0
    for (t, p) in sorted(pool):
        f = emb / f"{pc.slug(t)}__{p}.npz"
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=True)
        X = np.asarray(z["vectors"], dtype=float)
        if len(X) < MIN_N:
            continue
        cell = out["cells"].setdefault(f"{t}|{p}", {})
        cell["breadth_masked"] = _sampled_breadth(X, rng)
        n_sh += 1
    print(f"[29] breadth: {len(out['cells'])} cells; masked shards used = {n_sh}")
    BREADTH_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    pc.write_manifest(CFG, __file__, params=out["settings"], outputs=[BREADTH_JSON])


# ------------------------------------------------------------ phase: report

def _agg(rows):
    """Repetition-weighted indices over a set of per-document rows (Baes:
    sum(rating*repet)/sum(repet))."""
    nm = sum(r["n_match"] for r in rows)
    nc = sum(r["n_coll"] for r in rows)
    men = sum(r["n_mention"] for r in rows)
    if nm == 0 or men == 0:
        return None
    V = sum(r["sumV"] for r in rows) / nm
    A = sum(r["sumA"] for r in rows) / nm
    D = sum(r["sumD"] for r in rows) / nm
    return {"valence": V, "arousal": A, "dominance": D, "severity": (9 - V) + A,
            "pathologization": sum(r["n_path"] for r in rows) / nc if nc else float("nan"),
            "intensifier": sum(r["n_intens"] for r in rows) / men,
            "coverage": nm / nc if nc else float("nan"),
            "n_docs": len(rows), "n_mentions": men, "n_collocates": nc}


KEYS = ["valence", "arousal", "dominance", "severity", "pathologization", "intensifier"]


def _boot(rows, rng):
    if len(rows) < MIN_N:
        return {}
    vals = {k: [] for k in KEYS}
    n = len(rows)
    for _ in range(BOOT):
        samp = [rows[i] for i in rng.integers(n, size=n)]
        a = _agg(samp)
        if a is None:
            continue
        for k in KEYS:
            vals[k].append(a[k])
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
            for k, v in vals.items() if v}


def _trend(xs, ys):
    """OLS of a standardised series on standardised period index with HC3
    SEs (their reporting convention). Returns beta, p, n."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = ~np.isnan(ys)
    xs, ys = xs[ok], ys[ok]
    if len(xs) < 4 or np.std(ys) == 0:
        return None
    zx = (xs - xs.mean()) / xs.std()
    zy = (ys - ys.mean()) / ys.std()
    try:
        import statsmodels.api as sm
        m = sm.OLS(zy, sm.add_constant(zx)).fit(cov_type="HC3")
        return {"beta": round(float(m.params[1]), 3), "p": round(float(m.pvalues[1]), 4), "n": int(len(xs))}
    except Exception:
        b = float(np.polyfit(zx, zy, 1)[0])
        return {"beta": round(b, 3), "p": None, "n": int(len(xs))}


def _period_mid(p):
    a, b = p.split("-")
    return (int(a) + int(b)) / 2


def phase_report():
    rng = np.random.default_rng(SEED)
    import gzip
    gz = DOC_STATS.with_suffix(".jsonl.gz")
    if DOC_STATS.exists():
        fh = open(DOC_STATS, encoding="utf-8")
    elif gz.exists():   # committed copies are gzipped (the plain file exceeds the sync limit)
        fh = gzip.open(gz, "rt", encoding="utf-8")
    else:
        raise SystemExit(f"{DOC_STATS} missing — run --phase affect first")
    rows_by = defaultdict(list)
    by_year_parent = defaultdict(list)
    with fh as f:
        for line in f:
            r = json.loads(line)
            if ONLY_TERMS and r["term"] not in ONLY_TERMS:
                continue
            if r["n_mention"] == 0:
                continue   # resume placeholder: no valid span in the stored sentences
            r["term"] = MERGE.get(r["term"], r["term"])
            if r["period"]:
                rows_by[(r["term"], r["period"])].append(r)
            if r["term"] == PARENT and r.get("year"):
                by_year_parent[int(r["year"])].append(r)
    breadth = json.loads(BREADTH_JSON.read_text(encoding="utf-8")) if BREADTH_JSON.exists() else {"cells": {}}
    n_docs_period = Counter()
    for d in load_docs():
        if d.get("period"):
            n_docs_period[d["period"]] += 1
    periods = sorted(n_docs_period)
    terms = sorted({t for t, _ in rows_by}, key=lambda t: -sum(len(v) for (tt, _), v in rows_by.items() if tt == t))

    table = {}
    trends = {}
    for t in terms:
        series = {}
        for p in periods:
            rows = rows_by.get((t, p), [])
            if len(rows) < MIN_N:
                continue
            a = _agg(rows)
            if a is None:
                continue
            a = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in a.items()}
            a["ci95"] = _boot(rows, rng)
            a["family_share"] = round(len(rows) / n_docs_period[p], 4)
            b = breadth["cells"].get(f"{t}|{p}", {})
            if not b:   # merged label: take the largest-pool raw cell
                cands = [breadth["cells"][k] for k in breadth["cells"]
                         if k.endswith(f"|{p}") and MERGE.get(k.split("|")[0]) == t]
                if cands:
                    b = max(cands, key=lambda c: (c.get("breadth_mpnet") or c.get("breadth_masked") or {}).get("n_pool", 0))
            for k in ("breadth_mpnet", "breadth_masked"):
                if b.get(k):
                    a[k] = b[k]["mean"]; a[k + "_sd"] = b[k]["sd"]
            series[p] = a
        if not series:
            continue
        table[t] = series
        xs = [_period_mid(p) for p in series]
        trends[t] = {}
        for k in KEYS + ["family_share", "breadth_mpnet", "breadth_masked"]:
            ys = [series[p].get(k, np.nan) for p in series]
            tr = _trend(xs, ys)
            if tr:
                trends[t][k] = tr

    # --- parent per year vs Baes et al. (2023) published psychology series
    comparison = None
    yearly = {}
    for y in sorted(by_year_parent):
        rows = by_year_parent[y]
        if len(rows) >= MIN_N:
            a = _agg(rows)
            if a:
                yearly[y] = {k: round(a[k], 4) for k in ("valence", "arousal", "severity")} | {"n_docs": len(rows)}
    if BAES_TRAUMA.exists() and PARENT == "trauma" and yearly:
        theirs = {}
        with open(BAES_TRAUMA, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    theirs[int(row["year"])] = {"severity": float(row["sev_word"]), "arousal": float(row["aro_word"]),
                                                "valence_rev": float(row["val_word"])}
                except (KeyError, ValueError):
                    continue
        common = sorted(set(theirs) & set(yearly))
        if len(common) >= 5:
            def corr(a, b):
                return round(float(np.corrcoef(a, b)[0, 1]), 3)
            ours_sev = [yearly[y]["severity"] for y in common]; th_sev = [theirs[y]["severity"] for y in common]
            ours_aro = [yearly[y]["arousal"] for y in common]; th_aro = [theirs[y]["arousal"] for y in common]
            ours_val = [9 - yearly[y]["valence"] for y in common]; th_val = [theirs[y]["valence_rev"] for y in common]
            comparison = {"years": [common[0], common[-1]], "n_years": len(common),
                          "severity": {"r": corr(ours_sev, th_sev), "ours_mean": round(np.mean(ours_sev), 3),
                                       "theirs_mean": round(np.mean(th_sev), 3)},
                          "arousal": {"r": corr(ours_aro, th_aro), "ours_mean": round(np.mean(ours_aro), 3),
                                      "theirs_mean": round(np.mean(th_aro), 3)},
                          "valence_reversed": {"r": corr(ours_val, th_val), "ours_mean": round(np.mean(ours_val), 3),
                                               "theirs_mean": round(np.mean(th_val), 3)},
                          "trend_theirs_severity": _trend(common, th_sev),
                          "trend_ours_severity": _trend(common, ours_sev),
                          "note": "Their series: psychology abstracts 1970-2017 (Baes et al. 2023, "
                                  "data/output/df.csv in the SemanticSeverity repo). Ours: this corpus, "
                                  "bare parent term, same collocate rule. Levels comparable; the corpora "
                                  "differ in construction (theirs journal-bounded, ours term-bounded)."}

    # --- collocate tables: top collocates per term x period, and the movers
    #     between the first and last adequately-sized period (what drives a
    #     valence/arousal trend is readable here)
    coll = defaultdict(Counter)
    if COLLOCATES.exists():
        with open(COLLOCATES, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                t = MERGE.get(r["term"], r["term"])
                if r.get("period"):
                    coll[(t, r["period"])].update(dict(r["counts"]))
    collocates = {}
    norms = load_warriner() if WARRINER.exists() else {}
    for t in table:
        ps = [p for p in table[t] if table[t][p]["n_docs"] >= 100]
        entry = {"top": {p: coll[(t, p)].most_common(25) for p in table[t] if coll.get((t, p))}}
        if len(ps) >= 2:
            a, b = coll[(t, ps[0])], coll[(t, ps[-1])]
            na, nb = sum(a.values()) or 1, sum(b.values()) or 1
            words = {w for w, c in a.items() if c >= 5} | {w for w, c in b.items() if c >= 5}
            delta = {w: b[w] / nb - a[w] / na for w in words}
            def tag(w):
                r = norms.get(w)
                return [w, round(delta[w] * 1000, 2), round(r[0], 2) if r else None, round(r[1], 2) if r else None]
            up = sorted(words, key=lambda w: -delta[w])[:20]
            down = sorted(words, key=lambda w: delta[w])[:20]
            entry["movers"] = {"from": ps[0], "to": ps[-1], "unit": "per-mille of collocate tokens; [word, delta, valence, arousal]",
                               "up": [tag(w) for w in up], "down": [tag(w) for w in down]}
        collocates[t] = entry

    # --- control-term breadth (the null for the breadth series)
    controls = {}
    for c in CFG.controls:
        ser = {}
        for p in periods:
            b = breadth["cells"].get(f"{c}|{p}", {})
            if b:
                ser[p] = {k: b[k]["mean"] for k in ("breadth_mpnet", "breadth_masked") if b.get(k)}
        if ser:
            xs = [_period_mid(p) for p in ser]
            controls[c] = {"series": ser,
                           "trends": {k: _trend(xs, [ser[p].get(k, np.nan) for p in ser])
                                      for k in ("breadth_mpnet", "breadth_masked")}}

    out = {"settings": {"window": WINDOW, "min_n": MIN_N, "boot": BOOT, "seed": SEED, "parent": PARENT,
                        "breadth": breadth.get("settings"), "merge": MERGE, "intensifiers": sorted(INTENSIFIERS),
                        "pathology_lexicon": sorted(PATHOLOGY)},
           "periods": periods, "n_docs_period": dict(n_docs_period),
           "table": table, "trends": trends, "collocates": collocates, "control_breadth": controls, "parent_by_year": yearly, "comparison_baes2023": comparison}
    (OUT / "creep_measures.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    # --- text report
    L = [f"CONCEPT-CREEP MEASURES on the {CFG['concept']} corpus  (window +/-{WINDOW}, cell floor {MIN_N} docs, "
         f"{BOOT} document bootstraps)", ""]
    L.append("== trends (standardised OLS beta on period midpoint, HC3; * p<.05) ==")
    hdr = f"  {'term':30}" + "".join(f"{k[:11]:>12}" for k in KEYS + ["fam_share", "br_mpnet", "br_masked"])
    L.append(hdr)
    for t in terms:
        if t not in trends:
            continue
        cells = []
        for k in KEYS + ["family_share", "breadth_mpnet", "breadth_masked"]:
            tr = trends[t].get(k)
            if not tr:
                cells.append(f"{'-':>12}")
            else:
                star = "*" if (tr["p"] is not None and tr["p"] < .05) else " "
                cells.append(f"{tr['beta']:>+10.2f}{star} ")
        L.append(f"  {t:30}" + "".join(cells))
    L.append("")
    for t in terms:
        if t not in table:
            continue
        L.append(f"== {t} ==")
        L.append(f"  {'period':10}{'n':>7}{'val':>7}{'aro':>7}{'sev':>7}{'path':>7}{'intens':>8}{'cover':>7}"
                 f"{'share':>8}{'br_mp':>8}{'br_mask':>9}")
        for p, a in table[t].items():
            L.append(f"  {p:10}{a['n_docs']:>7}{a['valence']:>7.2f}{a['arousal']:>7.2f}{a['severity']:>7.2f}"
                     f"{a['pathologization']:>7.3f}{a['intensifier']:>8.3f}{a['coverage']:>7.2f}{a['family_share']:>8.3f}"
                     f"{a.get('breadth_mpnet', float('nan')):>8.3f}{a.get('breadth_masked', float('nan')):>9.3f}")
        L.append("")
    for t in terms:
        mv = collocates.get(t, {}).get("movers")
        if not mv or t not in (PARENT, *[x for x in terms if table.get(x) and len(table[x]) >= 6][:6]):
            continue
        L.append(f"== collocate movers: {t}  ({mv['from']} -> {mv['to']}; delta per mille; [valence, arousal]) ==")
        L.append("  up:   " + ", ".join(f"{w} {d:+.1f} [{v},{a}]" for w, d, v, a in mv["up"][:12]))
        L.append("  down: " + ", ".join(f"{w} {d:+.1f} [{v},{a}]" for w, d, v, a in mv["down"][:12]))
        L.append("")
    if controls:
        L.append("== control-term breadth (generic method words from the same abstracts; the null for the breadth trends) ==")
        L.append(f"  {'control':16}" + "".join(f"{p[:9]:>10}" for p in periods) + f"{'beta_mp':>10}{'beta_mask':>11}")
        for c, v in controls.items():
            for k, lab in (("breadth_mpnet", "mp"), ("breadth_masked", "mask")):
                row = "".join(f"{v['series'].get(p, {}).get(k, float('nan')):>10.3f}" for p in periods)
                tr = v["trends"].get(k)
                b = f"{tr['beta']:+.2f}{'*' if tr and tr['p'] is not None and tr['p'] < .05 else ' '}" if tr else "-"
                L.append(f"  {c + ' ' + lab:16}" + row + (f"{b:>10}" if lab == "mp" else f"{'':>10}{b:>11}"))
        L.append("  (a family term's breadth trend is only interpretable net of these: a control that narrows as much as the "
                 "term is an encoder/register effect, not a conceptual one)")
        L.append("")
    if comparison:
        L.append("== bare parent per year vs Baes et al. (2023) psychology series ==")
        for k in ("severity", "arousal", "valence_reversed"):
            c = comparison[k]
            L.append(f"  {k:18} r = {c['r']:+.3f}   ours {c['ours_mean']:.2f}  theirs {c['theirs_mean']:.2f}"
                     f"   ({comparison['years'][0]}-{comparison['years'][1]}, {comparison['n_years']} years)")
        L.append(f"  severity trend: theirs {comparison['trend_theirs_severity']}  ours {comparison['trend_ours_severity']}")
        L.append("")
    L.append("Reading guide: valence 1-9 (9 = happy); arousal 1-9 (9 = agitated); severity = (9 - valence) + arousal; "
             "pathologization = share of collocate tokens in the 17-word disease lexicon; intensifier = share of mentions "
             "with an adjectival intensifier; coverage = share of collocate tokens with a Warriner norm; "
             "family_share = documents mentioning the term / corpus documents in the period (term-bounded corpus); "
             "breadth = mean pairwise cosine distance, 50 x 10 draws (mpnet sentence vectors / masked SciBERT doc vectors).")
    (OUT / "creep_measures_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    pc.write_manifest(CFG, __file__, params=out["settings"],
                      outputs=[OUT / "creep_measures.json", OUT / "creep_measures_report.txt"],
                      notes="Concept-creep instrument (Baes/Haslam/Vylomova) re-implemented on this corpus for the "
                            "head-to-head in Paper 1; see docstring for sources.")


if __name__ == "__main__":
    {"affect": phase_affect, "breadth": phase_breadth, "report": phase_report}[PHASE]()
