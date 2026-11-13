"""
31_ground_truth_cooccurrence.py — the historical ground-truth tests
(Historical_Ground_Truth_Claims_v1.md; Young 1995, Leys 2000, Fassin &
Rechtman 2009), as vocabulary co-occurrence series with pre-registered cells.

Every cell is a share: documents in a DENOMINATOR set whose title+abstract
matches a VOCABULARY, per 5-year period and per year, with a document
bootstrap 95% CI. The vocabularies, denominators, thresholds and the
SEEN/OPEN status of each cell live in the config block `ground_truth:` and
were fixed in the Lab Notebook (Session 46) BEFORE this script was first run
on the corpus. The script evaluates each criterion and prints PASS / FAIL /
NA with the numbers, so the scoring is mechanical; the reading is not.

Three kinds of claim earn different things (memo §0): VALIDITY cells (the
books agree) test the instrument; DISCRIMINATING cells (they disagree) are
adjudicated, never counted toward validity; FORECAST cells (written before
the data existed) are the strongest test and count against the historian
when they fail, provided the validity cells passed.

Text source: abstracts re-read from the RIS exports the config lists (as 28
does); lineage_docs keeps only mention sentences. Denominators are TERM sets
from lineage_docs `terms` (PTSD spellings merged as in 27/28/29).

Addendum A (2026-10-05) additions:
  * every document's vocabulary flags are written to
    results/<slug>/ground_truth_doc_flags.jsonl (doc_key as 26 builds it), so
    27 can turn a vocabulary into a citation pseudo-label (@abuse, @veteran);
  * if results/<slug>/instrument_mentions.jsonl (from 28) exists, every
    instrument family becomes a pseudo-vocabulary `inst:<family>` matched by
    DOI or title, so a cell can read the MEASUREMENT facet (A5: `inst:biological measure`).

USAGE
    python3 31_ground_truth_cooccurrence.py --config configs/ptsd_trauma.yaml
    options: --boot 500  --seed 42  --min-n 15  --show-annual (print annual rows for annual cells)
"""

import glob
import hashlib
import json
import re
import time
from collections import defaultdict
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


BOOT = int(_opt("--boot", 500))
SEED = int(_opt("--seed", 42))
MIN_N = int(_opt("--min-n", 15))
SHOW_ANNUAL = "--show-annual" in _ARGV
OUT = CFG.results
DOCS = CFG.path("docs_path")
GT = CFG.get("ground_truth") or {}
if not GT:
    raise SystemExit("config has no ground_truth: block (vocabularies, denominators, cells)")
_CIT = CFG.get("citation", {}) or {}
_m = _CIT.get("merge")
MERGE = ({"ptsd": "PTSD", "posttraumatic stress disorder": "PTSD",
          "complex ptsd": "complex PTSD", "complex posttraumatic stress disorder": "complex PTSD"}
         if _m is None else dict(_m))


# ------------------------------------------------------------------ text

def norm_title(title):
    t = re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()
    return t[:120] if len(t) >= 25 else ""


TAG_RE = re.compile(r"^([A-Z][A-Z0-9])  - (.*)$")


def parse_ris(path):
    recs, cur, last = [], {}, None
    for line in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        m = TAG_RE.match(line)
        if not m:
            if cur and last:
                cur[last][-1] += " " + line.strip()
            continue
        tag, val = m.group(1), m.group(2).strip()
        if tag == "ER":
            if cur.get("TI"):
                recs.append(cur)
            cur, last = {}, None
            continue
        cur.setdefault(tag, []).append(val)
        last = tag
    if cur.get("TI"):
        recs.append(cur)
    return recs


def load_docs_with_text():
    docs, by_doi, by_title = [], {}, {}
    with open(DOCS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            d["_terms"] = sorted({MERGE.get(pc.canon(t), pc.canon(t)) for t in (d.get("terms") or [])})
            docs.append(d)
            doi = (d.get("doi") or "").lower().strip()
            if doi:
                by_doi[doi] = d
            nt = norm_title(d.get("title"))
            if nt:
                by_title.setdefault(nt, d)
    sources = (CFG.get("corpus") or {}).get("sources") or []
    n_files = 0
    for s in sources:
        for g in s.get("files") or []:
            for path in sorted(glob.glob(g, recursive=True)):
                n_files += 1
                if path.endswith(".jsonl"):
                    with open(path, encoding="utf-8") as f:
                        for line in f:
                            try:
                                w = json.loads(line)
                            except Exception:
                                continue
                            d = by_doi.get((w.get("doi") or "").lower().strip()) or by_title.get(norm_title(w.get("title")))
                            if d is not None and "_text" not in d:
                                ab = (w.get("abstract") or w.get("text") or "").strip()
                                if ab:
                                    d["_text"] = f"{d.get('title', '')}. {ab}"
                    continue
                for r in parse_ris(path):
                    doi = (r.get("DO", [""])[0] or "").lower().strip().replace("https://doi.org/", "").rstrip(".")
                    d = by_doi.get(doi) if doi else None
                    if d is None:
                        d = by_title.get(norm_title(r.get("TI", [""])[0]))
                    if d is None or "_text" in d:
                        continue
                    ab = (r.get("AB", [""])[0] or "") or (r.get("N2", [""])[0] if r.get("N2") else "")
                    if ab.strip():
                        d["_text"] = f"{d.get('title', '')}. {ab}"
    have = [d for d in docs if "_text" in d]
    print(f"[31] exports read: {n_files}; documents with abstract text: {len(have):,} of {len(docs):,}")
    if not have:
        raise SystemExit("[31] no abstracts found — run from the code directory so corpus.sources globs resolve")
    return have


# ------------------------------------------------------------ vocabularies

def vocab_regex(words):
    """'cortisol|hippocamp*|heart rate' -> one word-boundary regex; '*' = \\w*."""
    parts = []
    for w in (words.split("|") if isinstance(words, str) else words):
        w = w.strip().lower()
        if not w:
            continue
        p = re.escape(w).replace(r"\*", r"\w*").replace(r"\ ", r"[\s\-]+")
        parts.append(p)
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b", re.I)


def denominator(doc, spec):
    """spec: 'all' | 'term:PTSD' | 'terms:a,b' | 'venue:regex' | 'anyterm:a,b' (= terms)"""
    if spec == "all":
        return True
    kind, _, arg = spec.partition(":")
    if kind in ("term", "terms", "anyterm"):
        want = {MERGE.get(pc.canon(t), pc.canon(t)) for t in arg.split(",")}
        return any(t in want for t in doc["_terms"])
    if kind == "venue":
        return re.search(arg, doc.get("journal") or "", re.I) is not None
    raise SystemExit(f"unknown denominator spec {spec!r}")


def period_of(doc):
    return doc.get("period")


def doc_key_of(d):
    """The document key 26 assigns (doi: or t:sha1(title|year)), so flags can be
    joined to the citation graph in 27."""
    doi = (d.get("doi") or "").strip().lower()
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^doi:\s*", "", doi)
    if doi:
        return "doi:" + doi
    basis = pc.normalize_text(d.get("title") or "") + "|" + str(d.get("year") or "")
    return "t:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def instrument_flags(docs):
    """{family: bool array over docs} from 28's instrument_mentions.jsonl, joined
    by DOI or normalised title. Empty if 28 has not run."""
    path = OUT / "instrument_mentions.jsonl"
    if not path.exists():
        return {}
    by_doi = {(d.get("doi") or "").lower().strip(): i for i, d in enumerate(docs) if d.get("doi")}
    by_title = {}
    for i, d in enumerate(docs):
        nt = norm_title(d.get("title"))
        if nt:
            by_title.setdefault(nt, i)
    fam_docs = defaultdict(set)
    n_lines = n_hit = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            m = json.loads(line)
            i = by_doi.get((m.get("doi") or "").lower().strip())
            if i is None:
                i = by_title.get(norm_title(m.get("title")))
            if i is None:
                continue
            n_hit += 1
            for ins in m.get("instruments") or []:
                fam_docs[ins["family"]].add(i)
    out = {}
    for fam, idx in fam_docs.items():
        a = np.zeros(len(docs), dtype=bool)
        a[list(idx)] = True
        out["inst:" + fam] = a
    print(f"[31] instrument mentions: {n_lines:,} documents in 28's output, {n_hit:,} joined to abstracts; "
          f"{len(out)} families registered as inst:<family> pseudo-vocabularies")
    return out


# ------------------------------------------------------------------ main

def boot_ci(hits, rng):
    n = len(hits)
    if n == 0:
        return (None, None)
    h = np.asarray(hits, dtype=np.int8)
    idx = rng.integers(n, size=(BOOT, n))
    means = h[idx].mean(1)
    return (round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4))


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    docs = load_docs_with_text()
    vocabs = {k: vocab_regex(v) for k, v in (GT.get("vocabularies") or {}).items()}
    print(f"[31] {len(vocabs)} vocabularies: {', '.join(vocabs)}")
    # match every vocabulary once per document
    flags = {k: np.zeros(len(docs), dtype=bool) for k in vocabs}
    for i, d in enumerate(docs):
        txt = d["_text"]
        for k, rx in vocabs.items():
            if rx.search(txt):
                flags[k][i] = True
    periods = sorted({d["period"] for d in docs if d.get("period")})
    years = sorted({int(d["year"]) for d in docs if d.get("year")})

    # series: for each (denominator, vocab) -> per period and per year share + CI + n
    series = {}
    def get_series(den, voc):
        key = f"{voc} | {den}"
        if key in series:
            return series[key]
        den_mask = np.array([denominator(d, den) for d in docs])
        out = {"by_period": {}, "by_year": {}, "denominator": den, "vocabulary": voc}
        for p in periods:
            sel = den_mask & np.array([d.get("period") == p for d in docs])
            n = int(sel.sum())
            if n >= MIN_N:
                h = flags[voc][sel]
                out["by_period"][p] = {"n": n, "share": round(float(h.mean()), 4), "ci": boot_ci(h, rng)}
        for y in years:
            sel = den_mask & np.array([d.get("year") == y for d in docs])
            n = int(sel.sum())
            if n >= MIN_N:
                h = flags[voc][sel]
                out["by_year"][y] = {"n": n, "share": round(float(h.mean()), 4), "ci": boot_ci(h, rng)}
        series[key] = out
        return out

    # --- special denominators that are term shares rather than vocab matches:
    #     'termshare' cells compare term sets (C1b, C1c) — handled as vocabularies
    #     over doc["_terms"] by registering pseudo-vocabs
    for name, spec in (GT.get("term_sets") or {}).items():
        want = {MERGE.get(pc.canon(t), pc.canon(t)) for t in spec}
        flags[name] = np.array([any(t in want for t in d["_terms"]) for d in docs])
        vocabs[name] = None
    # venue sets: a "vocabulary" that is true when the journal name matches
    for name, rx in (GT.get("venue_sets") or {}).items():
        r = re.compile(rx, re.I)
        flags[name] = np.array([r.search(d.get("journal") or "") is not None for d in docs])
        vocabs[name] = None
    # measurement facet: 28's instrument families as pseudo-vocabularies (Addendum A5)
    for name, a in instrument_flags(docs).items():
        flags[name] = a
        vocabs[name] = None
    # per-document flags for 27 (descent facet pseudo-labels, Addendum A2-A4)
    flag_names = [k for k in flags if not k.startswith("inst:")]
    fp = OUT / "ground_truth_doc_flags.jsonl"
    with open(fp, "w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            on = [k for k in flag_names if flags[k][i]]
            f.write(json.dumps({"doc_key": doc_key_of(d), "year": d.get("year"), "period": d.get("period"),
                                "flags": on}) + "\n")
    print(f"[31] wrote {fp.name}: {len(docs):,} documents x {len(flag_names)} flags")

    # --- evaluate cells
    results = []
    for cell in GT.get("cells") or []:
        cid, kind = cell["id"], cell.get("kind", "")
        s = get_series(cell["denominator"], cell["vocab"])
        bp = s["by_period"]; by = s["by_year"]
        def share(p):
            return bp.get(p, {}).get("share")
        def ci(p):
            return bp.get(p, {}).get("ci", (None, None))
        def yr_mean(a, b):
            vals = [by[y]["share"] for y in by if a <= y <= b]
            return (round(float(np.mean(vals)), 4), len(vals)) if vals else (None, 0)
        def yr_ci(a, b):
            # pooled CI over the years in range: approximate by averaging CI bounds
            lo = [by[y]["ci"][0] for y in by if a <= y <= b and by[y]["ci"][0] is not None]
            hi = [by[y]["ci"][1] for y in by if a <= y <= b and by[y]["ci"][1] is not None]
            return (round(float(np.mean(lo)), 4), round(float(np.mean(hi)), 4)) if lo else (None, None)
        test = cell["test"]; typ = test["type"]; res, detail = "NA", ""
        try:
            if typ == "share_at_least":       # share(period) >= value
                v = share(test["period"]); res = "NA" if v is None else ("PASS" if v >= test["value"] else "FAIL")
                detail = f"{test['period']}: {v} (>= {test['value']}?)"
            elif typ == "share_at_most":
                v = share(test["period"]); res = "NA" if v is None else ("PASS" if v <= test["value"] else "FAIL")
                detail = f"{test['period']}: {v} (<= {test['value']}?)"
            elif typ == "monotone_increase":  # each consecutive listed period higher than the last
                ps = test["periods"]; vals = [share(p) for p in ps]
                if any(v is None for v in vals):
                    res = "NA"
                else:
                    res = "PASS" if all(vals[i + 1] > vals[i] for i in range(len(vals) - 1)) else "FAIL"
                detail = " -> ".join(f"{p}:{v}" for p, v in zip(ps, vals))
            elif typ == "monotone_decrease":
                ps = test["periods"]; vals = [share(p) for p in ps]
                if any(v is None for v in vals):
                    res = "NA"
                else:
                    res = "PASS" if all(vals[i + 1] < vals[i] for i in range(len(vals) - 1)) else "FAIL"
                detail = " -> ".join(f"{p}:{v}" for p, v in zip(ps, vals))
            elif typ == "greater_than_other":  # share(vocab, period) > share(other vocab, same denominator, period) for each period
                other = get_series(cell["denominator"], test["other"])
                ps = test["periods"]; ok, parts = [], []
                for p in ps:
                    a, b = share(p), other["by_period"].get(p, {}).get("share")
                    if a is None or b is None:
                        ok.append(None)
                    else:
                        ok.append(a > b)
                    parts.append(f"{p}: {a} vs {b}")
                res = "NA" if any(o is None for o in ok) else ("PASS" if all(ok) else "FAIL")
                detail = "; ".join(parts)
            elif typ == "plurality":           # vocab share > every other listed vocab share in period
                p = test["period"]; a = share(p); others = {}
                for o in test["others"]:
                    others[o] = get_series(cell["denominator"], o)["by_period"].get(p, {}).get("share")
                if a is None or any(v is None for v in others.values()):
                    res = "NA"
                else:
                    res = "PASS" if all(a > v for v in others.values()) else "FAIL"
                detail = f"{p}: {cell['vocab']} {a} vs " + ", ".join(f"{k} {v}" for k, v in others.items())
            elif typ == "exceeds_ci":          # lower CI of some period > upper CI of reference period
                ref = test["reference"]; ps = test["periods"]
                ru = ci(ref)[1]; parts, hits = [], []
                for p in ps:
                    lo = ci(p)[0]
                    hits.append(None if (lo is None or ru is None) else lo > ru)
                    parts.append(f"{p}: {share(p)} [{ci(p)[0]},{ci(p)[1]}]")
                res = "NA" if all(h is None for h in hits) else ("PASS" if any(h for h in hits if h) else "FAIL")
                detail = f"ref {ref}: {share(ref)} [{ci(ref)[0]},{ru}]; " + "; ".join(parts)
            elif typ == "within_ci":           # share in each listed period <= upper CI of reference
                ref = test["reference"]; ps = test["periods"]; ru = ci(ref)[1]
                hits = [None if (share(p) is None or ru is None) else share(p) <= ru for p in ps]
                res = "NA" if any(h is None for h in hits) else ("PASS" if all(hits) else "FAIL")
                detail = f"ref {ref} upper {ru}; " + "; ".join(f"{p}: {share(p)}" for p in ps)
            elif typ == "annual_step":         # mean(after) > mean(before) beyond averaged CIs; flat = overlapping
                b0, b1 = test["before"]; a0, a1 = test["after"]
                mb, nb = yr_mean(b0, b1); ma, na = yr_mean(a0, a1)
                cb, ca = yr_ci(b0, b1), yr_ci(a0, a1)
                if mb is None or ma is None:
                    res = "NA"
                else:
                    step = ca[0] is not None and cb[1] is not None and ca[0] > cb[1]
                    res = "PASS" if (step if test.get("expect", "step") == "step" else not step) else "FAIL"
                detail = f"{b0}-{b1}: {mb} [{cb[0]},{cb[1]}] ({nb} yrs) vs {a0}-{a1}: {ma} [{ca[0]},{ca[1]}] ({na} yrs); expect {test.get('expect', 'step')}"
            elif typ == "descriptive":
                res = "REPORTED"; detail = "; ".join(f"{p}: {share(p)}" for p in periods if share(p) is not None)
            else:
                res = "NA"; detail = f"unknown test type {typ}"
        except Exception as e:  # noqa
            res, detail = "ERROR", repr(e)
        results.append({"id": cid, "claim": cell.get("claim"), "kind": kind, "status": cell.get("status", "OPEN"),
                        "supports": cell.get("supports"), "denominator": cell["denominator"], "vocab": cell["vocab"],
                        "test": test, "result": res, "detail": detail})

    out = {"settings": {"boot": BOOT, "seed": SEED, "min_n": MIN_N, "merge": MERGE},
           "vocabularies": GT.get("vocabularies"), "term_sets": GT.get("term_sets"), "venue_sets": GT.get("venue_sets"),
           "series": series, "cells": results, "doc_flags_file": str(fp)}
    (OUT / "ground_truth_cooccurrence.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")

    L = [f"HISTORICAL GROUND-TRUTH TESTS — {CFG['concept']}  ({len(docs):,} documents with abstracts; "
         f"{BOOT} document bootstraps; cell floor {MIN_N})", "",
         "Kinds: VALIDITY = books agree (tests the instrument); DISCRIMINATING = books disagree (adjudicated, never "
         "counted toward validity); FORECAST = written before the data. Status SEEN = looked at before pre-registration "
         "(report as post-hoc match); OPEN = pre-registered in Session 46 before this run.", ""]
    L.append("== cells ==")
    for r in results:
        L.append(f"  {r['id']:<8}{r['result']:<9}{r['kind']:<15}{str(r['status']):<20}{(r['supports'] or ''):<10} {r['claim']}")
        L.append(f"          {r['detail']}")
    L.append("")
    L.append("== series (share of denominator documents matching the vocabulary; 5-year periods) ==")
    for key, s in series.items():
        L.append(f"  {key}")
        L.append("    " + "  ".join(f"{p[:4]}:{v['share']:.3f}({v['n']})" for p, v in s["by_period"].items()))
        if SHOW_ANNUAL and s["by_year"]:
            L.append("    annual: " + " ".join(f"{y}:{v['share']:.2f}" for y, v in s["by_year"].items()))
    rep = OUT / "ground_truth_cooccurrence_report.txt"
    rep.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    pc.write_manifest(CFG, __file__, params=out["settings"], outputs=[OUT / "ground_truth_cooccurrence.json", rep, fp],
                      notes="Pre-registered vocabulary co-occurrence cells from Historical_Ground_Truth_Claims_v1.md; "
                            "vocabularies and thresholds fixed in Lab Notebook Session 46 before the first run.")
    print(f"[31] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
