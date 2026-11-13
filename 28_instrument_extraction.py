"""
28_instrument_extraction.py — the measure / operationalisation stratum.

The thesis is concept–MEASURE co-stabilisation: a concept and the instruments
that operationalise it stabilise (or fail to) together. The semantic layer
measures the concept's dispersion per period; this script measures the
instrument inventory per term x period from the same documents, so the two
series can be paired. It also answers the reviewer's question "is this the
operationalised construct or colloquial usage?" with a number: the share of
documents mentioning the term that name any instrument at all.

WHAT IS DETECTED (title + abstract, from the RIS exports the config lists;
lineage_docs keeps only mention sentences, so abstracts are re-read here):
  1. Seeded instruments — config `instruments.seed`: [canonical, alias, ...]
     including acronyms. Matched case-insensitively as whole phrases.
  2. Generic named instruments — a capitalised multiword name ending in an
     instrument head noun (Scale, Inventory, Questionnaire, Checklist,
     Interview, Index, Schedule, Battery, Survey, Measure, Task, Paradigm,
     Test), optionally followed by a parenthesised acronym; e.g. "Impact of
     Event Scale (IES-R)", "Brief Self-Control Scale (BSCS)", "Stroop task".
  3. Acronym resolution within a document: an acronym introduced after a
     detected name resolves to that name for the rest of the document;
     bare acronyms resolve only through the seed list. Unresolved bare
     acronyms are counted but not treated as instruments.
Identity: names are canonicalised (lower-case, punctuation collapsed) and a
FAMILY key strips version markers (-5, -R, -C, -M, -SR, -II, "revised",
"short form", "for DSM-5", ...) so PCL, PCL-C and PCL-5 are one family with
three versions. Version switches are reported per family (DSM-5 era, etc.).

OUTPUTS results/<slug>/
  instrument_mentions.jsonl   one line per document: instruments found
  instrument_measures.json    per term x period: share naming an instrument,
                              families, normalised entropy, 1-HHI, top families,
                              version shares; per family: first period, series
  instrument_measures_report.txt
Paired with concept dispersion (29's masked breadth per cell, if
creep_breadth.json exists): Spearman rho across periods per term between
instrument dispersion (1-HHI) and concept breadth — the co-stabilisation
reading is CONVERGENT when both fall together, DIVERGENT when the instrument
inventory proliferates while the concept disperses, and so on. Reported as
measured, with n periods; no verdict is attached.

USAGE
    python3 28_instrument_extraction.py --config configs/ptsd_trauma.yaml
    python3 28_instrument_extraction.py --config configs/ego_depletion.yaml
    options: --terms "a,b"  --min-n 15  --no-generic (seed list only)  --show 20 (print top names found)
"""

import glob
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import pipeline_config as pc
from revision_core import normalize_instrument_version, document_id

CFG, _ARGV = pc.load()


def _opt(name, default=None):
    for i, a in enumerate(_ARGV):
        if a == name and i + 1 < len(_ARGV):
            return _ARGV[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


ONLY = [pc.canon(t) for t in (_opt("--terms") or "").split(",") if t.strip()]
_CIT = CFG.get("citation", {}) or {}
_m = _opt("--merge") or _CIT.get("merge")
if _m is None:
    MERGE = {"ptsd": "PTSD", "posttraumatic stress disorder": "PTSD",
             "complex ptsd": "complex PTSD", "complex posttraumatic stress disorder": "complex PTSD"}
elif isinstance(_m, str):
    MERGE = dict(kv.split("=", 1) for kv in _m.split(";") if "=" in kv)
else:
    MERGE = dict(_m)
MIN_N = int(_opt("--min-n", 15))
GENERIC = "--no-generic" not in _ARGV
SHOW = int(_opt("--show", 25))
OUT = CFG.results
DOCS = CFG.path("docs_path")
BREADTH = OUT / "creep_breadth.json"
if not BREADTH.exists() and CFG.get("revision_baseline_results"):
    BREADTH = Path(CFG["revision_baseline_results"]) / "creep_breadth.json"

HEADS = ("Scale", "Inventory", "Questionnaire", "Checklist", "Interview", "Index", "Schedule",
         "Battery", "Survey", "Measure", "Task", "Paradigm", "Test", "Screen", "Instrument", "Subscale")
_HEAD_RE = "|".join(HEADS)
# capitalised run of 1-7 words then a head noun; allow lower-case function words inside ("Impact of Event Scale")
NAME_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z0-9\-']*|of|for|and|the|in|on)(?:[\s\-]+(?:[A-Z][A-Za-z0-9\-']*|of|for|and|the|in|on)){0,7}"
    r"[\s\-]+(?:" + _HEAD_RE + r"))\b(?:\s*\(\s*([A-Z][A-Za-z0-9\-/]{1,12})\s*\))?")
ACRO_RE = re.compile(r"\b([A-Z]{2,}[A-Za-z0-9\-]{0,8})\b")
STOP_FIRST = {"the", "a", "an", "this", "these", "those", "our", "their", "its", "results", "methods", "method",
              "study", "data", "participants", "we", "it", "each", "all", "both", "no", "two", "three", "one",
              "first", "second", "using", "used", "in", "of", "for", "and", "on", "with", "as", "a", "to",
              "self", "post", "pre", "high", "low", "new", "further", "future", "present", "current",
              # sentence-initial verbs that would otherwise start a "name"
              "administered", "measured", "assessed", "completed", "included", "employed", "developed", "validated",
              "applied", "compared", "examined", "tested", "investigated", "evaluated", "based", "adapted", "modified",
              "translated", "analyzed", "analysed", "conducted", "reported", "revised", "following", "after", "before",
              "during", "within", "between", "among", "however", "although", "while", "when", "because", "thus",
              "main", "primary", "secondary", "dependent", "independent", "outcome", "additional", "standardized",
              "standardised", "objective", "subjective", "multiple", "several", "various", "other", "same", "similar"}
# generic phrases that have the shape of an instrument name but are not one
BLOCK = {"main outcome measure", "main outcome measures", "outcome measure", "outcome measures", "primary outcome measure",
         "dependent measure", "self report measure", "self-report measure", "repeated measure", "repeated measures",
         "exact test", "t test", "chi square test", "chi-square test", "post hoc test", "likelihood ratio test",
         "newman keuls test", "mann whitney test", "wilcoxon test", "fisher exact test",
         "national survey", "body mass index", "mass index", "quality index", "national longitudinal survey",
         "general social survey", "health survey", "household survey", "systematic review", "literature review"}
VERSION_RE = re.compile(r"(?:[\s\-]+(?:5|iv|ii|iii|r|c|m|s|sr|sf|a|b|short form|revised|version \d+|\d+(?:\.\d+)?"
                        r"|for dsm[\s\-]*(?:5|iv)|dsm[\s\-]*(?:5|iv)|civilian version|military version|specific|"
                        r"child(?:ren)?(?:'s)? version|parent version|brief|short|long|extended|abbreviated))+$")


def canon_name(s):
    s = pc.canon(s)   # hyphen/space/prefix variants collapse: post-traumatic = posttraumatic
    s = re.sub(r"[^a-z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def family_of(name):
    f = VERSION_RE.sub("", name).strip()
    f = re.sub(r"^(the|a|an)\s+", "", f)
    return f or name


def version_of(name, fam):
    v = name[len(fam):].strip(" -") if name.startswith(fam) else ""
    return v or "base"


# ---------------------------------------------------------------- seed list

def load_seed():
    """config instruments.seed: list of [canonical, alias, alias...]. Returns
    {alias_canon: canonical_canon} and the set of canonical families."""
    spec = (CFG.get("instruments") or {}).get("seed") or []
    alias = {}
    for row in spec:
        if isinstance(row, str):
            row = [row]
        canon = canon_name(row[0])
        for a in row:
            alias[canon_name(a)] = canon
    return alias


SEED = load_seed()
SEED_PATS = sorted(SEED, key=len, reverse=True)
def _alias_pat(a):
    # tokens may be joined by space or any dash in the text ("PCL-5", "PCL 5", "IES‐R")
    return r"[\s\-]+".join(re.escape(tok) for tok in a.split())


def _raw_alias(a):
    """The alias as written in the config (before canonicalisation)."""
    return _RAW.get(a, a)


_RAW = {}
for _row in (CFG.get("instruments") or {}).get("seed") or []:
    for _a in ([_row] if isinstance(_row, str) else _row):
        _RAW[canon_name(_a)] = _a


def _is_acronym(a):
    raw = _raw_alias(a)
    return " " not in raw.strip() and len(raw) <= 10 and raw.upper() == raw and any(ch.isalpha() for ch in raw)


# Acronyms (BRIEF, ACE, DES, BIS...) are matched CASE-SENSITIVELY, as written in
# the config; everything else case-insensitively. Found 2026-10-01: "BRIEF"
# matched the word "brief" in 1,000 abstracts.
_NAMES = [a for a in SEED_PATS if not _is_acronym(a)]
_ACROS = [a for a in SEED_PATS if _is_acronym(a)]
SEED_RE = re.compile(r"\b(" + "|".join(_alias_pat(a) for a in _NAMES) + r")\b", re.I) if _NAMES else None
ACRO_SEED_RE = re.compile(r"\b(" + "|".join(_alias_pat(_raw_alias(a)) for a in _ACROS) + r")\b") if _ACROS else None
SEED_ALIAS_RES = [(re.compile(_alias_pat(a) + r"$", re.I), SEED[a], a) for a in SEED_PATS]


def seed_lookup(surface):
    """Canonical family for a matched surface. The dictionary key is the
    canonicalised surface; if canonicalisation disagrees with the regex that
    matched (an odd character inside the surface), fall back to the alias
    pattern that actually matched. Returns (canonical, alias) or None."""
    cs = canon_name(surface)
    if cs in SEED:
        return SEED[cs], cs
    for rx, canon, alias in SEED_ALIAS_RES:
        if rx.match(surface.strip()):
            return canon, alias
    return None


# ---------------------------------------------------------------- detection

def detect(text):
    """Return list of (family, version, surface, how) for one document."""
    found = []
    acro_map = {}
    raw = pc.normalize_text(text or "")
    low = raw.lower()
    # 1. seeded names (case-insensitive) and seeded acronyms (case-sensitive)
    seed_hits = []
    if SEED_RE:
        seed_hits += [m.group(1) for m in SEED_RE.finditer(raw)]
    if ACRO_SEED_RE:
        seed_hits += [m.group(1) for m in ACRO_SEED_RE.finditer(raw)]
    if seed_hits:
        for surf in seed_hits:
            hit = seed_lookup(surf)
            if hit is None:
                continue
            canon, alias = hit
            fam = family_of(canon)
            # The generic name does not identify the Tangney instrument.
            if fam == "self control scale" and alias == "self control scale":
                fam = "self control scale unspecified"
            ver = version_of(alias, family_of(alias))   # the version marker on the alias actually used (PCL-5 -> "5")
            found.append((fam, ver, surf, "seed"))
    # 2. generic named instruments (+ acronym introduced in parentheses)
    if GENERIC:
        for m in NAME_RE.finditer(raw):
            surf, acro = m.group(1), m.group(2)
            words = surf.split()
            if len(words) < 2 or words[0].lower() in STOP_FIRST:
                continue
            if sum(1 for w in words if w[:1].isupper()) < 2:
                continue
            name = canon_name(surf)
            if name in BLOCK or family_of(name) in BLOCK:
                continue
            if SEED_RE and name in SEED:
                continue   # already counted as seed
            fam = family_of(name)
            found.append((fam, version_of(name, fam), surf, "generic"))
            if acro:
                acro_map[acro.upper()] = (fam, version_of(name, fam))
        # 3. later bare uses of an acronym introduced in this document
        for m in ACRO_RE.finditer(raw):
            a = m.group(1).upper()
            if a in acro_map:
                fam, ver = acro_map[a]
                found.append((fam, ver, m.group(1), "acronym"))
    # dedupe per document: one line per (family, version)
    seen = {}
    for fam, ver, surf, how in found:
        ver = normalize_instrument_version(fam, ver)
        key = (fam, ver)
        if key not in seen:
            seen[key] = {"family": fam, "version": ver, "surface": surf, "how": how}
    return list(seen.values())


# ---------------------------------------------------------------- corpus text

def norm_title(title):
    t = re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()
    return t if len(t) >= 25 else ""


def doc_index():
    """lineage docs keyed by DOI and by normalised title -> doc."""
    by_doi, by_title, docs = {}, {}, []
    with open(DOCS, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["year"] > CFG.get("revision_analysis_year_max", 9999):
                continue
            docs.append(d)
            doi = (d.get("doi") or "").lower().strip()
            if doi:
                by_doi[doi] = d
            nt = norm_title(d.get("title"))
            if nt:
                by_title.setdefault(nt, d)
    return docs, by_doi, by_title


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


def iter_texts():
    """Yield (doc, text) for every corpus document whose abstract can be found
    in the listed exports (RIS `AB`/`N2`, or JSONL `abstract`/`text`)."""
    docs, by_doi, by_title = doc_index()
    sources = (CFG.get("corpus") or {}).get("sources") or []
    seen = set()
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
                            if d is None or id(d) in seen:
                                continue
                            ab = (w.get("abstract") or w.get("text") or "").strip()
                            if ab:
                                seen.add(id(d)); yield d, f"{d.get('title', '')}. {ab}"
                    continue
                for r in parse_ris(path):
                    doi = (r.get("DO", [""])[0] or "").lower().strip().replace("https://doi.org/", "").rstrip(".")
                    d = by_doi.get(doi) if doi else None
                    if d is None:
                        d = by_title.get(norm_title(r.get("TI", [""])[0]))
                    if d is None or id(d) in seen:
                        continue
                    ab = (r.get("AB", [""])[0] or r.get("N2", [""])[0] if r.get("N2") else r.get("AB", [""])[0]) or ""
                    if ab.strip():
                        seen.add(id(d)); yield d, f"{d.get('title', '')}. {ab}"
    print(f"[28] exports read: {n_files}; documents with abstract text: {len(seen):,} of {len(docs):,}")
    if not seen:
        raise SystemExit("[28] no abstracts found — check corpus.sources globs resolve from code/ (run from the code directory)")


# ---------------------------------------------------------------- measures

def hhi(counts):
    v = np.asarray(list(counts.values()), float)
    v = v / v.sum()
    return float((v ** 2).sum())


def norm_entropy(counts):
    v = np.asarray(list(counts.values()), float); v = v[v > 0] / v.sum()
    if len(v) < 2:
        return 0.0
    return float(-(v * np.log(v)).sum() / np.log(len(v)))


def spearman(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return (round(float(r), 3), round(float(p), 4)) if r == r else (None, None)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    per_cell = defaultdict(lambda: {"n": 0, "n_named": 0, "fam": Counter(), "ver": Counter()})
    per_doc_out = open(OUT / "instrument_mentions.jsonl", "w", encoding="utf-8")
    surface = Counter()
    n_docs = 0
    for d, text in iter_texts():
        n_docs += 1
        inst = detect(text)
        terms = sorted({MERGE.get(pc.canon(t), pc.canon(t)) for t in (d.get("terms") or [])})
        if ONLY:
            terms = [t for t in terms if t in ONLY]
        p = d.get("period")
        for t in terms:
            c = per_cell[(t, p)]
            c["n"] += 1
            if inst:
                c["n_named"] += 1
            for family in {i["family"] for i in inst}:
                c["fam"][family] += 1
            for i in inst:
                c["ver"][(i["family"], i["version"])] += 1
        for i in inst:
            surface[(i["family"], i["how"])] += 1
        if inst:
            per_doc_out.write(json.dumps({"document_id": document_id(d), "doi": d.get("doi"), "title": d.get("title"), "year": d.get("year"),
                                          "period": p, "terms": terms, "instruments": inst}) + "\n")
    per_doc_out.close()
    print(f"[28] {n_docs:,} documents scanned in {time.time() - t0:.0f}s; "
          f"{sum(1 for _ in open(OUT / 'instrument_mentions.jsonl')):,} name an instrument")

    # breadth (concept dispersion) per cell, if 29 has run
    breadth = {}
    if BREADTH.exists():
        for k, v in json.loads(BREADTH.read_text(encoding="utf-8")).get("cells", {}).items():
            if v.get("breadth_masked"):
                t, p = k.split("|")
                t = MERGE.get(pc.canon(t), pc.canon(t))
                if (t, p) not in breadth or v["breadth_masked"]["n_pool"] > breadth[(t, p)][1]:
                    breadth[(t, p)] = (v["breadth_masked"]["mean"], v["breadth_masked"]["n_pool"])
    breadth = {k: v[0] for k, v in breadth.items()}

    terms = sorted({t for t, _ in per_cell}, key=lambda t: -sum(c["n"] for (tt, _), c in per_cell.items() if tt == t))
    periods = sorted({p for _, p in per_cell if p})
    table, pairing, families = {}, {}, {}
    for t in terms:
        ser = {}
        for p in periods:
            c = per_cell.get((t, p))
            if not c or c["n"] < MIN_N:
                continue
            fam = c["fam"]
            ser[p] = {"n_docs": c["n"], "share_naming_instrument": round(c["n_named"] / c["n"], 4),
                      "n_families": len(fam), "n_mentions": int(sum(fam.values())),
                      "entropy": round(norm_entropy(fam), 4) if fam else None,
                      "dispersion_1_minus_hhi": round(1 - hhi(fam), 4) if fam else None,
                      "top_families": fam.most_common(6),
                      "breadth_masked": breadth.get((t, p))}
        if ser:
            table[t] = ser
            xs = [ser[p]["dispersion_1_minus_hhi"] for p in ser if ser[p]["dispersion_1_minus_hhi"] is not None
                  and ser[p]["breadth_masked"] is not None]
            ys = [ser[p]["breadth_masked"] for p in ser if ser[p]["dispersion_1_minus_hhi"] is not None
                  and ser[p]["breadth_masked"] is not None]
            if len(xs) >= 4:
                dx = list(np.diff(xs)); dy = list(np.diff(ys))
                pairing[t] = {"n_periods": len(xs),
                              "spearman_instrument_dispersion_vs_breadth": spearman(xs, ys),
                              "spearman_first_differences": spearman(dx, dy) if len(dx) >= 4 else (None, None),
                              "note": "two monotone series correlate whatever their relation; the first-difference "
                                      "correlation asks whether period-to-period moves co-occur"}
        # per-family series for this term (version switches)
        fseries = defaultdict(lambda: defaultdict(Counter))
        for p in periods:
            c = per_cell.get((t, p))
            if not c:
                continue
            for (fam, ver), n in c["ver"].items():
                fseries[fam][p][ver] += n
        top = [f for f, _ in Counter({f: sum(sum(v.values()) for v in s.values()) for f, s in fseries.items()}).most_common(8)]
        families[t] = {f: {"first_period": min(fseries[f]), "by_period": {p: dict(v) for p, v in sorted(fseries[f].items())}}
                       for f in top}

    out = {"settings": {"min_n": MIN_N, "generic": GENERIC, "seed_families": sorted(set(SEED.values())),
                        "heads": HEADS}, "periods": periods, "table": table, "families": families,
           "pairing_with_concept_breadth": pairing,
           "most_detected": [{"family": f, "how": h, "n": n} for (f, h), n in surface.most_common(60)]}
    (OUT / "instrument_measures.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    L = [f"INSTRUMENT / OPERATIONALISATION STRATUM — {CFG['concept']}  (cell floor {MIN_N}; "
         f"seed families {len(set(SEED.values()))}; generic detection {'on' if GENERIC else 'off'})", ""]
    L.append(f"== most detected families (document count; how = seed / generic / acronym) ==")
    for (f, h), n in surface.most_common(SHOW):
        L.append(f"  {f[:60]:60} {h:8} {n:>6}")
    L.append("")
    for t in terms:
        if t not in table:
            continue
        L.append(f"== {t} ==")
        L.append(f"  {'period':10}{'n':>7}{'named%':>8}{'fams':>6}{'entropy':>9}{'1-HHI':>7}{'breadth':>9}  top families")
        for p, r in table[t].items():
            L.append(f"  {p:10}{r['n_docs']:>7}{100 * r['share_naming_instrument']:>7.1f}%{r['n_families']:>6}"
                     f"{(r['entropy'] if r['entropy'] is not None else float('nan')):>9.3f}"
                     f"{(r['dispersion_1_minus_hhi'] if r['dispersion_1_minus_hhi'] is not None else float('nan')):>7.3f}"
                     f"{(r['breadth_masked'] if r['breadth_masked'] is not None else float('nan')):>9.3f}  "
                     + "; ".join(f"{f[:28]} {n}" for f, n in r["top_families"][:4]))
        pr = pairing.get(t)
        if pr:
            L.append(f"  instrument dispersion vs concept breadth: Spearman rho = {pr['spearman_instrument_dispersion_vs_breadth'][0]} "
                     f"(p = {pr['spearman_instrument_dispersion_vs_breadth'][1]}, {pr['n_periods']} periods); "
                     f"on first differences rho = {pr['spearman_first_differences'][0]} (p = {pr['spearman_first_differences'][1]})")
        fams = families.get(t, {})
        if fams:
            L.append("  families (first period; versions by period):")
            for f, v in fams.items():
                vers = "; ".join(f"{p}: " + ", ".join(f"{ver} {n}" for ver, n in sorted(c.items(), key=lambda x: -x[1])[:3])
                                 for p, c in list(v["by_period"].items())[-4:])
                L.append(f"    {f[:40]:40} first {v['first_period']}  {vers}")
        L.append("")
    L.append("Reading guide: named% = share of documents mentioning the term that name any instrument (operationalised vs "
             "colloquial usage); 1-HHI = instrument dispersion (0 = one instrument, ->1 = many equally); breadth = 29's "
             "masked-target document breadth for the same cell. Families collapse versions (PCL / PCL-C / PCL-5). "
             "Read the paired correlation on first differences: both series trend, so the level correlation is not evidence of coupling.")
    (OUT / "instrument_measures_report.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    pc.write_manifest(CFG, __file__, params=out["settings"],
                      outputs=[OUT / "instrument_mentions.jsonl", OUT / "instrument_measures.json",
                               OUT / "instrument_measures_report.txt"])
    print(f"[28] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
