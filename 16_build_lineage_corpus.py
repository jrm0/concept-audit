"""
16_build_lineage_corpus.py — Assemble the document-level lineage corpus.

Merges every family source (RIS exports incl. subfolders + any family
works.jsonl) into ONE document-level dataset for the lineage analysis:

  data/family/lineage_docs.jsonl   one row per unique document:
      doi, year, period, journal, title, abstract, index_terms,
      terms (which family terms actually appear in title+abstract),
      mention_sentences {term: [sentences]}   <- the embedding input

Design notes:
- DOCUMENT is the unit (session-10 lesson: sentences are not independent).
- Term presence is recomputed from the text itself, so ancestor exports
  (retrieved by query, no per-record match info) get correct term labels.
- Control-term mentions ("participants", "results", "significant", "measure")
  are extracted from the same abstracts — the drift null costs nothing.
- 5-year periods from 1975.

Usage (from code/):
    python3 16_build_lineage_corpus.py [--config configs/<concept>.yaml]
                                        [--allow-unlisted]

All paths, the family file, period bounds and control terms come from the
config (default configs/ego_depletion.yaml). See pipeline_config.py.

CORPUS SPEC (2026-09-08). The config's `corpus:` block is the retrieval log
made machine-readable (principle 10: the log is the methods section). Every
RIS export under ris_root must match a `corpus.sources[].files` glob or a
`corpus.ignore[].files` glob; anything else is an ERROR (principle 6:
provenance separate at retrieval, merged in code — an export nobody wrote
down is an export nobody can defend). Each document carries `sources` (ids,
unioned on merge) and `stratum` (register by RIS TY tag: journal / thesis /
book / other; principle 7: registers are strata, not ingredients). A
coverage audit per source (abstract %, DOI %, type mix, kept/merged) is
printed and written to the manifest (principle 8: no analysis before the
audit).

MENTION RULES (2026-09-13, Session 29). Which occurrences count as a mention
is decided in pipeline_config (mention_rules / valid_spans) and shared with
17, so the sentence 16 keeps and the occurrence 17 masks are the same one.
Defaults: a match nested inside a LONGER family term does not count
("trauma" inside "trauma-informed"; "traumatic stress" inside
"post-traumatic stress disorder"), and "(PTSD)" after a long form is
stripped from the context. Config `mentions: {exclude_nested: false,
strip_acronym_parentheticals: false}` reproduces the pre-Session-29
behaviour as a sensitivity run. Consequence to keep in mind: a document whose
only "trauma" is inside "vicarious trauma" is no longer a "trauma" document —
the parent's shard is bare uses, the compound's shard is the compound.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pipeline_config as pc
from pipeline_config import canon, term_pattern

CFG, _ARGV = pc.load()

RIS_ROOT = Path(CFG["ris_root"])
OUT_PATH = Path(CFG["docs_path"])
FAMILY_DIR = OUT_PATH.parent
FAMILY_CSV = Path(CFG["family_csv"])
FAMILY_WORKS_GLOB = CFG.get("family_works_glob")   # optional OpenAlex works

PERIOD_YEARS = int(CFG["periods"]["width"])
YEAR_MIN, YEAR_MAX = int(CFG["periods"]["year_min"]), int(CFG["periods"]["year_max"])

CONTROL_TERMS = CFG.controls
CORPUS = CFG.get("corpus", {}) or {}
ALLOW_UNLISTED = "--allow-unlisted" in _ARGV
RULES = pc.mention_rules(CFG)      # see pipeline_config: mention extraction

TAG_RE = re.compile(r"^([A-Z][A-Z0-9])  - (.*)$")


def _match_any(path, globs):
    """Path matches any glob (relative to code/, POSIX separators)."""
    from fnmatch import fnmatch
    p = Path(path).as_posix()
    for g in globs:
        g = str(g)
        # '**' should match across directories; fnmatch's '*' already does
        if fnmatch(p, g.replace("**", "*")):
            return True
    return False


def source_of(path):
    """(source_id, entry) for an export, or ('IGNORED', why) / (None, None)."""
    for s in CORPUS.get("sources", []) or []:
        if _match_any(path, s.get("files", [])):
            return s["id"], s
    for ig in CORPUS.get("ignore", []) or []:
        g = ig.get("files")
        if g and _match_any(path, [g] if isinstance(g, str) else g):
            return "IGNORED", ig.get("why", "")
    return None, None


def stratum_of(ty):
    """Register stratum from the RIS TY tag (config corpus.strata)."""
    strata = CORPUS.get("strata") or {}
    for name, tys in strata.items():
        if ty in tys:
            return name
    return "journal" if not strata else "other"
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# canon(): hyphens and spaces are the SAME separator. 'ego-depletion' and
# 'ego depletion' are one term, not two. Keeping them apart split the focal
# term's documents across two rows AND collided in 17, whose filename slug
# maps both to 'ego_depletion' — so the second variant's documents were
# silently skipped as an existing checkpoint. (Session-8 surface-form lesson
# at the level of term identity.) Now lives in pipeline_config.


def load_family_terms():
    terms = {}
    with open(FAMILY_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = canon(row.get("term"))
            if t and t not in terms:
                terms[t] = (row.get("role") or "").strip()
    return terms


def compile_patterns(term_names):
    """Match either separator, so one canonical term captures all variants."""
    return {t: term_pattern(t) for t in term_names}


def parse_ris(path):
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    recs, cur, last = [], {}, None
    for line in text.splitlines():
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


def year_of(rec):
    for tag in ("Y1", "PY", "DA"):
        for v in rec.get(tag, []):
            m = re.search(r"(19|20)\d{2}", v)
            if m:
                return int(m.group(0))
    return None


def period_of(year):
    return pc.period_of(year, CFG)


ERRATUM_RE = re.compile(
    r"^\W*(erratum(\s+to)?|correction(\s+to)?|corrigendum(\s+to)?)\b[:\s]*",
    re.I)
NONALNUM_RE = re.compile(r"[^a-z0-9]+")


def norm_title(title):
    """Normalized title for CROSS-DATABASE dedupe.

    The same article indexed in PsycInfo, Business Source and SocINDEX gets a
    DIFFERENT accession number in each and often a DOI in only some of them,
    so accession/DOI matching alone leaves triplicates. Titles match. Errata
    and corrections are folded into the original (they are the same document
    for usage purposes, and previously inflated cluster sizes).
    Returns "" when the title is too short to be a safe key.
    """
    t = ERRATUM_RE.sub("", (title or "").strip().strip("'\""))
    t = NONALNUM_RE.sub(" ", t.lower()).strip()
    return t[:120] if len(t) >= 25 else ""


def main():
    family = load_family_terms()
    pats = compile_patterns(list(family) + CONTROL_TERMS)
    fam_pats = {t: pats[t] for t in family}
    docs = {}
    by_title = {}      # normalized title -> canonical key
    stats = Counter()
    excluded_by_term = Counter()   # family mentions dropped by the rules

    audit = defaultdict(Counter)   # source_id -> counters (principle 8)

    def merge(key, dbs, source=None):
        """Same document from another database: keep one record, UNION the
        provenance tags (a doc indexed in 3 databases carries all 3) and the
        source ids."""
        if dbs:
            docs[key]["dbs"] = sorted(set(docs[key].get("dbs", [])) | set(dbs))
        if source:
            docs[key]["sources"] = sorted(set(docs[key].get("sources", []))
                                          | {source})
            audit[source]["merged_into_existing"] += 1
        stats["merged_duplicates"] += 1

    def add_doc(key, year, journal, title, abstract, index_terms, dbs,
                source=None, stratum="journal"):
        period = period_of(year)
        if period is None or not abstract:
            stats["skipped_no_abstract_or_year"] += 1
            if source:
                audit[source]["skipped_no_abstract_or_year"] += 1
            return None
        text = f"{title}. {abstract}"
        sentences = [s.strip() for s in SENT_RE.split(text)
                     if 30 <= len(s.strip()) <= 1200]
        mentions = defaultdict(list)
        for s in sentences:
            # (1) strip "(ACRONYM)" after a family term — definitional device
            if RULES.get("strip_acronym_parentheticals", True):
                s, n_cut = pc.strip_acronym_parentheticals(s, fam_pats)
                stats["parentheticals_stripped"] += n_cut
            # (2) a match nested inside a longer family term does not count;
            #     a parenthetical-only occurrence does not count
            spans = pc.all_spans(s, pats)
            for t in spans:
                ok = pc.valid_spans(t, spans, s, RULES)
                if ok:
                    mentions[t].append(s)
                else:
                    stats["mentions_excluded_nested_or_paren"] += 1
                    if t in family:
                        excluded_by_term[t] += 1
        fam_terms = [t for t in mentions if t in family]
        if not fam_terms:
            stats["skipped_no_family_term"] += 1
            if source:
                audit[source]["skipped_no_family_term"] += 1
            return None  # no family term actually present
        stats["kept"] += 1
        if source:
            audit[source]["kept"] += 1
            audit[source][f"stratum:{stratum}"] += 1
        docs[key] = {
            "doi": key if not key.startswith("AN:") else "",
            "year": year, "period": period,
            "journal": journal, "title": title,
            "index_terms": index_terms,
            "dbs": sorted(set(dbs)),   # source database(s) — migration tracking
            "sources": [source] if source else [],   # corpus-spec source ids
            "stratum": stratum,                      # register (RIS TY)
            "terms": sorted(fam_terms),
            "roles": sorted({family[t] for t in fam_terms}),
            "mention_sentences": {t: v[:6] for t, v in mentions.items()},
        }
        return key

    # 1. All RIS files, recursively.
    #    DB tags are ground truth for provenance; the file a record came in
    #    is only a convenience. Dedupe order: DOI, then normalized title.
    all_ris = sorted(RIS_ROOT.rglob("*.ris"))
    ris_files, unlisted, ignored = [], [], []
    for f in all_ris:
        sid, entry = source_of(f)
        if sid == "IGNORED":
            ignored.append((f, entry))
        elif sid is None:
            unlisted.append(f)
        else:
            ris_files.append((f, sid, entry))
    if ignored:
        print(f"Ignoring {len(ignored)} export(s) listed under corpus.ignore:")
        for f, why in ignored[:5]:
            print(f"   {f}  — {why}")
        if len(ignored) > 5:
            print(f"   ... and {len(ignored) - 5} more")
    if unlisted:
        msg = (f"{len(unlisted)} RIS export(s) under {RIS_ROOT} match no "
               f"corpus.sources or corpus.ignore entry:\n   "
               + "\n   ".join(str(u) for u in unlisted[:10]))
        if not ALLOW_UNLISTED:
            raise SystemExit(msg + "\nAdd them to the config's corpus block "
                             "(with query, dates, role, question) or to "
                             "corpus.ignore. Override with --allow-unlisted "
                             "(they will be ingested with source=UNLISTED).")
        print("WARNING: " + msg + "\n   ingesting as source=UNLISTED")
        ris_files += [(u, "UNLISTED", {}) for u in unlisted]
    listed_missing = [s["id"] for s in CORPUS.get("sources", []) or []
                      if not any(sid == s["id"] for _, sid, _ in ris_files)]
    if listed_missing:
        print(f"WARNING: corpus.sources with no export on disk: {listed_missing}")

    for f, sid, entry in ris_files:
        recs = parse_ris(f)
        n_before, merged_before = len(docs), stats["merged_duplicates"]
        for r in recs:
            audit[sid]["records"] += 1
            if (r.get("AB", [""])[0] or "").strip():
                audit[sid]["with_abstract"] += 1
            if (r.get("DO", [""])[0] or "").strip():
                audit[sid]["with_doi"] += 1
            ty = (r.get("TY", [""])[0] or "").strip()
            audit[sid][f"TY:{ty or '?'}"] += 1
            stratum = stratum_of(ty)

            doi = (r.get("DO", [""])[0] or "").lower().strip()
            doi = doi.replace("https://doi.org/", "").rstrip(".")
            title = r.get("TI", [""])[0]
            nt = norm_title(title)
            dbs = [v.strip() for v in r.get("DB", []) if v.strip()]

            if doi and doi in docs:               # same DOI
                merge(doi, dbs, sid)
                continue
            if nt and nt in by_title:             # same title, other database
                merge(by_title[nt], dbs, sid)
                continue

            an = (r.get("AN", [""])[0] or "").strip()
            key = doi or (f"AN:{an}" if an else f"T:{nt or title[:80].lower()}")
            if key in docs:
                merge(key, dbs, sid)
                continue
            made = add_doc(key, year_of(r),
                           (r.get("JO", [""])[0] or r.get("T2", [""])[0]
                            or "").strip(),
                           title, (r.get("AB", [""])[0] or ""),
                           [k.strip() for k in r.get("KW", []) if k.strip()],
                           dbs, source=sid, stratum=stratum)
            if made and nt:
                by_title[nt] = made
        print(f"{f.name} [{sid}]: +{len(docs) - n_before} docs, "
              f"{stats['merged_duplicates'] - merged_before} merged "
              f"(total {len(docs)})")

    # 2. OpenAlex-derived family works, if present (adds field metadata later)
    works_files = sorted(Path().glob(FAMILY_WORKS_GLOB)) if FAMILY_WORKS_GLOB else []
    for jf in works_files:
        n_before = len(docs)
        for line in open(jf, encoding="utf-8"):
            try:
                w = json.loads(line)
            except json.JSONDecodeError:
                continue
            doi = (w.get("doi") or "").lower().replace(
                "https://doi.org/", "").rstrip(".")
            nt = norm_title(w.get("title", ""))
            if doi and doi in docs:
                merge(doi, ["OpenAlex"], "openalex_works")
                continue
            if nt and nt in by_title:
                merge(by_title[nt], ["OpenAlex"], "openalex_works")
                continue
            key = doi or (f"T:{nt}" if nt else None)
            if not key or key in docs:
                continue
            made = add_doc(key, w.get("year"), w.get("journal", ""),
                           w.get("title", ""), w.get("abstract", ""),
                           w.get("index_terms", []), ["OpenAlex"],
                           source="openalex_works", stratum="journal")
            if made and nt:
                by_title[nt] = made
        print(f"{jf.name}: +{len(docs) - n_before} docs (total {len(docs)})")

    if not docs:
        raise SystemExit("No documents ingested (no exports on disk for this "
                         "corpus spec?) — nothing written.")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for d in docs.values():
            f.write(json.dumps(d) + "\n")

    print(f"\nRECORD ACCOUNTING: kept {stats['kept']}, "
          f"merged duplicates {stats['merged_duplicates']}, "
          f"no family term {stats['skipped_no_family_term']}, "
          f"no abstract/year {stats['skipped_no_abstract_or_year']}")
    print(f"MENTION RULES {RULES}: parentheticals stripped "
          f"{stats['parentheticals_stripped']}; sentence-mentions excluded "
          f"(nested in a longer family term, or parenthetical-only) "
          f"{stats['mentions_excluded_nested_or_paren']}")
    if excluded_by_term:
        print("  excluded by term: " + "; ".join(
            f"{t} {n}" for t, n in excluded_by_term.most_common(12)))

    # Report: term x period document counts (the power table for the analysis)
    tp = defaultdict(Counter)
    for d in docs.values():
        for t in d["terms"]:
            tp[t][d["period"]] += 1
    periods = sorted({d["period"] for d in docs.values()})
    print(f"\n{len(docs)} documents -> {OUT_PATH}\n")
    print("DOCUMENTS PER TERM PER PERIOD (lineage power table):")
    print(f"{'term':<26}" + "".join(f"{p[2:4]}-{p[7:9]:>3}" for p in periods))
    for t in sorted(tp, key=lambda x: -sum(tp[x].values())):
        role = load_family_terms().get(t, "")
        print(f"{t:<26}" + "".join(f"{tp[t].get(p, 0):>7}" for p in periods)
              + f"   [{role}]")

    dbc = Counter()
    for d in docs.values():
        for db in d.get("dbs", []):
            dbc[db] += 1
    if dbc:
        print("\nDATABASE PROVENANCE (a record may carry several):")
        for db, c in dbc.most_common():
            print(f"  {db}: {c}")

    # --- Coverage audit per source (principle 8) ---
    thr = CORPUS.get("audit", {}) or {}
    min_ab = float(thr.get("min_abstract_share", 0.0))
    print("\nCOVERAGE AUDIT (per corpus source):")
    print(f"{'source':<24}{'records':>8}{'AB%':>6}{'DOI%':>6}{'kept':>7}"
          f"{'merged':>8}{'noTerm':>8}  strata / types")
    audit_rows = {}
    for sid in sorted(audit):
        a = audit[sid]
        n = a["records"] or 1
        strata = {k.split(":", 1)[1]: v for k, v in a.items()
                  if k.startswith("stratum:")}
        types = {k.split(":", 1)[1]: v for k, v in a.items()
                 if k.startswith("TY:")}
        flag = (" <-- abstracts below threshold"
                if a["records"] and a["with_abstract"] / n < min_ab else "")
        if a["records"] and a["skipped_no_family_term"] / n > 0.4:
            flag += (" <-- >40% of records lack a family term in title/abstract"
                     " (full-text expander?)")
        print(f"{sid:<24}{a['records']:>8}{100*a['with_abstract']//n:>5}%"
              f"{100*a['with_doi']//n:>5}%{a['kept']:>7}"
              f"{a['merged_into_existing']:>8}{a['skipped_no_family_term']:>8}  "
              f"{strata} {dict(sorted(types.items(), key=lambda x: -x[1])[:4])}{flag}")
        audit_rows[sid] = {**{k: v for k, v in a.items()}, "strata": strata}
    strat_tot = Counter(d.get("stratum", "?") for d in docs.values())
    print(f"\nDocuments by register stratum: {dict(strat_tot)}")

    power_path = FAMILY_DIR / "lineage_power_table.csv"
    with open(power_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term"] + periods)
        for t in sorted(tp):
            w.writerow([t] + [tp[t].get(p, 0) for p in periods])
    print(f"\nPower table -> {power_path}")
    pc.write_manifest(
        CFG, __file__,
        params={"ris_root": str(RIS_ROOT), "family_csv": str(FAMILY_CSV),
                "family_works_glob": FAMILY_WORKS_GLOB,
                "ris_files": [[str(p), sid] for p, sid, _ in ris_files],
                "ignored_exports": [str(p) for p, _ in ignored],
                "allow_unlisted": ALLOW_UNLISTED,
                "periods": dict(CFG["periods"]), "control_terms": CONTROL_TERMS,
                "n_docs": len(docs), "accounting": dict(stats),
                "mention_rules": RULES,
                "mentions_excluded_by_term": dict(excluded_by_term),
                "db_provenance": dict(dbc),
                "strata": dict(strat_tot),
                "corpus_spec": {k: v for k, v in CORPUS.items()
                                if k != "sources"},
                "corpus_sources": [{k: v for k, v in s.items()}
                                   for s in CORPUS.get("sources", []) or []],
                "coverage_audit": audit_rows},
        outputs=[OUT_PATH, power_path])
    print("Next: python3 17_embed_documents.py   (local, needs torch)")


if __name__ == "__main__":
    main()
