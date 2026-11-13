"""
26_pull_openalex_citations.py — Build the CITATION LAYER for a lineage corpus.

Session 38 (2026-09-23). Companion to the semantic layer (16/17/18); feeds the
multilayer graph (semantic closeness and citation as two edge types over the
same document nodes).

WHAT IT COLLECTS
  outbound  each corpus document's OpenAlex record, including `referenced_works`
            (what it cites) and `cited_by_count`.
  inbound   for a chosen subset, the works that cite a corpus document
            (`filter=cites:W...`), including works OUTSIDE the corpus.

  The intra-corpus edge set is complete from the OUTBOUND pass alone: any
  citation from one corpus document to another appears in the citing
  document's `referenced_works`. The INBOUND pass buys only reach beyond the
  corpus ("who outside is citing us"), and it is ~50x more expensive per
  document because `cites:` cannot be batched per target. Scope it.

RESOLUTION: TWO ROUTES, AND WHY
  Route A (doi)    50 DOIs per list request. Cheap.
  Route B (title)  one list request per document, `title.search` + year, with a
                   normalized-title similarity check. For documents with no DOI.

  Route B exists for a substantive reason, not just coverage. Crossref went live
  in June 2000 (crossref.org/about/history), so pre-2000 articles carry a DOI
  only where a publisher later registered its backfile — a decision that varies
  by publisher size and digitization budget. Joining on DOI alone therefore
  builds a publisher-prestige bias into exactly the early periods where a
  concept's birth is visible. In this corpus DOI coverage runs 61-73% before
  2000 and 92-95% after 2015. The audit phase reports what route B recovers and
  whether its recovered documents differ in venue profile from route A's - that
  comparison is the bias check, and it belongs in the methods section.

CREDITS (OpenAlex changed its model on 2026-02-13: api_key required, credit-based)
  filter/list request = 10 credits; ANY `.search` filter = 100 credits (10x);
  100,000 credits/day on the free tier. Exhausted credits come back as HTTP 429.
  Measured on this corpus (129,977 docs, 113,983 with DOI):
    outbound, route A   2,280 batched requests             22,800 credits   (15 min)  DONE 2026-09-24
    titles (OpenAlex search)  15,848 x 100               1,584,800 credits  (~16 days) — do not
    crossref            15,218 requests, no key, no credits     0            (~1.5 h)  <- use this
    outbound-crossref   recovered DOIs / 50 x 10          a few thousand      (minutes)
    inbound, all works  113,983 x 1+ pages              >1,140,000 credits  (~12 days) — scope it
  The daily budget is tracked in the log across runs; 429 after one wait = day exhausted.

OUTPUTS (all under data/<slug>/citations/)
  openalex_works.jsonl   one line per resolved corpus document
  openalex_cites.jsonl   one line per inbound citing work (with its target)
  unresolved.jsonl       documents no route could resolve, with the reason
  retrieval_log.jsonl    one line per completed unit — resume reads this
Audit tables go to results/<slug>/ and a manifest is written there, per the
standing rule that a run without a manifest is not citable.

KEYS  .secrets/api_keys.yaml (git-ignored):
        openalex: "<your key from openalex.org/settings/api>"
        openalex_mailto: "you@example.edu"     # also used for Crossref's polite pool

USAGE
    python3 26_pull_openalex_citations.py --config configs/ptsd_trauma.yaml --dry-run
    python3 26_pull_openalex_citations.py --config configs/ptsd_trauma.yaml --phase outbound
    python3 26_pull_openalex_citations.py --config configs/ptsd_trauma.yaml --phase crossref --probe
    python3 26_pull_openalex_citations.py --config configs/ptsd_trauma.yaml --phase crossref
    python3 26_pull_openalex_citations.py --config configs/ptsd_trauma.yaml --phase outbound-crossref
    python3 26_pull_openalex_citations.py --config configs/ptsd_trauma.yaml --phase titles --allow-search \
            --max-year 1999 --limit 200
        # OpenAlex title search costs 100 credits per document: SAMPLE the pre-2000 residual
        # (200 docs = 20,000 credits) and read the hit rate before paying for the rest.
        # --sample draws at random (seed 42) instead of file order; --stratum journal and
        # --min-year 2000 restrict the pool; every run (and --phase audit) ends with the
        # hit rate of all title searches logged so far, by stratum / period / source.
    python3 26_pull_openalex_citations.py --config configs/ptsd_trauma.yaml --phase inbound \
            --min-cited-by 5 --max-works 2000
    python3 26_pull_openalex_citations.py --config configs/ptsd_trauma.yaml --phase audit
    python3 26_pull_openalex_citations.py --config ... --phase titles --max-seconds 150
        # bounded chunk for a 180-s shell; repeat the call, the log resumes each time
"""

import datetime as dt
import difflib
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import yaml

import pipeline_config as pc

CFG, _ARGV = pc.load()


def _opt(name, default=None):
    for i, a in enumerate(_ARGV):
        if a == name and i + 1 < len(_ARGV):
            return _ARGV[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


PHASE = _opt("--phase", "outbound")   # outbound | crossref | outbound-crossref | titles | inbound | audit
ALLOW_SEARCH = "--allow-search" in _ARGV       # titles phase bills 100 credits/doc; opt in explicitly
PROBE = "--probe" in _ARGV                     # crossref: try ONE document and print the raw candidates
DRY_RUN = "--dry-run" in _ARGV
CREDIT_BUDGET = int(_opt("--credit-budget", 90000))
MAX_SECONDS = float(_opt("--max-seconds", 0)) or None   # stop cleanly after N s; log resumes
_T_START = time.time()
MIN_CITED_BY = int(_opt("--min-cited-by", 1))
MAX_WORKS = int(_opt("--max-works", 0)) or None       # inbound: cap works per run
TITLE_SIM = float(_opt("--title-sim", 0.92))
TITLES_LIMIT = int(_opt("--limit", 0)) or None        # titles: try at most N documents this run
TITLES_MAX_YEAR = int(_opt("--max-year", 0)) or None  # titles: only documents with year <= this
TITLES_MIN_YEAR = int(_opt("--min-year", 0)) or None  # titles: only documents with year >= this
TITLES_STRATUM = _opt("--stratum")                     # titles: only this register stratum (e.g. journal)
TITLES_SAMPLE = "--sample" in _ARGV                    # titles: random sample (seed 42) instead of file order
YEAR_SLACK = int(_opt("--year-slack", 1))

API = "https://api.openalex.org/works"
LIST_CREDITS = 10             # filter / list request
SEARCH_CREDITS = 100          # any `.search` filter is billed 10x (openalex pricing: $0.001 vs $0.0001)
CROSSREF_API = "https://api.crossref.org/works"
CROSSREF_SLEEP = 0.15         # starting pace; re-set from X-Rate-Limit headers on first reply
_cr_pace = CROSSREF_SLEEP
_cr_announced = False
ALLOW_COMMON_POOL = "--allow-common-pool" in _ARGV
DOI_BATCH = 50                # OpenAlex allows up to 50 piped values per filter
PER_PAGE = 200
SLEEP = 0.12                  # filter/batch requests: ~8 req/s was fine for 2,280 batches
SEARCH_SLEEP = 0.6            # title.search requests are heavier; 8/s drew sustained 429s
BACKOFF = (10, 30, 90, 180, 300)   # 429/5xx waits; ~10 min total before a clean stop
_pace_mult = 1.0              # adaptive: doubles on 429 (cap 8x), eases back on success

WORK_FIELDS = ",".join([
    "id", "doi", "display_name", "publication_year", "type",
    "primary_location", "referenced_works", "cited_by_count",
])
CITE_FIELDS = ",".join([
    "id", "doi", "display_name", "publication_year", "type", "primary_location",
])

OUT_DIR = Path(CFG["docs_path"]).parent / "citations"
WORKS_OUT = OUT_DIR / "openalex_works.jsonl"
XREF_OUT = OUT_DIR / "crossref_dois.jsonl"     # DOIs recovered by title via Crossref
CITES_OUT = OUT_DIR / "openalex_cites.jsonl"
UNRESOLVED_OUT = OUT_DIR / "unresolved.jsonl"
LOG_PATH = OUT_DIR / "retrieval_log.jsonl"

KEYS_PATH = Path(".secrets/api_keys.yaml")

_credits_used = 0


# ------------------------------------------------------------------ plumbing

def keys():
    if not KEYS_PATH.exists():
        raise SystemExit(
            f"No {KEYS_PATH}. Create it (git-ignored) with:\n"
            "  openalex: \"<key from openalex.org/settings/api>\"\n"
            "  openalex_mailto: \"you@example.edu\"\n"
            "Since 2026-02-13 OpenAlex requires an API key; without one you get "
            "100 credits/day, which is a smoke test and nothing more."
        )
    k = yaml.safe_load(open(KEYS_PATH, encoding="utf-8")) or {}
    if not k.get("openalex"):
        raise SystemExit(f"{KEYS_PATH} has no 'openalex' key.")
    return k


def norm_doi(d):
    if not d:
        return ""
    d = str(d).strip().lower()
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d)
    d = re.sub(r"^doi:\s*", "", d)
    return d


def doc_key(d):
    """Stable id for a corpus document: its DOI, else a hash of title+year."""
    doi = norm_doi(d.get("doi"))
    if doi:
        return "doi:" + doi
    basis = pc.normalize_text(d.get("title") or "") + "|" + str(d.get("year") or "")
    return "t:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]


def load_log():
    done = set()
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["unit"])
                except Exception:
                    continue
    return done


def credits_today():
    """Credits already logged today, across ALL runs and phases — so the
    budget is a per-day figure, not per-process. Outbound + titles in one
    evening was 22,800 + 90,000 > OpenAlex's 100,000/day; this stops that."""
    today = dt.date.today().isoformat()
    total = 0
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if str(rec.get("timestamp", ""))[:10] == today:
                    total += int(rec.get("credits", 0) or 0)
    return total


def log_unit(unit, **fields):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"unit": unit, "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    rec.update(fields)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def append_jsonl(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def spend(n=LIST_CREDITS):
    global _credits_used
    if n == 0:
        return
    if MAX_SECONDS and time.time() - _T_START > MAX_SECONDS:
        raise Budget(f"--max-seconds {MAX_SECONDS:.0f} reached after {_credits_used} "
                     f"credits; the log resumes on the next call")
    _credits_used += n
    if _credits_used > CREDIT_BUDGET:
        raise Budget(f"credit budget {CREDIT_BUDGET} reached "
                     f"({_credits_used} spent); rerun tomorrow, the log resumes")


class Budget(Exception):
    pass


def get(params, pace=SLEEP, credits=LIST_CREDITS):
    """One list request. Returns parsed JSON.

    429/5xx: wait through BACKOFF (~10 min total), and slow the base pace
    for the rest of the run each time a 429 appears. If the limit persists
    past the whole schedule, stop CLEANLY (Budget) — the log resumes.
    Never exit hard on a rate limit; only on a rejected key / exhausted day.
    """
    global _pace_mult
    k = keys()
    q = dict(params)
    q["api_key"] = k["openalex"]
    if k.get("openalex_mailto"):
        q["mailto"] = k["openalex_mailto"]
    url = API + "?" + urllib.parse.urlencode(q)
    spend(credits)
    for attempt in range(len(BACKOFF) + 1):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "conceptual-cartography/1.0 (research; contact in mailto param)"
            })
            with urllib.request.urlopen(req, timeout=90) as r:
                time.sleep(pace * _pace_mult)
                _pace_mult = max(1.0, _pace_mult * 0.98)     # ease back slowly
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                _pace_mult = min(8.0, _pace_mult * 2)        # and stay slower
                if attempt >= 1:
                    # a 429 that survives the first wait is not a burst limit: OpenAlex
                    # answers 429 when the day's credits are gone (learned 2026-09-25).
                    raise Budget(
                        "HTTP 429 persisted after backoff — almost certainly the daily "
                        f"credit allowance is exhausted ({_credits_used:,} logged today; "
                        "check openalex.org/settings/usage). Stopping cleanly; the log resumes.")
            if e.code in (429, 500, 502, 503, 504) and attempt < len(BACKOFF):
                wait = BACKOFF[attempt]
                print(f"  [http {e.code}] waiting {wait}s (pace now {pace * _pace_mult:.2f}s/req)")
                time.sleep(wait)
                continue
            if e.code in (401, 403, 409):
                raise SystemExit(
                    f"HTTP {e.code} from OpenAlex — key rejected or daily credits "
                    f"exhausted ({_credits_used:,} spent today). The log resumes tomorrow."
                )
            if e.code in (429, 500, 502, 503, 504):
                raise Budget(f"HTTP {e.code} persisted through {sum(BACKOFF)}s of backoff; "
                             f"stopping cleanly, the log resumes")
            raise
        except urllib.error.URLError as e:
            wait = BACKOFF[min(attempt, len(BACKOFF) - 1)]
            print(f"  [net {e}] retrying in {wait}s")
            time.sleep(wait)
    raise Budget("repeated network failures; stopping cleanly, the log resumes")


def venue_of(work):
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name") or ""


def corpus_docs():
    p = CFG.path("docs_path")
    if not p.exists():
        raise SystemExit(f"docs_path not found: {p} — run 16 first")
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ------------------------------------------------------------ phase: outbound

def phase_outbound():
    done = load_log()
    batch, pending = [], []
    stats = Counter()
    for d in corpus_docs():
        doi = norm_doi(d.get("doi"))
        if not doi:
            stats["no_doi"] += 1
            continue
        stats["with_doi"] += 1
        pending.append((doc_key(d), doi, d))

    batches = [pending[i:i + DOI_BATCH] for i in range(0, len(pending), DOI_BATCH)]
    todo = [(i, b) for i, b in enumerate(batches) if f"doi_batch:{i}" not in done]
    print(f"[outbound] {stats['with_doi']} docs with DOI, {stats['no_doi']} without; "
          f"{len(batches)} batches, {len(todo)} remaining")
    print(f"[outbound] estimated credits for the remainder: {len(todo) * LIST_CREDITS:,}")
    if DRY_RUN:
        return

    for i, b in todo:
        dois = "|".join(doi for _, doi, _ in b)
        try:
            js = get({"filter": f"doi:{dois}", "per-page": PER_PAGE, "select": WORK_FIELDS})
        except Budget as e:
            print(f"[stop] {e}")
            return
        by_doi = {norm_doi(w.get("doi")): w for w in js.get("results", [])}
        rows, missing = [], []
        for key, doi, d in b:
            w = by_doi.get(doi)
            if not w:
                missing.append({"doc_key": key, "doi": doi, "year": d.get("year"),
                                "title": d.get("title"), "reason": "doi_not_in_openalex"})
                continue
            rows.append({
                "doc_key": key, "route": "doi", "match_score": 1.0,
                "openalex_id": w["id"], "doi": doi,
                "corpus_year": d.get("year"), "openalex_year": w.get("publication_year"),
                "corpus_journal": d.get("journal"), "openalex_venue": venue_of(w),
                "period": d.get("period"), "terms": d.get("terms"), "dbs": d.get("dbs"),
                "type": w.get("type"),
                "cited_by_count": w.get("cited_by_count", 0),
                "referenced_works": w.get("referenced_works", []),
                "n_refs": len(w.get("referenced_works") or []),
            })
        append_jsonl(WORKS_OUT, rows)
        append_jsonl(UNRESOLVED_OUT, missing)
        log_unit(f"doi_batch:{i}", n_in=len(b), n_resolved=len(rows), n_missing=len(missing),
                 credits=LIST_CREDITS)
        print(f"  batch {i + 1}/{len(batches)}: {len(rows)} resolved, {len(missing)} missing "
              f"({_credits_used:,} credits)")



# ------------------------------------------------------------ phase: crossref

def crossref_lookup(title, year, mailto):
    """Ask Crossref for the DOI of a title. Free, no key, polite pool via mailto.
    Returns the raw candidate list (each: DOI, title, issued, container-title, score)."""
    q = {"query.bibliographic": re.sub(r"\s+", " ", title)[:300],
         "rows": 3,
         "filter": f"from-pub-date:{year - YEAR_SLACK},until-pub-date:{year + YEAR_SLACK}",
         "select": "DOI,title,issued,container-title,score"}
    if mailto:
        q["mailto"] = mailto
    url = CROSSREF_API + "?" + urllib.parse.urlencode(q)
    ua = f"conceptual-cartography/1.0 (mailto:{mailto})" if mailto else "conceptual-cartography/1.0"
    global _cr_pace, _cr_announced
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": ua}),
                                        timeout=60) as r:
                # Crossref tells you its current allowance; pace to it, with headroom.
                lim = r.headers.get("X-Rate-Limit-Limit")
                itv = r.headers.get("X-Rate-Limit-Interval", "1s")
                if lim:
                    try:
                        per_s = float(lim) / float(itv.rstrip("s") or 1)
                        _cr_pace = max(CROSSREF_SLEEP, 1.5 / per_s)   # ~2/3 of the allowance
                    except ValueError:
                        pass
                if not _cr_announced:
                    print(f"  [crossref] pool={'polite' if mailto else 'COMMON'} "
                          f"limit={lim or '?'}/{itv} -> pace {_cr_pace:.2f}s/req")
                    _cr_announced = True
                time.sleep(_cr_pace)
                return (json.loads(r.read().decode("utf-8")).get("message") or {}).get("items") or []
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                ra = e.headers.get("Retry-After") if e.headers else None
                wait = int(ra) if (ra and ra.isdigit()) else (5, 15, 45, 90)[attempt]
                _cr_pace = min(5.0, _cr_pace * 2)
                print(f"  [crossref http {e.code}] waiting {wait}s (pace now {_cr_pace:.2f}s/req)")
                time.sleep(wait)
                continue
            if e.code == 404:
                return []
            raise
        except urllib.error.URLError as e:
            time.sleep((5, 15, 45, 90)[attempt])
    raise Budget("Crossref unreachable after retries; stopping cleanly, the log resumes")


def _cr_year(item):
    try:
        return int((item.get("issued") or {}).get("date-parts", [[None]])[0][0])
    except Exception:
        return None


def phase_crossref():
    """DOI discovery for documents whose export carried none. Costs no OpenAlex
    credits. The 630-odd DOI-misses (DOI present, unknown to OpenAlex) are NOT
    retried here — Crossref would only hand back the same DOI."""
    k = keys() if KEYS_PATH.exists() else {}
    mailto = k.get("crossref_mailto") or k.get("openalex_mailto")
    if not mailto and not ALLOW_COMMON_POOL:
        raise SystemExit(
            "[crossref] no email in .secrets/api_keys.yaml — add\n"
            "    openalex_mailto: \"you@example.edu\"\n"
            "so Crossref routes us to its polite pool (the anonymous pool is throttled at "
            "this volume and it is bad manners to lean on it). Or pass --allow-common-pool.")
    done = load_log()
    pending = []
    for d in corpus_docs():
        if norm_doi(d.get("doi")):
            continue
        key = doc_key(d)
        if f"crossref:{key}" in done:
            continue
        if (d.get("title") or "").strip() and d.get("year"):
            pending.append((key, d))
    print(f"[crossref] {len(pending)} DOI-less documents to try; no credits; "
          f"~{len(pending) * (CROSSREF_SLEEP + 0.25) / 60:.0f} min at polite pace"
          + ("" if mailto else "  (no mailto in secrets: common pool, slower)"))
    if DRY_RUN:
        return
    if PROBE:
        key, d = pending[0]
        print(f"\n[probe] {d.get('year')}  {d.get('title')}\n        corpus journal: {d.get('journal')}")
        for it in crossref_lookup(d["title"], int(d["year"]), mailto):
            t = (it.get("title") or [""])[0]
            sim = difflib.SequenceMatcher(None, pc.normalize_text(d["title"]),
                                          pc.normalize_text(t)).ratio()
            print(f"   score={it.get('score', 0):.1f} sim={sim:.3f} year={_cr_year(it)} "
                  f"doi={it.get('DOI')}\n      {t[:110]}\n      {(it.get('container-title') or [''])[0]}")
        return

    found = 0
    for n, (key, d) in enumerate(pending, 1):
        yr = int(d["year"])
        try:
            items = crossref_lookup(d["title"], yr, mailto)
        except Budget as e:
            print(f"[stop] {e}")
            return
        want = pc.normalize_text(d["title"])
        best, best_sim = None, 0.0
        for it in items:
            got = pc.normalize_text((it.get("title") or [""])[0])
            sim = difflib.SequenceMatcher(None, want, got).ratio()
            cy = _cr_year(it)
            if cy is not None and abs(cy - yr) > YEAR_SLACK:
                continue
            if sim > best_sim:
                best, best_sim = it, sim
        if best and best_sim >= TITLE_SIM:
            append_jsonl(XREF_OUT, [{
                "doc_key": key, "doi": norm_doi(best.get("DOI")), "match_sim": round(best_sim, 4),
                "crossref_score": best.get("score"), "corpus_year": yr, "crossref_year": _cr_year(best),
                "corpus_journal": d.get("journal"),
                "crossref_journal": (best.get("container-title") or [""])[0],
            }])
            log_unit(f"crossref:{key}", found=True, sim=round(best_sim, 4), credits=0)
            found += 1
        else:
            log_unit(f"crossref:{key}", found=False, sim=round(best_sim, 4), credits=0)
        if n % 200 == 0:
            print(f"  {n}/{len(pending)}  DOIs found so far: {found} ({100 * found / n:.0f}%)")
    print(f"[crossref] done: {found} DOIs recovered of {len(pending)} tried")


def phase_outbound_crossref():
    """OpenAlex records for the DOIs Crossref recovered. Same cheap batched
    route as outbound; separate log namespace so outbound's own log is untouched."""
    if not XREF_OUT.exists():
        raise SystemExit("run --phase crossref first")
    docs = {}
    for d in corpus_docs():
        if not norm_doi(d.get("doi")):
            docs[doc_key(d)] = d
    pending = []
    with open(XREF_OUT, encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            d = docs.get(x["doc_key"])
            if d and x.get("doi"):
                pending.append((x["doc_key"], x["doi"], d, x))
    done = load_log()
    batches = [pending[i:i + DOI_BATCH] for i in range(0, len(pending), DOI_BATCH)]
    def bkey(b):
        return "xref_batch:" + hashlib.sha1("|".join(doi for _, doi, _, _ in b).encode()).hexdigest()[:12]
    todo = [b for b in batches if bkey(b) not in done]
    print(f"[outbound-crossref] {len(pending)} recovered DOIs in {len(batches)} batches, "
          f"{len(todo)} remaining; ~{len(todo) * LIST_CREDITS:,} credits")
    if DRY_RUN:
        return
    for b in todo:
        dois = "|".join(doi for _, doi, _, _ in b)
        try:
            js = get({"filter": f"doi:{dois}", "per-page": PER_PAGE, "select": WORK_FIELDS})
        except Budget as e:
            print(f"[stop] {e}")
            return
        by_doi = {norm_doi(w.get("doi")): w for w in js.get("results", [])}
        rows, missing = [], []
        for key, doi, d, x in b:
            w = by_doi.get(doi)
            if not w:
                missing.append({"doc_key": key, "doi": doi, "year": d.get("year"),
                                "title": d.get("title"), "reason": "crossref_doi_not_in_openalex"})
                continue
            rows.append({
                "doc_key": key, "route": "doi_via_crossref", "match_score": x.get("match_sim"),
                "openalex_id": w["id"], "doi": doi,
                "corpus_year": d.get("year"), "openalex_year": w.get("publication_year"),
                "corpus_journal": d.get("journal"), "openalex_venue": venue_of(w),
                "period": d.get("period"), "terms": d.get("terms"), "dbs": d.get("dbs"),
                "type": w.get("type"),
                "cited_by_count": w.get("cited_by_count", 0),
                "referenced_works": w.get("referenced_works", []),
                "n_refs": len(w.get("referenced_works") or []),
            })
        append_jsonl(WORKS_OUT, rows)
        append_jsonl(UNRESOLVED_OUT, missing)
        log_unit(bkey(b), n_in=len(b), n_resolved=len(rows), n_missing=len(missing),
                 credits=LIST_CREDITS)
    print(f"[outbound-crossref] done ({_credits_used:,} credits today)")

# -------------------------------------------------------------- phase: titles

def phase_titles():
    """One list request per DOI-less document. Expensive; run it deliberately."""
    done = load_log()
    # DOIs that OpenAlex did not recognise in the outbound pass get a second
    # chance here by title (~2-3% of DOI'd docs; reason doi_not_in_openalex).
    doi_misses = set()
    if UNRESOLVED_OUT.exists():
        with open(UNRESOLVED_OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    u = json.loads(line)
                except Exception:
                    continue
                if u.get("reason") == "doi_not_in_openalex":
                    doi_misses.add(u["doc_key"])
    pending, n_noDoi, n_miss = [], 0, 0
    for d in corpus_docs():
        key = doc_key(d)
        has_doi = bool(norm_doi(d.get("doi")))
        if has_doi and key not in doi_misses:
            continue
        if f"title:{key}" in done:
            continue
        if TITLES_MAX_YEAR and int(d.get("year") or 0) > TITLES_MAX_YEAR:
            continue
        if TITLES_MIN_YEAR and int(d.get("year") or 0) < TITLES_MIN_YEAR:
            continue
        if TITLES_STRATUM and (d.get("stratum") or "") != TITLES_STRATUM:
            continue
        if (d.get("title") or "").strip() and d.get("year"):
            pending.append((key, d))
            if has_doi:
                n_miss += 1
            else:
                n_noDoi += 1
    if TITLES_SAMPLE:   # a representative sample, not the first N in file order
        import random
        random.Random(42).shuffle(pending)
    if TITLES_LIMIT:
        pending = pending[:TITLES_LIMIT]
    print(f"[titles] {len(pending)} documents to try by OpenAlex title search "
          f"({n_noDoi} without DOI, {n_miss} DOI-misses); "
          f"estimated credits {len(pending) * SEARCH_CREDITS:,} at {SEARCH_CREDITS}/search "
          f"= ~{len(pending) * SEARCH_CREDITS / 100000:.1f} days of free-tier allowance")
    print("[titles] Run --phase crossref first (free); use this only for the residual.")
    if DRY_RUN:
        return
    if not ALLOW_SEARCH:
        raise SystemExit("[titles] refusing to spend search credits without --allow-search")

    for n, (key, d) in enumerate(pending, 1):
        title = re.sub(r"[^\w\s\-]", " ", d["title"])[:250].strip()
        yr = int(d["year"])
        try:
            js = get({
                "filter": (f"title.search:{title},"
                           f"from_publication_date:{yr - YEAR_SLACK}-01-01,"
                           f"to_publication_date:{yr + YEAR_SLACK}-12-31"),
                "per-page": 10, "select": WORK_FIELDS,
            }, pace=SEARCH_SLEEP, credits=SEARCH_CREDITS)
        except Budget as e:
            print(f"[stop] {e}")
            return
        want = pc.normalize_text(d["title"])
        best, best_score = None, 0.0
        for w in js.get("results", []):
            got = pc.normalize_text(w.get("display_name") or "")
            s = difflib.SequenceMatcher(None, want, got).ratio()
            if s > best_score:
                best, best_score = w, s
        if best and best_score >= TITLE_SIM:
            append_jsonl(WORKS_OUT, [{
                "doc_key": key, "route": "title" if not norm_doi(d.get("doi")) else "title_after_doi_miss",
                "match_score": round(best_score, 4),
                "openalex_id": best["id"], "doi": norm_doi(best.get("doi")),
                "corpus_year": yr, "openalex_year": best.get("publication_year"),
                "corpus_journal": d.get("journal"), "openalex_venue": venue_of(best),
                "period": d.get("period"), "terms": d.get("terms"), "dbs": d.get("dbs"),
                "type": best.get("type"),
                "cited_by_count": best.get("cited_by_count", 0),
                "referenced_works": best.get("referenced_works", []),
                "n_refs": len(best.get("referenced_works") or []),
            }])
            log_unit(f"title:{key}", resolved=True, score=round(best_score, 4),
                     credits=SEARCH_CREDITS)
        else:
            append_jsonl(UNRESOLVED_OUT, [{
                "doc_key": key, "doi": "", "year": yr, "title": d.get("title"),
                "reason": "no_title_match", "best_score": round(best_score, 4)}])
            log_unit(f"title:{key}", resolved=False, score=round(best_score, 4),
                     credits=SEARCH_CREDITS)
        if n % 100 == 0:
            print(f"  {n}/{len(pending)} ({_credits_used:,} credits)")
    titles_summary()


def titles_summary():
    """Hit rate of every title search ever logged, by stratum / period / source,
    so a sample run answers 'is the DOI-less residual recoverable?'."""
    if not LOG_PATH.exists():
        return
    hits = {}
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                u = json.loads(line)
            except Exception:
                continue
            if str(u.get("unit", "")).startswith("title:"):
                hits[u["unit"][6:]] = bool(u.get("resolved"))
    if not hits:
        return
    by = {"stratum": defaultdict(lambda: [0, 0]), "period": defaultdict(lambda: [0, 0]),
          "source": defaultdict(lambda: [0, 0])}
    for d in corpus_docs():
        k = doc_key(d)
        if k not in hits:
            continue
        h = hits[k]
        by["stratum"][d.get("stratum") or "?"][0] += 1; by["stratum"][d.get("stratum") or "?"][1] += h
        by["period"][d.get("period") or "?"][0] += 1; by["period"][d.get("period") or "?"][1] += h
        for sid in (d.get("sources") or ["?"]):
            by["source"][sid][0] += 1; by["source"][sid][1] += h
    n, r = len(hits), sum(hits.values())
    print(f"\n[titles] all title searches logged so far: {n} tried, {r} resolved ({100 * r / n:.1f}%)")
    for dim in ("stratum", "period", "source"):
        print(f"  by {dim}:")
        for k in sorted(by[dim], key=lambda x: -by[dim][x][0]):
            t, h = by[dim][k]
            print(f"    {str(k):<28} tried {t:>4}  resolved {100 * h / max(t, 1):>5.1f}%")


# ------------------------------------------------------------- phase: inbound

def phase_inbound():
    """Works citing our documents. One request per target (cites: cannot be
    batched per target — a piped query returns the union without telling you
    which target each citing work hit). Cursor-paged for highly cited works."""
    if not WORKS_OUT.exists():
        raise SystemExit("run --phase outbound first")
    done = load_log()
    targets = []
    with open(WORKS_OUT, encoding="utf-8") as f:
        for line in f:
            w = json.loads(line)
            if w.get("cited_by_count", 0) >= MIN_CITED_BY and \
                    f"cites:{w['openalex_id']}" not in done:
                targets.append(w)
    targets.sort(key=lambda w: -w.get("cited_by_count", 0))
    if MAX_WORKS:
        targets = targets[:MAX_WORKS]
    est_pages = sum(max(1, -(-t.get("cited_by_count", 0) // PER_PAGE)) for t in targets)
    print(f"[inbound] {len(targets)} targets with cited_by_count >= {MIN_CITED_BY}; "
          f"~{est_pages} requests, ~{est_pages * LIST_CREDITS:,} credits")
    if DRY_RUN:
        return

    for n, t in enumerate(targets, 1):
        wid = t["openalex_id"].rsplit("/", 1)[-1]
        cursor, rows, pages = "*", [], 0
        try:
            while cursor:
                js = get({"filter": f"cites:{wid}", "per-page": PER_PAGE,
                          "cursor": cursor, "select": CITE_FIELDS})
                for w in js.get("results", []):
                    rows.append({
                        "cited_doc_key": t["doc_key"], "cited_openalex_id": t["openalex_id"],
                        "citing_openalex_id": w["id"], "citing_doi": norm_doi(w.get("doi")),
                        "citing_year": w.get("publication_year"),
                        "citing_venue": venue_of(w), "citing_type": w.get("type"),
                    })
                pages += 1
                cursor = (js.get("meta") or {}).get("next_cursor")
                if pages > 60:                     # ~12k citing works; enough
                    break
        except Budget as e:
            append_jsonl(CITES_OUT, rows)
            print(f"[stop] {e}")
            return
        append_jsonl(CITES_OUT, rows)
        log_unit(f"cites:{t['openalex_id']}", n_citing=len(rows), pages=pages,
                 credits=pages * LIST_CREDITS)
        if n % 25 == 0:
            print(f"  {n}/{len(targets)} targets ({_credits_used:,} credits)")


# --------------------------------------------------------------- phase: audit

def phase_audit():
    """Coverage and edge density. Read this before interpreting anything."""
    if not WORKS_OUT.exists():
        raise SystemExit("nothing collected yet")
    works, by_key, id_to_key = [], {}, {}
    with open(WORKS_OUT, encoding="utf-8") as f:
        for line in f:
            w = json.loads(line)
            works.append(w)
            by_key[w["doc_key"]] = w
            id_to_key[w["openalex_id"]] = w["doc_key"]

    total_docs = sum(1 for _ in corpus_docs())
    per_period = Counter()
    per_period_res = Counter()
    # resolution by register stratum, corpus source and venue category: a
    # citation-layer claim about a habitat is only as good as that habitat's
    # resolution (found 2026-10-01: the ego-depletion family corpus resolves
    # at 59% overall because its H.W. Wilson / CINAHL / EconLit exports and
    # its thesis stratum carry few DOIs — the applied habitats where the
    # migration finding lives are the under-resolved ones).
    cat = pc.venue_categorizer(CFG)
    by_stratum, by_stratum_res = Counter(), Counter()
    by_source, by_source_res = Counter(), Counter()
    by_cat, by_cat_res = Counter(), Counter()
    by_term, by_term_res = Counter(), Counter()
    per_period_j, per_period_j_res, per_period_thesis = Counter(), Counter(), Counter()
    for d in corpus_docs():
        per_period[d.get("period", "?")] += 1
        hit = doc_key(d) in by_key
        if hit:
            per_period_res[d.get("period", "?")] += 1
        st = d.get("stratum") or "?"
        if st == "journal":
            per_period_j[d.get("period", "?")] += 1
            per_period_j_res[d.get("period", "?")] += hit
        elif st == "thesis":
            per_period_thesis[d.get("period", "?")] += 1
        by_stratum[st] += 1; by_stratum_res[st] += hit
        for sid in (d.get("sources") or ["?"]):
            by_source[sid] += 1; by_source_res[sid] += hit
        vc = cat(d.get("journal") or "")
        by_cat[vc] += 1; by_cat_res[vc] += hit
        for t in (d.get("terms") or []):
            by_term[t] += 1; by_term_res[t] += hit

    # intra-corpus edges, from the outbound pass alone
    edges = 0
    edges_by_period = Counter()
    for w in works:
        for r in (w.get("referenced_works") or []):
            if r in id_to_key:
                edges += 1
                edges_by_period[w.get("period", "?")] += 1

    route = Counter(w.get("route") for w in works)
    print(f"\n=== resolution ===")
    print(f"corpus documents      {total_docs:,}")
    print(f"resolved to OpenAlex  {len(works):,} ({100 * len(works) / max(total_docs, 1):.1f}%)"
          f"   [doi {route['doi']:,} | doi via crossref {route['doi_via_crossref']:,} | "
          f"title {route['title']:,} | title after doi-miss {route['title_after_doi_miss']:,}]")
    print(f"\nby period (all | journal stratum only | thesis share of period):")
    for p in sorted(per_period):
        n, r = per_period[p], per_period_res[p]
        nj, rj = per_period_j[p], per_period_j_res[p]
        print(f"  {p:<10} n={n:>6}  resolved {100 * r / max(n, 1):>5.1f}%  "
              f"journal {100 * rj / max(nj, 1):>5.1f}% (n={nj:,})  thesis {100 * per_period_thesis[p] / max(n, 1):>4.1f}%  "
              f"intra-corpus refs out {edges_by_period[p]:>7,}")
    def _table(title, n, r, key=None):
        print(f"\n{title}")
        for k in sorted(n, key=key or (lambda x: -n[x])):
            print(f"  {str(k):<34} n={n[k]:>6}  resolved {100 * r[k] / max(n[k], 1):>5.1f}%")
    _table("by register stratum:", by_stratum, by_stratum_res)
    _table("by corpus source (a document can be in several):", by_source, by_source_res)
    _table("by venue category (config regexes on the EBSCO journal field):", by_cat, by_cat_res)
    _table("by term:", by_term, by_term_res)
    print(f"\n=== intra-corpus citation graph ===")
    print(f"nodes {len(works):,}   edges {edges:,}   "
          f"mean out-degree {edges / max(len(works), 1):.2f}")
    print("(edges are complete from the outbound pass: any corpus->corpus citation "
          "appears in the citing document's referenced_works)")

    # the bias check route B exists for: does the title route recover a
    # different kind of venue than the DOI route?
    ven = defaultdict(Counter)
    for w in works:
        ven[w.get("route")][w.get("openalex_venue") or "(none)"] += 1
    print(f"\n=== route comparison (top venues) ===")
    for r in ("doi", "doi_via_crossref", "title", "title_after_doi_miss"):
        if not ven[r]:
            continue
        top = ", ".join(f"{v} ({c})" for v, c in ven[r].most_common(6))
        print(f"  {r:<6} {top}")
    print("If the title route's venues look systematically smaller/older, the "
          "DOI-only join was carrying a publisher-prestige bias — say so in the methods.")

    titles_summary()

    if CITES_OUT.exists():
        inb = sum(1 for _ in open(CITES_OUT, encoding="utf-8"))
        print(f"\n=== inbound ===\ncollected citing edges: {inb:,}")

    out = CFG.results / "citation_layer_audit.json"
    out.write_text(json.dumps({
        "corpus_docs": total_docs,
        "resolved": len(works),
        "route_counts": dict(route),
        "per_period": {p: {"n": per_period[p], "resolved": per_period_res[p],
                           "n_journal": per_period_j[p], "resolved_journal": per_period_j_res[p],
                           "n_thesis": per_period_thesis[p],
                           "intra_refs_out": edges_by_period[p]} for p in per_period},
        "intra_corpus_edges": edges,
        "by_stratum": {k: {"n": by_stratum[k], "resolved": by_stratum_res[k]} for k in by_stratum},
        "by_source": {k: {"n": by_source[k], "resolved": by_source_res[k]} for k in by_source},
        "by_venue_category": {k: {"n": by_cat[k], "resolved": by_cat_res[k]} for k in by_cat},
        "by_term": {k: {"n": by_term[k], "resolved": by_term_res[k]} for k in by_term},
    }, indent=2), encoding="utf-8")
    print(f"\n[audit] {out}")


# ------------------------------------------------------------------- dispatch

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    global _credits_used
    _credits_used = credits_today()
    print(f"[26] concept={CFG['concept']}  phase={PHASE}  "
          f"budget={CREDIT_BUDGET:,} credits/day, {_credits_used:,} already spent today"
          f"{'  (dry run)' if DRY_RUN else ''}")
    t0 = time.time()
    phases = {"outbound": phase_outbound, "crossref": phase_crossref,
              "outbound-crossref": phase_outbound_crossref, "titles": phase_titles,
              "inbound": phase_inbound, "audit": phase_audit}
    if PHASE not in phases:
        raise SystemExit(f"unknown --phase {PHASE}; one of {', '.join(phases)}")
    phases[PHASE]()
    if not DRY_RUN and PHASE != "audit":
        pc.write_manifest(
            CFG, __file__,
            params={"phase": PHASE, "credit_budget": CREDIT_BUDGET,
                    "min_cited_by": MIN_CITED_BY, "max_works": MAX_WORKS,
                    "title_sim": TITLE_SIM, "year_slack": YEAR_SLACK,
                    "credits_spent_this_run": _credits_used},
            outputs=[WORKS_OUT, XREF_OUT, CITES_OUT, UNRESOLVED_OUT, LOG_PATH],
            notes=("OpenAlex credit model since 2026-02-13; list request = 10 credits. "
                   "Intra-corpus edges are complete from the outbound pass."))
    print(f"[26] done in {time.time() - t0:.0f}s, {_credits_used:,} credits this run")


if __name__ == "__main__":
    main()
