"""
25_popular_news.py — Retrieve the POPULAR STRATUM: news articles mentioning
the family terms, from the Guardian Open Platform (full text, 1999-) and the
NYT Article Search API (headline + abstract + lead paragraph, 1851-).
Session 33 (2026-09-15).

Writes one jsonl per (outlet, year) in the schema 16 ingests
(corpus.sources[].files ending in .jsonl):
    {"id": url, "url": url, "doi": "", "year": 2004, "date": "2004-03-12",
     "outlet": "The Guardian", "section": "World news", "title": ..., "text": ...,
     "dbs": ["Guardian"], "ty": "NEWS", "index_terms": [...], "retrieved_by": ["trauma"]}
The same article retrieved by several terms is written once (url-deduped
within the run and across re-runs of the same file); `retrieved_by` records
every term that found it. 16 recomputes term presence from the text, so a
hit whose only mention is in the unreturned NYT body is dropped there — that
loss is counted in 16's "no family term" line and is the price of the NYT
API returning only the head of the article.

Retrieval log (corpus principle 10 — the log is the methods section):
    data/<slug>/retrieval_log.jsonl   one line per completed (outlet, term, window):
        outlet, term, from, to, hits_reported, docs_written, calls, timestamp
Windows already in the log are skipped, so the script resumes across days —
both APIs have daily call limits (developer tiers; see the key pages), so a
full pull of "trauma" is several days of short runs.

Rate limits are respected with fixed sleeps (Guardian 1 call/s; NYT 5
calls/min); HTTP 429 backs off and retries three times, then the run stops
cleanly with everything so far logged.

Keys: .secrets/api_keys.yaml (git-ignored): nyt_article_search, guardian.

Usage:
    python3 25_popular_news.py --config configs/ptsd_trauma_popular.yaml --dry-run
    python3 25_popular_news.py --config configs/ptsd_trauma_popular.yaml --source guardian
    python3 25_popular_news.py --config configs/ptsd_trauma_popular.yaml --source nyt --max-calls 450
    python3 25_popular_news.py --config ... --terms "trauma,PTSD" --years 1999-2005
    python3 25_popular_news.py --config ... --source nyt --terms "trauma,PTSD" --nyt-relevance 3
        # relevance-ranked, stop after 3 pages with no head match (see NYT_RELEVANCE_STOP)
"""

import csv
import datetime as dt
import json
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


SOURCE = _opt("--source", "all")             # guardian | nyt | all
MAX_CALLS = int(_opt("--max-calls", 400))     # per outlet per run (stay under daily caps)
DRY_RUN = "--dry-run" in _ARGV
# NYT relevance mode (Session 34): the API searches the full body but returns
# only the head; ~90% of oldest-first hits never show the term in what we get.
# Relevance ranking front-loads head-matched articles (probe: 10/10), so
# --nyt-relevance N paginates by relevance and stops after N consecutive pages
# with no head match. Logged as "relevance-stop" — a documented truncation,
# not the completeness of oldest-first. Head match = term in headline,
# abstract, lead paragraph or snippet (the same text 16 will see).
NYT_RELEVANCE_STOP = int(_opt("--nyt-relevance", 0))
OUT_DIR = Path(CFG["docs_path"]).parent
LOG_PATH = OUT_DIR / "retrieval_log.jsonl"
RESULTS = CFG.results
KEYS_PATH = Path(".secrets/api_keys.yaml")

OUTLETS = {
    "guardian": {"name": "The Guardian", "db": "Guardian", "first_year": 1999, "sleep": 1.1},
    "nyt": {"name": "The New York Times", "db": "NYT",
            "first_year": int(CFG["periods"]["year_min"]), "sleep": 12.5},
}


def load_keys():
    if not KEYS_PATH.exists():
        raise SystemExit(f"{KEYS_PATH} not found — create it from the template in the repo "
                         "(git-ignored) and paste the keys.")
    k = yaml.safe_load(open(KEYS_PATH)) or {}
    return {"guardian": (k.get("guardian") or "").strip(),
            "nyt": (k.get("nyt_article_search") or "").strip()}


def year_range():
    yr = _opt("--years")
    if yr:
        a, b = yr.split("-")
        return int(a), int(b)
    return int(CFG["periods"]["year_min"]), int(CFG["periods"]["year_max"])


def terms_to_search():
    t = _opt("--terms")
    if t:
        return [x.strip() for x in t.split(",") if x.strip()]
    return list((CFG.get("corpus") or {}).get("queries", {}).get("popular_terms") or [])


def get_json(url, sleep, tries=3):
    """GET with fixed pacing; 429 -> back off; returns (json, ok)."""
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research corpus builder"})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8", errors="replace")
            time.sleep(sleep)
            return json.loads(body), True
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < tries - 1:
                wait = 60 * (attempt + 1)
                print(f"    429 rate-limited; waiting {wait}s")
                time.sleep(wait)
                continue
            print(f"    HTTP {e.code}: {e.reason}")
            return None, False
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            print(f"    request failed: {e}")
            time.sleep(sleep)
            return None, False
    return None, False


# ----------------------------------------------------------------- outlets

def guardian_window(term, y0, y1, key, budget):
    """Yield article dicts for `term` in [y0, y1]; returns (docs, hits, calls)."""
    base = "https://content.guardianapis.com/search"
    docs, calls, page, pages, hits = [], 0, 1, 1, None
    while page <= pages and calls < budget:
        q = {"q": f'"{term}"', "from-date": f"{y0}-01-01", "to-date": f"{y1}-12-31",
             "page-size": 200, "page": page, "type": "article", "order-by": "oldest",
             "show-fields": "headline,trailText,bodyText,byline", "show-tags": "keyword",
             "api-key": key}
        data, ok = get_json(base + "?" + urllib.parse.urlencode(q), OUTLETS["guardian"]["sleep"])
        calls += 1
        if not ok:
            return docs, hits, calls, False
        resp = data.get("response", {})
        hits = resp.get("total", hits)
        pages = resp.get("pages", 1)
        for r in resp.get("results", []):
            f = r.get("fields", {}) or {}
            date = (r.get("webPublicationDate") or "")[:10]
            body = (f.get("bodyText") or "")[:20000]
            text = " ".join(x for x in [f.get("trailText") or "", body] if x).strip()
            docs.append({"id": r.get("webUrl"), "url": r.get("webUrl"), "doi": "",
                         "year": int(date[:4]) if date[:4].isdigit() else None, "date": date,
                         "outlet": OUTLETS["guardian"]["name"], "section": r.get("sectionName") or "",
                         "title": f.get("headline") or r.get("webTitle") or "",
                         "text": text, "dbs": [OUTLETS["guardian"]["db"]], "ty": "NEWS",
                         "index_terms": [t.get("webTitle") for t in (r.get("tags") or []) if t.get("webTitle")],
                         "retrieved_by": [term]})
        page += 1
    return docs, hits, calls, page > pages


def nyt_window(term, d0, d1, key, budget):
    """NYT Article Search: 10 per page, page <= 100. The API (2026) returns
    NO hit count (meta.hits is null) and a `fq` filter silently zeroes the
    results (Session 33 probe), so: no fq (document_type is checked client-
    side), paginate until a short page, and when the 100-page cap is reached
    with a full page, CONTINUE FROM THE LAST DATE SEEN (sort=oldest) — a
    date cursor instead of a month split, no refetching. Returns
    (docs, None, calls, complete)."""
    base = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    docs, calls = [], 0
    if NYT_RELEVANCE_STOP:
        return nyt_window_relevance(term, d0, d1, key, budget, base)
    cursor = d0
    while calls < budget:
        page, last_date, full_last_page = 0, None, False
        while page <= 100 and calls < budget:
            q = {"q": f'"{term}"', "begin_date": cursor.strftime("%Y%m%d"),
                 "end_date": d1.strftime("%Y%m%d"), "page": page, "sort": "oldest",
                 "api-key": key}
            data, ok = get_json(base + "?" + urllib.parse.urlencode(q), OUTLETS["nyt"]["sleep"])
            calls += 1
            if not ok:
                return docs, None, calls, False
            got = (data.get("response") or {}).get("docs") or []
            for r in got:
                if (r.get("document_type") or "article") != "article":
                    continue
                date = (r.get("pub_date") or "")[:10]
                last_date = date or last_date
                head = ((r.get("headline") or {}).get("main")) or ""
                text = " ".join(x for x in [r.get("abstract") or "", r.get("lead_paragraph") or "",
                                            r.get("snippet") or ""] if x).strip()
                docs.append({"id": r.get("web_url") or r.get("_id"), "url": r.get("web_url"), "doi": "",
                             "year": int(date[:4]) if date[:4].isdigit() else None, "date": date,
                             "outlet": OUTLETS["nyt"]["name"],
                             "section": r.get("section_name") or r.get("news_desk") or "",
                             "title": head, "text": text, "dbs": [OUTLETS["nyt"]["db"]], "ty": "NEWS",
                             "index_terms": [k.get("value") for k in (r.get("keywords") or []) if k.get("value")],
                             "retrieved_by": [term]})
            if len(got) < 10:
                return docs, None, calls, True          # short page: window exhausted
            full_last_page = True
            page += 1
        # page cap reached with full pages: move the cursor to the last date seen
        if not full_last_page or not last_date:
            return docs, None, calls, True
        try:
            nxt = dt.date.fromisoformat(last_date)
        except ValueError:
            return docs, None, calls, True
        if nxt <= cursor:                                # > 1,010 hits on a single day: accept the loss
            print(f"    NYT: more than 1,010 hits on {cursor} — beyond the API's reach; continuing next day")
            nxt = cursor + dt.timedelta(days=1)
        cursor = nxt
        if cursor > d1:
            return docs, None, calls, True
    return docs, None, calls, False                       # budget exhausted mid-window


def _nyt_doc(r, term):
    date = (r.get("pub_date") or "")[:10]
    head = ((r.get("headline") or {}).get("main")) or ""
    text = " ".join(x for x in [r.get("abstract") or "", r.get("lead_paragraph") or "",
                                r.get("snippet") or ""] if x).strip()
    return {"id": r.get("web_url") or r.get("_id"), "url": r.get("web_url"), "doi": "",
            "year": int(date[:4]) if date[:4].isdigit() else None, "date": date,
            "outlet": OUTLETS["nyt"]["name"],
            "section": r.get("section_name") or r.get("news_desk") or "",
            "title": head, "text": text, "dbs": [OUTLETS["nyt"]["db"]], "ty": "NEWS",
            "index_terms": [k.get("value") for k in (r.get("keywords") or []) if k.get("value")],
            "retrieved_by": [term]}


def nyt_window_relevance(term, d0, d1, key, budget, base):
    """--nyt-relevance N: relevance-ranked pages; stop after N consecutive
    pages with no head match, or at the 100-page cap. Only head-matched
    documents are kept (the others would be dropped by 16 anyway)."""
    pat = pc.term_pattern(term)
    docs, calls, page, empty_run = [], 0, 0, 0
    while page <= 100 and calls < budget:
        q = {"q": f'"{term}"', "begin_date": d0.strftime("%Y%m%d"),
             "end_date": d1.strftime("%Y%m%d"), "page": page, "api-key": key}
        data, ok = get_json(base + "?" + urllib.parse.urlencode(q), OUTLETS["nyt"]["sleep"])
        calls += 1
        if not ok:
            return docs, None, calls, False
        got = (data.get("response") or {}).get("docs") or []
        matched = 0
        for r in got:
            if (r.get("document_type") or "article") != "article":
                continue
            d = _nyt_doc(r, term)
            if pat.search(pc.normalize_text(d["title"] + " " + d["text"])):
                docs.append(d)
                matched += 1
        if len(got) < 10:
            return docs, None, calls, True
        empty_run = empty_run + 1 if matched == 0 else 0
        if empty_run >= NYT_RELEVANCE_STOP:
            return docs, None, calls, True          # relevance-stop: logged as complete
        page += 1
    return docs, None, calls, True


def month_windows(year):
    for m in range(1, 13):
        d0 = dt.date(year, m, 1)
        d1 = (dt.date(year + 1, 1, 1) if m == 12 else dt.date(year, m + 1, 1)) - dt.timedelta(days=1)
        yield d0, d1


# ------------------------------------------------------------------ writing

class YearFiles:
    """news_<outlet>_<year>.jsonl writers with url-dedupe across terms and runs."""

    def __init__(self, outlet):
        self.outlet = outlet
        self.seen = defaultdict(set)     # year -> urls
        self.handles = {}
        self.written = Counter()

    def path(self, year):
        return OUT_DIR / f"news_{self.outlet}_{year}.jsonl"

    def _open(self, year):
        if year not in self.handles:
            p = self.path(year)
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    for line in f:
                        try:
                            self.seen[year].add(json.loads(line).get("id"))
                        except json.JSONDecodeError:
                            pass
            self.handles[year] = open(p, "a", encoding="utf-8")
        return self.handles[year]

    def write(self, doc):
        y = doc.get("year")
        if not y or not doc.get("id"):
            return 0
        if doc["id"] in self.seen[y]:
            return 0
        self.seen[y].add(doc["id"])
        self._open(y).write(json.dumps(doc, ensure_ascii=False) + "\n")
        self.written[y] += 1
        return 1

    def close(self):
        for h in self.handles.values():
            h.close()


def load_log():
    done = set()
    if LOG_PATH.exists():
        for line in open(LOG_PATH, encoding="utf-8"):
            try:
                r = json.loads(line)
                done.add((r["outlet"], r["term"], r["from"], r["to"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def log_window(rec):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# --------------------------------------------------------------------- main

def probe_nyt(key):
    """--probe: one call per query variant, raw response summarised. Session 33
    diagnosis: 450 windows returned HTTP 200 with hits=0."""
    base = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    variants = {
        "B plain q (reference)":          {"q": "trauma", "begin_date": "19850101", "end_date": "19851231"},
        "F fq headline only":             {"fq": 'headline:("trauma")', "begin_date": "19850101", "end_date": "19851231", "sort": "oldest"},
        "G fq headline OR abstract":      {"fq": 'headline:("trauma") OR abstract:("trauma")', "begin_date": "19850101", "end_date": "19851231", "sort": "oldest"},
        "H fq headline OR abstract OR lead": {"fq": 'headline:("trauma") OR abstract:("trauma") OR lead_paragraph:("trauma")', "begin_date": "19850101", "end_date": "19851231", "sort": "oldest"},
        "I fq type_of_material":          {"q": "trauma", "fq": 'type_of_material:("News")', "begin_date": "19850101", "end_date": "19851231"},
    }
    for name, q in variants.items():
        q = dict(q, **{"api-key": key})
        url = base + "?" + urllib.parse.urlencode(q)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research corpus builder"})
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode("utf-8", errors="replace")
            status = "200"
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            status = str(e.code)
        try:
            data = json.loads(body)
            resp = data.get("response") or {}
            meta = resp.get("meta") or {}
            docs = resp.get("docs") or []
            first = docs[0] if docs else {}
            import re as _re
            in_head = sum(1 for d in docs if _re.search(r"trauma", " ".join([((d.get("headline") or {}).get("main") or ""),
                          d.get("abstract") or "", d.get("lead_paragraph") or ""]), _re.I))
            summ = (f"status={data.get('status')} hits={meta.get('hits')} docs={len(docs)} term_in_head={in_head}/{len(docs)}"
                    + (f" first={((first.get('headline') or {}).get('main') or '')[:60]!r} "
                       f"date={str(first.get('pub_date'))[:10]} type={first.get('document_type')}" if first else "")
                    + ("" if "response" in data else f" RAW={body[:300]}"))
        except json.JSONDecodeError:
            summ = f"non-JSON: {body[:300]}"
        print(f"  [{name}] HTTP {status}: {summ}")
        time.sleep(OUTLETS["nyt"]["sleep"])


def main():
    if "--probe" in _ARGV:
        probe_nyt(load_keys()["nyt"])
        return
    terms = terms_to_search()
    y_lo, y_hi = year_range()
    outlets = ["guardian", "nyt"] if SOURCE == "all" else [SOURCE]
    keys = load_keys() if not DRY_RUN else {"guardian": "x", "nyt": "x"}
    done = load_log()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Popular stratum: outlets {outlets}, terms {terms}, years {y_lo}-{y_hi}, "
          f"max {MAX_CALLS} calls per outlet this run; {len(done)} windows already logged")

    summary = Counter()
    for outlet in outlets:
        key = keys[outlet]
        if not key and not DRY_RUN:
            print(f"[{outlet}] no key in {KEYS_PATH} — skipped")
            continue
        yf = YearFiles(outlet)
        calls_left = MAX_CALLS
        stopped = False
        first = max(y_lo, OUTLETS[outlet]["first_year"])
        for term in terms:
            for year in range(first, y_hi + 1):
                if stopped:
                    break
                win = (outlet, term, f"{year}-01-01", f"{year}-12-31")
                if win in done:
                    continue
                if DRY_RUN:
                    summary[f"{outlet}:planned windows"] += 1
                    continue
                if calls_left <= 0:
                    print(f"[{outlet}] call budget for this run exhausted at {term} {year}; resume later")
                    stopped = True
                    break
                t0 = time.time()
                if outlet == "guardian":
                    docs, hits, calls, complete = guardian_window(term, year, year, key, calls_left)
                else:
                    docs, hits, calls, complete = nyt_window(term, dt.date(year, 1, 1), dt.date(year, 12, 31), key, calls_left)
                calls_left -= calls
                n = sum(yf.write(d) for d in docs)
                summary[f"{outlet}:docs"] += n
                summary[f"{outlet}:calls"] += calls
                status = "ok" if complete else "INCOMPLETE"
                print(f"[{outlet}] {term!r} {year}: hits {hits}, +{n} new docs, {calls} calls, {status} ({time.time() - t0:.0f}s)")
                if complete:
                    log_window({"outlet": outlet, "term": term, "from": win[2], "to": win[3],
                                "hits_reported": hits, "docs_written": n, "calls": calls,
                                "mode": (f"relevance-stop{NYT_RELEVANCE_STOP}" if outlet == "nyt" and NYT_RELEVANCE_STOP else "complete"),
                                "timestamp": dt.datetime.now().isoformat(timespec="seconds")})
                else:
                    stopped = True
                    print(f"[{outlet}] stopping this run (window incomplete — not logged, will be retried)")
        yf.close()
        if yf.written:
            print(f"[{outlet}] written this run by year: {dict(sorted(yf.written.items()))}")

    print("\nSUMMARY:", dict(summary))
    if DRY_RUN:
        print("(dry run: nothing fetched)")
        return
    pc.write_manifest(
        CFG, __file__,
        params={"source": SOURCE, "terms": terms, "years": [y_lo, y_hi], "max_calls": MAX_CALLS,
                "summary": dict(summary), "retrieval_log": str(LOG_PATH),
                "apis": {"guardian": "content.guardianapis.com/search (type=article, q=\"term\", full bodyText)",
                         "nyt": "api.nytimes.com/svc/search/v2/articlesearch (q=\"term\", sort=oldest, no fq — "
                                "a fq filter zeroes results; document_type checked client-side; "
                                "headline+abstract+lead_paragraph+snippet only; no hit counts returned)"}},
        outputs=[OUT_DIR],
        notes="Windows are logged only when complete; re-run to resume. NYT windows past "
              "the 100-page cap continue from the last date seen (date cursor).")
    print("Next: python3 16_build_lineage_corpus.py --config configs/ptsd_trauma_popular.yaml")


if __name__ == "__main__":
    main()
