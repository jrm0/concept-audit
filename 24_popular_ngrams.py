"""
24_popular_ngrams.py — Salience curves for the family terms from Google Books
Ngrams (frequency only, no contexts). Session 33 (2026-09-15).

What this is for: Haslam's own popular-usage evidence is Ngram frequency; the
lead/lag question (does salience precede breadth?) needs a salience curve.
This gives one per term, 1800–2019, as relative frequency per year. It does
NOT feed the embedding pipeline — there are no sentences here.

Source: the Ngram Viewer's JSON endpoint
    https://books.google.com/ngrams/json?content=<terms>&year_start=..&year_end=..&corpus=<id>&smoothing=0
Undocumented but stable for years; rate-limited; may refuse scripted traffic
(HTTP 429 / a captcha page). The script is polite (one request per query
group, a pause between them) and says clearly when it is blocked — the
fallback is the bulk 1-gram/2-gram files (very large) or a manual export from
the Viewer, both of which write the same CSV.

Case: the Viewer is case-sensitive unless case_insensitive=true is passed —
it is passed. Multi-word terms are looked up as-is (ngrams up to 5 words).
Hyphenated forms ("post-traumatic") are separate ngrams in Google's index;
each surface form the config lists is requested and the CSV keeps them apart.

Reads:   configs/<concept>_popular.yaml -> corpus.queries.popular_terms
         (or the family CSV when the list is absent)
Writes:  data/<slug>/ngram_<corpus>.csv     term, surface_form, year, frequency
         results/<slug>/manifest_24_popular_ngrams.json

Usage:
    python3 24_popular_ngrams.py --config configs/ptsd_trauma_popular.yaml
    python3 24_popular_ngrams.py --config ... --corpus en-2019    # default
    python3 24_popular_ngrams.py --config ... --corpus en-US-2019 # or en-GB-2019
    python3 24_popular_ngrams.py --config ... --terms "trauma,PTSD"
"""

import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pipeline_config as pc

CFG, _ARGV = pc.load()


def _opt(name, default=None):
    for i, a in enumerate(_ARGV):
        if a == name and i + 1 < len(_ARGV):
            return _ARGV[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


CORPUS_ID = _opt("--corpus", "en-2019")
YEAR_START, YEAR_END = int(_opt("--start", 1800)), int(_opt("--end", 2019))
OUT_DIR = Path(CFG["docs_path"]).parent
RESULTS = CFG.results
ENDPOINT = "https://books.google.com/ngrams/json"
PAUSE = 4.0            # seconds between requests; be a good citizen
GROUP = 6              # terms per request (the Viewer accepts up to 12)


def surface_forms(term):
    """Google indexes hyphenated and solid spellings as different ngrams:
    request each. 'posttraumatic stress disorder' -> also 'post-traumatic
    stress disorder' and 'post traumatic stress disorder'."""
    forms = {term}
    for pre in ("post", "pre", "non", "anti", "inter", "intra"):
        if term.startswith(pre) and len(term) > len(pre) + 3 and term[len(pre)] != " " and term[len(pre)] != "-":
            rest = term[len(pre):]
            forms.add(f"{pre}-{rest}")
            forms.add(f"{pre} {rest}")
    if "-" in term:
        forms.add(term.replace("-", " "))
        forms.add(term.replace("-", ""))
    return sorted(forms)


def fetch(forms):
    q = {"content": ",".join(forms), "year_start": YEAR_START, "year_end": YEAR_END,
         "corpus": CORPUS_ID, "smoothing": 0, "case_insensitive": "true"}
    url = ENDPOINT + "?" + urllib.parse.urlencode(q)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; ngram salience curves)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode("utf-8", errors="replace")
    if body.lstrip().startswith("<"):
        raise RuntimeError("HTML instead of JSON — the Viewer is refusing scripted "
                           "access (captcha/rate limit). Wait, or export manually.")
    return json.loads(body)


def main():
    terms = [t.strip() for t in (_opt("--terms") or "").split(",") if t.strip()]
    if not terms:
        terms = list((CFG.get("corpus") or {}).get("queries", {}).get("popular_terms") or [])
    if not terms:
        terms = list(pc.family_patterns(CFG))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"ngram_{CORPUS_ID}.csv"
    rows, missing, blocked = [], [], False
    print(f"Ngram salience: {len(terms)} terms, corpus {CORPUS_ID}, {YEAR_START}-{YEAR_END}")
    for i in range(0, len(terms), GROUP):
        group = terms[i:i + GROUP]
        forms = []
        for t in group:
            forms += surface_forms(t)
        try:
            data = fetch(forms)
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as e:
            print(f"  BLOCKED on {group}: {e}")
            blocked = True
            break
        got = {d["ngram"].lower().replace(" (all)", ""): d["timeseries"] for d in data}
        for t in group:
            hit = False
            for f in surface_forms(t):
                ts = got.get(f.lower())
                if ts is None:
                    continue
                hit = True
                for k, v in enumerate(ts):
                    rows.append([t, f, YEAR_START + k, v])
            if not hit:
                missing.append(t)
        print(f"  {group}: {sum(1 for t in group if t not in missing)}/{len(group)} found")
        time.sleep(PAUSE)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["term", "surface_form", "year", "frequency"])
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {out_path}")
    if missing:
        print(f"no ngram found for: {missing}")
    if blocked:
        print("Stopped early (blocked). Re-run later; existing rows were written. "
              "Fallback: Ngram Viewer manual export (same columns), or the bulk files.")
    pc.write_manifest(
        CFG, __file__,
        params={"corpus": CORPUS_ID, "year_start": YEAR_START, "year_end": YEAR_END,
                "terms": terms, "missing": missing, "blocked": blocked, "n_rows": len(rows),
                "endpoint": ENDPOINT, "note": "relative frequency per year (share of all "
                "ngrams of that length in the corpus year); Viewer JSON endpoint, "
                "case-insensitive, smoothing 0"},
        outputs=[out_path])


if __name__ == "__main__":
    main()
