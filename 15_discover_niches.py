"""
15_discover_niches.py — DISCOVERY-first niche mapping (no prior assumptions).

Why: naming the destinations we already know about (criminology, education,
economics...) is confirmatory sampling — it cannot find unexpected niches,
which is precisely what a speciation study needs. This script instead asks the
corpus where the concept actually went, using two signals already present in
the RIS exports:

  VENUES      — 2,800+ journals; a journal is a proxy for a research community
  INDEX TERMS — APA controlled vocabulary; a proxy for topical territory

For each, it computes ENTRY TIME (first sustained appearance) and GROWTH, so
niches are ranked by when the concept arrived and how fast it took hold —
the empirical signature of niche colonisation.

Outputs (data/family/):
  niches_venues.csv       journal, n, first_year, entry_decade, recent_share
  niches_indexterms.csv   term, n, first_year, growth ratio (recent vs early)
  niches_emerging.csv     terms/venues whose share is rising fastest — the
                          candidate NEW niches, incl. ones nobody predicted

Usage (from code/):
    python3 15_discover_niches.py                    # all .ris under data/ebsco exports
    python3 15_discover_niches.py "data/ebsco exports/Psyc_SC-SR-WP"
"""

import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

IN_ROOT = Path("data/ebsco exports")
OUT_DIR = Path("data/family")
TAG_RE = re.compile(r"^([A-Z][A-Z0-9])  - (.*)$")
EARLY_END = 2004          # "early" period boundary for growth ratios
RECENT_START = 2015       # "recent" period start
MIN_N = 8                 # ignore very rare venues/terms


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


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else IN_ROOT
    files = sorted(root.rglob("*.ris"))
    if not files:
        raise SystemExit(f"No .ris under {root}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    seen = set()
    venue_years, term_years = defaultdict(list), defaultdict(list)
    n_recs = 0
    for f in files:
        for r in parse_ris(f):
            key = (r.get("DO", [""])[0] or r.get("AN", [""])[0]
                   or r.get("TI", [""])[0][:80]).lower()
            if key in seen:
                continue
            seen.add(key)
            n_recs += 1
            y = year_of(r)
            if y is None:
                continue
            j = (r.get("JO", [""])[0] or r.get("T2", [""])[0] or "").strip()
            if j:
                venue_years[j].append(y)
            for k in r.get("KW", []):
                k = k.strip()
                if k:
                    term_years[k].append(y)
    print(f"Parsed {len(files)} files; {n_recs} unique records; "
          f"{len(venue_years)} venues; {len(term_years)} index terms\n")

    # CORPUS BASELINE — essential. Raw growth ratios are meaningless because
    # the corpus itself grows: in the first run "Female" and "Male" showed
    # ~8.8x "growth" simply because there are more recent papers. A niche is
    # emerging only if its SHARE of the corpus rises, i.e. it grows faster
    # than the field does.
    all_years = [y for years in venue_years.values() for y in years]
    corpus_early = sum(1 for y in all_years if y <= EARLY_END) or 1
    corpus_recent = sum(1 for y in all_years if y >= RECENT_START) or 1
    print(f"Corpus baseline: {corpus_early} records <= {EARLY_END}, "
          f"{corpus_recent} >= {RECENT_START} "
          f"(raw growth {corpus_recent/corpus_early:.1f}x — all ratios below "
          f"are relative to this)\n")

    def profile(d, label, path):
        rows = []
        for name, years in d.items():
            n = len(years)
            if n < MIN_N:
                continue
            years_sorted = sorted(years)
            first = years_sorted[0]
            # entry = year of the 5th record (sustained presence, not a one-off)
            entry = years_sorted[min(4, n - 1)]
            early = sum(1 for y in years if y <= EARLY_END)
            recent = sum(1 for y in years if y >= RECENT_START)
            # share of corpus in each window, then relative change
            share_early = (early + 0.5) / corpus_early
            share_recent = (recent + 0.5) / corpus_recent
            rel = share_recent / share_early
            rows.append({
                "name": name, "n": n, "first_year": first,
                "entry_year": entry, "median_year": years_sorted[n // 2],
                "n_early": early, "n_recent": recent,
                "share_early_pct": round(100 * early / corpus_early, 3),
                "share_recent_pct": round(100 * recent / corpus_recent, 3),
                "relative_growth": round(rel, 2),
            })
        rows.sort(key=lambda r: -r["n"])
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"{label}: {len(rows)} above n>={MIN_N} -> {path}")
        return rows

    v = profile(venue_years, "Venues", OUT_DIR / "niches_venues.csv")
    t = profile(term_years, "Index terms", OUT_DIR / "niches_indexterms.csv")

    # Emerging niches = high growth AND late entry (colonisation signature)
    # emerging = grows faster than the corpus (relative_growth > 1) by a clear
    # margin, AND entered late — the colonisation signature
    emerging = [r for r in (v + t)
                if r["relative_growth"] >= 2.5 and r["entry_year"] >= 2005]
    emerging.sort(key=lambda r: -r["relative_growth"])
    with open(OUT_DIR / "niches_emerging.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(v[0].keys()))
        w.writeheader()
        w.writerows(emerging)

    print(f"\nEMERGING NICHES — grew >=2.5x FASTER than the corpus, entered "
          f">=2005 ({len(emerging)} found)")
    print(f"{'n':>6}  {'entry':>5}  {'rel.growth':>10}  name")
    for r in emerging[:28]:
        print(f"{r['n']:>6}  {r['entry_year']:>5}  "
              f"{r['relative_growth']:>10}  {r['name'][:60]}")

    print("\nDECLINING / ANCESTRAL (share falling fastest — territory the "
          "concept is LEAVING):")
    declining = sorted([r for r in (v + t) if r["n"] >= 25],
                       key=lambda r: r["relative_growth"])[:15]
    print(f"{'n':>6}  {'entry':>5}  {'rel.growth':>10}  name")
    for r in declining:
        print(f"{r['n']:>6}  {r['entry_year']:>5}  "
              f"{r['relative_growth']:>10}  {r['name'][:60]}")

    print("\nEARLIEST-ESTABLISHED (entry <=1990, the ancestral territory):")
    anc = sorted([r for r in v if r["entry_year"] <= 1990], key=lambda r: -r["n"])
    for r in anc[:15]:
        print(f"{r['n']:>6}  {r['entry_year']:>5}  {r['name'][:62]}")

    print("\nRead niches_emerging.csv for candidate NEW niches — including ones "
          "not predicted. Cross-check any surprise against a database census "
          "(see EBSCO_Retrieval_Plan) before committing sampling budget.")


if __name__ == "__main__":
    main()
