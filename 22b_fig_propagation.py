"""
22b_fig_propagation.py — Figure 5: correction propagation across venues.

Grouped bars per venue: % citing the 1998 coinage, the 2010 supportive
meta-analysis (milestone control), any RRR, and any crisis source. The
visual argument: coinage bars are tall everywhere (the boundary passes
citations); critique bars collapse outside the home field and sport.

Data: per-document flags from results/citation_semantics.csv (coinage,
RRRs) and the venue-level meta/crisis rates parsed from
results/reference_portfolio_report.txt (23b's output).
Output: results/figures/fig5_propagation.svg/.png
Usage:  python3 22b_fig_propagation.py
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

SEM = Path("results/citation_semantics.csv")
REP = Path("results/reference_portfolio_report.txt")
OUT = Path("results/figures")
SHOW = ["core/other psych", "work/org", "health/med", "education", "sport"]
LABELS = {"core/other psych": "core/other\npsychology",
          "work/org": "work /\norganizational",
          "health/med": "health /\nmedical",
          "education": "education", "sport": "sport"}


def main():
    rows = list(csv.DictReader(open(SEM, encoding="utf-8")))
    agg = defaultdict(lambda: [0, 0, 0])      # venue -> [n, cites98, anyRRR]
    for r in rows:
        a = agg[r["venue_cat"]]
        a[0] += 1
        a[1] += int(r["cites_1998"])
        a[2] += int(int(r["cites_rrr2016"]) or int(r["cites_rrr2021"]))

    # meta + crisisANY per venue from 23b's report table
    meta, crisis = {}, {}
    if REP.exists():
        for line in REP.read_text(encoding="utf-8").splitlines():
            m = re.match(r"(\S[\w /]*?)\s{2,}(\d+)\s+(\d+)%\s+(\d+)%\s+"
                         r"(\d+)%\s+(\d+)%", line)
            if m:
                v = m.group(1).strip()
                meta[v] = int(m.group(4))
                crisis[v] = int(m.group(6))

    venues = [v for v in SHOW if agg[v][0] >= 10 or v == "sport"]
    x = np.arange(len(venues))
    series = [
        ("cites 1998 coinage", "0.35",
         [100 * agg[v][1] / agg[v][0] for v in venues]),
        ("cites 2010 meta (milestone control)", "#7fa8d9",
         [meta.get(v, np.nan) for v in venues]),
        ("cites an RRR", "crimson",
         [100 * agg[v][2] / agg[v][0] for v in venues]),
        ("cites ANY crisis source", "#f4a3b0",
         [crisis.get(v, np.nan) for v in venues]),
    ]
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, (lab, col, vals) in enumerate(series):
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col,
                      edgecolor="white", lw=.5)
        for b, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                        ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{LABELS[v]}\n(n={agg[v][0]})" for v in venues],
                       fontsize=8.5)
    ax.set_ylabel("% of 2017+ ego-depletion papers")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("The concept travels; the correction doesn't\n"
                 "Applied venues cite the coinage like the home field — "
                 "and the critiques at a fraction of its rate",
                 loc="left", fontsize=10)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"fig5_propagation.{ext}", bbox_inches="tight")
    print(f"-> {OUT}/fig5_propagation.svg/.png")


if __name__ == "__main__":
    main()
