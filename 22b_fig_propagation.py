"""
22b_fig_propagation.py — Figure 5: correction propagation across venues.

Grouped bars per venue: % citing the 1998 coinage, the 2010 supportive
meta-analysis (milestone control), any RRR, and any crisis source. The
visual argument: coinage bars are tall everywhere (the boundary passes
citations); critique bars collapse outside the home field and sport.

Data: per-document flags from results/<slug>/citation_semantics.csv
(coinage, correction set) and the venue-level milestone/crisis rates parsed
from results/<slug>/reference_portfolio_report.txt (23b's output).
Output: results/<slug>/figures/fig5_propagation.svg/.png
Usage:  python3 22b_fig_propagation.py [--config configs/<concept>.yaml]
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

import pipeline_config as pc             # noqa: E402

CFG, _ARGV = pc.load()
CIT = CFG.get("citation", {}) or {}
RESULTS = CFG.results
SEM = RESULTS / "citation_semantics.csv"
REP = RESULTS / "reference_portfolio_report.txt"
OUT = RESULTS / "figures"
VC = pc.venue_categorizer(CFG)
SHOW = [VC.core] + [n for n in VC.names if n != VC.core]
LABELS = {n: n.replace("/", " /\n") for n in SHOW}
LABELS[VC.core] = "core/other\npsychology"
COINAGE = CIT.get("coinage")
CORRECTION = list(CIT.get("correction_set", []))
MILESTONE = CIT.get("milestone_control", "milestone")
FOCAL = CFG["concept"]


def main():
    rows = list(csv.DictReader(open(SEM, encoding="utf-8")))
    agg = defaultdict(lambda: [0, 0, 0])      # venue -> [n, coinage, correction]
    for r in rows:
        a = agg[r["venue_cat"]]
        a[0] += 1
        a[1] += int(r.get(f"cites_{COINAGE}", 0) or 0)
        a[2] += int(any(int(r.get(f"cites_{l}", 0) or 0) for l in CORRECTION))

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

    venues = [v for v in SHOW if agg[v][0] >= 10]
    x = np.arange(len(venues))
    series = [
        (f"cites coinage ({COINAGE})", "0.35",
         [100 * agg[v][1] / agg[v][0] for v in venues]),
        (f"cites {MILESTONE} (milestone control)", "#7fa8d9",
         [meta.get(v, np.nan) for v in venues]),
        ("cites a correction-set paper", "crimson",
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
    ax.set_ylabel(f"% of {CIT.get('year_min', '')}+ '{FOCAL}' papers")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    sub = (CFG.get("figures", {}) or {}).get("annotations", {}).get(
        "propagation", "citation rates by venue category")
    ax.set_title(f"Correction propagation for '{FOCAL}'\n{sub}",
                 loc="left", fontsize=10)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"fig5_propagation.{ext}", bbox_inches="tight")
    print(f"-> {OUT}/fig5_propagation.svg/.png")
    pc.write_manifest(CFG, __file__,
                      params={"venues": venues, "coinage": COINAGE,
                              "correction_set": CORRECTION,
                              "milestone": MILESTONE},
                      outputs=[OUT / "fig5_propagation.svg"])


if __name__ == "__main__":
    main()
