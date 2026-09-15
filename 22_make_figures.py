"""
22_make_figures.py — Figures for the ego-depletion lineage paper.

Design principle: the figures must show what the data DID, which is not
branching. 21 found the ego-depletion usage cloud unimodal in every period
(dip p 0.71-0.99) while k-means still partitioned it stably — so a
phylogenetic tree with discrete clades would be a picture of an artifact.
These four views show a distribution that STRETCHES and MIGRATES.

  fig1_ridgeline   Per-period density of documents along ONE fixed axis (the
                   1998->2020s drift axis). Ridgeline / "joyplot" form: each
                   period is a curve, stacked by time. A split would appear
                   as a curve growing two humps; a continuum appears as one
                   hump sliding and widening. This is the honest picture of
                   the central negative result, and it doubles as the
                   validity evidence.
  fig2_migration   Stacked area: share of the term's documents by journal
                   category over time, with the term's own control terms
                   plotted as reference lines. The niche-colonisation result.
  fig3_trajectory  All terms' period centroids projected into 2D (PCA fitted
                   on centroids only), each term a path with arrows and time
                   labels. Shows ego depletion born beside its ancestors,
                   moving away to 2015-19, then turning back.
  fig4_drift       Excess drift and cross-term distances over time with the
                   dated events (1998 coinage, 2010 meta, 2014 re-analysis,
                   2016 RRR, 2021 RRR) as vertical rules.

Outputs SVG (vector, editable, journal-ready) + PNG to results/<slug>/figures/.
Usage:  python3 22_make_figures.py [--config configs/<concept>.yaml]

Focal term, control terms, events, venue-category regexes, which terms/pairs
to draw, and the free-text subtitles all come from the config. The subtitles
are ANNOTATIONS (previously hard-typed result numbers); they are not computed
here — see figures.annotations in the YAML.
"""

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from scipy.spatial.distance import cosine  # noqa: E402
from sklearn.decomposition import PCA    # noqa: E402

import pipeline_config as pc             # noqa: E402
from pipeline_config import slug         # noqa: E402

CFG, _ARGV = pc.load()
FIG = CFG.get("figures", {}) or {}
ANN = FIG.get("annotations", {}) or {}

EMB = Path(CFG["emb_dir"])
DOCS = Path(CFG["docs_path"])
RESULTS = CFG.results
OUT = RESULTS / "figures"
FOCAL = pc.canon(CFG["concept"])
CONTROLS = set(CFG.controls)
EVENTS = pc.events(CFG, short=True) or pc.events(CFG)
VENUE_CAT = pc.venue_categorizer(CFG)
BASELINE_CAT = FIG.get("migration_baseline_category", VENUE_CAT.names[0])
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 130})


def mid_year(p):
    a, b = p.split("-")
    return (int(a) + int(b)) / 2


def load_shards(term):
    out = {}
    for f in sorted(EMB.glob(f"{slug(term)}__*.npz")):
        z = np.load(f, allow_pickle=True)
        out[f.stem.split("__")[1]] = z["vectors"].astype(np.float64)
    return out


def control_reference():
    ref = {}
    for c in CONTROLS:
        for f in EMB.glob(f"{slug(c)}__*.npz"):
            p = f.stem.split("__")[1]
            z = np.load(f, allow_pickle=True)
            s, n = ref.get(p, (0, 0))
            ref[p] = (s + z["vectors"].astype(np.float64).sum(0),
                      n + len(z["vectors"]))
    return {p: s / n for p, (s, n) in ref.items() if n}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight",
                    transparent=False)
    plt.close(fig)
    print(f"  -> {OUT / name}.svg / .png")


# ---------------------------------------------------------------- fig 1
def fig_ridgeline():
    sh = load_shards(FOCAL)
    ref = control_reference()
    periods = [p for p in sorted(sh) if len(sh[p]) >= 15 and p in ref]
    if len(periods) < 3:
        print("  (skipped ridgeline: too few periods)")
        return
    C = {p: (sh[p] - ref[p]) for p in periods}          # control-centered
    axis = C[periods[-1]].mean(0) - C[periods[0]].mean(0)
    axis /= np.linalg.norm(axis) + 1e-12                # earliest -> latest
    proj = {p: C[p] @ axis for p in periods}
    allv = np.concatenate(list(proj.values()))
    lo, hi = np.percentile(allv, [0.5, 99.5])
    grid = np.linspace(lo, hi, 400)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    step = 1.0
    for i, p in enumerate(periods):
        v = proj[p]
        bw = 1.06 * v.std() * len(v) ** (-1 / 5) + 1e-9   # Silverman
        d = np.exp(-0.5 * ((grid[:, None] - v[None, :]) / bw) ** 2).sum(1)
        d = d / d.max() * 0.92
        y0 = i * step
        ax.fill_between(grid, y0, y0 + d, color=plt.cm.viridis(i / len(periods)),
                        alpha=.75, lw=0)
        ax.plot(grid, y0 + d, color="0.15", lw=.9)
        ax.plot([np.mean(v)], [y0 + .04], "v", color="0.1", ms=5)
        ax.text(lo, y0 + .12, f"{p}  (n={len(v)})", fontsize=8, va="bottom")
    ax.set_yticks([])
    ax.set_xlabel(f"position along the '{FOCAL}' drift axis\n"
                  "(earliest usage $\\rightarrow$ latest usage; "
                  "control-term centered)")
    sub = ANN.get("ridgeline", "triangles mark period means")
    ax.set_title(f"Per-period usage of '{FOCAL}' along its drift axis\n{sub}",
                 fontsize=10, loc="left")
    ax.set_xlim(lo, hi)
    save(fig, "fig1_ridgeline")


# ---------------------------------------------------------------- fig 2
def fig_migration():
    docs = [json.loads(l) for l in open(DOCS, encoding="utf-8")]
    cat = VENUE_CAT
    names = VENUE_CAT.names
    per, tot = defaultdict(Counter), Counter()
    base_per, base_tot = Counter(), Counter()
    for d in docs:
        c = cat(d["journal"])
        base_per[(d["period"], c)] += 1
        base_tot[d["period"]] += 1
        if FOCAL in d["terms"]:
            per[d["period"]][c] += 1
            tot[d["period"]] += 1
    periods = [p for p in sorted(tot) if tot[p] >= 20]
    x = [mid_year(p) for p in periods]
    Y = np.array([[100 * per[p][c] / tot[p] for p in periods] for c in names])

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colors = plt.cm.tab20(np.linspace(0, .55, len(names)))
    ax.stackplot(x, Y, labels=names, colors=colors, alpha=.9, lw=.4,
                 edgecolor="white")
    base = [100 * base_per[(p, BASELINE_CAT)] / base_tot[p]
            for p in periods]
    ax.plot(x, base, "k--", lw=1.6,
            label=f"whole corpus: {BASELINE_CAT} share (baseline)")
    for yr, lab in EVENTS.items():
        if min(x) <= yr <= max(x):
            ax.axvline(yr, color="0.25", lw=.8, ls=":")
            ax.text(yr, 102, lab, rotation=90, fontsize=7, va="bottom",
                    ha="center", color="0.25")
    ax.set_ylim(0, 100)
    ax.set_xlim(min(x), max(x))
    ax.set_ylabel(f"% of '{FOCAL}' documents")
    sub = ANN.get("migration", "share of documents by venue category")
    ax.set_title(f"Where '{FOCAL}' is published, by venue category\n{sub}",
                 fontsize=10, loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), frameon=False,
              fontsize=8)
    save(fig, "fig2_migration")


# ---------------------------------------------------------------- fig 3
def fig_trajectory():
    ref = control_reference()
    terms, cents = [], {}
    for f in EMB.glob("*.npz"):
        t = f.stem.split("__")[0].replace("_", " ")
        if t in CONTROLS:
            continue
        terms.append(t)
    # Only well-sampled terms: a centroid built from ~20 documents is mostly
    # noise and, being an outlier, would dominate the PCA and squash the
    # differences we care about (this happened with "strength model", n=21-71).
    MIN_DOCS = int(FIG.get("trajectory_min_docs", 60))
    for t in sorted(set(terms)):
        sh = load_shards(t)
        c = {p: (sh[p] - ref[p]).mean(0) for p in sh
             if len(sh[p]) >= MIN_DOCS and p in ref}
        if len(c) >= 3:
            cents[t] = c
    if not cents:
        print("  (skipped trajectory)")
        return
    keys = [(t, p) for t in cents for p in cents[t]]
    M = np.array([cents[t][p] for t, p in keys])
    XY = PCA(n_components=2).fit_transform(M)
    pos = {k: XY[i] for i, k in enumerate(keys)}

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    wanted = [pc.canon(t) for t in FIG.get("trajectory_terms", [])] or sorted(cents)
    show = [t for t in wanted if t in cents]
    cols = plt.cm.tab10(np.linspace(0, .9, len(show)))
    for t, col in zip(show, cols):
        ps = sorted(cents[t])
        pts = np.array([pos[(t, p)] for p in ps])
        lw = 2.6 if t == FOCAL else 1.2
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=col, lw=lw, ms=3.5,
                alpha=.95 if t == FOCAL else .65, zorder=3 if t == FOCAL else 2)
        for j in range(len(pts) - 1):
            ax.annotate("", xy=pts[j + 1], xytext=pts[j],
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=lw, alpha=.9 if t == FOCAL else .5))
        ax.text(*pts[-1], f"  {t}", color=col, fontsize=8,
                fontweight="bold" if t == FOCAL else "normal", va="center")
        if t == FOCAL:
            for p, xy in zip(ps, pts):
                ax.text(xy[0], xy[1] - .004, p[:4], fontsize=6.5, ha="center",
                        color="0.25")
    ax.set_xlabel("PC1 of period centroids")
    ax.set_ylabel("PC2")
    y0, y1 = CFG["periods"]["year_min"], CFG["periods"]["year_max"]
    sub = ANN.get("trajectory", f"'{FOCAL}' in bold")
    ax.set_title(f"Trajectories of terms through usage space, {y0}–{y1}\n{sub}",
                 fontsize=10, loc="left")
    save(fig, "fig3_trajectory")


# ---------------------------------------------------------------- fig 4
def fig_drift():
    drift = RESULTS / "lineage_drift.csv"
    dist = RESULTS / "term_distances_centered.csv"      # preferred (18 --center)
    if not dist.exists():                                # fall back to the raw run
        dist = RESULTS / "term_distances.csv"
        print("  (panel B: no centered distances found, using raw term_distances.csv)")
    if not drift.exists():
        print("  (skipped drift: run 18 first)")
        return
    rows = list(csv.DictReader(open(drift, encoding="utf-8")))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
    ax = axes[0]
    drift_terms = [(pc.canon(t), str(c)) for t, c in
                   FIG.get("drift_terms", [[FOCAL, "navy"]])]
    for t, col in drift_terms:
        r = [(mid_year(x["period"]), float(x["excess_drift"]))
             for x in rows if x["term"] == t and x["excess_drift"]]
        if len(r) < 2:
            continue
        r.sort()
        ax.plot(*zip(*r), "-o", ms=3, color=col,
                lw=2 if t == FOCAL or col == "crimson" else 1, label=t)
    ax.axhline(0, color="k", lw=.8)
    ax.set_ylabel("excess drift over control terms")
    ax.set_title("A. Drift from origin (raw space)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    ax.text(1999, 0.001, "above 0 = drifting faster\nthan generic method words",
            fontsize=7, color="0.35")

    ax = axes[1]
    if dist.exists():
        dr = list(csv.DictReader(open(dist, encoding="utf-8")))
        pairs = [(pc.canon(a), pc.canon(b), str(c)) for a, b, c in
                 FIG.get("distance_pairs", [])]
        for a, b, col in pairs:
            aa, bb = sorted([a, b])
            s = [(mid_year(x["period"]), float(x["cosine_distance"]))
                 for x in dr if x["term_a"] == aa and x["term_b"] == bb]
            if len(s) < 2:
                continue
            s.sort()
            ax.plot(*zip(*s), "-o", ms=3, color=col, lw=1.8,
                    label=f"{a} — {b}")
        ax.set_ylabel("cosine distance (control-centered)")
        ax.set_title("B. Distance between terms", fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=7.5)
    for ax in axes:
        for yr, lab in EVENTS.items():
            lo, hi = ax.get_xlim()
            if lo <= yr <= hi:
                ax.axvline(yr, color="0.3", lw=.8, ls=":")
                ax.text(yr, ax.get_ylim()[1], lab, rotation=90, fontsize=6.5,
                        va="top", ha="right", color="0.3")
        ax.set_xlabel("year (period midpoint)")
    fig.suptitle(f"{FOCAL.capitalize()}: drift and divergence against dated "
                 f"events", fontsize=11, x=.09, ha="left")
    save(fig, "fig4_drift_divergence")


if __name__ == "__main__":
    print("Building figures ...")
    done, failed = [], []
    for fn in (fig_ridgeline, fig_migration, fig_trajectory, fig_drift):
        try:
            fn()
            done.append(fn.__name__)
        except Exception as e:            # keep going; report which failed
            failed.append(f"{fn.__name__}: {type(e).__name__}: {e}")
            print(f"  !! {failed[-1]}")
    pc.write_manifest(CFG, __file__,
                      params={"figures_built": done, "failed": failed,
                              "annotations": ANN,
                              "venue_categories": VENUE_CAT.names},
                      outputs=[OUT])
    print(f"\nAll figures -> {OUT}/ (SVG for editing, PNG for viewing)")
