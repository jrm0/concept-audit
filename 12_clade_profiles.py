"""
12_clade_profiles.py (v2) — Clade profiles with the session-8 artifact fixes.

WHAT WENT WRONG IN v1 (and why this version exists):
Clusters tracked the target term's SURFACE FORM, not its meaning. well-being
split perfectly into "wellbeing" vs "well-being"; reaction time split into
singular / plural / "response time"; dissonance split into bare vs modified.
SciBERT's vector for the target span encodes that span's own tokenization
strongly enough that GMM recovers orthography before sense. Every quantitative
guard passed (ARI 1.0, silhouette 0.60, dBIC +2.8, gap 0.0) — only reading the
sentences caught it.

THREE FIXES IMPLEMENTED HERE:
 1. CANONICAL-SPAN SUBSTITUTION — the matched variant is replaced with one
    canonical string before embedding, so every mention contributes identical
    target tokens and only CONTEXT can vary. This removes the artifact at its
    source.
 2. JINGLE / JANGLE SEPARATION —
      jingle (polysemy): cluster WITHIN a single surface form. Reported as the
        primary, artifact-free clade analysis (per-variant, largest group).
      jangle (synonymy): compare centroids ACROSS surface forms (§5.3). Whether
        "reaction time" and "response time" are one concept is a jangle
        question, not a clade question.
 3. VARIANT-ALIGNMENT DIAGNOSTIC — clusters are crosstabbed against the matched
    variant and scored by NORMALIZED MUTUAL INFORMATION (not raw purity: if one
    variant is 95% of the data, every cluster is 95% that variant by base rate).
    NMI >= VARIANT_NMI_FLAG demotes the concept to ARTIFACT(variant).
    Verified: session-8 artifacts score NMI 1.0; base-rate dominance 0.002.

Reads:  data/concepts/sentences_{slug}.csv  (from 11)
        concepts_pilot.csv                  (needs a `canonical` column)
Writes: data/concepts/embeddings_{slug}.npz
        results/clade_profiles.csv / _detail.json
        results/cluster_samples_{slug}.txt
        results/variant_diagnostics.csv     cluster x variant crosstabs
        results/jangle_comparisons.csv      cross-variant centroid similarity

Usage (LOCAL — needs torch + SciBERT):
    python3 12_clade_profiles.py
    python3 12_clade_profiles.py ego_depletion working_memory
"""

import csv
import json
import random
import re
import sys
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# --- Configuration (embedding params match 02/03 exactly) ---
MODEL_NAME = "allenai/scibert_scivocab_uncased"
EXTRACTION_LAYER = "last_four_mean"
MAX_LENGTH = 256
MAX_SENTENCES = 1500
NUM_RANDOM_BASELINE = 500
SEED = 42
MAX_K = 6
N_GMM_INITS = 50
N_ARI_RUNS = 100
PCA_DIMS = 5
N_SAMPLES_PER_CLUSTER = 8
MIN_VARIANT_N = 200          # min sentences to run a within-variant analysis

# Classification thresholds (provisional; tune against qualitative reading)
ARI_ROBUST = 0.60
SIL_SEPARATED = 0.10
MEV_UNIMODAL = 0.30
DELTA_BIC_MIN = 1.0
VARIANT_PURITY_FLAG = 0.80   # descriptive only (see variant_diagnostic)
VARIANT_NMI_FLAG = 0.15      # NMI(cluster, variant) above this = artifact

DATA = Path("data/concepts")
RESULTS = Path("results")
CONFIG = "concepts_pilot.csv"


# ---------- config / text handling ----------

def load_config(path=CONFIG):
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            slug = row["concept_slug"].strip()
            terms = [t.strip() for t in (row.get("variants") or "").split("|")
                     if t.strip()]
            anchor = (row.get("retrieval_anchor") or "").strip()
            if anchor:
                terms.append(anchor)
            terms = sorted(set(terms), key=len, reverse=True)
            canonical = (row.get("canonical") or "").strip() or \
                slug.replace("_", " ")
            cfg[slug] = {
                "canonical": canonical,
                "terms": terms,
                "patterns": [(t.lower(),
                              re.compile(r"\b" + re.escape(t).replace(r"\ ", r"\s+")
                                         + r"\b", re.I))
                             for t in terms],
            }
    return cfg


def canonicalize(sentence, patterns, canonical):
    """Replace the matched variant with the canonical string.

    Returns (new_sentence, (start, end) of canonical span, matched_variant).
    This is FIX 1: after substitution every mention contributes identical
    target tokens, so clusters cannot be driven by the target's spelling.
    """
    for name, pat in patterns:
        m = pat.search(sentence)
        if m:
            new = sentence[:m.start()] + canonical + sentence[m.end():]
            return new, (m.start(), m.start() + len(canonical)), name
    return sentence, None, None


# ---------- embedding ----------

def embed(sentences_spans, tokenizer, model, torch):
    """sentences_spans: list of (text, span or None)."""
    out = []
    for text, span in sentences_spans:
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=MAX_LENGTH, return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            res = model(**enc, output_hidden_states=True)
        hs = res.hidden_states
        layer = (torch.stack(hs[-4:]).mean(dim=0)
                 if EXTRACTION_LAYER == "last_four_mean" else hs[-1])
        idx = []
        if span:
            for i, (a, b) in enumerate(offsets):
                if a == b:
                    continue
                if a < span[1] and b > span[0]:
                    idx.append(i)
        vec = layer[0, idx, :].mean(dim=0) if idx else layer[0, 0, :]
        out.append(vec.numpy())
    return np.array(out)


# ---------- metrics ----------

def self_similarity(embs, baseline):
    from scipy.spatial.distance import cosine
    n = len(embs)
    if n < 2:
        return {"raw": None, "corrected": None}
    if n > 300:
        idx = np.random.RandomState(SEED).choice(n, 300, replace=False)
        embs = embs[idx]
    sims = [1 - cosine(embs[i], embs[j])
            for i, j in combinations(range(len(embs)), 2)]
    raw = float(np.mean(sims))
    return {"raw": raw, "corrected": raw - baseline, "std": float(np.std(sims))}


def mev_of(embs):
    from sklearn.decomposition import PCA
    if len(embs) < 3:
        return {"mev": None}
    pca = PCA(n_components=min(10, len(embs) - 1)).fit(embs)
    return {"mev": float(pca.explained_variance_ratio_[0]),
            "top_3": [float(v) for v in pca.explained_variance_ratio_[:3]]}


def gmm_clustering(embs):
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.mixture import GaussianMixture
    if len(embs) < 10:
        return {"best_k": 1, "silhouette": None, "ari_stability_mean": None,
                "labels": None, "delta_bic_per_n": None}
    dims = min(PCA_DIMS, len(embs) - 1, embs.shape[1])
    reduced = PCA(n_components=dims).fit_transform(embs)
    bic_k1 = float(GaussianMixture(n_components=1, n_init=5,
                                   random_state=SEED).fit(reduced).bic(reduced))
    best_k, best_sil, by_k, best_labels = 1, -1, {}, None
    for k in range(2, min(MAX_K + 1, max(3, len(embs) // 5))):
        try:
            g = GaussianMixture(n_components=k, n_init=N_GMM_INITS,
                                random_state=SEED)
            labels = g.fit_predict(reduced)
            sil = silhouette_score(reduced, labels)
            by_k[k] = {"silhouette": float(sil), "bic": float(g.bic(reduced))}
            if sil > best_sil:
                best_k, best_sil, best_labels = k, sil, labels
        except Exception:
            continue
    aris = []
    if best_k > 1:
        ref = GaussianMixture(n_components=best_k, n_init=10,
                              random_state=0).fit_predict(reduced)
        for run in range(N_ARI_RUNS):
            lab = GaussianMixture(n_components=best_k, n_init=1,
                                  random_state=run + 1).fit_predict(reduced)
            aris.append(adjusted_rand_score(ref, lab))
    bic_best = by_k.get(best_k, {}).get("bic")
    dbic = (bic_k1 - bic_best) / len(embs) if bic_best is not None else None
    return {"best_k": best_k,
            "silhouette": float(best_sil) if best_sil > -1 else None,
            "ari_stability_mean": float(np.mean(aris)) if aris else None,
            "delta_bic_per_n": dbic,
            "results_by_k": by_k,
            "labels": best_labels.tolist() if best_labels is not None else None}


def variant_diagnostic(labels, variants):
    """FIX 3: crosstab clusters x matched variant; flag artifact clusters.

    Raw purity is NOT sufficient: if one variant is 95% of all sentences, every
    cluster will be ~95% that variant by base rate alone. So the flag uses
    NORMALIZED MUTUAL INFORMATION between cluster labels and variant labels,
    which is scale-free and base-rate aware. NMI ~1 = clusters encode the
    surface form (the session-8 artifact); NMI ~0 = variants are spread across
    clusters as expected. Purity is retained as descriptive context.
    """
    from sklearn.metrics import normalized_mutual_info_score
    if not labels:
        return {"max_purity": None, "nmi": None, "flagged": False, "table": {}}
    table = {}
    for c in sorted(set(labels)):
        vs = Counter(variants[i] for i, l in enumerate(labels) if l == c)
        n = sum(vs.values())
        top, top_n = vs.most_common(1)[0]
        table[str(c)] = {"n": n, "top_variant": top,
                         "purity": round(top_n / n, 3) if n else 0,
                         "breakdown": dict(vs)}
    max_purity = max((v["purity"] for v in table.values()), default=0)
    nmi = (float(normalized_mutual_info_score(variants, labels))
           if len(set(variants)) > 1 else 0.0)
    return {"max_purity": max_purity, "nmi": round(nmi, 3),
            "flagged": nmi >= VARIANT_NMI_FLAG, "table": table}


def classify(mev, gmm, diag):
    """POLYSEMOUS / BORDERLINE / TIGHT / VAGUE, with artifact demotion."""
    k = gmm.get("best_k") or 1
    ari = gmm.get("ari_stability_mean")
    sil = gmm.get("silhouette")
    dbic = gmm.get("delta_bic_per_n")
    m = mev.get("mev")
    structured = (k > 1 and ari is not None and ari >= ARI_ROBUST
                  and sil is not None and sil >= SIL_SEPARATED)
    bic_ok = dbic is not None and dbic >= DELTA_BIC_MIN

    if diag.get("flagged"):
        # FIX 3: clusters align with surface form -> not a sense distinction
        return "ARTIFACT(variant)"
    if structured and bic_ok:
        return "POLYSEMOUS"
    if structured and dbic is not None and dbic > 0:
        return "BORDERLINE"
    if m is not None and m >= MEV_UNIMODAL:
        return "TIGHT"
    return "VAGUE"


# ---------- main ----------

def main():
    import torch
    from transformers import AutoModel, AutoTokenizer
    from scipy.spatial.distance import cosine

    cfg = load_config()
    wanted = sys.argv[1:] or [p.stem.replace("sentences_", "")
                              for p in sorted(DATA.glob("sentences_*.csv"))]
    RESULTS.mkdir(exist_ok=True)
    print(f"Loading {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    prepared, pool = {}, []
    for slug in wanted:
        path = DATA / f"sentences_{slug}.csv"
        if not path.exists() or slug not in cfg:
            print(f"--- {slug}: missing sentences or config, skipping")
            continue
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        if len(rows) > MAX_SENTENCES:
            random.seed(SEED)
            rows = random.sample(rows, MAX_SENTENCES)
        c = cfg[slug]
        items, variants, kept = [], [], []
        for r in rows:
            new, span, var = canonicalize(r["sentence"], c["patterns"],
                                          c["canonical"])
            if span is None:
                continue
            items.append((new, span))
            variants.append(var)
            kept.append(r)
        print(f"\n--- {slug}: embedding {len(items)} sentences "
              f"(canonical='{c['canonical']}'; variants "
              f"{dict(Counter(variants))})")
        embs = embed(items, tokenizer, model, torch)
        np.savez_compressed(DATA / f"embeddings_{slug}.npz", embeddings=embs,
                            variant=np.array(variants, dtype=object),
                            source=np.array([r["source_type"] for r in kept]),
                            year=np.array([r["year"] for r in kept]),
                            journal=np.array([r["journal"] for r in kept],
                                             dtype=object))
        prepared[slug] = {"rows": kept, "embs": embs, "variants": variants}
        pool.extend(t for t, _ in items[:150])

    # Anisotropy baseline
    print(f"\n--- anisotropy baseline ({NUM_RANDOM_BASELINE} sentences)")
    random.seed(SEED)
    base_sents = random.sample(pool, min(NUM_RANDOM_BASELINE, len(pool)))
    base = embed([(s, None) for s in base_sents], tokenizer, model, torch)
    n = min(len(base), 300)
    baseline = float(np.mean([1 - cosine(base[i], base[j])
                              for i, j in combinations(range(n), 2)]))
    print(f"    baseline = {baseline:.4f}")

    summary, detail, diag_rows, jangle_rows = [], [], [], []
    for slug, P in prepared.items():
        rows, embs, variants = P["rows"], P["embs"], P["variants"]
        ss, mv = self_similarity(embs, baseline), mev_of(embs)
        gm = gmm_clustering(embs)
        diag = variant_diagnostic(gm.get("labels"), variants)
        label = classify(mv, gm, diag)

        # FIX 2a (jingle): clean within-variant analysis on the largest form
        vc = Counter(variants)
        within = None
        if vc:
            top_var, top_n = vc.most_common(1)[0]
            if top_n >= MIN_VARIANT_N and len(vc) > 1:
                idx = [i for i, v in enumerate(variants) if v == top_var]
                sub = embs[idx]
                w_mv, w_gm = mev_of(sub), gmm_clustering(sub)
                w_diag = {"flagged": False}
                within = {"variant": top_var, "n": top_n,
                          "mev": w_mv.get("mev"),
                          "best_k": w_gm.get("best_k"),
                          "silhouette": w_gm.get("silhouette"),
                          "ari": w_gm.get("ari_stability_mean"),
                          "delta_bic_per_n": w_gm.get("delta_bic_per_n"),
                          "classification": classify(w_mv, w_gm, w_diag)}

        # FIX 2b (jangle): cross-variant centroid comparison (§5.3)
        # NOTE: the null must be SPLIT-HALF centroid similarity of a group to
        # itself. Comparing centroid-to-centroid cosine against mean pairwise
        # within-group cosine is apples-to-oranges (centroids are averages, so
        # far less noisy) and produced nonsense in the first run.
        def split_half_centroid(X, reps=30):
            rng = np.random.RandomState(0)
            vals = []
            for _ in range(reps):
                p = rng.permutation(len(X))
                h = len(X) // 2
                if h < 5:
                    continue
                vals.append(1 - cosine(X[p[:h]].mean(axis=0),
                                       X[p[h:2 * h]].mean(axis=0)))
            return float(np.mean(vals)) if vals else None

        big_vars = [v for v, n_ in vc.items() if n_ >= 60]
        for a, b in combinations(sorted(big_vars), 2):
            ia = [i for i, v in enumerate(variants) if v == a]
            ib = [i for i, v in enumerate(variants) if v == b]
            cross = float(1 - cosine(embs[ia].mean(axis=0),
                                     embs[ib].mean(axis=0)))
            sa, sb = split_half_centroid(embs[ia]), split_half_centroid(embs[ib])
            ratio = (cross / ((sa + sb) / 2)) if (sa and sb) else None
            verdict = ("" if ratio is None else
                       "SAME(jangle)" if ratio > 0.995 else
                       "near" if ratio > 0.98 else "DISTINCT")
            jangle_rows.append([slug, a, b, len(ia), len(ib),
                                round(cross, 4),
                                round(sa, 4) if sa else "",
                                round(sb, 4) if sb else "",
                                round(ratio, 4) if ratio else "", verdict])

        print(f"\n{slug}: {label}")
        print(f"   n={len(rows)} self-sim={ss['corrected']:.4f} "
              f"MEV={mv['mev']:.3f} K={gm['best_k']} ARI={gm['ari_stability_mean']} "
              f"sil={gm['silhouette']} dBIC/n={gm['delta_bic_per_n']}")
        print(f"   variant NMI={diag['nmi']} purity_max={diag['max_purity']} "
              f"flagged={diag['flagged']}")
        if within:
            print(f"   [within '{within['variant']}' n={within['n']}] "
                  f"K={within['best_k']} dBIC/n={within['delta_bic_per_n']} "
                  f"-> {within['classification']}")

        for c_id, d in diag["table"].items():
            diag_rows.append([slug, c_id, d["n"], d["top_variant"], d["purity"],
                              json.dumps(d["breakdown"])])

        if gm.get("labels"):
            with open(RESULTS / f"cluster_samples_{slug}.txt", "w",
                      encoding="utf-8") as f:
                f.write(f"{slug} — {label} — K={gm['best_k']} "
                        f"(canonical-substituted; variant purity "
                        f"{diag['max_purity']})\n")
                f.write("Judge: are these real SENSES, or register/topic/"
                        "variant artifacts?\n\n")
                for c in range(gm["best_k"]):
                    idx = [i for i, l in enumerate(gm["labels"]) if l == c]
                    d = diag["table"].get(str(c), {})
                    f.write(f"===== cluster {c} (n={len(idx)}; top variant "
                            f"'{d.get('top_variant')}' {d.get('purity')}) =====\n")
                    for i in idx[:N_SAMPLES_PER_CLUSTER]:
                        r = rows[i]
                        f.write(f"  [{r['journal'][:26]} {r['year']} "
                                f"{r['source_type']}] {r['sentence'][:210]}\n")
                    f.write("\n")

        summary.append([slug, label, len(rows),
                        round(ss["corrected"], 4) if ss["corrected"] else "",
                        round(mv["mev"], 4) if mv["mev"] else "",
                        gm["best_k"],
                        round(gm["silhouette"], 4) if gm["silhouette"] else "",
                        round(gm["ari_stability_mean"], 3)
                        if gm["ari_stability_mean"] else "",
                        round(gm["delta_bic_per_n"], 3)
                        if gm["delta_bic_per_n"] is not None else "",
                        diag["max_purity"], diag["nmi"], diag["flagged"],
                        within["classification"] if within else "",
                        within["best_k"] if within else "",
                        within["n"] if within else ""])
        detail.append({"concept": slug, "classification": label,
                       "n": len(rows), "self_similarity": ss, "mev": mv,
                       "clustering": {k: v for k, v in gm.items()
                                      if k != "labels"},
                       "variant_diagnostic": diag, "within_variant": within,
                       "anisotropy_baseline": baseline})

    with open(RESULTS / "clade_profiles.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept", "classification", "n", "self_sim_corrected",
                    "mev", "best_k", "silhouette", "ari_stability",
                    "delta_bic_per_n", "variant_max_purity", "variant_nmi", "variant_flagged",
                    "within_variant_class", "within_variant_k",
                    "within_variant_n"])
        w.writerows(summary)
    with open(RESULTS / "variant_diagnostics.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept", "cluster", "n", "top_variant", "purity",
                    "breakdown"])
        w.writerows(diag_rows)
    with open(RESULTS / "jangle_comparisons.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept", "variant_a", "variant_b", "n_a", "n_b",
                    "cross_centroid_cos", "splithalf_a", "splithalf_b",
                    "ratio_cross_to_self", "verdict"])
        w.writerows(jangle_rows)
    with open(RESULTS / "clade_profiles_detail.json", "w") as f:
        json.dump(detail, f, indent=2, default=str)

    print(f"\n=== Done ===")
    print(f"  {RESULTS}/clade_profiles.csv        headline (now with variant flags)")
    print(f"  {RESULTS}/variant_diagnostics.csv   cluster x variant crosstabs")
    print(f"  {RESULTS}/jangle_comparisons.csv    cross-variant similarity (§5.3)")
    print(f"  {RESULTS}/cluster_samples_*.txt     READ THESE — still required")
    print("\nInterpretation: ARTIFACT(variant) = clusters track surface form. "
          "The 'within_variant' columns are the CLEAN polysemy test.")


if __name__ == "__main__":
    main()
