"""
27_citation_communities.py — The CITATION LAYER read against the SEMANTIC layer:
citation communities as structural niches, and a pre-registered test of the
"speciation by coinage" reading of trauma (Session 32).

Session 38 (2026-09-28). Input: data/<slug>/citations/openalex_works.jsonl (from 26).
Edges = intra-corpus references (complete from 26's outbound pass).

PRE-REGISTERED PREDICTIONS (Jonathan, 2026-09-28; stated before this ran)
  P1  trauma-informed, secondary traumatic stress and vicarious trauma each
      concentrate in citation communities distinct from bare trauma's, with
      same-term citation ("insulation") well above the label-shuffle null.
  P2  trauma-informed's internal semantic split (child-welfare/education
      *practice* vs nursing *care*, Session 32) corresponds to a citation
      sub-community split — operationalised here as alignment between Leiden
      sub-communities of the trauma-informed subgraph and venue categories.
  P3  PTSD does NOT separate from bare trauma by citation: convergent by
      semantics, continuous by citation.
  P4  (what would embarrass the reading) compounds citationally continuous
      with the parent -> lexical productivity without lineage separation.

MEASURES
  community distribution   share of a term's documents per Leiden community;
                           top-community share; normalised entropy
  distinctness             JS divergence between a term's community
                           distribution and bare trauma's
  insulation               among intra-corpus references from documents
                           carrying term T, the share pointing to documents
                           carrying T; ratio and z against a null that
                           permutes term labels among nodes WITHIN period
  flow matrix              observed / expected citations term_i -> term_j
                           under the same null

Nothing here is interpreted without reading: the report prints each
community's top venues, top terms and period profile.

Addendum A (2026-10-05): the descent facet can be read for PSEUDO-LABELS as
well as terms. Config `citation.groups` defines them and they enter the
flow / insulation machinery exactly like terms, under the name "@label":
    groups:
      ancestor: {terms: [traumatic neurosis, war neurosis, ...]}   any listed term
      analytic: {venue: "psychoanal|analytic"}                     journal-name regex
      abuse:    {flag: A_abuse}                                    a 31 vocabulary flag
                                                                   (results/<slug>/ground_truth_doc_flags.jsonl)
      cat:      {venue_categories: true}                           one label per config venue category
--window Y1-Y2 restricts the flow / insulation counts (observed and null) to
references whose CITING document falls in the window; --flows-only skips the
Leiden partition and the subgraph (P2/P2b) so a windowed run is quick. Output
names carry _y<year_min> and _w<window> when non-default, so the Session 38/39
results are never overwritten.

USAGE
    python3 27_citation_communities.py --config configs/ptsd_trauma.yaml
    python3 27_citation_communities.py --config configs/ptsd_trauma.yaml --null venue
    python3 27_citation_communities.py --config configs/ptsd_trauma.yaml --year-min 1980 --window 1980-1994 \
        --flows-only --focus "trauma,PTSD,@ancestor,@analytic"
    python3 27_citation_communities.py --config configs/ego_depletion.yaml --parent "self control" \
        --focus "ego depletion,self control,self regulation,willpower,strength model,executive function,delay of gratification" \
        --compounds "ego depletion,strength model" --subgraph-term "ego depletion" --merge ""
"""

import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import pipeline_config as pc
from revision_core import distinct_edge_shares

CFG, _ARGV = pc.load()


def _opt(name, default=None):
    for i, a in enumerate(_ARGV):
        if a == name and i + 1 < len(_ARGV):
            return _ARGV[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


YEAR_MIN = int(_opt("--year-min", 1995))
PERMS = int(_opt("--perms", 200))
RESOLUTION = float(_opt("--resolution", 1.0))
SEED = int(_opt("--seed", CFG.get("analysis", {}).get("seed", 42)))
MIN_COMM = int(_opt("--min-community", 200))
TOP_N = 6
NULL = _opt("--null", "period")            # period | venue  (venue = period x venue-category strata)
N_ITER = int(_opt("--iterations", -1))     # Leiden iterations; -1 = until stable (slow on 1M edges)
FLOWS_ONLY = "--flows-only" in _ARGV       # skip Leiden + subgraph; flows/insulation only (Addendum A)
_w = _opt("--window")                      # "1980-1994": citing-document years for flows/insulation
WINDOW = tuple(int(x) for x in _w.split("-")) if _w else None
if WINDOW and WINDOW[0] < YEAR_MIN:
    raise SystemExit(f"--window starts before --year-min ({WINDOW[0]} < {YEAR_MIN}); set --year-min {WINDOW[0]}")

WORKS = Path(CFG["docs_path"]).parent / "citations" / "openalex_works.jsonl"
OUT = CFG.results

# Term lists are CASE-SPECIFIC. Defaults = the trauma case (Session 38); any other
# concept passes them on the command line (or in config `citation:`):
#   --parent "self control"   the reference term for JS divergence (the parent lineage)
#   --focus  "a,b,c"          terms to report
#   --compounds "a,b"         the candidate daughter lineages (P1)
#   --merge  "ptsd=PTSD;posttraumatic stress disorder=PTSD"   label merges
#   --subgraph-term "trauma informed"   the term whose internal split P2 tests
_CIT = CFG.get("citation", {}) or {}
def _list(opt, key, default):
    v = _opt(opt) or _CIT.get(key)
    if v is None:
        return default
    return [x.strip() for x in v.split(",")] if isinstance(v, str) else list(v)
_m = _opt("--merge") or _CIT.get("merge")
if _m is None:
    MERGE = {"ptsd": "PTSD", "posttraumatic stress disorder": "PTSD",
             "complex ptsd": "complex PTSD", "complex posttraumatic stress disorder": "complex PTSD"}
elif isinstance(_m, str):
    MERGE = dict(kv.split("=", 1) for kv in _m.split(";") if "=" in kv)
else:
    MERGE = dict(_m)
PARENT = _opt("--parent") or _CIT.get("parent") or "trauma"
FOCUS = _list("--focus", "focus_terms", [
    "trauma", "PTSD", "trauma informed", "secondary traumatic stress", "vicarious trauma",
    "complex PTSD", "moral injury", "collective trauma", "intergenerational trauma",
    "posttraumatic stress", "traumatic stress", "psychological trauma", "bereavement",
    "acute stress disorder", "adjustment disorder"])
COMPOUNDS = _list("--compounds", "compounds", ["trauma informed", "secondary traumatic stress", "vicarious trauma"])
SUBGRAPH_TERM = _opt("--subgraph-term") or _CIT.get("subgraph_term") or (COMPOUNDS[0] if COMPOUNDS else None)
# the "->2nd" column: the sibling / second parent each term's citations are
# compared against (trauma case: PTSD; ego depletion: self regulation). Default =
# first focus term that is neither the parent nor a compound.
SECOND = _opt("--second") or _CIT.get("second") or next(
    (t for t in FOCUS if t != PARENT and t not in COMPOUNDS), None)
GROUPS = dict(_CIT.get("groups") or {})


def load_doc_flags():
    """{doc_key: set(flags)} from 31's ground_truth_doc_flags.jsonl (run 31 first)."""
    fp = OUT / "ground_truth_doc_flags.jsonl"
    if not fp.exists() and CFG.get("revision_baseline_results"):
        fp = Path(CFG["revision_baseline_results"]) / fp.name
    if not fp.exists():
        return None
    out = {}
    with open(fp, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["doc_key"]] = set(d.get("flags") or [])
    return out


def apply_groups(nodes):
    """Append '@label' pseudo-terms to each node's term list per config citation.groups."""
    if not GROUPS:
        return []
    import re
    cat = pc.venue_categorizer(CFG)
    flags = None
    labels = []
    for label, spec in GROUPS.items():
        spec = spec or {}
        if "terms" in spec:
            want = {MERGE.get(t, t) for t in spec["terms"]}
            sel = lambda n, want=want: any(t in want for t in n["terms"])
            labels.append(("@" + label, sel))
        elif "venue" in spec:
            rx = re.compile(spec["venue"], re.I)
            sel = lambda n, rx=rx: rx.search(n["venue"] or "") is not None
            labels.append(("@" + label, sel))
        elif "flag" in spec:
            if flags is None:
                flags = load_doc_flags()
                if flags is None:
                    raise SystemExit("citation.groups uses a 31 flag but results/<slug>/ground_truth_doc_flags.jsonl "
                                     "is missing: run 31_ground_truth_cooccurrence.py first")
            fl = spec["flag"]
            sel = lambda n, fl=fl, flags=flags: fl in flags.get(n["key"], ())
            labels.append(("@" + label, sel))
        elif spec.get("venue_categories"):
            for name in cat.names:
                sel = lambda n, name=name, cat=cat: cat(n["venue"]) == name
                labels.append((f"@{label}:{name}", sel))
        else:
            raise SystemExit(f"citation.groups.{label}: need one of terms / venue / flag / venue_categories")
    counts = Counter()
    for n in nodes:
        for lab, sel in labels:
            if sel(n):
                n["terms"].append(lab)
                counts[lab] += 1
    for lab, _ in labels:
        print(f"[27] pseudo-label {lab:<28} {counts[lab]:>8,} nodes")
    return [lab for lab, _ in labels]


# ---------------------------------------------------------------- load graph

def load():
    nodes, id2idx = [], {}
    with open(WORKS, encoding="utf-8") as f:
        work_ids = Counter(json.loads(line)["openalex_id"] for line in f)
    # Repeated resolutions may be duplicate corpus records or erroneous matches.
    # Exclude all colliding resolutions until adjudicated; never silently duplicate nodes.
    with open(WORKS, encoding="utf-8") as f:
        for line in f:
            w = json.loads(line)
            if work_ids[w["openalex_id"]] != 1:
                continue
            y = w.get("corpus_year")
            if y is None or y < YEAR_MIN or y > CFG.get("revision_analysis_year_max", 9999):
                continue
            terms = sorted({MERGE.get(t, t) for t in (w.get("terms") or [])})
            id2idx[w["openalex_id"]] = len(nodes)
            nodes.append({"key": w["doc_key"], "oa": w["openalex_id"], "year": int(y),
                          "period": w.get("period"), "terms": terms,
                          "venue": w.get("openalex_venue") or w.get("corpus_journal") or "",
                          "refs": w.get("referenced_works") or []})
    src, dst = [], []
    for i, n in enumerate(nodes):
        for r in sorted(set(n["refs"])):
            j = id2idx.get(r)
            if j is not None and j != i:
                src.append(i); dst.append(j)
        n.pop("refs")
    return nodes, np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)


def term_masks(nodes):
    terms = sorted({t for n in nodes for t in n["terms"]})
    bit = {t: 1 << k for k, t in enumerate(terms)}
    mask = np.zeros(len(nodes), dtype=np.int64)
    for i, n in enumerate(nodes):
        for t in n["terms"]:
            mask[i] |= bit[t]
    return terms, bit, mask


# ---------------------------------------------------------------- communities

def leiden(n_nodes, src, dst):
    """Cached under results/<slug>/stability/ (shared with 27b) so re-runs with a
    different null or term list do not recompute a multi-minute partition."""
    cache = OUT / "stability" / f"leiden_s{SEED}_r{RESOLUTION}_i{N_ITER}_y{YEAR_MIN}.npz"
    if cache.exists():
        z = np.load(cache)
        # the graph may have grown since the partition was cached (a later 26
        # phase added resolved documents): a partition is only valid for the
        # graph it was computed on. Found 2026-10-01 on ego depletion, where a
        # 200-document title sample added 58 nodes after the first 27 run.
        same = (len(z["memb"]) == n_nodes and ("n_edges" not in z.files or int(z["n_edges"]) == len(src)))
        if same:
            return z["memb"], float(z["Q"]), f"leiden/RBConfiguration (cached, iterations={N_ITER})"
        stale = cache.with_name(cache.stem + f"_stale_n{len(z['memb'])}.npz")
        cache.rename(stale)
        print(f"[27] cached partition was for {len(z['memb']):,} nodes, graph now has {n_nodes:,}: "
              f"recomputing (old cache kept as {stale.name})")
    import igraph as ig
    import leidenalg as la
    g = ig.Graph(n=n_nodes, edges=list(zip(src.tolist(), dst.tolist())), directed=False)
    g.simplify()
    part = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=RESOLUTION,
                             seed=SEED, n_iterations=N_ITER)
    memb = np.array(part.membership)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, memb=memb, Q=part.modularity, n_edges=len(src))
    return memb, part.modularity, f"leiden/RBConfiguration (iterations={N_ITER})"


def js_divergence(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / p.sum() if p.sum() else p; q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)
    def kl(a, b):
        nz = a > 0
        return float(np.sum(a[nz] * np.log2(a[nz] / b[nz])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def norm_entropy(counts):
    c = np.asarray(counts, float); c = c[c > 0]
    if len(c) <= 1:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(c)))


# ---------------------------------------------------------------- insulation

def shuffled_masks(mask, strata, rng):
    """Permute term-label vectors among nodes within each stratum.
    --null period : strata = period                      (preserves time + prevalence)
    --null venue  : strata = period x venue category     (also preserves where each
                    period's labels sit by broad venue category; exact-journal
                    effects require the separate journal null)"""
    out = mask.copy()
    for p in np.unique(strata):
        idx = np.where(strata == p)[0]
        out[idx] = mask[rng.permutation(idx)]
    return out


def flow_counts(mask, src, dst, bits):
    """observed citations term_i -> term_j (a doc may carry several terms)."""
    ma, mb = mask[src], mask[dst]
    M = np.zeros((len(bits), len(bits)), dtype=np.int64)
    sel_a = [(ma & b) != 0 for b in bits]
    sel_b = [(mb & b) != 0 for b in bits]
    for i in range(len(bits)):
        if not sel_a[i].any():
            continue
        for j in range(len(bits)):
            M[i, j] = int(np.count_nonzero(sel_a[i] & sel_b[j]))
    return M


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    nodes, src, dst = load()
    N, E = len(nodes), len(src)
    print(f"[27] {CFG['concept']}  nodes={N:,} (year >= {YEAR_MIN})  intra-corpus edges={E:,}")
    apply_groups(nodes)
    terms, bit, mask = term_masks(nodes)
    # flows / insulation can be restricted to references whose citing document is in --window
    if WINDOW:
        yrs = np.array([n["year"] for n in nodes])
        wsel = (yrs[src] >= WINDOW[0]) & (yrs[src] <= WINDOW[1])
        fsrc, fdst = src[wsel], dst[wsel]
        print(f"[27] window {WINDOW[0]}-{WINDOW[1]}: {len(fsrc):,} of {E:,} references have a citing document in the window")
    else:
        fsrc, fdst = src, dst
    periods = np.array([n["period"] or "?" for n in nodes])
    if NULL not in {"period", "venue", "journal", "degree"}:
        raise ValueError("null must be period, venue, journal, or degree")
    if NULL == "venue":
        cat = pc.venue_categorizer(CFG)
        strata = np.array([f"{n['period']}|{cat(n['venue'])}" for n in nodes])
    elif NULL == "journal":
        strata = np.array([f"{n['period']}|{n['venue'].strip().lower()}" for n in nodes])
    elif NULL == "degree":
        cat = pc.venue_categorizer(CFG)
        indegree = np.floor(np.log2(1 + np.bincount(dst, minlength=N))).astype(int)
        outdegree = np.floor(np.log2(1 + np.bincount(src, minlength=N))).astype(int)
        strata = np.array([f"{n['period']}|{cat(n['venue'])}|{indegree[i]}|{outdegree[i]}" for i,n in enumerate(nodes)])
    else:
        strata = periods
    stratum_groups = defaultdict(list)
    for i, label in enumerate(strata):
        stratum_groups[str(label)].append(i)
    stratum_indices = [np.array(v) for v in stratum_groups.values()]
    print(f"[27] null = label shuffle within {'period x venue-category' if NULL == 'venue' else 'period'} "
          f"(mode={NULL}; {len(np.unique(strata))} strata)")
    focus = [t for t in FOCUS if t in bit]
    if PARENT not in bit:
        raise SystemExit(f"parent term '{PARENT}' not in corpus terms: {sorted(bit)[:30]}")
    fbits = [bit[t] for t in focus]

    # --- communities
    if FLOWS_ONLY:
        memb, Q, algo = np.zeros(N, dtype=np.int64), float("nan"), "skipped (--flows-only)"
    else:
        memb, Q, algo = leiden(N, src, dst)
    sizes = Counter(memb.tolist())
    big = [c for c, s in sizes.most_common() if s >= MIN_COMM]
    print(f"[27] {algo} resolution={RESOLUTION}: {len(sizes)} communities, modularity={Q:.3f}; "
          f"{len(big)} with >= {MIN_COMM} nodes covering "
          f"{100 * sum(sizes[c] for c in big) / N:.1f}% of nodes")

    # readable community profiles
    prof = {}
    for c in big:
        idx = np.where(memb == c)[0]
        ven = Counter(nodes[i]["venue"] for i in idx if nodes[i]["venue"])
        tc = Counter(t for i in idx for t in nodes[i]["terms"])
        pc_ = Counter(periods[i] for i in idx)
        prof[int(c)] = {"n": int(len(idx)),
                        "top_venues": ven.most_common(TOP_N),
                        "top_terms": [(t, n, round(n / len(idx), 3)) for t, n in tc.most_common(TOP_N)],
                        "periods": sorted(pc_.items())}

    # --- term x community distributions, distinctness vs bare trauma
    comm_index = {c: k for k, c in enumerate(big)}
    K = len(big)
    dist = {}
    for t, b in zip(focus, fbits):
        idx = np.where((mask & b) != 0)[0]
        v = np.zeros(K)
        for i in idx:
            k = comm_index.get(memb[i])
            if k is not None:
                v[k] += 1
        dist[t] = {"n_docs": int(len(idx)), "vec": v}
    base = dist[PARENT]["vec"] if PARENT in dist else None
    term_rows = []
    for t in focus:
        v = dist[t]["vec"]
        if v.sum() == 0:
            continue
        top_share = float(v.max() / dist[t]["n_docs"])
        top_c = int(big[int(v.argmax())])
        # over-representation: communities where term share > 2x its global share
        glob = dist[t]["n_docs"] / N
        over = [int(big[k]) for k in range(K) if sizes[big[k]] and (v[k] / sizes[big[k]]) > 2 * glob]
        in_over = float(sum(v[k] for k in range(K) if int(big[k]) in over) / v.sum())
        term_rows.append({
            "term": t, "n_docs": dist[t]["n_docs"], "top_community": top_c,
            "top_community_share": round(top_share, 3),
            "top_community_share_conditional_large": round(float(v.max() / v.sum()), 3),
            "share_in_large_communities": round(float(v.sum() / dist[t]["n_docs"]), 3),
            "norm_entropy": round(norm_entropy(v), 3),
            "distribution_scope": "entropy, overrepresentation and JS conditional on communities >= min_community",
            "share_in_overrepresented_comms": round(in_over, 3),
            "n_overrepresented_comms": len(over),
            "js_vs_parent": round(js_divergence(v, base), 4) if base is not None and t != PARENT else None,
        })

    # --- insulation + flow, with time-stratified label-shuffle null
    rng = np.random.default_rng(SEED)
    obs, edges_from, obs_shares = distinct_edge_shares(mask, fsrc, fdst, fbits)
    print(f"[27] null: {PERMS} {NULL} label permutations over {len(fsrc):,} edges ...")
    null = np.zeros((PERMS, len(fbits), len(fbits)))
    null_shares = np.full_like(null, np.nan)
    for p in range(PERMS):
        shuffled = mask.copy()
        for idx in stratum_indices:
            shuffled[idx] = mask[rng.permutation(idx)]
        null[p], _, null_shares[p] = distinct_edge_shares(shuffled, fsrc, fdst, fbits)
        if (p + 1) % 50 == 0:
            print(f"    {p + 1}/{PERMS}")
    mu, sd = null.mean(0), null.std(0)
    ratio = np.divide(obs, mu, out=np.full_like(mu, np.nan), where=mu > 0)
    z = np.divide(obs-mu, sd, out=np.full_like(mu, np.nan), where=sd > 0)
    share_mu, share_sd = np.nanmean(null_shares, axis=0), np.nanstd(null_shares, axis=0)
    share_ratio = np.divide(obs_shares, share_mu, out=np.full_like(mu, np.nan), where=share_mu > 0)
    share_z = np.divide(obs_shares-share_mu, share_sd, out=np.full_like(mu, np.nan), where=share_sd > 0)
    ins_rows = []
    for i, t in enumerate(focus):
        if edges_from[i] == 0:
            continue
        ins_rows.append({
            "term": t, "refs_out_to_focus": int(obs[i].sum()), "edges_from_term_docs": int(edges_from[i]),
            "same_term_edges": int(obs[i,i]),
            "same_term_share_obs": float(obs_shares[i,i]),
            "same_term_share_null": float(share_mu[i,i]),
            "insulation_ratio": float(share_ratio[i,i]),
            "insulation_z": float(share_z[i,i]),
            "null_share_sd": float(share_sd[i,i]),
            "null_valid_draws": int(np.isfinite(null_shares[:,i,i]).sum()),
            "count_enrichment_ratio": float(ratio[i,i]),
            "to_parent_ratio": float(ratio[i,focus.index(PARENT)]) if PARENT in focus else None,
            "to_second_ratio": float(ratio[i,focus.index(SECOND)]) if SECOND in focus else None,
            "second": SECOND if SECOND in focus else None,
        })

    # --- P2: trauma-informed sub-communities vs venue category
    p2 = None
    if SUBGRAPH_TERM and SUBGRAPH_TERM in bit and not FLOWS_ONLY:
        import igraph as ig
        import leidenalg as la
        ti = np.where((mask & bit[SUBGRAPH_TERM]) != 0)[0]
        tiset = set(ti.tolist()); remap = {v: k for k, v in enumerate(ti.tolist())}
        e = [(remap[a], remap[b]) for a, b in zip(src.tolist(), dst.tolist()) if a in tiset and b in tiset]
        g = ig.Graph(n=len(ti), edges=e, directed=False); g.simplify()
        part = la.find_partition(g, la.RBConfigurationVertexPartition, resolution_parameter=RESOLUTION,
                                 seed=SEED, n_iterations=-1)
        sub = np.array(part.membership)
        cat = pc.venue_categorizer(CFG)
        subsz = Counter(sub.tolist())
        keep = [c for c, s in subsz.most_common() if s >= 40][:6]
        table = {}
        for c in keep:
            idx = ti[sub == c]
            cc = Counter(cat(nodes[i]["venue"]) for i in idx)
            tv = Counter(nodes[i]["venue"] for i in idx if nodes[i]["venue"]).most_common(4)
            table[int(c)] = {"n": int(len(idx)), "venue_categories": cc.most_common(5), "top_venues": tv}
        # Cramer's V between sub-community and venue category
        cats = sorted({cat(nodes[i]["venue"]) for i in ti})
        ct = np.zeros((len(keep), len(cats)))
        for a, c in enumerate(keep):
            for i in ti[sub == c]:
                ct[a, cats.index(cat(nodes[i]["venue"]))] += 1
        n = ct.sum(); rs = ct.sum(1, keepdims=True); cs = ct.sum(0, keepdims=True)
        exp = rs @ cs / max(n, 1)
        chi2 = float(((ct - exp) ** 2 / np.maximum(exp, 1e-9)).sum())
        V = math.sqrt(chi2 / (n * max(min(ct.shape) - 1, 1))) if n else 0.0
        p2 = {"subgraph_term": SUBGRAPH_TERM, "n_docs": int(len(ti)), "intra_edges": len(e), "n_subcommunities": len(subsz),
              "modularity": round(part.modularity, 3), "cramers_v_subcomm_x_venue_category": round(V, 3),
              "subcommunities": table}

    # --- P2b: anchor citation by sub-community and venue category
    #     (does the correction travel with the concept?). Anchors come from the
    #     config's citation.anchors / anchors_extra as [label, "doi:..." or
    #     title, year]; titles are matched against the corpus (lineage_docs)
    #     and mapped to graph nodes through the document key, so nothing is
    #     asserted about a paper that is not in the corpus and resolved.
    anchors_out = None
    anchor_specs = list(_CIT.get("anchors") or []) + list(_CIT.get("anchors_extra") or [])
    if p2 is not None and anchor_specs:
        import difflib
        import hashlib
        import re
        key2idx = {n["key"]: i for i, n in enumerate(nodes)}
        titles = []   # (norm title, year, key)
        with open(CFG["docs_path"], encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                doi = (d.get("doi") or "").strip().lower()
                doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi)
                if doi:
                    key = "doi:" + doi
                else:
                    basis = pc.normalize_text(d.get("title") or "") + "|" + str(d.get("year") or "")
                    key = "t:" + hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
                titles.append((pc.normalize_text(d.get("title") or "").lower(), d.get("year"), key, d.get("title")))
        resolved = {}
        for spec in anchor_specs:
            label, q, yr = (list(spec) + [None, None])[:3]
            node = None; how = None
            if isinstance(q, str) and q.lower().startswith("doi:"):
                node = key2idx.get("doi:" + q[4:].strip().lower()); how = "doi"
            elif isinstance(q, str):
                want = pc.normalize_text(q).lower()
                best, bs = None, 0.0
                for nt, y, key, raw in titles:
                    if yr and y and abs(int(y) - int(yr)) > 1:
                        continue
                    sc = difflib.SequenceMatcher(None, want, nt).ratio()
                    if want and want in nt:          # query is the title's first clause (subtitle dropped)
                        sc = max(sc, 0.95)
                    if sc > bs:
                        best, bs = (key, raw), sc
                if best and bs >= 0.85:
                    node = key2idx.get(best[0]); how = f"title ({bs:.2f}): {best[1][:90]}"
            if node is None:
                how = None if how is None or how == "doi" else how + "  [matched a corpus title but that document is not in the graph]"
            resolved[label] = {"node": node, "how": how, "year": yr or (nodes[node]["year"] if node is not None else None)}
        # who cites each anchor: in-graph citers
        citers = {label: set() for label in resolved}
        anchor_idx = {a["node"]: label for label, a in resolved.items() if a["node"] is not None}
        for a, b in zip(src.tolist(), dst.tolist()):
            if b in anchor_idx:
                citers[anchor_idx[b]].add(a)
        cat = pc.venue_categorizer(CFG)
        win = int(_CIT.get("year_min") or 0)   # propagation window (post-correction), e.g. 2017
        ti_list = ti.tolist()
        def rates(idx):
            window_idx = [i for i in idx if (not win or nodes[i]["year"] >= win)]
            details = {}
            lag = int(_CIT.get("anchor_lag_years", 1))
            for label,a in resolved.items():
                if a["node"] is None or a["year"] is None:
                    continue
                eligible = [i for i in window_idx if nodes[i]["year"] >= a["year"]+lag]
                k = sum(i in citers[label] for i in eligible)
                details[label] = {"eligible_from": max(win, a["year"]+lag), "n": len(eligible),
                                  "citers": k, "share": k/len(eligible) if eligible else None, "lag":lag}
            return {"n_docs_in_window": len(window_idx), "anchor_details": details,
                    **{label: a["share"] for label,a in details.items()}}
        by_sub = {int(c): rates(ti[sub == c].tolist()) for c in keep}
        # robustness: the anchors are hubs INSIDE the subgraph, so a sub-community
        # can be partly defined by citing one (every RRR citer links to the RRR
        # node). Re-partition the subgraph with the anchor nodes deleted and
        # report the same rates — if the split survives, it is not the anchors'.
        anc_local = {remap[n] for n in anchor_idx if n in remap}
        keep_local = [k for k in range(len(ti)) if k not in anc_local]
        remap2 = {k: j for j, k in enumerate(keep_local)}
        e2 = [(remap2[a], remap2[b]) for a, b in e if a in remap2 and b in remap2]
        g2 = ig.Graph(n=len(keep_local), edges=e2, directed=False); g2.simplify()
        part2 = la.find_partition(g2, la.RBConfigurationVertexPartition, resolution_parameter=RESOLUTION,
                                  seed=SEED, n_iterations=-1)
        sub2 = np.array(part2.membership)
        keep2 = [c for c, n_ in Counter(sub2.tolist()).most_common() if n_ >= 40][:6]
        by_sub_noanchor = {}
        for c in keep2:
            idx = [ti[keep_local[k]] for k in np.where(sub2 == c)[0]]
            cc = Counter(cat(nodes[i]["venue"]) for i in idx).most_common(3)
            r = rates(idx)
            r["n_total"] = len(idx); r["venue_categories"] = cc
            by_sub_noanchor[int(c)] = r
        by_cat = {}
        for cname in sorted({cat(nodes[i]["venue"]) for i in ti_list}):
            by_cat[cname] = rates([i for i in ti_list if cat(nodes[i]["venue"]) == cname])
        anchors_out = {"resolved": {l: {"how": a["how"], "year": a["year"], "in_graph": a["node"] is not None,
                                        "in_graph_citers": len(citers[l])}
                                    for l, a in resolved.items()},
                       "window_year_min": win or None, "coinage": _CIT.get("coinage"),
                       "correction_set": _CIT.get("correction_set") or [],
                       "rates_by_subcommunity": by_sub, "rates_by_subcommunity_anchors_removed": by_sub_noanchor,
                       "rates_by_venue_category": by_cat,
                       "all_subgraph": rates(ti_list)}

    # --- write
    res = {
        "graph": {"nodes": N, "edges": E, "year_min": YEAR_MIN, "algorithm": algo,
                  "resolution": RESOLUTION, "modularity": round(Q, 4) if not FLOWS_ONLY else None, "n_communities": len(sizes) if not FLOWS_ONLY else None,
                  "n_communities_ge_min": len(big), "min_community": MIN_COMM},
        "communities": prof if not FLOWS_ONLY else {},
        "term_distribution": term_rows if not FLOWS_ONLY else [],
        "insulation": ins_rows,
        "flow_ratio": {"terms": focus, "obs": obs.tolist(), "ratio_obs_over_null": np.round(ratio, 2).tolist(),
                       "z": np.round(z, 1).tolist(), "perms": PERMS},
        "p2_subgraph": p2,
        "p2b_anchor_citation": anchors_out,
        "settings": {"parent": PARENT, "focus": focus, "compounds": COMPOUNDS, "merge": MERGE,
                     "subgraph_term": SUBGRAPH_TERM, "second": SECOND, "null": NULL, "iterations": N_ITER,
                     "window": list(WINDOW) if WINDOW else None, "flows_only": FLOWS_ONLY, "groups": GROUPS,
                     "edges_in_window": int(len(fsrc)), "year_max": CFG.get("revision_analysis_year_max"),
                     "identity_policy": "exclude all repeated OpenAlex resolutions pending adjudication; unique source-target reference edges",
                     "insulation_estimand": "observed same-term share / mean of per-permutation same-term shares; denominator = all distinct outgoing intra-corpus edges from source-term documents",
                     "null_strata": len(stratum_indices),
                     "nodes_in_non_singleton_strata": sum(len(x) for x in stratum_indices if len(x)>1),
                     "degree_null": "period x venue category x log2(1+in-degree) x log2(1+out-degree) bins; approximate degree adjustment, not degree-preserving rewiring",
                     "flow_ratio_estimand": "count enrichment; distinct from insulation share ratio"},
        "predictions_trauma_case": {"P1": "compounds distinct from bare trauma; insulation >> null",
                                    "P2": "trauma-informed sub-communities align with practice vs care venues",
                                    "P3": "PTSD continuous with bare trauma by citation",
                                    "P4": "embarrassment condition: compounds continuous with parent"},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "" if NULL == "period" else f"_{NULL}null"
    if YEAR_MIN != 1995:
        suffix += f"_y{YEAR_MIN}"
    if WINDOW:
        suffix += f"_w{WINDOW[0]}-{WINDOW[1]}"
    def finite_json(x):
        if isinstance(x, dict): return {k: finite_json(v) for k,v in x.items()}
        if isinstance(x, list): return [finite_json(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x): return None
        return x
    (OUT / f"citation_communities{suffix}.json").write_text(json.dumps(finite_json(res), indent=2, default=str, allow_nan=False), encoding="utf-8")

    # human report
    L = []
    L.append(f"CITATION COMMUNITIES — {CFG['concept']}  (nodes {N:,}, edges {E:,}, year >= {YEAR_MIN})")
    L.append(f"{algo}, resolution {RESOLUTION}, modularity {Q:.3f}; {len(big)} communities >= {MIN_COMM} nodes")
    L.append(f"parent = {PARENT}; null = within-{NULL} label shuffle, {PERMS} perms"
             + (f"; flows restricted to citing documents in {WINDOW[0]}-{WINDOW[1]} ({len(fsrc):,} references)" if WINDOW else "")
             + "\n")
    L.append("== communities (read before interpreting) ==" + ("  [skipped: --flows-only]" if FLOWS_ONLY else ""))
    for c in ([] if FLOWS_ONLY else big[:20]):
        p = prof[int(c)]
        L.append(f"  C{c}  n={p['n']:,}")
        L.append("     venues: " + "; ".join(f"{v} ({n})" for v, n in p["top_venues"]))
        L.append("     terms:  " + "; ".join(f"{t} {s:.0%}" for t, n, s in p["top_terms"]))
        L.append("     periods:" + " ".join(f"{k}:{v}" for k, v in p["periods"]))
    L.append("\n== term x community (P1/P3) ==")
    L.append(f"  {'term':<28}{'docs':>7}{'topC':>7}{'topShare':>10}{'entropy':>9}{'inOver':>8}{'JSvsParent':>12}")
    for r in ([] if FLOWS_ONLY else term_rows):
        L.append(f"  {r['term']:<28}{r['n_docs']:>7}{r['top_community']:>7}{r['top_community_share']:>10.3f}"
                 f"{r['norm_entropy']:>9.3f}{r['share_in_overrepresented_comms']:>8.3f}"
                 f"{(r['js_vs_parent'] if r['js_vs_parent'] is not None else float('nan')):>12.4f}")
    L.append(f"\n== insulation: distinct-edge share ratio; {NULL} label-shuffle null ==")
    L.append("  ->parent and ->2nd are COUNT enrichment ratios, not share ratios.")
    L.append(f"  (->parent = citations to '{PARENT}' docs vs null; ->2nd = to '{SECOND}' docs vs null)")
    L.append(f"  {'term':<28}{'edges':>8}{'refs':>8}{'obs':>7}{'null':>7}{'ratio':>7}{'z':>8}{'->parent':>10}{'->2nd':>8}")
    for r in ins_rows:
        L.append(f"  {r['term']:<28}{r['edges_from_term_docs']:>8}{r['refs_out_to_focus']:>8}{r['same_term_share_obs']:>7.3f}"
                 f"{r['same_term_share_null']:>7.3f}{r['insulation_ratio']:>7.2f}{r['insulation_z']:>8.1f}"
                 f"{(r['to_parent_ratio'] or float('nan')):>10.2f}{(r['to_second_ratio'] or float('nan')):>8.2f}")
    if p2:
        L.append(f"\n== P2: {p2['subgraph_term']} subgraph ==")
        L.append(f"  docs {p2['n_docs']:,}, intra edges {p2['intra_edges']:,}, "
                 f"{p2['n_subcommunities']} sub-communities, modularity {p2['modularity']}, "
                 f"Cramer's V (sub-community x venue category) = {p2['cramers_v_subcomm_x_venue_category']}")
        for c, t in p2["subcommunities"].items():
            L.append(f"  S{c} n={t['n']}: " + "; ".join(f"{k} {v}" for k, v in t["venue_categories"]))
            L.append("       " + "; ".join(f"{v} ({n})" for v, n in t["top_venues"]))
    if anchors_out:
        A = anchors_out
        labels = [l for l, a in A["resolved"].items() if a["in_graph"]]
        L.append(f"\n== P2b: anchor citation inside the {p2['subgraph_term']} subgraph "
                 f"(documents from {A['window_year_min'] or 'all years'}; share of docs citing each anchor) ==")
        for l, a in A["resolved"].items():
            L.append(f"  {l:<18} {'UNRESOLVED (not in corpus / not resolved to OpenAlex)' if a['how'] is None else a['how']}"
                     + (f"  in-graph citers: {a['in_graph_citers']}" if a["in_graph"] else ""))
        cols = labels
        hdr = f"  {'group':<26}{'n':>6}" + "".join(f"{c[:12]:>14}" for c in cols)
        L.append(hdr)
        L.append("  n is the broad window count; each anchor has its own eligible denominator in anchor_details (default publication year + 1).")
        def row(name, r):
            return f"  {name:<26}{r['n_docs_in_window']:>6}" + "".join(
                f"{(r.get(c) if r.get(c) is not None else float('nan')):>14.3f}" for c in cols)
        L.append(row("all", A["all_subgraph"]))
        for c, r in A["rates_by_subcommunity"].items():
            L.append(row(f"S{c}", r))
        for c, r in A["rates_by_venue_category"].items():
            L.append(row(f"venue: {c}", r))
        if A.get("rates_by_subcommunity_anchors_removed"):
            L.append("  -- sub-communities re-detected with the anchor nodes REMOVED (so no group is defined by citing an anchor) --")
            for c, r in A["rates_by_subcommunity_anchors_removed"].items():
                lab = f"T{c} " + "/".join(f"{k.split('/')[0]}{v}" for k, v in r["venue_categories"][:2])
                L.append(row(lab[:26], r))
        if A.get("coinage") and A.get("correction_set"):
            L.append(f"  (coinage = {A['coinage']}; correction set = {', '.join(A['correction_set'])}; "
                     "a group that cites the coinage at the field's rate but the corrections below it is "
                     "a candidate for source-level review of citation context)")
    (OUT / f"citation_communities_report{suffix}.txt").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    pc.write_manifest(CFG, __file__,
                      params={"year_min": YEAR_MIN, "perms": PERMS, "resolution": RESOLUTION, "seed": SEED,
                              "iterations": N_ITER, "null": NULL, "min_community": MIN_COMM,
                              "parent": PARENT, "focus_terms": focus, "compounds": COMPOUNDS,
                              "merge": MERGE, "subgraph_term": SUBGRAPH_TERM,
                              "window": list(WINDOW) if WINDOW else None, "flows_only": FLOWS_ONLY, "groups": GROUPS},
                      outputs=[OUT / f"citation_communities{suffix}.json",
                               OUT / f"citation_communities_report{suffix}.txt"],
                      notes="pre-registered P1-P4 in the docstring; null = within-period term-label shuffle")
    print(f"[27] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
