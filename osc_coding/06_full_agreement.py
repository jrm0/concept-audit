"""
06_full_agreement.py — SET-BASED consensus + agreement for the full
99-study LLM construct coding (coding_sheet_llm_full.csv).

v2 — WHY SET-BASED. Each study was coded with up to THREE constructs
(construct_1..3): typically the primary IV, the primary DV, and a
secondary construct. v1 of this script compared construct_1 only, which
re-created the pilot's study-10 mistake: two coders who both found
{stereotypes, illusory recollection} but ranked them differently were
scored as disagreeing. Study 008 in the v1 adjudication queue was exactly
this — all three models listed both constructs, in different slots.
Agreement must therefore be computed over construct SETS, with rank
reported separately.

Procedure:
  1. Per (study, model): pool constructs from the 5 masked runs (all
     slots), merge variants by relaxed matching (normalized exact, or
     token containment), keep those appearing in >=2 runs. Support =
     number of runs mentioning it (any slot).
  2. Across models: merge the model sets the same way. Each consensus
     construct's MODEL SUPPORT = how many models' sets contain it.
  3. Tiers per study:  A = >=1 construct supported by ALL 3 models;
     B = >=1 by 2 models; C = no cross-model construct (adjudicate).
     RANK agreement (all models put a matched construct first) reported
     separately — it is information, not a gate.
  4. Contamination: does the unmasked run introduce constructs absent
     from the model's masked set?
  5. Human anchor: Jonathan's pilot codings (3 slots) vs consensus sets.

Outputs:
  data/consensus_constructs.csv   pilot_id, tier, consensus_1..3 (with
                                  model-support counts), rank_agreement
  data/full_agreement_report.txt
Usage (from code/osc_coding/):
    python3 06_full_agreement.py
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("data")
LLM_PATH = DATA / "coding_sheet_llm_full.csv"
HUMAN_PATH = DATA / "coding_sheet_pilot.csv"
PILOT_MAP = DATA / "pilot_10_studies.csv"
OUT_CONSENSUS = DATA / "consensus_constructs.csv"
OUT_REPORT = DATA / "full_agreement_report.txt"

MIN_RUNS = 2          # construct must appear in >=2 of 5 masked runs
SLOTS = ("construct_1", "construct_2", "construct_3")


def norm(term):
    t = (term or "").lower().strip()
    t = re.sub(r"[‐-―]", "-", t)
    t = re.sub(r"[^a-z0-9\- ]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


STOP = {"the", "a", "an", "of", "in", "on", "for", "and", "effect",
        "effects", "task", "paradigm"}


def toks(t):
    return {w for w in norm(t).replace("-", " ").split() if w not in STOP}


def relaxed_match(a, b):
    if not a or not b:
        return False
    if norm(a) == norm(b):
        return True
    ta, tb = toks(a), toks(b)
    return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


class Groups:
    """Greedy equivalence-grouping of construct strings by relaxed match.
    Representative = shortest member (most canonical)."""

    def __init__(self):
        self.groups = []          # list of (rep, Counter(member->count))

    def add(self, term, weight=1):
        for i, (rep, members) in enumerate(self.groups):
            if relaxed_match(rep, term) or any(relaxed_match(m, term)
                                               for m in members):
                members[term] += weight
                if len(norm(term)) < len(norm(rep)) and len(toks(term)) > 0:
                    self.groups[i] = (term, members)
                return
        self.groups.append((term, Counter({term: weight})))

    def ranked(self):
        return sorted(((rep, sum(m.values())) for rep, m in self.groups),
                      key=lambda x: -x[1])


def main():
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    rows = [r for r in csv.DictReader(open(LLM_PATH, encoding="utf-8"))]
    masked = defaultdict(list)        # (pid, model) -> [run rows]
    unmasked = defaultdict(list)      # (pid, model) -> [constructs]
    for r in rows:
        pid = r["pilot_id"].strip().zfill(3)
        cs = [r[s] for s in SLOTS if (r.get(s) or "").strip()]
        if not cs:
            continue
        if r["run"] == "unmasked":
            unmasked[(pid, r["coder"])].extend(cs)
        elif r["run"].isdigit():
            masked[(pid, r["coder"])].append(cs)
    models = sorted({m for (_, m) in masked})
    pids = sorted({p for (p, _) in masked})
    emit(f"{len(pids)} studies x {models}; set-based agreement over "
         f"{sum(len(v) for v in masked.values())} masked runs x <=3 slots")

    # 1. per-model sets, with first-slot tally for rank analysis
    model_set = {}                    # (pid, model) -> [(rep, support)]
    model_first = {}                  # (pid, model) -> modal construct_1
    for (pid, m), runs in masked.items():
        g = Groups()
        firsts = Counter()
        for cs in runs:
            seen_this_run = []
            for c in cs:
                if not any(relaxed_match(c, s) for s in seen_this_run):
                    g.add(c)
                    seen_this_run.append(c)
            firsts[norm(cs[0])] += 1
        kept = [(rep, sup) for rep, sup in g.ranked() if sup >= MIN_RUNS]
        model_set[(pid, m)] = kept[:4]
        model_first[(pid, m)] = firsts.most_common(1)[0][0] if firsts else ""

    # 2-3. cross-model consensus sets + tiers
    tier, consensus, rank_agree = {}, {}, {}
    for p in pids:
        g = Groups()
        contributors = defaultdict(set)   # rep-group index by model
        for m in models:
            for rep, sup in model_set.get((p, m), []):
                g.add(rep, weight=0)      # build groups
        # count model support per group
        support = []
        for rep, _members in g.groups:
            ms = {m for m in models
                  if any(relaxed_match(rep, r2)
                         for r2, _s in model_set.get((p, m), []))}
            support.append((rep, ms))
        support.sort(key=lambda x: (-len(x[1]), len(norm(x[0]))))
        cons = [(rep, len(ms)) for rep, ms in support if len(ms) >= 2][:3]
        consensus[p] = cons
        best = len(support[0][1]) if support else 0
        tier[p] = "A" if best == 3 else ("B" if best == 2 else "C")
        firsts = [model_first.get((p, m), "") for m in models
                  if model_first.get((p, m), "")]
        rank_agree[p] = (len(firsts) == 3 and
                         all(relaxed_match(firsts[0], f) for f in firsts[1:]))
    tc = Counter(tier.values())
    emit(f"\n=== TIERS (set-based) ===")
    emit(f"  A (a construct in ALL 3 models' sets): {tc['A']}")
    emit(f"  B (in 2 of 3):                         {tc['B']}")
    emit(f"  C (no cross-model construct):          {tc['C']}")
    emit(f"  Primary-rank agreement (all models rank a matched construct "
         f"first): {sum(rank_agree.values())}/{len(pids)}")

    # 4. contamination, set version
    emit("\n=== TITLE CONTAMINATION (unmasked adds constructs absent from "
         "the masked set) ===")
    for m in models:
        n, added = 0, 0
        for p in pids:
            um = unmasked.get((p, m), [])
            ms = [rep for rep, _ in model_set.get((p, m), [])]
            if um and ms:
                n += 1
                if any(not any(relaxed_match(u, s) for s in ms)
                       for u in um[:1]):     # primary unmasked construct
                    added += 1
        emit(f"  {m}: unmasked primary not in masked set for {added}/{n} "
             f"({100*added/max(n,1):.0f}%)")

    # 5. human anchor (sets)
    if HUMAN_PATH.exists() and PILOT_MAP.exists():
        p2s = {r["pilot_id"].strip().zfill(2):
               (r.get("study_num") or "").strip().zfill(3)
               for r in csv.DictReader(open(PILOT_MAP, encoding="utf-8"))
               if (r.get("study_num") or "").strip()}
        human = defaultdict(list)
        for r in csv.DictReader(open(HUMAN_PATH, encoding="utf-8")):
            if r.get("coder", "") in ("claude", "gpt", "gemini"):
                continue
            sid = p2s.get(r["pilot_id"].strip().zfill(2))
            if sid:
                human[sid] += [r[s] for s in SLOTS
                               if (r.get(s) or "").strip()]
        if human:
            hit_p, hit_any = [], []
            for p, hs in human.items():
                cons = [rep for rep, _ in consensus.get(p, [])]
                if cons and relaxed_match(hs[0], cons[0]):
                    hit_p.append(p)
                if cons and any(relaxed_match(h, c) for h in hs
                                for c in cons):
                    hit_any.append(p)
            emit(f"\n=== HUMAN ANCHOR (n={len(human)}) ===")
            emit(f"  human primary = consensus primary: {len(hit_p)}")
            emit(f"  human set overlaps consensus set:  {len(hit_any)}")
            for p, hs in sorted(human.items()):
                cons = ["%s(%d)" % c for c in consensus.get(p, [])]
                mark = "OK " if p in hit_any else "DIFF"
                emit(f"  [{mark}] {p}: human={hs} | consensus={cons}")

    # write consensus sheet
    with open(OUT_CONSENSUS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pilot_id", "tier", "rank_agreement",
                    "consensus_1", "support_1", "consensus_2", "support_2",
                    "consensus_3", "support_3",
                    *[f"set_{m}" for m in models]])
        for p in pids:
            cons = consensus.get(p, [])
            flat = []
            for i in range(3):
                flat += list(cons[i]) if i < len(cons) else ["", ""]
            w.writerow([p, tier[p], int(rank_agree[p]), *flat,
                        *["; ".join(f"{rep}({sup})" for rep, sup in
                                    model_set.get((p, m), []))
                          for m in models]])
    emit(f"\nConsensus sheet -> {OUT_CONSENSUS}")

    queue = [p for p in pids if tier[p] == "C"]
    emit(f"\nADJUDICATION QUEUE ({len(queue)} studies):")
    for p in queue:
        emit(f"  study {p}:")
        for m in models:
            emit(f"    {m}: " + "; ".join(
                f"{rep}({sup})" for rep, sup in model_set.get((p, m), [])))
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
