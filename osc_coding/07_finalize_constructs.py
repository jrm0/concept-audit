"""
07_finalize_constructs.py — Produce the LOCKED construct assignment for
Paper 2 from the set-based consensus (06) plus the unmasked runs.

Decisions locked with Jonathan (session 17):
  * SET-BASED consensus is the primary approach; rank is secondary.
  * MASKED consensus is the backbone. Rationale: the OSC-99 are famous
    BECAUSE of the replication project, so a title-recognizing model may
    retrieve construct labels from the replication literature — a channel
    for the DV (replication outcome) to leak into the predictor. Masked
    coding is computed blind to the outcome; also 5 masked runs vs 1
    unmasked run per model = better sampling.
  * Unmasked constructs are NOT discarded: title-carried constructs are
    legitimate (titles name constructs). Any unmasked construct supported
    by >=2 models and absent from the masked consensus set JOINS the set,
    flagged title_carried=1.
  * Consensus primary ranking stands unless Jonathan overrides
    (override column provided; blank = accept).

Reads:  data/consensus_constructs.csv (from 06, incl. manual fix for 150)
        data/coding_sheet_llm_full.csv (for the unmasked runs)
Writes: data/final_constructs.csv — pilot_id, construct_1..4,
        title_carried flags, tier, override column (blank)
Usage:  python3 07_finalize_constructs.py
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

DATA = Path("data")
SLOTS = ("construct_1", "construct_2", "construct_3")


def clean_label(t):
    """Strip LLM gloss artifacts: parenthetical explanations and trailing
    qualifiers — 'association (low-level associative processing)' is the
    construct 'association' plus a gloss, not a searchable term."""
    t = re.sub(r"\s*\([^)]*\)\s*$", "", (t or "").strip())
    t = re.sub(r"\s*\([^)]*\)", " ", t)          # mid-string parentheticals
    return re.sub(r"\s+", " ", t).strip()


def norm(t):
    t = clean_label(t).lower()
    t = re.sub(r"[‐-―]", "-", t)
    t = re.sub(r"[^a-z0-9\- ]", "", t)
    return re.sub(r"\s+", " ", t)


STOP = {"the", "a", "an", "of", "in", "on", "for", "and", "effect",
        "effects", "task", "paradigm"}


def toks(t):
    return {w for w in norm(t).replace("-", " ").split() if w not in STOP}


def match(a, b):
    if not a or not b:
        return False
    if norm(a) == norm(b):
        return True
    ta, tb = toks(a), toks(b)
    return bool(ta) and bool(tb) and (ta <= tb or tb <= ta)


def main():
    # zfill: Excel strips leading zeros from pilot_ids on any hand-edit
    cons = {r["pilot_id"].strip().zfill(3): r for r in csv.DictReader(
        open(DATA / "consensus_constructs.csv", encoding="utf-8"))}

    # unmasked constructs by study, with model support
    um = defaultdict(lambda: defaultdict(set))   # pid -> construct -> models
    for r in csv.DictReader(open(DATA / "coding_sheet_llm_full.csv",
                                 encoding="utf-8")):
        if r["run"] != "unmasked":
            continue
        pid = r["pilot_id"].strip().zfill(3)
        for s in SLOTS:
            c = clean_label(r.get(s))
            if c:
                # merge into an existing unmasked variant if one matches
                tgt = next((k for k in um[pid] if match(k, c)), c)
                um[pid][tgt].add(r["coder"])

    # Expert relabels live in construct_overrides.csv so they SURVIVE
    # re-runs of this script (hand-edits to final_constructs.csv would be
    # overwritten, and Excel strips leading zeros from pilot_ids).
    overrides = {}
    ov_path = DATA / "construct_overrides.csv"
    if ov_path.exists():
        for r in csv.DictReader(open(ov_path, encoding="utf-8")):
            overrides[r["pilot_id"].strip().zfill(3)] = \
                clean_label(r["new_construct_1"])

    out_rows = []
    n_added = 0
    for pid, r in sorted(cons.items()):
        base = [clean_label(r[k])
                for k in ("consensus_1", "consensus_2", "consensus_3")
                if clean_label(r[k])]
        if pid in overrides:
            new1 = overrides[pid]
            base = [new1] + [b for b in base if not match(b, new1)]
        flags = [0] * len(base)
        # title-carried additions: unmasked, >=2 models, not already in set
        adds = [(c, len(ms)) for c, ms in um.get(pid, {}).items()
                if len(ms) >= 2 and not any(match(c, b) for b in base)]
        adds.sort(key=lambda x: -x[1])
        for c, _n in adds[: 4 - len(base)]:
            base.append(c)
            flags.append(1)
            n_added += 1
        row = {"pilot_id": pid, "tier": r["tier"],
               "rank_agreement": r.get("rank_agreement", ""),
               "override_primary": ""}
        for i in range(4):
            row[f"construct_{i+1}"] = base[i] if i < len(base) else ""
            row[f"title_carried_{i+1}"] = flags[i] if i < len(flags) else ""
        out_rows.append(row)

    fields = (["pilot_id", "tier", "rank_agreement", "override_primary"]
              + [f"construct_{i}" for i in (1, 2, 3, 4)]
              + [f"title_carried_{i}" for i in (1, 2, 3, 4)])
    with open(DATA / "final_constructs.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"{len(out_rows)} studies -> {DATA / 'final_constructs.csv'}")
    print(f"title-carried constructs added to sets: {n_added}")
    print("override_primary column is blank — Jonathan fills only where he "
          "disagrees with construct_1.")


if __name__ == "__main__":
    main()
