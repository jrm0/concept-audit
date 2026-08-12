"""
03_pilot_agreement.py — Pilot agreement analysis (coding_protocol.md §Pilot analysis).

Reads:
  data/coding_sheet_pilot.csv — human coding (coder=jonathan, run=1)
  data/coding_sheet_llm.csv   — LLM codings from 02_run_llm_coding.py

Reports (printed + data/pilot_agreement_report.txt):
  1. Within-model consistency: share of the 5 masked runs agreeing with the
     model's modal construct_1, per study
  2. Cross-model agreement on modal construct_1
  3. Human-LLM agreement on construct_1 (exact match after normalization);
     disagreements listed for your relaxed-match judgment — the >=8/10
     decision rule counts exact + documented-relaxed matches
  4. Masked vs. unmasked shift: does the unmasked run diverge from the
     masked modal answer? (contamination signal)

Usage: python3 03_pilot_agreement.py
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("data")
HUMAN_PATH = DATA_DIR / "coding_sheet_pilot.csv"
LLM_PATH = DATA_DIR / "coding_sheet_llm.csv"
REPORT_PATH = DATA_DIR / "pilot_agreement_report.txt"


def norm(term):
    t = (term or "").lower().strip()
    t = re.sub(r"[‐-―]", "-", t)      # unicode hyphens
    t = re.sub(r"[^a-z0-9\- ]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def load(path):
    if not path.exists():
        raise SystemExit(f"Missing {path} — run the earlier steps first.")
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if norm(r.get("construct_1"))]


def main():
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    human_rows = [r for r in load(HUMAN_PATH)]
    llm_rows = load(LLM_PATH)

    def pid_norm(p):
        return (p or "").strip().zfill(2)  # '1' and '01' are the same study

    human = {pid_norm(r["pilot_id"]): norm(r["construct_1"]) for r in human_rows}
    if not human:
        raise SystemExit("No human codings found in coding_sheet_pilot.csv — "
                         "fill in your rows first (coder, construct_1...).")

    # Organize LLM masked runs (runs 1-5) and unmasked run
    masked = defaultdict(list)     # (pid, model) -> [construct_1,...]
    unmasked = {}                  # (pid, model) -> construct_1
    for r in llm_rows:
        key = (pid_norm(r["pilot_id"]), r["coder"])
        if r["run"] == "unmasked":
            unmasked[key] = norm(r["construct_1"])
        elif r["run"].isdigit():
            masked[key].append(norm(r["construct_1"]))

    models = sorted({m for (_, m) in masked})
    pids = sorted({p for (p, _) in masked} | set(human))

    emit("=== 1. Within-model consistency (masked runs agreeing with modal) ===")
    modal = {}
    for model in models:
        shares = []
        for pid in pids:
            runs = masked.get((pid, model), [])
            if not runs:
                continue
            top, n = Counter(runs).most_common(1)[0]
            modal[(pid, model)] = top
            shares.append(n / len(runs))
            emit(f"  {model} study {pid}: modal='{top}' ({n}/{len(runs)})")
        if shares:
            emit(f"  {model} mean consistency: {sum(shares)/len(shares):.2f}")
        emit()

    emit("=== 2. Cross-model agreement on modal construct_1 ===")
    agree_all = 0
    for pid in pids:
        answers = {m: modal.get((pid, m)) for m in models if modal.get((pid, m))}
        vals = list(answers.values())
        all_same = len(set(vals)) == 1 and len(vals) == len(models)
        agree_all += all_same
        emit(f"  study {pid}: {answers} {'AGREE' if all_same else ''}")
    if pids and models:
        emit(f"  full agreement on {agree_all}/{len(pids)} studies")
    emit()

    emit("=== 3. Human vs. LLM modal (decision rule: >=8/10 exact+relaxed) ===")
    for model in models:
        exact = 0
        disagreements = []
        for pid in pids:
            h, m = human.get(pid), modal.get((pid, model))
            if not h or not m:
                continue
            if h == m:
                exact += 1
            else:
                disagreements.append((pid, h, m))
        emit(f"  {model}: exact matches {exact}/{len(human)}")
        for pid, h, m in disagreements:
            emit(f"    study {pid}: human='{h}' vs {model}='{m}'  "
                 f"<- judge: same construct? (relaxed match)")
    emit()

    emit("=== 3b. Ranked-SET overlap (credits construct_2/3 on both sides) ===")
    emit("(match = any overlap between human's ranked constructs and constructs")
    emit(" a model produced in >=2 masked runs at any rank)")
    human_sets = {}
    for r in human_rows:
        pid = pid_norm(r["pilot_id"])
        s = {norm(r.get(f"construct_{i}", "")) for i in (1, 2, 3)}
        human_sets[pid] = {x for x in s if x}
    model_counts = defaultdict(Counter)
    for r in llm_rows:
        if r["run"].isdigit():
            key = (pid_norm(r["pilot_id"]), r["coder"])
            for i in (1, 2, 3):
                c = norm(r.get(f"construct_{i}", ""))
                if c:
                    model_counts[key][c] += 1
    for model in models:
        matches, misses = 0, []
        for pid in sorted(human_sets):
            mset = {c for c, n in model_counts.get((pid, model), Counter()).items()
                    if n >= 2}
            if not mset:
                continue
            if human_sets[pid] & mset:
                matches += 1
            else:
                misses.append((pid, human_sets[pid], mset))
        emit(f"  {model}: set-overlap matches {matches}/{len(human_sets)}")
        for pid, hs, ms in misses:
            emit(f"    study {pid}: human={sorted(hs)} vs model={sorted(ms)}")
    emit()

    emit("=== 4. Masked vs. unmasked (contamination check) ===")
    for model in models:
        shifts = []
        for pid in pids:
            mm, um = modal.get((pid, model)), unmasked.get((pid, model))
            if mm and um and mm != um:
                shifts.append((pid, mm, um))
        emit(f"  {model}: {len(shifts)} shift(s)")
        for pid, mm, um in shifts:
            emit(f"    study {pid}: masked='{mm}' -> unmasked='{um}'")
    emit()
    emit("Interpretation notes: relaxed matches are YOUR call — document each "
         "one. Shifts in section 4 toward famous construct labels suggest "
         "leakage of replication-lore into coding.")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
