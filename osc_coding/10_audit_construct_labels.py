"""
10_audit_construct_labels.py — Are our construct labels the FIELD'S terms,
the STUDY'S terms, or the CODER'S paraphrases?

The EBSCO searches surfaced the problem: several consensus labels return
ZERO PsycInfo hits ("dynamic mental processing"). Two different situations
hide under a zero:
  ATTESTED-ISOLATED  the label appears in the original study's own
                     title/abstract but the field never adopted it
                     ("value heuristic", coined by OSC study 165 itself).
                     This is real construct isolation — a MEASUREMENT,
                     and a candidate Paper-2 predictor.
  PARAPHRASE         the label does NOT appear in the study's own text —
                     the LLM described the construct instead of naming it
                     ("dynamic mental processing"). This is coder error to
                     fix by relabeling from the study's own vocabulary.
Plus the healthy case:
  ATTESTED-COMMON    in the study's text AND used by the field.

Reads:  data/final_constructs.csv, data/full_texts/study_*.txt,
        ../data/abstracts/*.jsonl (field-usage counts)
Writes: data/construct_label_audit.csv
Usage:  python3 10_audit_construct_labels.py
"""
import csv, glob, json, re
from pathlib import Path

DATA = Path("data")

def pat_of(term):
    body = re.escape(term.lower()).replace(r"\ ", "[-\\s]+").replace(r"\-", "[-\\s]+")
    return re.compile(r"\b" + body + r"\b", re.I)

def main():
    rows = list(csv.DictReader(open(DATA / "final_constructs.csv", encoding="utf-8")))
    # study texts (unmasked)
    texts = {}
    for f in (DATA / "full_texts").glob("study_*.txt"):
        m = re.match(r"study_(\d+)$", f.stem)
        if m:
            texts[m.group(1)] = f.read_text(encoding="utf-8").lower()
    # field counts from the pool
    constructs = sorted({r[f"construct_{i}"].lower().strip()
                         for r in rows for i in (1, 2, 3, 4)
                         if r.get(f"construct_{i}")})
    pats = {c: pat_of(c) for c in constructs}
    field = {c: 0 for c in constructs}
    for f in glob.glob("../data/abstracts/*.jsonl"):
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = ((d.get("title") or "") + ". " + (d.get("abstract") or "")).lower()
            for c, p in pats.items():
                if c[:4] in t and p.search(t):
                    field[c] += 1

    out = []
    counts = {"ATTESTED-COMMON": 0, "ATTESTED-ISOLATED": 0, "PARAPHRASE": 0}
    for r in rows:
        pid = r["pilot_id"].strip().zfill(3)
        text = texts.get(pid, "")
        for i in (1, 2, 3, 4):
            c = (r.get(f"construct_{i}") or "").lower().strip()
            if not c:
                continue
            in_study = bool(pats[c].search(text))
            n_field = field[c]
            if in_study and n_field >= 10:
                cls = "ATTESTED-COMMON"
            elif in_study:
                cls = "ATTESTED-ISOLATED"
            else:
                cls = "PARAPHRASE"
            counts[cls] += 1
            out.append([pid, i, c, int(in_study), n_field, cls,
                        r.get(f"title_carried_{i}", "")])
    with open(DATA / "construct_label_audit.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pilot_id", "slot", "construct", "in_study_text",
                    "field_docs", "class", "title_carried"])
        w.writerows(out)
    print(f"{len(out)} construct assignments audited -> "
          f"{DATA / 'construct_label_audit.csv'}")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    prim = [o for o in out if o[1] == 1]
    print(f"\nPRIMARY constructs only ({len(prim)}):")
    for k in counts:
        print(f"  {k}: {sum(1 for o in prim if o[5] == k)}")
    print("\nPARAPHRASE primaries (relabel from the study's own vocabulary):")
    for o in prim:
        if o[5] == "PARAPHRASE":
            print(f"  study {o[0]}: '{o[2]}'")

if __name__ == "__main__":
    main()
