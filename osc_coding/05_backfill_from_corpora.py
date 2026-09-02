"""
05_backfill_from_corpora.py — Fill the 45 MISSING OSC study texts from
corpora we have since built (abstracts pool + lineage corpus).

When 04 ran (session ~6), 45 of the 99 OSC originals had no abstract via the
S2 API (mostly JPSP 2008 and JEP:LMC 2008 short reports; publisher withheld).
We paused rather than chase them manually, on the hunch that the corpora
being built for the lineage study would sweep them up. They did: 39 of the
45 titles now have abstracts in data/abstracts/*.jsonl or the lineage
corpus. This script writes them into osc_coding/data/full_texts/ in the
exact format 02_run_llm_coding.py expects (study_XXX.txt + masked variant),
updates full_studies.csv status to 'backfilled', and prints the residual
manual-download list.

Masked variant: same as 01 — the TITLE line is replaced with '[MASKED]'
(the contamination check: can the LLM identify the construct without seeing
the title?).

Usage (from code/osc_coding/):
    python3 05_backfill_from_corpora.py
"""

import csv
import glob
import json
import re
from pathlib import Path

DATA = Path("data")
TEXTS = DATA / "full_texts"
POOLS = (glob.glob("../data/abstracts/*.jsonl")
         + ["../data/family/lineage_docs.jsonl"])


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()[:100]


def main():
    rows = list(csv.DictReader(open(DATA / "full_studies.csv",
                                    encoding="utf-8")))
    missing = {norm(r["title"]): r for r in rows if r["status"] == "MISSING"}
    print(f"{len(missing)} studies marked MISSING; searching "
          f"{len(POOLS)} corpus files ...")

    filled = {}
    for f in POOLS:
        for line in open(f, encoding="utf-8"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = norm(d.get("title"))
            if key not in missing or key in filled:
                continue
            ab = (d.get("abstract") or "").strip()
            if not ab and d.get("mention_sentences"):
                # lineage corpus keeps sentences, not the full abstract
                ab = " ".join(s for ss in d["mention_sentences"].values()
                              for s in ss)
            if len(ab) < 200:
                continue
            filled[key] = (d, ab, Path(f).name)

    for key, (d, ab, src) in filled.items():
        r = missing[key]
        sid = r["study_id"]
        head = (f"TITLE: {r['title']}\nJOURNAL: {r['journal']}\n"
                f"DOI: {d.get('doi','')}\nSOURCE: corpus backfill ({src})\n"
                f"\nABSTRACT:\n{ab}\n")
        (TEXTS / f"study_{sid}.txt").write_text(head, encoding="utf-8")
        masked = head.replace(f"TITLE: {r['title']}", "TITLE: [MASKED]")
        (TEXTS / f"study_{sid}_masked.txt").write_text(masked,
                                                       encoding="utf-8")
        r["status"] = "backfilled"
        r["abstract_source"] = src
        print(f"  study {sid} <- {src}")

    with open(DATA / "full_studies.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    left = [r for r in rows if r["status"] == "MISSING"]
    print(f"\nBackfilled {len(filled)}; {len(left)} still need the abstract "
          f"pasted by hand (copy the abstract into the file shown):")
    for r in left:
        print(f"  study {r['study_id']} | {r['journal']:<8} | {r['title']}")
        print(f"      -> edit data/full_texts/study_{r['study_id']}.txt")
    print("\nThen: python3 02_run_llm_coding.py full   "
          "(runs all 99 x 3 models x masked/unmasked)")


if __name__ == "__main__":
    main()
