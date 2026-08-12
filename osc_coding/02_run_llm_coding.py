"""
02_run_llm_coding.py — Automated LLM construct coding for the OSC pilot.

Implements the procedure in coding_protocol.md:
  - 3 model families (Claude / GPT / Gemini)
  - per study: 5 independent MASKED runs (provider default sampling, fresh context each)
  - per study: 1 UNMASKED run (contamination check)
  - prompt variant v1b on the first 3 studies (1 masked run each, per model)

Each API call is a brand-new conversation containing ONLY the prompt — no
shared context between studies or runs. That is the point: it keeps the
masking intact.

Outputs:
  data/coding_sheet_llm.csv  — one row per (study x model x run), same columns
                               as coding_sheet_pilot.csv (merge after review)
  data/llm_raw/              — every raw response, for audit
  data/llm_run_log.csv       — call log (model, status, latency)

Resumable: completed (study, model, run) combinations are skipped on re-run.

Setup — NEVER paste real keys into this or any file that syncs or commits.
Put them in code/osc_coding/.env (gitignored; see .env.example), then load
them into your terminal before running:
    source .env
Models with a missing key are skipped with a warning, so you can run the
available ones now and fill in the rest later.

Usage (note: python3, per Jonathan's standing rule):
    python3 02_run_llm_coding.py

MODEL NAMES drift — check each provider's docs if a call errors, and update
the MODELS dict below (or override via env vars CLAUDE_MODEL / OPENAI_MODEL /
GEMINI_MODEL). Record the exact model strings used in the lab notebook.

Usage (from code/osc_coding/, venv active):
    python3 02_run_llm_coding.py           # everything
    python3 02_run_llm_coding.py claude    # one model family only
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

# --- Configuration ---

MODELS = {
    "claude": os.environ.get("CLAUDE_MODEL", "claude-sonnet-5"),
    "gpt": os.environ.get("OPENAI_MODEL", "gpt-4o"),
    "gemini": os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
}
KEYS = {
    "claude": os.environ.get("ANTHROPIC_API_KEY", ""),
    "gpt": os.environ.get("OPENAI_API_KEY", ""),
    "gemini": os.environ.get("GEMINI_API_KEY", ""),
}

# Sampling: PROVIDER DEFAULTS, deliberately. Newer Anthropic models reject a
# manual `temperature`, so for cross-model symmetry no provider gets one.
# Record in methods as: "provider default sampling settings (July 2026)".
MASKED_RUNS = 5

# Mode: "pilot" (default) or "full" — pass `full` on the command line to code
# the whole RP:P frame from data/full_texts/ (built by 04_fetch_all_studies.py).
FULL_MODE = "full" in sys.argv[1:]
VARIANT_STUDIES = 0 if FULL_MODE else 3  # v1b prompt check was a pilot diagnostic

DATA_DIR = Path("data")
if FULL_MODE:
    TEXT_DIR = DATA_DIR / "full_texts"
    RAW_DIR = DATA_DIR / "llm_raw_full"
    OUT_PATH = DATA_DIR / "coding_sheet_llm_full.csv"
    LOG_PATH = DATA_DIR / "llm_run_log_full.csv"
else:
    TEXT_DIR = DATA_DIR / "pilot_texts"
    RAW_DIR = DATA_DIR / "llm_raw"
    OUT_PATH = DATA_DIR / "coding_sheet_llm.csv"
    LOG_PATH = DATA_DIR / "llm_run_log.csv"

# Canonical prompt — mirrors llm_prompt.md (keep in sync!)
PROMPT_V1_INTRO = (
    "You are assisting with a study-coding task in psychology. Below is the "
    "abstract of a psychology paper. Your task is to identify the central "
    "theoretical construct(s) of the study — the abstract psychological "
    "concept(s) whose behavior the study is fundamentally about."
)
PROMPT_V1B_INTRO = (
    "Read the following psychology abstract and name the main theoretical "
    "concept or concepts the research is testing claims about."
)
PROMPT_BODY = """

Rules:
- A construct is a theoretical concept (e.g., "cognitive dissonance"), not an operationalization (e.g., "essay-writing task") and not a population or method.
- Use the authors' own terminology as it appears in the text, in lowercase.
- List up to 3 constructs, ranked from most to least central. Do not pad — if the study is about one construct, list one.
- Base your judgment only on the text provided. Do not use outside knowledge about the paper, its authors, or subsequent literature.

Respond in exactly this format:
construct_1: <term>
construct_2: <term or NONE>
construct_3: <term or NONE>
rationale: <one sentence>

Abstract:
"""


def build_prompt(text, variant="v1"):
    intro = PROMPT_V1B_INTRO if variant == "v1b" else PROMPT_V1_INTRO
    return intro + PROMPT_BODY + text


# --- Provider calls (each call = one fresh conversation) ---

RETRYABLE = {429, 500, 502, 503, 504}  # rate limits / transient overload


def post_json(url, headers, payload, max_tries=6):
    """POST with automatic backoff on transient errors (429/5xx).

    Overloaded-model 503s and rate-limit 429s are retried with exponential
    backoff (2, 4, 8... seconds, up to ~1 min per call). Non-retryable errors
    (e.g., 404 unknown model) fail immediately with the provider's message.
    """
    for attempt in range(max_tries):
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        if r.status_code < 400:
            return r.json()
        if r.status_code in RETRYABLE and attempt < max_tries - 1:
            wait = min(2 ** (attempt + 1), 60)
            print(f"    transient HTTP {r.status_code}; retrying in {wait}s "
                  f"({attempt + 1}/{max_tries - 1})")
            time.sleep(wait)
            continue
        raise RuntimeError(f"HTTP {r.status_code} — {r.text[:300]}")


def call_claude(prompt):
    data = post_json(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": KEYS["claude"],
                 "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        payload={"model": MODELS["claude"], "max_tokens": 300,
                 "messages": [{"role": "user", "content": prompt}]})
    return data["content"][0]["text"]


def call_gpt(prompt):
    data = post_json(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {KEYS['gpt']}"},
        payload={"model": MODELS["gpt"], "max_tokens": 300,
                 "messages": [{"role": "user", "content": prompt}]})
    return data["choices"][0]["message"]["content"]


def call_gemini(prompt):
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{MODELS['gemini']}:generateContent")
    data = post_json(
        url,
        headers={"x-goog-api-key": KEYS["gemini"]},
        payload={"contents": [{"parts": [{"text": prompt}]}],
                 "generationConfig": {"maxOutputTokens": 300}})
    return data["candidates"][0]["content"]["parts"][0]["text"]


CALLERS = {"claude": call_claude, "gpt": call_gpt, "gemini": call_gemini}


def parse_response(text):
    """Extract construct_1..3 and rationale; tolerate minor format drift."""
    out = {"construct_1": "", "construct_2": "", "construct_3": "", "rationale": ""}
    for key in out:
        m = re.search(rf"{key}\s*:\s*(.+)", text, re.I)
        if m:
            val = m.group(1).strip().strip('"').strip()
            if val.upper() != "NONE":
                out[key] = val.lower() if key.startswith("construct") else val
    return out


def read_study_texts():
    """Return {pilot_id: {"masked": text, "unmasked": text}} for usable studies."""
    studies = {}
    for f in sorted(TEXT_DIR.glob("study_*_masked.txt")):
        pid = f.stem.replace("study_", "").replace("_masked", "")
        masked = f.read_text(encoding="utf-8")
        unmasked_path = TEXT_DIR / f"study_{pid}.txt"
        unmasked = unmasked_path.read_text(encoding="utf-8") if unmasked_path.exists() else masked
        if "ABSTRACT NOT FOUND" in masked:
            print(f"SKIPPING study {pid}: abstract still missing — paste it in "
                  f"and re-run.")
            continue
        studies[pid] = {"masked": masked, "unmasked": unmasked}
    return studies


def load_done():
    done = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add((row["pilot_id"], row["coder"], row["run"]))
    return done


def main():
    model_args = [a for a in sys.argv[1:] if a in MODELS]  # "full" is a mode, not a model
    only = model_args[0] if model_args else None
    families = [m for m in MODELS if not only or m == only]
    active = [m for m in families if KEYS[m]]
    for m in families:
        if not KEYS[m]:
            print(f"WARNING: no API key for {m} — skipping "
                  f"(set the env var and re-run; resumable).")
    if not active:
        sys.exit("No API keys found. Set at least one of ANTHROPIC_API_KEY / "
                 "OPENAI_API_KEY / GEMINI_API_KEY.")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    studies = read_study_texts()
    print(f"Studies ready: {sorted(studies)}")
    print(f"Models: {', '.join(f'{m}={MODELS[m]}' for m in active)}")

    done = load_done()
    new_sheet = not OUT_PATH.exists()
    with open(OUT_PATH, "a", newline="", encoding="utf-8") as outf, \
         open(LOG_PATH, "a", newline="", encoding="utf-8") as logf:
        out = csv.writer(outf)
        log = csv.writer(logf)
        if new_sheet:
            out.writerow(["pilot_id", "coder", "run", "masked",
                          "construct_1", "construct_2", "construct_3", "notes"])
            log.writerow(["pilot_id", "model", "run", "status", "seconds"])

        # Build the run plan: 5 masked + 1 unmasked (v1), + v1b masked on first 3
        plan = []
        for pid in sorted(studies):
            for run in range(1, MASKED_RUNS + 1):
                plan.append((pid, f"{run}", "yes", "v1"))
            plan.append((pid, "unmasked", "no", "v1"))
        for pid in sorted(studies)[:VARIANT_STUDIES]:
            plan.append((pid, "v1b", "yes", "v1b"))

        for model in active:
            print(f"\n===== {model} ({MODELS[model]}) =====")
            for pid, run, masked, variant in plan:
                if (pid, model, run) in done:
                    continue
                text = studies[pid]["masked" if masked == "yes" else "unmasked"]
                prompt = build_prompt(text, variant)
                t0 = time.time()
                try:
                    reply = CALLERS[model](prompt)
                    status = "ok"
                except Exception as e:
                    reply = ""
                    status = f"ERROR: {e}"
                    print(f"  [{pid} run {run}] {status}")
                secs = round(time.time() - t0, 1)
                log.writerow([pid, model, run, status, secs])
                logf.flush()
                if status != "ok":
                    time.sleep(2)
                    continue
                (RAW_DIR / f"{model}_study{pid}_run{run}.txt").write_text(
                    reply, encoding="utf-8")
                parsed = parse_response(reply)
                out.writerow([pid, model, run, masked,
                              parsed["construct_1"], parsed["construct_2"],
                              parsed["construct_3"], parsed["rationale"]])
                outf.flush()
                print(f"  [{pid} run {run}] {parsed['construct_1'] or '??'}")
                time.sleep(0.5)

    print(f"\nDone. LLM codings -> {OUT_PATH}; raw responses -> {RAW_DIR}/")
    print("Next: run 03_pilot_agreement.py (or merge with your hand-coding and "
          "compute agreement per coding_protocol.md).")


if __name__ == "__main__":
    main()
