# OSC Construct-Coding Protocol (pilot, v1)

*Applies to the human coder(s) and, with the masked prompt, to the LLM coders. Corpus protocol §2 Stage A is the governing document.*

## Task

For each study, identify the **central theoretical construct(s)** — the abstract psychological concept(s) whose causal or correlational behavior the study is fundamentally about. Rank up to 3 by centrality.

## Definitions and rules

1. **Construct, not operationalization.** "Ego depletion" is a construct; "time spent squeezing a handgrip" is an operationalization. Code the construct.
2. **Construct, not effect.** If the study tests whether X influences Y, the constructs are X and Y (ranked by which the study is *about* — usually the one in the title/first sentence). "The anchoring effect" counts as a construct when the effect *is* the theoretical object.
3. **Use the authors' term** as it appears in the text, lowercased ("working memory," not "WM" or your own gloss). If the authors use two names interchangeably, record the more frequent one and note the alternate in `notes`.
4. **Up to 3, ranked.** `construct_1` = most central. Leave 2–3 blank rather than padding.
5. **No outcome knowledge.** Do not look up whether the study replicated before coding. (For LLMs, this is handled by the masked prompt; for humans, it's self-discipline — note in `notes` if you already know the study.)

## Materials

- `data/pilot_texts/study_NN.txt` — human coders use these (title + authors + abstract)
- `data/pilot_texts/study_NN_masked.txt` — LLM coders use these (abstract only)
- `data/coding_sheet_pilot.csv` — one row per (study × coder × run)

## Coding sheet fields

| Field | Value |
|---|---|
| `pilot_id` | 01–10 |
| `coder` | `jonathan`, `human2` (if available), `claude`, `gpt`, `gemini` |
| `run` | 1–5 for LLM coders (independent runs); 1 for humans |
| `masked` | `yes`/`no` — which text version was used |
| `construct_1..3` | ranked constructs, authors' terminology, lowercase |
| `notes` | alternates, ambiguities, prior familiarity with the study |

## LLM procedure (per model: Claude, GPT, Gemini)

1. Fresh conversation per study per run (no shared context between studies).
2. Paste the prompt from `llm_prompt.md` with the **masked** text. 5 runs per study.
3. Additionally: 1 run per study with the **unmasked** text (title/authors included) — this is the contamination check. If unmasked coding shifts toward famous post-hoc framings, we have measurable leakage.
4. Record every run as its own row.

## Pilot analysis (before scaling to all 100)

- **Human–LLM agreement** on `construct_1` (exact match after normalization; then relaxed match allowing synonyms — document every relaxed judgment).
- **Within-model consistency** across the 5 runs (modal answer share).
- **Cross-model agreement** on modal answers.
- **Masked vs. unmasked shift** rate.
- Decision rule: if LLM modal `construct_1` agrees with the human coder on ≥8/10 studies (exact or documented-relaxed), scale to all 100 with human spot-checks; otherwise revise the prompt/rules and re-pilot.
