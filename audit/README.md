# Concept Audit prototype — code/audit

*Started 12 November 2026. Steps 1 to 4 of five are built. Step 5, the report and the review cards, is not implemented. The task definitions the code follows are in `Concept_Audit_Task_Definitions_v1.md`, approved 12 November 2026.*

This folder holds the prototype described in `Prototype_Kickoff_Prompt_v1.md`. It builds on `code/reading_pilot.py`, which it imports and never edits. It reads the twelve prepared documents from `revision_work/2026-10-21/reading_pilot/packets/` and the 120 abstracts from the two `source_review_packet.jsonl` files. It refuses to open the reserved evaluation pool or the identity catalog.

## Commands

Run these from the Conceptual Coherence folder.

```
python3 code/audit/audit.py foundations
```
Registers the sources, proposes study boundaries, checks every passage address, tests the quotation checker on the real documents, and writes a run record. Takes a few seconds. Calls no model.

```
python3 code/audit/audit.py check-quotes --file revision_work/2026-11-12/concept_audit/definitions/worked_example_quotes.json
```
Checks a list of quotations against the sources and lists any that are not where they claim to be.

```
python3 code/audit/audit.py show E02:P0013
```
Prints one passage so you can read a quotation in place. `E02:P0013:0-120` prints characters 0 to 120 of it.

```
python3 code/audit/audit.py check-models
```
Reports, for Anthropic, OpenAI and Google, whether a key is on file, whether the service can be reached, and which models the key may use. Also reports whether Ollama is running on this computer and which open-weights models are installed; run it in Terminal on the Mac for that, because the assistant's sandbox cannot see programs on the Mac. Costs nothing and never prints a key. Keys go in `code/.secrets/api_keys.yaml` on lines beginning `anthropic:`, `openai:` and `google:`.

```
python3 code/audit/audit.py try-model --provider local --model llama3.1:8b
```
Two live calls that test the plumbing. The model copies a sentence from one passage, then copies the final sentence of a whole paper, which it can only do if it was given all of the text. The quotation checker judges both. A wrong copy is a finding about the model; a failed call is a fault in the plumbing. Works with `--provider anthropic`, `openai` or `google` too, where it also gives a first measured cost.

```
python3 code/audit/audit.py roles --provider local --model qwen3.6:35b --docs E02,E05
```
Step 2. A model reads each document and records what part every task, scale or intervention played, and in whose study. The reply is checked against the definitions, and every quotation in it is checked against the source. Nothing the model says is dropped: a record that breaks a rule, or whose quotations cannot be found, is kept and marked "needs attention". Leave out `--docs` to read the eight report documents. Read the result in `ROLES_SUMMARY.md` in the run folder.

```
python3 code/audit/audit.py claims --provider local --model qwen3.6:35b --docs E02,E06
```
Step 3. A model records each proposition the document makes about the concept, in six stages that are never merged: what is proposed, whose it is, the test, the reported result, the authors' interpretation, and what stays unresolved. The reply and its quotations are checked as in step 2. The checks enforce the rule that a hypothesis is never counted as a finding: a result recorded where no test is recorded is flagged. Read the result in `CLAIMS_SUMMARY.md` in the run folder. The concept and its related terms come from `config.json`.

Both commands also print a short list of what the model may have left out: named tasks that appear in no role record, and sentences marked as hypotheses or arguments that no claim record cites. These lists are word matches. They over-collect on purpose and only say where to look.

```
python3 code/audit/audit.py tests --provider local --model qwen3.6:35b
```
Step 4a. Runs 29 constructed tests against one system. Each test is a short invented document whose right answer follows from how it was written: a sentence that calls something "the dependent variable" makes it an outcome wherever the sentence sits. The answers are graded by rule, not by a model. There are three families. Invariance: something irrelevant changes (a passage moves to another section) and the answer must not. Directional: something relevant changes ("Smith measured" becomes "we measured") and the answer must follow. Withholding: evidence is removed and the system must say so. Add `--workflow full` to test the reading together with its second pass, `--only R-LOC-1,C-FLG-1` to run named cases, `--variants 2` for a second set of invented names. Read the result in `TESTS.md`. Passing shows a capability on constructed cases; it does not show accuracy on real papers.

```
python3 code/audit/audit.py compare --docs E02,E06
```
Step 4b. Runs three systems on the same documents and lays their records side by side, with the time, tokens and cost of each. Plain: a model from a different family reads the whole document once. Retrieval: a model reads only the passages a keyword search picked (about 40% of the text). Full: a model reads the whole document, and a second pass then tries to prove the first reading wrong, rechecking who did what, looking for passages that contradict each record, and working through the lists of what may have been left out. The models for each arm are set in `config.json` under "comparison", or with `--plain`, `--retrieval` and `--full` (for example `--plain local:llama3.1:8b`). Read the result in `COMPARISON.md`. Agreement between the systems shows where to look. It is not evidence that a reading is correct.

Some open models "think" before they answer, which takes time and uses up part of the reply allowance. For a local model, add `--thinking off` to skip that, or `--thinking on` to require it; the default leaves the choice to the model. The run record says which was used.

A reply is saved the first time it is received. Asking the same model the same thing again reuses the saved reply, so reruns while debugging cost nothing and take no time. Add `--fresh` to ask again. Saved replies live in `revision_work/2026-11-12/concept_audit/reply_cache/`.

```
python3 -m unittest discover -s code/audit -p 'test_*.py' -v
```
Runs the step 1 tests.

## What each file does

| File | In plain terms |
|---|---|
| `foundations.py` | Gives every document and study an identifier. Gives every stretch of text an address. Checks whether a quotation is really in the source. Attaches the two labels to every output. |
| `runlog.py` | Keeps the record. Each run gets a folder that is never overwritten, with a manifest, a log of events, and one file for every model call. |
| `providers.py` | The ways of reaching a model: Anthropic, OpenAI, Google, an open-weights model running on this computer through Ollama, and a stand-in that is not a model and is used to test the record-keeping. None of the real ones has yet been run against a live service. |
| `roles.py` | Step 2. Shows a document to a model passage by passage, reads the reply, checks each record against the approved definitions and each quotation against the source, and works out which operations are cited only and which hold more than one role. |
| `claims.py` | Step 3. The same for claim records: asks for the six stages of each proposition, checks them against the definitions and the source, and marks tests reported second-hand. |
| `testsuite.py` | Step 4a. The constructed test documents, what each must and must not produce, and the grader. |
| `workflow.py` | Step 4b. The three systems: plain reading, reading of retrieved passages, and the full workflow with its second pass. |
| `compare.py` | Step 4b. Matches the three systems' records and counts where they agree and differ. |
| `audit.py` | The commands above. |
| `test_foundations.py` | Tests of the code itself, for steps 1 to 4, including checks that the step 4 grader passes a right answer and fails a wrong one. These are separate from the constructed tests in `testsuite.py`, which test a model's reading. |
| `config.json` | Where runs are written; which models to use (not set); prices (not set); a spending ceiling (not set). |
| `prompts/` | One file per prompt. A run records a fingerprint of the exact wording it used. |

## What a run record holds

Runs are written to `revision_work/2026-11-12/concept_audit/runs/`, one folder each.

- `manifest.json`: the command, the time, the version of this code, a fingerprint of every code file and every input, the prompts and models used, token counts and cost.
- `events.jsonl`: what happened, in order.
- `model_calls/0001.json` and so on: for each call, the prompt exactly as sent, the model asked for, the model that answered, the settings, the complete raw reply, token counts, the time taken, and the cost. A failed call is recorded too.
- The outputs of the command.

Cost is tokens multiplied by a price entered in `config.json`. If no price has been entered the record says "not recorded". It never guesses.

## The two labels

Every output record carries a basis and a check status, and the code refuses to write one without them.

- Basis: recorded or computed · a model's reading · a researcher's judgment.
- Checked: unchecked · constructed tests · source-reviewed · held-out evaluation.

Everything step 1 produces is "recorded or computed". Every role record from step 2 and claim record from step 3 is "a model's reading" and "unchecked"; the quotation outcomes inside it are "recorded or computed". The quotation checker and the address check are labeled "constructed tests" only when the test suite passes in the same run. The study boundaries are labeled "unchecked": they are pattern matches on headings.

## The quotation checker's answers

| Answer | Meaning |
|---|---|
| exact | The words are in the passage cited, character for character. |
| exact_after_normalization | The same, once curly quotes, dashes and spacing are tidied. Letters and digits are never changed. |
| exact_ignoring_spacing | Every letter, digit and punctuation mark matches in order; only spaces or line breaks differ. Common with text taken from PDFs. |
| exact_with_elision | The quotation contains "..." (in the middle or at an end) and every part is in the passage cited, in order. |
| found_in_other_passage | The words are in the document, not in the passage cited. |
| not_found | The words are not in the document. |

Only the first four count as located. Each is counted separately, so the stricter figure can always be read off.

## Not built

The report and the review cards. The `roles` and `claims` commands have read real papers with a local model. The `tests` and `compare` commands have been tested with prepared replies only; no live model has yet been through them. Model access is not yet confirmed: run `check-models` once the keys are in place.
