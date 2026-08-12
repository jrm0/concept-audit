# LLM Coding Prompt (v1 — use verbatim; log any deviation)

*Use with the MASKED text files. One fresh conversation per study per run. Do not mention replication, the RP:P, or study outcomes anywhere in the conversation.*

---

You are assisting with a study-coding task in psychology. Below is the abstract of a psychology paper. Your task is to identify the central theoretical construct(s) of the study — the abstract psychological concept(s) whose behavior the study is fundamentally about.

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
[PASTE MASKED TEXT HERE]

---

## Prompt-sensitivity variant (v1b — run on 3 of the 10 studies)

Replace the first paragraph with: "Read the following psychology abstract and name the main theoretical concept or concepts the research is testing claims about." Keep the rules and format identical. If v1 and v1b modal answers diverge, flag for discussion before scaling.
