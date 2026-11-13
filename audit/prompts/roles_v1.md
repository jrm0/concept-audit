You are reading one research document to record what part each instrument, task or procedure played in it. Work only from the text below. Do not use anything you know from elsewhere about these studies, these tasks or these authors.

WHAT YOU HAVE
{{coverage}}
The text is divided into passages. Each begins with a line such as [P0013 | archive label: METHODS]. That line is not part of the paper. The label was assigned by an archive, is often wrong, and is never evidence of anything.

DEFINITIONS

Operation. Anything the researchers had participants do, gave them, or used to score them: a task, a questionnaire, a scale, an intervention, a procedure. Name it as the paper does. List one operation per task or instrument; put what was scored from it (error rate, reaction time) in "scored".

Whose. Every record says whose study it describes.
- own: a study reported in this document.
- cited: a study by someone else, or an earlier study by the same authors, that this document describes.
- unclear: the text does not let you tell.

Roles. What part the operation played in that study.
- manipulated: the researchers varied it. Different people, or the same people at different times, got different versions. You can name two conditions that differ on it.
- comparator: the version or condition that the manipulated one is set against (the control, the baseline, the comparison).
- manipulation_check: measured to confirm that the manipulation did what was intended.
- outcome: measured as the result the study's test is about; the dependent variable.
- measured: measured, not varied by the researchers, and entered into the test as something other than the outcome. Give its position: predictor, mediator, moderator, or not_stated.
- covariate: measured and used only to adjust, describe the sample, or exclude.
- applied_to_all: given identically to every participant, so that it is neither varied nor scored as the outcome and no comparison rests on it.
- insufficient_evidence: the operation is named, but the text you have supports no role. Give the reason: text_missing (you were given only part of the document, and the part that would say is absent) or text_silent (you were given the full text and it does not say).

RULES

1. Location is not role. Decide the role from what the text says was done. A sentence about procedure counts the same in an abstract, a methods section or a discussion.
2. Who is not what. "Smith measured X" and "we measured X" give X the same role and a different owner.
3. An operation may have more than one role in a study. Give each role its own evidence.
4. Withhold by role. Record every role the text supports. Where the text supports none, use insufficient_evidence. Never fill a gap from what is generally true of a task.
5. A name does not fix a role. The same task can be a manipulation in one study and an outcome in another.
6. Every role needs evidence: at least one quotation copied from one passage exactly, character for character, with that passage's number. Copy odd symbols, spellings and typing errors as they appear. Do not correct anything. Do not join text from two passages. Keep quotations short: the clause that shows the use is enough.
7. Strength. stated: the text says how the operation was used. inferred: the text describes a design from which the use follows, without saying so.
8. Include operations from studies the document cites, when the document says what the operation was or did in that study. Mark them cited and name the cited work as the text names it.
9. If the document reports no study of its own, every operation is cited or unclear.
10. Leave out statistical methods and software (t tests, meta-analysis, analysis packages). They are not operations.

OUTPUT

Reply with one JSON object and nothing else. No commentary, no code fence.

{
  "own_studies": "one" | "several" | "none",
  "studies": [
    {"number": "1", "label": "the paper's name for the study", "passage_id": "P0000", "quote": "exact words that introduce the study"}
  ],
  "operations": [
    {
      "operation": "name as the paper gives it",
      "variant": "version or condition, or null",
      "scored": "what was scored from it, or null",
      "study": "1" for Study 1, "0" when the document reports a single study, null for a cited study,
      "whose": "own" | "cited" | "unclear",
      "cited_work": "the cited work as the text names it, or null",
      "mention": {"passage_id": "P0000", "quote": "exact words where the operation is named"},
      "roles": [
        {
          "role": "one of the role names above",
          "position": "predictor" | "mediator" | "moderator" | "not_stated" for measured; otherwise null,
          "strength": "stated" | "inferred",
          "reason": "text_missing" | "text_silent" for insufficient_evidence; otherwise null,
          "evidence": [{"passage_id": "P0000", "quote": "exact words"}]
        }
      ]
    }
  ]
}

"studies" lists only the document's own studies, and is empty when "own_studies" is "one" or "none".
For insufficient_evidence, "evidence" may be an empty list; "mention" is still required.

DOCUMENT
{{document}}
