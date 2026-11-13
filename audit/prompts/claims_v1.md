You are reading one research document to record what it proposes about a concept, and what became of each proposal: who holds it, what test addressed it, what result was reported, how the authors read that result, and what they leave open. Work only from the text below. Do not use anything you know from elsewhere about these studies, these theories or these authors.

THE CONCEPT
{{concept}}
Related terms that also count: {{neighbors}}

WHAT YOU HAVE
{{coverage}}
The text is divided into passages. Each begins with a line such as [P0013 | archive label: METHODS]. That line is not part of the paper. The label was assigned by an archive, is often wrong, and is never evidence of anything.

WHICH PROPOSITIONS TO RECORD
A proposition is a statement that something is the case. Record a proposition when both of these hold:
- the document itself marks it: a numbered hypothesis, a stated aim or prediction, "we argue", "we propose", a theory the document sets out in order to test or dispute it, or a conclusion; and
- it involves the concept or one of the related terms.
Record each such proposition once. Do not record background facts, definitions, or methods.

THE SIX STAGES
Fill each stage separately, from the text that belongs to it.

1. Proposition. State it in one sentence of your own, and quote the passage that states it.
2. Whose. Who holds it?
   - authors: the authors of this document put it forward as their own.
   - cited: the document attributes it to someone else. Name them as the text does.
   - not_attributed: the text presents it without saying whose it is.
   And the authors' stance toward it: assert (they claim it), test_without_asserting (they test it and do not claim it in advance), report (they describe it without taking a side), dispute (they argue against it).
3. Test. What in this document addresses the proposition?
   - present: describe the test. Say whether it is the authors' own (own) or a test by someone else that the document reports (cited).
   - none_in_document: you have the full text and it contains no test of this proposition.
   - insufficient_evidence: you were given only part of the document and cannot tell.
4. Reported result. What does the document report was found? Give numbers as printed. Use one item per study when a proposition was tested more than once. The same three statuses apply as for the test.
5. Authors' interpretation. What do the authors say the result means for the proposition? Mark each statement hedged ("may", "suggests", "might") or asserted. Use none_stated when they do not say.
6. Unresolved. What do the authors themselves say is left open, limited or unexplained? Use none_stated when they do not say.

RULES
1. Never merge stages. A hypothesis is not a result. "This supports our account" is the authors' interpretation and belongs at stage 5, not stage 4.
2. Never fill one stage from another. A stated hypothesis does not show that it was tested. If the results are not in your text, stage 1 stands and stage 4 is insufficient_evidence.
3. Attribution follows the text. A theory the authors describe in order to test or criticize belongs to its proponents, not to the authors.
4. A proposition may sit at several stages at once, and may have two results that differ.
5. Do not judge. Whether the result truly supports the proposition is not a stage and is not yours to decide.
6. "Not distinct" is not "the same". A statement that two things did not differ, or were not distinguished, is recorded as that and nothing stronger. A result that was not significant is recorded as not significant, not as evidence of no effect.
7. Evidence. Every stage you fill needs at least one quotation copied from one passage exactly, character for character, with that passage's number. Copy odd symbols, spellings and typing errors as they appear. Do not correct anything. Do not join text from two passages. Keep quotations short.
8. Flag. If two stages do not match - for example the interpretation contradicts the reported result - say so in "flag", in one sentence. Do not change any stage because of it. Otherwise "flag" is null.

OUTPUT
Reply with one JSON object and nothing else. No commentary, no code fence.

{
  "propositions": [
    {
      "proposition": "one sentence in your own words",
      "statement_evidence": {"passage_id": "P0000", "quote": "exact words"},
      "holder": "authors" | "cited" | "not_attributed",
      "holder_name": "the holder as the text names them, or null",
      "stance": "assert" | "test_without_asserting" | "report" | "dispute",
      "whose_evidence": {"passage_id": "P0000", "quote": "exact words that show whose it is or the stance"} or null,
      "test": {
        "status": "present" | "none_in_document" | "insufficient_evidence",
        "whose": "own" | "cited" | null,
        "studies": ["1", "2"] for Studies 1 and 2, ["0"] when the document reports a single study, [] otherwise,
        "description": "what was done, in one sentence, or null",
        "evidence": [{"passage_id": "P0000", "quote": "exact words"}]
      },
      "results": {
        "status": "present" | "none_in_document" | "insufficient_evidence",
        "items": [{"study": "1" | "0" | null, "summary": "what was found, numbers as printed", "evidence": [{"passage_id": "P0000", "quote": "exact words"}]}]
      },
      "interpretation": {
        "status": "present" | "none_stated",
        "items": [{"summary": "what the authors say it means", "hedging": "hedged" | "asserted", "evidence": [{"passage_id": "P0000", "quote": "exact words"}]}]
      },
      "unresolved": {
        "status": "present" | "none_stated",
        "items": [{"summary": "what is left open", "evidence": [{"passage_id": "P0000", "quote": "exact words"}]}]
      },
      "flag": "one sentence, or null"
    }
  ]
}

DOCUMENT
{{document}}
