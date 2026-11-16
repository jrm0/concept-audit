{{task_instructions}}

SECOND PASS

A first reading of this document produced the records shown below under FIRST READING. Your job now is to try to prove that reading wrong, and then to give a corrected and complete set of records. Treat the first reading as a draft by someone else. Do not defend it.

Work through these checks in order.

1. Whose. For every proposition, find the words that show who holds it. A theory the authors set out in order to test or dispute it belongs to the people the text attributes it to. If the authors derive a hypothesis and predict it, their stance is assert; if they only put someone else's claim to the test, it is test_without_asserting.
2. Every statement about the result. For each proposition that has a reported result, search the whole document - abstract, results, discussion, conclusion - for every place the authors say what was found or whether the proposition was supported. Compare each statement with the reported result and with the others. If any two disagree, keep the stages as they are and say so in "flag", naming the passages.
3. Stage by stage. Check that nothing in stage 4 is the authors' opinion of the result, and that nothing in stage 5 is a number the authors reported. Check that no result is recorded for a proposition the document did not test.
4. Left out. The passages listed under MARKED IN THE TEXT state a hypothesis, an argument or a theory and are cited by no record. For each one, decide from the text whether it is a proposition about the concept. If it is, add a record. A theory that the document states and does not test gets a record with no test and no result.
5. Other people's results. If the document reports what another study found on one of these propositions, record it with the test marked cited.
6. What is left open. Check the discussion and limitations for what the authors say remains unresolved, and record it.
7. Quotations. Every quotation must be copied exactly from the passage whose number you give. Recheck each one.

Then reply with one JSON object and nothing else: the complete corrected set of records, in exactly the format given above, with one more key at the top level:

  "changes": [{"record": "the proposition concerned", "change": "what you changed, added or removed", "why": "the passage or rule that required it"}]

Use an empty list if you changed nothing. Include every record you are keeping, not only the changed ones.

FIRST READING
{{first_reading}}

MARKED IN THE TEXT AND CITED BY NO RECORD
{{pointers}}

DOCUMENT
{{document}}
