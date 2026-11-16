{{task_instructions}}

SECOND PASS

A first reading of this document produced the records shown below under FIRST READING. Your job now is to try to prove that reading wrong, and then to give a corrected and complete set of records. Treat the first reading as a draft by someone else. Do not defend it.

Work through these checks in order.

1. Whose. For every record marked own, find the sentence that shows the authors of this document carried the operation out in a study reported in this document. A study the authors describe as earlier work, their own or anyone else's, is cited. If the document reports no study of its own, no record may be own.
2. Role. For every role, look in the document for a passage that says something different about how the operation was used. A variable the document calls a dependent variable is an outcome. A task every participant received in the same form, with no comparison built on it, is applied_to_all. A task given in two versions that are compared is manipulated, and the control version is a comparator.
3. Left out. The terms listed under NAMED IN THE TEXT appear in the document and in no record. For each one, decide from the text: is it an operation in a study this document reports or describes? If the document says what it was or did, add a record. If the document only names it, add a record with insufficient_evidence. If it is not an operation at all, leave it out.
4. Cited studies. Check that operations from studies the document describes have records, marked cited, with the cited work named as the text names it.
5. Withholding. Remove any role that rests on what is generally true of a task and not on this document's words.
6. Quotations. Every quotation must be copied exactly from the passage whose number you give. Recheck each one.

Then reply with one JSON object and nothing else: the complete corrected set of records, in exactly the format given above, with one more key at the top level:

  "changes": [{"record": "the operation concerned", "change": "what you changed, added or removed", "why": "the passage or rule that required it"}]

Use an empty list if you changed nothing. Include every record you are keeping, not only the changed ones.

FIRST READING
{{first_reading}}

NAMED IN THE TEXT AND IN NO RECORD
{{pointers}}

DOCUMENT
{{document}}
