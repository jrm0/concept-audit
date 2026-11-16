"""Concept Audit - step 3: claim records.

For each proposition a document puts forward, reports or disputes about the
concept, record six stages kept strictly apart: what is proposed, whose it
is, what test addressed it, what result was reported, how the authors read
that result, and what stays unresolved. Definitions:
Concept_Audit_Task_Definitions_v1.md, Task B.

As in step 2, a model does the reading and this file checks the reply
against the definitions and every quotation against the source. Records
that fail a check are kept and marked.

The record does not judge whether a result supports a proposition. It shows
the stages side by side.
"""
import re

import foundations as fd
import roles as roles_task

HOLDER = ("authors", "cited", "not_attributed")
STANCE = ("assert", "test_without_asserting", "report", "dispute")
TEST_STATUS = ("present", "none_in_document", "insufficient_evidence")
SAID_STATUS = ("present", "none_stated")
TEST_WHOSE = ("own", "cited")
HEDGING = ("hedged", "asserted")
PROMPT_ID = "claims_v1"
MAX_TOKENS = 16000
SYSTEM = roles_task.SYSTEM


def _evidence_list(source, items, sent, quotes):
    out = []
    for item in items if isinstance(items, list) else []:
        ev = roles_task._check_evidence(source, item, sent)
        quotes.append(ev)
        out.append(ev)
    return out


def _located(evidence):
    return any(e["outcome"] in fd.QUOTE_OK for e in evidence)


def validate(data, source, sent, full_text=True):
    """Check a parsed reply against the claim-record definitions and the source.

    full_text says whether the model was given the whole document. It
    matters for one rule: "none in this document" is a finding about the
    paper and may only be said with the whole paper in hand.
    """
    sid = source["source_id"]
    sent = set(sent)
    problems, quotes, records = [], [], []
    props = data.get("propositions")
    if not isinstance(props, list):
        problems.append("the reply has no list of propositions")
        props = []
    for n, p in enumerate(props, 1):
        rec = {"record_id": "%s.C%02d" % (sid, n), "source_id": sid, "problems": [], "notes": []}
        if not isinstance(p, dict):
            rec.update({"proposition": None, "status": "needs_attention"})
            rec["problems"].append("the entry is not an object")
            records.append(fd.labeled(rec, "model_reading", "unchecked", tool="claim_records"))
            continue
        ok = True
        # Stage 1
        text = p.get("proposition")
        rec["proposition"] = text
        if not isinstance(text, str) or not text.strip():
            rec["problems"].append("no proposition stated")
            ok = False
        ev = roles_task._check_evidence(source, p.get("statement_evidence"), sent)
        quotes.append(ev)
        rec["statement_evidence"] = ev
        if ev["outcome"] not in fd.QUOTE_OK:
            rec["problems"].append("the passage said to state the proposition was not located (%s)" % ev["outcome"])
            ok = False
        # Stage 2
        holder, stance = p.get("holder"), p.get("stance")
        if holder not in HOLDER:
            rec["problems"].append("holder is %r; expected one of %s" % (holder, ", ".join(HOLDER)))
            holder, ok = None, False
        if stance not in STANCE:
            rec["problems"].append("stance is %r; expected one of %s" % (stance, ", ".join(STANCE)))
            stance, ok = None, False
        rec["whose"] = {"holder": holder, "holder_name": p.get("holder_name"), "stance": stance, "evidence": None}
        if holder == "cited" and not p.get("holder_name"):
            rec["problems"].append("held by a cited source, but the source is not named")
        if p.get("whose_evidence"):
            wev = roles_task._check_evidence(source, p.get("whose_evidence"), sent)
            quotes.append(wev)
            rec["whose"]["evidence"] = wev
            if wev["outcome"] not in fd.QUOTE_OK:
                rec["problems"].append("the quotation for whose it is was not located (%s)" % wev["outcome"])
        elif holder == "cited":
            rec["problems"].append("held by a cited source, with no quotation showing the attribution")
        # Stage 3
        test = p.get("test") if isinstance(p.get("test"), dict) else {}
        t_status = test.get("status")
        if t_status not in TEST_STATUS:
            rec["problems"].append("test status is %r" % (t_status,))
            t_status, ok = None, False
        t_ev = _evidence_list(source, test.get("evidence"), sent, quotes)
        t_whose = test.get("whose") if t_status == "present" else None
        studies = test.get("studies") if isinstance(test.get("studies"), list) else [test.get("study")]
        studies = [str(x).strip().upper() for x in studies if x not in (None, "", "null")]
        rec["test"] = {"status": t_status, "whose": t_whose, "description": test.get("description"),
                       "study_ids": [fd.study_id(sid, x) for x in studies] if t_whose == "own" else [],
                       "evidence": t_ev, "evidence_located": _located(t_ev)}
        if t_status == "present":
            if t_whose not in TEST_WHOSE:
                rec["problems"].append("a test is recorded without saying whether it is own or cited")
                ok = False
            if not _located(t_ev):
                rec["problems"].append("no quotation for the test was located")
                ok = False
        if t_status == "none_in_document" and not full_text:
            rec["problems"].append('"none in this document" was said without the full text; it should be '
                                   "insufficient_evidence")
            ok = False
        if t_status == "insufficient_evidence" and full_text:
            rec["notes"].append("the model withheld on the test although it was given the full text")
        rec["second_hand"] = t_whose == "cited"
        # Stage 4
        results = p.get("results") if isinstance(p.get("results"), dict) else {}
        r_status = results.get("status")
        if r_status not in TEST_STATUS:
            rec["problems"].append("result status is %r" % (r_status,))
            r_status, ok = None, False
        r_items = []
        for item in results.get("items") if isinstance(results.get("items"), list) else []:
            if not isinstance(item, dict):
                continue
            ev_ = _evidence_list(source, item.get("evidence"), sent, quotes)
            st = item.get("study")
            st = None if st in (None, "", "null") else str(st).strip().upper()
            r_items.append({"study_id": fd.study_id(sid, st) if (st is not None and t_whose == "own") else None,
                            "summary": item.get("summary"), "evidence": ev_, "evidence_located": _located(ev_)})
        rec["results"] = {"status": r_status, "items": r_items}
        if r_status == "present":
            if not r_items:
                rec["problems"].append("a result is said to be present but none is given")
                ok = False
            for i, item in enumerate(r_items, 1):
                if not item["evidence_located"]:
                    rec["problems"].append("no quotation for result %d was located" % i)
                    ok = False
            if t_status != "present":
                rec["problems"].append("a result is recorded for a proposition with no test recorded")
                ok = False
        if r_status == "none_in_document" and not full_text:
            rec["problems"].append('"none in this document" was said of the result without the full text')
            ok = False
        # Stages 5 and 6
        for key, label in (("interpretation", "interpretation"), ("unresolved", "unresolved point")):
            block = p.get(key) if isinstance(p.get(key), dict) else {}
            status = block.get("status")
            if status not in SAID_STATUS:
                rec["problems"].append("%s status is %r" % (key, status))
                status, ok = None, False
            items = []
            for item in block.get("items") if isinstance(block.get("items"), list) else []:
                if not isinstance(item, dict):
                    continue
                ev_ = _evidence_list(source, item.get("evidence"), sent, quotes)
                entry = {"summary": item.get("summary"), "evidence": ev_, "evidence_located": _located(ev_)}
                if key == "interpretation":
                    entry["hedging"] = item.get("hedging")
                    if entry["hedging"] not in HEDGING:
                        rec["problems"].append("an interpretation is not marked hedged or asserted")
                        entry["hedging"] = None
                items.append(entry)
            if status == "present":
                if not items:
                    rec["problems"].append("%s is said to be present but none is given" % key)
                    ok = False
                for i, item in enumerate(items, 1):
                    if not item["evidence_located"]:
                        rec["problems"].append("no quotation for %s %d was located" % (label, i))
                        ok = False
            rec[key] = {"status": status, "items": items}
        if rec["interpretation"]["status"] == "present" and r_status != "present" and t_status != "present":
            rec["notes"].append("the authors interpret a result that this record does not hold")
        # The flag is the model's own remark and never part of a stage.
        flag = p.get("flag")
        rec["flag"] = fd.labeled({"text": flag}, "model_reading", "unchecked", tool="claim_flag") \
            if isinstance(flag, str) and flag.strip() else None
        # One quotation serving two stages is worth a look: it may be a stage merged into another.
        used = {}
        for stage, evs in (("proposition", [rec["statement_evidence"]]),
                           ("result", [e for i in r_items for e in i["evidence"]]),
                           ("interpretation", [e for i in rec["interpretation"]["items"] for e in i["evidence"]])):
            for e in evs:
                if e.get("quote"):
                    used.setdefault(" ".join(e["quote"].split()), set()).add(stage)
        rec["quotations_used_for_two_stages"] = [{"quote": q, "stages": sorted(s)} for q, s in used.items() if len(s) > 1]
        rec["stages_occupied"] = ["proposed"] + (["tested"] if t_status == "present" else []) + \
            (["result reported"] if r_status == "present" else []) + \
            (["interpreted"] if rec["interpretation"]["status"] == "present" else [])
        rec["status"] = "usable" if ok else "needs_attention"
        records.append(fd.labeled(rec, "model_reading", "unchecked", tool="claim_records"))
    return {"records": records, "problems": problems, "quotes": quotes}


_MARKED = re.compile(r"\b(Hypothes[ie]s\s+\d+[a-z]?\s*[:.]|H\d+[a-z]?\s*[:.]|[Ww]e\s+(?:hypothesi[sz]ed?|propose[d]?|"
                     r"argue[d]?|predict(?:ed)?|expect(?:ed)?|suggest(?:ed)?)\s+that|"
                     r"[Aa]ccording\s+to\s+(?:the\s+)?[\w' -]{3,60}?(?:model|theory|account|hypothesis|view)|"
                     r"(?:model|theory|account|hypothesis|view)\s+(?:suggests|posits|proposes|holds|assumes|predicts|"
                     r"states|claims)\s+that)")


def unrecorded_marked_statements(source, sent, records):
    """Passages that mark a proposition ("Hypothesis 2:", "we argue that") and are cited by no record.

    The companion to roles.unrecorded_mentions: a crude net for propositions
    a model may have left out. It matches wording only. A marked statement
    may have nothing to do with the concept, so this over-collects; it
    proposes where to look and decides nothing.
    """
    sent = set(sent)
    cited = set()
    for r in records:
        ev = r.get("statement_evidence") or {}
        for loc in ev.get("locators") or []:
            cited.add(fd.parse_locator(loc)["passage_id"])
        if ev.get("passage_id"):
            cited.add(ev["passage_id"])
    found = []
    for b in source["blocks"]:
        if b["passage_id"] not in sent:
            continue
        m = _MARKED.search(b["text"])
        if m and b["passage_id"] not in cited:
            start = max(0, m.start())
            found.append({"passage_id": b["passage_id"], "marker": m.group(1).strip(),
                          "text": b["text"][start:start + 160]})
    return fd.labeled({"in_no_record": found}, "recorded_or_computed", "unchecked", tool="unrecorded_marked_statements")


def extract_claims(run, provider, model, source, concept, neighbors, keep=None, prices=None, ceiling=None, fresh=False):
    """Ask a model for the claim records of one document and check the reply."""
    document, sent, left_out = roles_task.render_document(source, keep)
    full_text = source["level"] == "full_text" and keep is None
    result = {"source_id": source["source_id"], "level": source["level"], "prompt_id": PROMPT_ID, "concept": concept,
              "passages_sent": len(sent), "passages_left_out": left_out, "call": None, "problem": None}
    try:
        rec = run.call_model(provider, model, PROMPT_ID,
                             {"concept": concept, "neighbors": ", ".join(neighbors),
                              "coverage": roles_task.coverage_statement(source, partial=keep is not None),
                              "document": document},
                             system=SYSTEM, temperature=0, max_tokens=MAX_TOKENS,
                             purpose="claim records: " + source["source_id"], prices=prices,
                             spend_ceiling_usd=ceiling, fresh=fresh)
    except RuntimeError as exc:
        result["problem"] = "the model call failed: %s" % exc
        return result
    result["call"] = roles_task.call_info(rec)
    data, problem = roles_task.parse_reply(rec["output_text"])
    if problem:
        result["problem"] = "the reply could not be read: " + problem
        return result
    checked = validate(data, source, sent, full_text=full_text)
    summary = fd.quotation_summary([q for q in checked["quotes"] if q.get("quote")])
    records = checked["records"]
    result.update({
        "records": records, "reply_problems": checked["problems"],
        "possibly_left_out": unrecorded_marked_statements(source, sent, records),
        "counts": {
            "records": len(records), "usable": sum(r["status"] == "usable" for r in records),
            "tested_here": sum(1 for r in records if r.get("test", {}).get("status") == "present"
                               and not r.get("second_hand")),
            "second_hand": sum(1 for r in records if r.get("second_hand")),
            "no_test_in_document": sum(1 for r in records if r.get("test", {}).get("status") == "none_in_document"),
            "withheld": sum(1 for r in records if r.get("test", {}).get("status") == "insufficient_evidence"),
            "flags": sum(1 for r in records if r.get("flag")),
            "quotations": summary["total"], "quotations_located": summary["located_in_cited_passage"],
            "quotation_outcomes": summary["by_outcome"],
            "near_misses": sum(1 for q in checked["quotes"] if q.get("near_miss")),
        },
    })
    return result
