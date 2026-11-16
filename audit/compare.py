"""Concept Audit - step 4b: laying the three systems' records side by side.

Agreement between systems tells us where to look. It is not evidence that
a reading is correct: systems that share a model, a prompt or a blind spot
agree on the same mistakes.

Two matching rules, both approximate and both stated in the output:
  roles   two records are about the same operation when their names share
          most of their distinguishing words ("Stroop task" and "color word
          Stroop task")
  claims  two records are about the same proposition when they quote the
          same passage as stating it
"""
import foundations as fd
import roles as roles_task

_GENERIC = set("task tasks test tests scale scales subscale questionnaire measure measures inventory the of a an for and "
               "version item items rating ratings score scores question questions".split())


def _tokens(record):
    words = set(roles_task.operation_key(record.get("operation")).split()) - _GENERIC
    return words or set(roles_task.operation_key(record.get("operation")).split())


def _same_operation(a, b):
    if not a or not b:
        return False
    return a <= b or b <= a or len(a & b) / float(len(a | b)) >= 0.5


def _role_tuples(record):
    out = set()
    for x in record.get("roles", []):
        if not x.get("valid"):
            continue
        if x["role"] == "insufficient_evidence" or x.get("evidence_located"):
            out.add((record.get("whose"), x["role"]))
    return out


def compare_roles(by_arm):
    """by_arm: {arm: result or None}. Returns clusters, agreement figures and the disagreements."""
    arms = [a for a, r in by_arm.items() if r and not r.get("problem")]
    clusters = []
    for arm in arms:
        for rec in by_arm[arm].get("records", []):
            words = _tokens(rec)
            home = next((c for c in clusters if _same_operation(words, c["words"])), None)
            if home is None:
                home = {"label": rec.get("operation"), "words": words, "by_arm": {}}
                clusters.append(home)
            home["by_arm"].setdefault(arm, []).append(rec)
    rows, agree, union, withheld_agree = [], 0, 0, 0
    for c in clusters:
        tuples = {arm: set().union(*[_role_tuples(r) for r in c["by_arm"].get(arm, [])]) if c["by_arm"].get(arm) else set()
                  for arm in arms}
        for t in sorted(set().union(*tuples.values()), key=str):
            have = [a for a in arms if t in tuples[a]]
            union += 1
            if len(have) == len(arms):
                agree += 1
                withheld_agree += int(t[1] == "insufficient_evidence")
            rows.append({"operation": c["label"], "whose": t[0], "role": t[1], "arms_with": have,
                         "arms_without": [a for a in arms if a not in have],
                         "records": {a: [r["record_id"] for r in c["by_arm"].get(a, [])] for a in arms}})
        if not set().union(*tuples.values()):
            rows.append({"operation": c["label"], "whose": None, "role": "(no supported role)",
                         "arms_with": [a for a in arms if c["by_arm"].get(a)],
                         "arms_without": [a for a in arms if not c["by_arm"].get(a)],
                         "records": {a: [r["record_id"] for r in c["by_arm"].get(a, [])] for a in arms}})
    return fd.labeled({
        "arms": arms, "operations": len(clusters), "role_statements": union, "made_by_every_arm": agree,
        "of_which_withheld": withheld_agree,
        "agreements": [r for r in rows if not r["arms_without"]],
        "disagreements": [r for r in rows if r["arms_without"]],
        "matching_rule": "operations matched across arms by shared distinguishing words in their names (approximate)",
    }, "recorded_or_computed", "unchecked", tool="inter_system_agreement")


def _statement_passage(record):
    ev = record.get("statement_evidence") or {}
    if ev.get("locators"):
        return fd.parse_locator(ev["locators"][0])["passage_id"]
    return ev.get("passage_id")


CLAIM_FIELDS = ("holder", "stance", "test", "result", "flagged")


def _claim_values(record):
    return {"holder": (record.get("whose") or {}).get("holder"), "stance": (record.get("whose") or {}).get("stance"),
            "test": "%s%s" % ((record.get("test") or {}).get("status"),
                              ", cited" if record.get("second_hand") else ""),
            "result": (record.get("results") or {}).get("status"), "flagged": bool(record.get("flag"))}


def compare_claims(by_arm):
    arms = [a for a, r in by_arm.items() if r and not r.get("problem")]
    clusters = {}
    for arm in arms:
        for rec in by_arm[arm].get("records", []):
            key = _statement_passage(rec) or rec["record_id"]
            clusters.setdefault(key, {}).setdefault(arm, rec)
    field_counts = {f: {"compared": 0, "same": 0} for f in CLAIM_FIELDS}
    rows, in_all = [], 0
    for passage, members in sorted(clusters.items()):
        have = [a for a in arms if a in members]
        row = {"stated_in": passage, "proposition": next(iter(members.values())).get("proposition"),
               "arms_with": have, "arms_without": [a for a in arms if a not in members],
               "records": {a: members[a]["record_id"] for a in have}, "differences": {}}
        if len(have) == len(arms):
            in_all += 1
            values = {a: _claim_values(members[a]) for a in have}
            for f in CLAIM_FIELDS:
                field_counts[f]["compared"] += 1
                seen = {str(values[a][f]) for a in have}
                if len(seen) == 1:
                    field_counts[f]["same"] += 1
                else:
                    row["differences"][f] = {a: values[a][f] for a in have}
        rows.append(row)
    return fd.labeled({
        "arms": arms, "propositions": len(clusters), "recorded_by_every_arm": in_all, "fields": field_counts,
        "agreements": [r for r in rows if not r["arms_without"] and not r["differences"]],
        "disagreements": [r for r in rows if r["arms_without"] or r["differences"]],
        "matching_rule": "propositions matched across arms by the passage each record quotes as stating them (approximate)",
    }, "recorded_or_computed", "unchecked", tool="inter_system_agreement")


def arm_totals(results):
    """Totals for one arm over a list of results: what it produced and what it cost."""
    t = {"documents": len(results), "read": 0, "records": 0, "usable": 0, "quotations": 0, "quotations_located": 0,
         "calls": 0, "seconds": 0.0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "cost_complete": True,
         "replayed_calls": 0}
    for r in results:
        call = r.get("call") or {}
        if r.get("problem"):
            continue
        t["read"] += 1
        c = r["counts"]
        t["records"] += c["records"]; t["usable"] += c["usable"]
        t["quotations"] += c["quotations"]; t["quotations_located"] += c["quotations_located"]
        t["calls"] += call.get("calls", 1)
        t["seconds"] += call.get("seconds") or 0
        t["tokens_in"] += call.get("tokens_in") or 0
        t["tokens_out"] += call.get("tokens_out") or 0
        if call.get("cost_usd") is None:
            t["cost_complete"] = False
        else:
            t["cost_usd"] += call["cost_usd"]
        t["replayed_calls"] += int(bool(call.get("replayed")))
    t["seconds"] = round(t["seconds"], 1)
    t["cost_usd"] = round(t["cost_usd"], 4)
    return t
