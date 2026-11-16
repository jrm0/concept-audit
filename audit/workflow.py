"""Concept Audit - step 4b: the three systems that are compared.

  plain      a model reads the whole document once, with the definitions and
             clear instructions. For the comparison this model comes from a
             different family from the one used in the other two arms.
  retrieval  a model reads only the passages a keyword search picked out as
             likely to bear on the task. Cheaper; it can miss what the
             search missed, and it can never say "none in this document".
  full       a model reads the whole document, then a second pass tries to
             prove the first reading wrong: it rechecks who did what, looks
             for passages that contradict each record, and is shown the
             terms and marked statements that appear in no record.

All three reply in the same format, so their records can be laid side by
side. All three have every quotation checked. Cost and time are recorded
per arm.
"""
import json
import re

import foundations as fd
import roles as roles_task
import claims as claims_task

ARMS = ("plain", "retrieval", "full")
_ROLE_CUES = re.compile(r"\b(measur\w*|manipulat\w*|tasks?|scales?|questionnaires?|inventor\w*|assess\w*|instrument\w*|"
                        r"conditions?|assigned|randomi[sz]ed|dependent variables?|control(?:led)? (?:for|variables?)|"
                        r"participants|procedure|items?|rated|completed)\b", re.I)
_CLAIM_CUES = re.compile(r"\b(hypothes\w*|predict\w*|propos\w*|argu\w*|theor\w*|models?|accounts?|suggest\w*|support\w*|"
                         r"found|results?|significan\w*|effects?|limitation\w*|future|unclear|unresolved|mechanism\w*|"
                         r"replicat\w*|contrary|however|conclude\w*)\b", re.I)


def retrieve(source, task, share=0.4, floor_chars=9000):
    """Pick the passages a keyword search finds most likely to bear on the task.

    Deterministic: the same document and task always give the same passages.
    Passages are scored by how many cue words they hold, relative to their
    length; the best are kept until about `share` of the text is reached.
    Abstracts and titles are always kept. Returns the kept passage numbers.
    """
    cues = _ROLE_CUES if task == "roles" else _CLAIM_CUES
    body = [b for b in source["blocks"] if not fd._BACK_MATTER.match(b["section"])]
    total = sum(len(b["text"]) for b in body)
    budget = max(floor_chars, int(total * share))
    always = [b for b in body if re.match(r"^(abstract|title)", b["section"], re.I)]
    scored = []
    for b in body:
        if b in always or len(b["text"]) < 40:
            continue
        hits = len(cues.findall(b["text"]))
        if hits:
            scored.append((hits / max(1.0, len(b["text"]) / 600.0) + min(hits, 12) * 0.25, b["passage_id"], b))
    kept = {b["passage_id"] for b in always}
    used = sum(len(b["text"]) for b in always)
    for _, pid, b in sorted(scored, key=lambda x: (-x[0], x[1])):
        if used + len(b["text"]) > budget and len(kept) > len(always):
            continue
        kept.add(pid)
        used += len(b["text"])
    return sorted(kept)


def _instructions(run, prompt_id, variables):
    """The task prompt up to (not including) the document, filled in."""
    template = run.load_prompt(prompt_id)
    head = template.split("\nDOCUMENT\n")[0]
    return run.render(head, variables)


def _pointers_roles(result):
    left = result.get("possibly_left_out", {}).get("in_no_record", [])
    return "\n".join("- %s (first in %s)" % (e["phrase"], e["first_passage"]) for e in left) or "(none found)"


def _pointers_claims(result):
    left = result.get("possibly_left_out", {}).get("in_no_record", [])
    return "\n".join("- %s: %s" % (e["passage_id"], e["text"]) for e in left) or "(none found)"


def _first_reading(run, result):
    """The first pass's own reply, as the model gave it."""
    record = json.loads((run.dir / "model_calls" / ("%04d.json" % result["call"]["seq"])).read_text(encoding="utf-8"))
    data, problem = roles_task.parse_reply(record["output_text"])
    return json.dumps(data, ensure_ascii=False, indent=1) if not problem else record["output_text"]


def _add_calls(first, second):
    out = dict(second)
    for key in ("seconds", "tokens_in", "tokens_out", "cost_usd"):
        a, b = first.get(key), second.get(key)
        out[key] = None if a is None or b is None else round(a + b, 6)
    out["calls"] = 2
    out["replayed"] = bool(first.get("replayed")) and bool(second.get("replayed"))
    return out


def challenge(run, provider, model, task, source, first, concept=None, neighbors=None, prices=None, ceiling=None,
              fresh=False, keep=None):
    """The second pass of the full workflow: try to prove the first reading wrong."""
    document, sent, left_out = roles_task.render_document(source, keep)
    coverage = roles_task.coverage_statement(source, partial=keep is not None)
    whole = source["level"] == "full_text" and keep is None
    if task == "roles":
        prompt_id, module = "roles_challenge_v1", roles_task
        instructions = _instructions(run, roles_task.PROMPT_ID, {"coverage": coverage})
        pointers = _pointers_roles(first)
    else:
        prompt_id, module = "claims_challenge_v1", claims_task
        instructions = _instructions(run, claims_task.PROMPT_ID, {"coverage": coverage, "concept": concept,
                                                                  "neighbors": ", ".join(neighbors or [])})
        pointers = _pointers_claims(first)
    result = dict(first)
    result["workflow"] = "full"
    result["first_pass"] = {"counts": first.get("counts"), "call": first.get("call")}
    try:
        rec = run.call_model(provider, model, prompt_id,
                             {"task_instructions": instructions, "first_reading": _first_reading(run, first),
                              "pointers": pointers, "document": document},
                             system=module.SYSTEM, temperature=0, max_tokens=module.MAX_TOKENS + 4000,
                             purpose="%s, second pass: %s" % (task, source["source_id"]), prices=prices,
                             spend_ceiling_usd=ceiling, fresh=fresh)
    except RuntimeError as exc:
        result["second_pass_problem"] = "the second pass failed, so the first reading stands: %s" % exc
        return result
    call = roles_task.call_info(rec)
    data, problem = roles_task.parse_reply(rec["output_text"])
    if problem:
        result["second_pass_problem"] = "the second pass could not be read, so the first reading stands: " + problem
        result["call"] = _add_calls(first["call"], call)
        return result
    if task == "roles":
        checked = roles_task.validate(data, source, sent)
        summary = fd.quotation_summary([q for q in checked["quotes"] if q.get("quote")])
        result.update({
            "own_studies": checked["own_studies"], "studies": checked["studies"], "records": checked["records"],
            "reply_problems": checked["problems"], "derived": roles_task.derive(checked["records"]),
            "possibly_left_out": roles_task.unrecorded_mentions(source, sent, checked["records"]),
            "counts": {"records": len(checked["records"]),
                       "usable": sum(r["status"] == "usable" for r in checked["records"]),
                       "needs_attention": sum(r["status"] != "usable" for r in checked["records"]),
                       "cited_records": sum(r.get("whose") == "cited" for r in checked["records"]),
                       "quotations": summary["total"], "quotations_located": summary["located_in_cited_passage"],
                       "quotations_elsewhere_in_document": summary["by_outcome"].get("found_in_other_passage", 0),
                       "quotations_spacing_only": summary["by_outcome"].get("exact_ignoring_spacing", 0),
                       "quotation_outcomes": summary["by_outcome"],
                       "near_misses": sum(1 for q in checked["quotes"] if q.get("near_miss"))}})
    else:
        checked = claims_task.validate(data, source, sent, full_text=whole)
        summary = fd.quotation_summary([q for q in checked["quotes"] if q.get("quote")])
        records = checked["records"]
        result.update({
            "records": records, "reply_problems": checked["problems"],
            "possibly_left_out": claims_task.unrecorded_marked_statements(source, sent, records),
            "counts": {"records": len(records), "usable": sum(r["status"] == "usable" for r in records),
                       "tested_here": sum(1 for r in records if r.get("test", {}).get("status") == "present"
                                          and not r.get("second_hand")),
                       "second_hand": sum(1 for r in records if r.get("second_hand")),
                       "no_test_in_document": sum(1 for r in records if r.get("test", {}).get("status") == "none_in_document"),
                       "withheld": sum(1 for r in records if r.get("test", {}).get("status") == "insufficient_evidence"),
                       "flags": sum(1 for r in records if r.get("flag")),
                       "quotations": summary["total"], "quotations_located": summary["located_in_cited_passage"],
                       "quotation_outcomes": summary["by_outcome"],
                       "near_misses": sum(1 for q in checked["quotes"] if q.get("near_miss"))}})
    changes = data.get("changes") if isinstance(data.get("changes"), list) else []
    result["changes"] = fd.labeled({"items": changes}, "model_reading", "unchecked", tool="second_pass_changes")
    result["call"] = _add_calls(first["call"], call)
    return result


def run_arm(run, arm, provider, model, task, source, concept=None, neighbors=None, keep=None, prices=None,
            ceiling=None, fresh=False):
    """Run one arm on one document for one task. Returns the checked result."""
    if arm not in ARMS:
        raise ValueError("Unknown arm: %r" % (arm,))
    selected = None
    if arm == "retrieval" and source["level"] == "full_text":
        chosen = set(retrieve(source, task))
        inner = keep
        keep = (lambda b, chosen=chosen, inner=inner: b["passage_id"] in chosen and (inner is None or inner(b)))
        selected = sorted(chosen)
    if task == "roles":
        result = roles_task.extract_roles(run, provider, model, source, keep=keep, prices=prices, ceiling=ceiling,
                                          fresh=fresh)
    else:
        result = claims_task.extract_claims(run, provider, model, source, concept, neighbors or [], keep=keep,
                                            prices=prices, ceiling=ceiling, fresh=fresh)
    result["workflow"] = arm
    if selected is not None:
        result["retrieved_passages"] = selected
    if arm == "full" and not result.get("problem"):
        result = challenge(run, provider, model, task, source, result, concept, neighbors, prices, ceiling, fresh,
                           keep=keep)
    return result
