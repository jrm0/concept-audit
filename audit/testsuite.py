"""Concept Audit - step 4a: constructed tests for study roles and claim records.

Each test is a short invented document written so that the right answer
follows from how it was written. A sentence that says "time spent before
giving up was the dependent variable" makes that task an outcome, wherever
the sentence sits and whoever reads it.

Three families, from Harness_Tool_Inventory_v1.md section 8:
  invariance   something irrelevant changes; the answer must not
               (a passage moves to another section; a sentence is reworded)
  directional  something relevant changes; the answer must change with it
               (manipulated -> measured as outcome; "Smith measured" ->
               "we measured"; "we hypothesize" -> "we found")
  withholding  evidence is removed; the system must say so, not guess

Task and study names are invented or used in unfamiliar ways, so that a
model cannot pass by remembering what a task is usually for.

Passing shows a capability on constructed cases. It does not show accuracy
on real papers, and every report that cites these results must say so.
"""
import re

import foundations as fd

SUITE_VERSION = "2026-11-14.1"

VARIANTS = [
    {"dep": "letter-cancelling task", "out": "anagram task", "scale": "Brief Effort Scale",
     "other": "handgrip task", "cite": "Okafor and Lindqvist (2011)",
     "dep_re": r"letter.?cancel", "out_re": r"anagram", "scale_re": r"effort scale", "other_re": r"hand.?grip"},
    {"dep": "digit-sorting task", "out": "tracing puzzle", "scale": "State Strain Questionnaire",
     "other": "cold-water task", "cite": "Marchetti and Dube (2009)",
     "dep_re": r"digit.?sort", "out_re": r"tracing", "scale_re": r"strain", "other_re": r"cold.?water"},
]
# What the claim tests are "about", so that a model does not leave a proposition out as off-topic.
CONCEPT = "the weakening of self-control after prior exertion"
NEIGHBORS = ["persistence", "effort and exertion", "the reservoir model", "what the instruments in the document measure"]
SUBSTANTIVE = ("manipulated", "comparator", "manipulation_check", "outcome", "measured", "covariate", "applied_to_all")


def base_passages(v):
    """The invented study every test starts from, as named passages."""
    return {
        "intro1": ("INTRO", "Self-control is often said to weaken after use. %(cite)s reported that people who first "
                            "completed a demanding task gave up sooner on a later one." % v),
        "hyp": ("INTRO", "We hypothesized that participants who completed the hard version of the %(dep)s would "
                         "persist for less time on the %(out)s than participants who completed the easy version." % v),
        "sample": ("METHODS", "One hundred and twenty undergraduates took part in a single laboratory session."),
        "manip": ("METHODS", "Participants were randomly assigned to complete either a hard or an easy version of the "
                             "%(dep)s. In the hard version they followed two rules at once; in the easy version they "
                             "followed one." % v),
        "check": ("METHODS", "Afterwards they rated how effortful the task had been on the %(scale)s, which we used to "
                             "confirm that the hard version was more demanding." % v),
        "outcome": ("METHODS", "All participants then worked on the %(out)s. Time spent before giving up was the "
                               "dependent variable." % v),
        "covar": ("METHODS", "Age and gender were recorded and entered as control variables."),
        "res_check": ("RESULTS", "Participants in the hard condition rated the task as more effortful, t(118) = 4.10, "
                                 "p < .001."),
        "res_main": ("RESULTS", "They also persisted for less time on the %(out)s (M = 412 s) than those in the easy "
                                "condition (M = 530 s), t(118) = 2.45, p = .016." % v),
        "disc": ("DISCUSS", "These findings suggest that prior exertion may shorten later persistence."),
        "limit": ("DISCUSS", "We did not measure any physiological resource, so the mechanism remains open."),
    }


ORDER = ["intro1", "hyp", "sample", "manip", "check", "outcome", "covar", "res_check", "res_main", "disc", "limit"]


def build(case_id, named, order=None, level="full_text"):
    order = order or [k for k in ORDER if k in named] + [k for k in named if k not in ORDER]
    return fd.constructed_source(case_id.replace("-", ""), [named[k] for k in order], level=level), order


def pid(order, name):
    return "P%04d" % (order.index(name) + 1)


# --------------------------------------------------------------------------
# Reading a system's answer
# --------------------------------------------------------------------------
def _matches(record, pattern):
    text = "%s %s" % (record.get("operation") or "", record.get("variant") or "")
    return re.search(pattern, text, re.I) is not None


def ops(result, pattern, whose=None):
    return [r for r in result.get("records", []) if _matches(r, pattern) and (whose is None or r.get("whose") == whose)]


def roles_of(records):
    return [x for r in records for x in r.get("roles", []) if x.get("valid")]


def has_role(result, pattern, role, whose=None, position=None):
    for r in ops(result, pattern, whose):
        for x in r.get("roles", []):
            if x.get("valid") and x["role"] == role and (position is None or x.get("position") == position):
                return True
    return False


def lacks_role(result, pattern, role):
    return not has_role(result, pattern, role)


def no_substantive_role(result, pattern=None, stated_only=False):
    """True when nothing substantive is claimed for the operation (or for anything, if no pattern)."""
    records = ops(result, pattern) if pattern else result.get("records", [])
    for x in roles_of(records):
        if x["role"] in SUBSTANTIVE and (not stated_only or x.get("strength") == "stated"):
            return False
    return True


def props(result, passage=None, text=None):
    out = []
    for r in result.get("records", []):
        ev = r.get("statement_evidence") or {}
        cited = {ev.get("passage_id")} | {fd.parse_locator(l)["passage_id"] for l in ev.get("locators") or []}
        if passage is not None and passage not in cited:
            continue
        if text is not None and not re.search(text, r.get("proposition") or "", re.I):
            continue
        out.append(r)
    return out


def evidence_passages(items):
    return {e.get("passage_id") for i in items for e in i.get("evidence", [])}


def expect(description, test):
    return {"expect": description, "test": test}


# --------------------------------------------------------------------------
# The cases
# --------------------------------------------------------------------------
def role_cases(v, tag):
    cases = []
    dep, out, scale, other = (v[k] for k in ("dep_re", "out_re", "scale_re", "other_re"))
    P = base_passages(v)

    def add(cid, family, rule, named, expects, order=None, level="full_text", keep=None, note=None):
        source, order_ = build("%s%s" % (cid, tag), named, order, level)
        kept = None
        if keep is not None:
            wanted = {pid(order_, k) for k in keep}
            kept = (lambda b, wanted=wanted: b["passage_id"] in wanted)
        cases.append({"id": "%s%s" % (cid, tag), "task": "roles", "family": family, "rule": rule, "source": source,
                      "keep": kept, "expects": expects, "note": note})

    # Invariance: where a passage sits
    base = [expect("the %s is manipulated" % v["dep"], lambda r: has_role(r, dep, "manipulated", "own")),
            expect("the %s is the outcome" % v["out"], lambda r: has_role(r, out, "outcome", "own")),
            expect("the %s is a manipulation check" % v["scale"], lambda r: has_role(r, scale, "manipulation_check", "own"))]
    add("R-LOC-1", "invariance", "Location is not role", P, base)
    moved = dict(P)
    moved["outcome"] = ("DISCUSS", P["outcome"][1])
    add("R-LOC-2", "invariance", "Location is not role", moved, base,
        order=[k for k in ORDER if k != "outcome"] + ["outcome"],
        note="the sentence that names the outcome is moved to the end and labeled DISCUSS")
    add("R-LOC-3", "invariance", "Location is not role", {k: ("INTRO", t) for k, (s, t) in P.items()}, base,
        note="every passage carries the label INTRO, as in one of the real papers")
    # Invariance: rewording
    reworded = dict(P)
    reworded["manip"] = ("METHODS", "We gave each participant, at random, one of two versions of the %(dep)s: a hard "
                                    "one with two simultaneous rules, or an easy one with a single rule." % v)
    add("R-PAR-1", "invariance", "A reworded sentence keeps its role", reworded, base[:1])
    # Directional: manipulated -> measured as outcome
    as_outcome = dict(P)
    as_outcome["hyp"] = ("INTRO", "We hypothesized that people kept awake overnight would make more errors on the "
                                  "%(dep)s than people who had slept." % v)
    as_outcome["manip"] = ("METHODS", "Participants were randomly assigned to a night of normal sleep or a night "
                                      "without sleep in the laboratory.")
    as_outcome["outcome"] = ("METHODS", "All participants then completed the %(dep)s. The number of errors on it was "
                                        "the dependent variable." % v)
    for k in ("check", "res_check", "res_main", "disc"):
        del as_outcome[k]
    add("R-DIR-1", "directional", "Manipulated is not measured-as-outcome", as_outcome,
        [expect("the %s is the outcome" % v["dep"], lambda r: has_role(r, dep, "outcome", "own")),
         expect("the %s is not recorded as manipulated" % v["dep"], lambda r: lacks_role(r, dep, "manipulated"))],
        note="the same task that is manipulated in the base case is here only scored")
    add("R-DIR-2", "directional", "Manipulated is not measured-as-outcome", P,
        [expect("the %s is manipulated" % v["dep"], lambda r: has_role(r, dep, "manipulated", "own")),
         expect("the %s is not recorded as the outcome" % v["dep"], lambda r: lacks_role(r, dep, "outcome"))])
    # Directional: who did it
    cited = dict(P)
    cited["intro1"] = ("INTRO", "Self-control is often said to weaken after use. %(cite)s measured how long people "
                                "held the %(other)s as their dependent variable." % v)
    add("R-WHO-1", "directional", "Who is not what", cited,
        [expect("the %s is an outcome in the cited study" % v["other"], lambda r: has_role(r, other, "outcome", "cited")),
         expect("the %s is not recorded as the authors' own" % v["other"], lambda r: not ops(r, other, "own")),
         expect("the %s is the authors' own outcome" % v["out"], lambda r: has_role(r, out, "outcome", "own"))])
    ours = dict(P)
    ours["outcome"] = ("METHODS", "All participants then held the %(other)s. We measured how long they held it as "
                                  "our dependent variable." % v)
    ours["hyp"] = ("INTRO", "We hypothesized that the hard version of the %(dep)s would shorten the time people held "
                            "the %(other)s." % v)
    ours["res_main"] = ("RESULTS", "They also held the %(other)s for less time, t(118) = 2.45, p = .016." % v)
    add("R-WHO-2", "directional", "Who is not what", ours,
        [expect("the %s is the authors' own outcome" % v["other"], lambda r: has_role(r, other, "outcome", "own"))],
        note="the same sentence as R-WHO-1 with 'we' in place of the cited authors")
    earlier = {
        "a": ("INTRO", "Why do people sometimes choose effort for its own sake? This article reviews three accounts "
                       "and argues for the second."),
        "b": ("DISCUSS", "In line with this view, we recently showed that payment made contingent on effort in the "
                         "%(dep)s led people to choose harder versions of the %(out)s afterwards." % v),
        "c": ("DISCUSS", "We conclude that effort can acquire value, and we set out questions for future work."),
    }
    add("R-WHO-3", "directional", "Who is not what", earlier,
        [expect("the document is read as reporting no study of its own", lambda r: r.get("own_studies") == "none"),
         expect("the %s is not recorded as an own-study operation" % v["dep"], lambda r: not ops(r, dep, "own"))],
        note="an essay in which the authors mention their own earlier study; by the definitions that study is cited")
    # Withholding
    abstract = dict(P)
    abstract["abs"] = ("ABSTRACT", "Undergraduates (N = 120) were randomly assigned to a hard or an easy version of "
                                   "the %(dep)s. They also completed the %(scale)s and the %(out)s. Prior exertion "
                                   "shortened persistence." % v)
    add("R-WH-1", "withholding", "Withhold by role, not by operation", abstract,
        [expect("the %s is still recorded as manipulated" % v["dep"], lambda r: has_role(r, dep, "manipulated")),
         expect("no role is claimed for the %s" % v["scale"], lambda r: no_substantive_role(r, scale))],
        order=["abs"] + ORDER, keep=["abs", "intro1", "disc", "limit"],
        note="methods and results are removed; the abstract supports one role and is silent on another")
    add("R-WH-2", "withholding", "No evidence, no role",
        {"a": ("ABSTRACT", "Self-control failure is costly. We review theories of why it occurs and argue for a "
                           "motivational account. Implications for practice are discussed.")},
        [expect("no role is claimed for anything", lambda r: no_substantive_role(r))], level="abstract")
    named = dict(P)
    named["intro1"] = ("INTRO", P["intro1"][1] + " The %(other)s is among the most widely used tasks in this "
                                                 "literature." % v)
    add("R-WH-3", "withholding", "A name does not fix a role", named,
        [expect("no role is claimed for the %s, which is named and nothing more" % v["other"],
                lambda r: no_substantive_role(r, other))])
    followed = dict(P)
    followed["intro1"] = ("INTRO", "For their replication, %(cite)s chose a rule-switching task followed by the "
                                   "%(other)s." % v)
    add("R-WH-4", "withholding", "No outside knowledge", followed,
        [expect("no role is stated for the %s; the text says only that it followed" % v["other"],
                lambda r: no_substantive_role(r, other, stated_only=True))])
    # More than one role; applied to all; outcome named as dependent variable; comparator
    two = dict(P)
    two["covar"] = ("METHODS", "Scores on the %(scale)s were also entered as a moderator of the effect of condition "
                               "on persistence." % v)
    add("R-MUL-1", "directional", "More than one role is allowed", two,
        [expect("the %s is a manipulation check" % v["scale"], lambda r: has_role(r, scale, "manipulation_check", "own")),
         expect("the %s is also a measured moderator" % v["scale"],
                lambda r: has_role(r, scale, "measured", "own", position="moderator"))])
    everyone = dict(P)
    everyone["hyp"] = ("INTRO", "We hypothesized that five weeks of training would lengthen persistence on the "
                                "%(out)s after a tiring task." % v)
    everyone["manip"] = ("METHODS", "Participants were randomly assigned to five weeks of daily training or to keep a "
                                    "diary for five weeks. At the end, every participant in both groups first "
                                    "completed the same version of the %(dep)s to induce fatigue." % v)
    del everyone["check"], everyone["res_check"]
    everyone["res_main"] = ("RESULTS", "The training group persisted longer on the %(out)s than the diary group, "
                                       "t(118) = 2.45, p = .016." % v)
    add("R-ALL-1", "directional", "Applied to all is not manipulated", everyone,
        [expect("the %s is applied to all" % v["dep"], lambda r: has_role(r, dep, "applied_to_all", "own")),
         expect("the %s is not recorded as manipulated" % v["dep"], lambda r: lacks_role(r, dep, "manipulated")),
         expect("the training is manipulated", lambda r: has_role(r, r"train", "manipulated", "own"))])
    dv = dict(P)
    dv["outcome"] = ("METHODS", "Fatigue and persistence were the dependent variables. Fatigue was indexed by scores "
                                "on the %(scale)s; persistence by time spent on the %(out)s." % v)
    del dv["check"], dv["res_check"]
    add("R-DV-1", "directional", "A dependent variable is an outcome, not merely measured", dv,
        [expect("the %s is an outcome" % v["scale"], lambda r: has_role(r, scale, "outcome", "own")),
         expect("the %s is an outcome" % v["out"], lambda r: has_role(r, out, "outcome", "own"))])
    add("R-CMP-1", "directional", "The control version is the comparator", P,
        [expect("a comparator is recorded", lambda r: any(x["role"] == "comparator" for x in roles_of(r.get("records", []))))])
    swapped = dict(P)
    swapped["hyp"] = ("INTRO", "We hypothesized that answering a long questionnaire would raise interference on the "
                               "Stroop task.")
    swapped["manip"] = ("METHODS", "Participants were randomly assigned to complete either the long form or the short "
                                   "form of the %(scale)s." % v)
    swapped["outcome"] = ("METHODS", "All participants then completed the Stroop task. Stroop interference was the "
                                     "dependent variable.")
    del swapped["check"], swapped["res_check"]
    swapped["res_main"] = ("RESULTS", "Interference was greater after the long form, t(118) = 2.45, p = .016.")
    add("R-NAM-1", "directional", "A name does not fix a role", swapped,
        [expect("the Stroop task is the outcome", lambda r: has_role(r, r"stroop", "outcome", "own")),
         expect("the Stroop task is not recorded as manipulated", lambda r: lacks_role(r, r"stroop", "manipulated")),
         expect("the %s is manipulated" % v["scale"], lambda r: has_role(r, scale, "manipulated", "own"))],
        note="a familiar task is the outcome and a questionnaire is the manipulation")
    return cases


def claim_cases(v, tag):
    cases = []
    P = base_passages(v)

    def add(cid, family, rule, named, expects, order=None, level="full_text", keep=None, note=None):
        source, order_ = build("%s%s" % (cid, tag), named, order, level)
        kept = None
        if keep is not None:
            wanted = {pid(order_, k) for k in keep}
            kept = (lambda b, wanted=wanted: b["passage_id"] in wanted)
        made = {"id": "%s%s" % (cid, tag), "task": "claims", "family": family, "rule": rule, "source": source,
                "keep": kept, "expects": [], "note": note}
        made["expects"] = expects(order_)
        cases.append(made)

    def hyp_record(r, order):
        found = props(r, passage=pid(order, "hyp"))
        return found[0] if found else None

    add("C-STG-1", "directional", "Stages are kept apart", P, lambda o: [
        expect("the hypothesis has a record", lambda r: hyp_record(r, o) is not None),
        expect("its test is recorded as the authors' own", lambda r: (hyp_record(r, o) or {}).get("test", {}).get("status") == "present"
               and hyp_record(r, o)["test"]["whose"] == "own"),
        expect("its result is quoted from the results passage",
               lambda r: pid(o, "res_main") in evidence_passages((hyp_record(r, o) or {}).get("results", {}).get("items", []))),
        expect("the interpretation is marked hedged",
               lambda r: any(i.get("hedging") == "hedged" for i in (hyp_record(r, o) or {}).get("interpretation", {}).get("items", []))),
        expect("the stated limit is recorded as unresolved",
               lambda r: pid(o, "limit") in evidence_passages((hyp_record(r, o) or {}).get("unresolved", {}).get("items", []))),
    ])
    add("C-WH-1", "withholding", "No stage is filled from another", P, lambda o: [
        expect("the hypothesis still has a record", lambda r: hyp_record(r, o) is not None),
        expect("no result is recorded", lambda r: (hyp_record(r, o) or {"results": {"status": "present"}})["results"]["status"] != "present"),
        expect("the result is marked insufficient evidence, not 'none in this document'",
               lambda r: (hyp_record(r, o) or {}).get("results", {}).get("status") == "insufficient_evidence"),
    ], keep=[k for k in ORDER if not k.startswith("res_")], note="the results passages are removed")
    essay = {
        "hyp": ("INTRO", "We hypothesize that prior exertion shortens later persistence."),
        "b": ("INTRO", "No study has yet tested this directly. Here we set out the reasoning behind the hypothesis and "
                       "describe how it could be tested."),
        "c": ("DISCUSS", "Whether the hypothesis holds is an empirical question that remains open."),
    }
    add("C-HYP-1", "directional", "A hypothesis is not a finding", essay, lambda o: [
        expect("the hypothesis has a record", lambda r: hyp_record(r, o) is not None),
        expect("no test in this document", lambda r: (hyp_record(r, o) or {}).get("test", {}).get("status") == "none_in_document"),
        expect("no result is recorded", lambda r: (hyp_record(r, o) or {"results": {"status": "present"}})["results"]["status"] != "present"),
    ], order=["hyp", "b", "c"], note="'we hypothesize' with no test; the base case is the same claim with 'we found'")
    critic = {
        "hyp": ("INTRO", "According to the reservoir model (%(cite)s), self-control draws on a limited store that is "
                         "used up by effort." % v),
        "b": ("INTRO", "We argue that this model is mistaken. People stop trying because the task loses its value "
                       "for them, and no store runs dry."),
        "c": ("DISCUSS", "The reservoir model cannot explain why incentives restore performance."),
    }
    add("C-ATT-1", "directional", "Attribution follows the text", critic, lambda o: [
        expect("the reservoir model has a record", lambda r: hyp_record(r, o) is not None),
        expect("it is held by the cited source", lambda r: (hyp_record(r, o) or {}).get("whose", {}).get("holder") == "cited"),
        expect("the authors' stance is dispute", lambda r: (hyp_record(r, o) or {}).get("whose", {}).get("stance") == "dispute"),
    ], order=["hyp", "b", "c"], note="the authors describe a theory in order to criticize it")
    proponent = {
        "hyp": ("INTRO", "We propose the reservoir model: self-control draws on a limited store that is used up by "
                         "effort."),
        "b": ("INTRO", "%(cite)s have argued that this model is mistaken and that people stop trying because the "
                       "task loses its value." % v),
        "c": ("DISCUSS", "We maintain that a limited store best explains the pattern of findings."),
    }
    add("C-ATT-2", "directional", "Attribution follows the text", proponent, lambda o: [
        expect("the reservoir model has a record", lambda r: hyp_record(r, o) is not None),
        expect("it is held by the authors", lambda r: (hyp_record(r, o) or {}).get("whose", {}).get("holder") == "authors"),
        expect("the authors' stance is assert", lambda r: (hyp_record(r, o) or {}).get("whose", {}).get("stance") == "assert"),
    ], order=["hyp", "b", "c"], note="the same theory with proponent and critic exchanged")
    account = dict(P)
    account["disc"] = ("DISCUSS", "This supports our account.")
    add("C-INT-1", "directional", "An interpretation is not a result", account, lambda o: [
        expect("'This supports our account' is not offered as a reported result",
               lambda r: all(pid(o, "disc") not in evidence_passages(x.get("results", {}).get("items", []))
                             for x in r.get("records", []))),
        expect("the result is still quoted from the results passage",
               lambda r: pid(o, "res_main") in evidence_passages((hyp_record(r, o) or {}).get("results", {}).get("items", []))),
    ])
    theory = dict(P)
    theory["theory"] = ("INTRO", "According to the reservoir model (%(cite)s), self-control draws on a limited store "
                                 "that is used up by effort." % v)
    theory["limit"] = ("DISCUSS", "Our data do not speak to whether a limited store is involved.")
    add("C-THY-1", "directional", "A result for one proposition is not a result for another", theory, lambda o: [
        expect("the reservoir model has a record", lambda r: bool(props(r, passage=pid(o, "theory")))),
        expect("no result is recorded for the reservoir model",
               lambda r: all(x["results"]["status"] != "present" for x in props(r, passage=pid(o, "theory")))),
        expect("the tested hypothesis keeps its result",
               lambda r: (hyp_record(r, o) or {}).get("results", {}).get("status") == "present"),
    ], order=["intro1", "theory"] + ORDER[1:], note="a tested hypothesis sits beside an untested theory")
    second = dict(P)
    second["rep"] = ("INTRO", "A replication in 23 laboratories by %(cite)s found no such effect, d = 0.04." % v)
    add("C-SEC-1", "directional", "Other people's tests are second-hand", second, lambda o: [
        expect("the cited replication has a record marked second-hand",
               lambda r: any(x.get("second_hand") and (pid(o, "rep") in evidence_passages(x["results"]["items"])
                                                       or pid(o, "rep") in {e.get("passage_id") for e in x["test"]["evidence"]}
                                                       or x["statement_evidence"].get("passage_id") == pid(o, "rep"))
                             for x in r.get("records", []))),
        expect("the cited result is not attached to the authors' own test",
               lambda r: all(x.get("second_hand") for x in r.get("records", [])
                             if pid(o, "rep") in evidence_passages(x["results"]["items"]))),
    ], order=["intro1", "rep"] + ORDER[1:])
    clash = dict(P)
    clash["res_main"] = ("RESULTS", "Persistence on the %(out)s did not differ between the hard condition (M = 498 s) "
                                    "and the easy condition (M = 505 s), t(118) = 0.21, p = .83. The hypothesis was "
                                    "not supported." % v)
    clash["disc"] = ("DISCUSS", "As predicted, participants who completed the hard version persisted for less time. "
                                "The results supported our hypothesis.")
    add("C-FLG-1", "directional", "A mismatch between stages is flagged, not smoothed over", clash, lambda o: [
        expect("the hypothesis record carries a flag", lambda r: bool((hyp_record(r, o) or {}).get("flag"))),
        expect("the reported result is still the one in the results passage",
               lambda r: pid(o, "res_main") in evidence_passages((hyp_record(r, o) or {}).get("results", {}).get("items", []))),
    ], note="the Discussion says the opposite of the Results")
    distinct = dict(P)
    distinct["hyp"] = ("INTRO", "We hypothesized that the %(scale)s and the Daily Vigor Index measure different "
                                "things." % v)
    distinct["res_main"] = ("RESULTS", "Scores on the two instruments were not statistically distinct, t(118) = 0.84, "
                                       "p = .40.")
    distinct["disc"] = ("DISCUSS", "The two instruments could not be told apart in this sample.")
    for k in ("manip", "check", "outcome", "res_check", "limit"):
        del distinct[k]
    distinct["sample2"] = ("METHODS", "All participants completed both instruments in a counterbalanced order.")

    def no_equivalence(r, o):
        rec = hyp_record(r, o)
        if rec is None:
            return False
        said = " ".join([i.get("summary") or "" for i in rec["results"]["items"]] +
                        [i.get("summary") or "" for i in rec["interpretation"]["items"]])
        return re.search(r"\b(same (thing|construct)|identical|equivalent|one construct|interchangeable)\b", said, re.I) is None
    add("C-NDS-1", "withholding", "'Not distinct' is not 'the same'", distinct, lambda o: [
        expect("the hypothesis has a record", lambda r: hyp_record(r, o) is not None),
        expect("neither the result nor the interpretation is summarized as equivalence (a wording check)",
               lambda r: no_equivalence(r, o)),
    ], order=["intro1", "hyp", "sample", "sample2", "covar", "res_main", "disc"])
    add("C-WH-2", "withholding", "'None in this document' needs the whole document",
        {"hyp": ("ABSTRACT", "We hypothesized that prior exertion shortens later persistence. The design and "
                             "findings are given in the full report.")},
        lambda o: [
            expect("the hypothesis has a record", lambda r: hyp_record(r, o) is not None),
            expect("the test is marked insufficient evidence",
                   lambda r: (hyp_record(r, o) or {}).get("test", {}).get("status") == "insufficient_evidence"),
        ], order=["hyp"], level="abstract")
    return cases


def build_cases(variants=1, task=None, family=None, only=None):
    cases = []
    for n, v in enumerate(VARIANTS[:max(1, variants)]):
        tag = "" if n == 0 else "-v%d" % (n + 1)
        cases += role_cases(v, tag) + claim_cases(v, tag)
    if task:
        cases = [c for c in cases if c["task"] == task]
    if family:
        cases = [c for c in cases if c["family"] == family]
    if only:
        wanted = {x.strip() for x in only.split(",")}
        cases = [c for c in cases if c["id"] in wanted]
    return cases


def grade(case, result):
    """Apply a case's expectations to a system's checked output."""
    if result.get("problem"):
        return fd.labeled({"id": case["id"], "task": case["task"], "family": case["family"], "rule": case["rule"],
                           "outcome": "no_reply", "problem": result["problem"], "expectations": []},
                          "recorded_or_computed", "constructed_tests", tool="constructed_tests")
    checks = []
    for e in case["expects"]:
        try:
            ok = bool(e["test"](result))
        except Exception as exc:  # a malformed record fails the expectation; it does not stop the suite
            ok = False
            checks.append({"expect": e["expect"], "met": False, "error": "%s: %s" % (type(exc).__name__, exc)})
            continue
        checks.append({"expect": e["expect"], "met": ok})
    return fd.labeled({"id": case["id"], "task": case["task"], "family": case["family"], "rule": case["rule"],
                       "outcome": "pass" if all(c["met"] for c in checks) else "fail", "expectations": checks,
                       "note": case.get("note")},
                      "recorded_or_computed", "constructed_tests", tool="constructed_tests")


def summarize(graded):
    by_family, by_rule = {}, {}
    for g in graded:
        for table, key in ((by_family, "%s / %s" % (g["task"], g["family"])), (by_rule, g["rule"])):
            row = table.setdefault(key, {"cases": 0, "pass": 0, "fail": 0, "no_reply": 0})
            row["cases"] += 1
            row[g["outcome"]] += 1
    total = {"cases": len(graded), "pass": sum(g["outcome"] == "pass" for g in graded),
             "fail": sum(g["outcome"] == "fail" for g in graded),
             "no_reply": sum(g["outcome"] == "no_reply" for g in graded)}
    return {"suite_version": SUITE_VERSION, "total": total, "by_family": by_family, "by_rule": by_rule}
