"""Concept Audit prototype - command line.

Run from the Conceptual Coherence folder:

  python3 code/audit/audit.py foundations
      Step 1. Registers the 12 documents and the 120 abstracts, proposes
      study boundaries, checks every passage address, tests the quotation
      checker on the real sources, and writes a run record.

  python3 code/audit/audit.py check-quotes --file quotes.json
      Checks a list of quotations: [{"source_id": "E02", "passage_id":
      "P0006", "quote": "..."}]. Writes the results to a run folder.

  python3 code/audit/audit.py show E02:P0006:0-200
      Prints the text at an address, so a quotation can be read in place.

  python3 code/audit/audit.py check-models
      For Anthropic, OpenAI and Google: is a key on file, can the service be
      reached, and which models may the key use? Also: is Ollama running on
      this computer, and which open-weights models are installed? Costs
      nothing. Never prints a key.

  python3 code/audit/audit.py try-model --provider local --model llama3.1:8b
      Two live calls that test the plumbing: copy a sentence from a passage,
      then copy the final sentence of a whole paper (which shows the model
      was given all of it). Works with any provider.

  python3 code/audit/audit.py roles --provider local --model qwen3.6:35b --docs E02,E05
      Step 2. A model reads each document and records what part every task,
      scale or intervention played and in whose study. Each reply is checked
      against the approved definitions and every quotation against the
      source. Leave out --docs to read the eight report documents. A reply
      already saved for the same prompt and model is reused at no cost; add
      --fresh to ask again.

  python3 code/audit/audit.py claims --provider local --model qwen3.6:35b --docs E02,E06
      Step 3. A model records each proposition the document makes about the
      concept, in six stages kept apart: what is proposed, whose it is, the
      test, the reported result, the authors' interpretation, and what
      stays unresolved. Checked the same way as the roles.

Steps 4 and 5 (the constructed tests and three-way comparison, the report)
are not implemented.
"""
import argparse
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
if str(AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIT_DIR))
import foundations as fd  # noqa: E402
import providers  # noqa: E402
import roles as roles_task  # noqa: E402
import claims as claims_task  # noqa: E402
from runlog import Run, ROOT, sha256_file  # noqa: E402


def load_config():
    return json.loads((AUDIT_DIR / "config.json").read_text(encoding="utf-8"))


def run_unit_tests():
    """Run the step 1 tests and return a plain summary."""
    suite = unittest.defaultTestLoader.discover(str(AUDIT_DIR), pattern="test_foundations.py")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    return {"suite": "code/audit/test_foundations.py", "sha256": sha256_file(AUDIT_DIR / "test_foundations.py"),
            "run": result.testsRun, "failed": len(result.failures), "errors": len(result.errors),
            "skipped": len(result.skipped), "passed": result.wasSuccessful(), "log": stream.getvalue()}


def _pick(seed, key, n):
    """A repeatable pseudo-random number in [0, n): the same seed gives the same choice."""
    return int(hashlib.sha256((seed + "|" + key).encode()).hexdigest(), 16) % n


def quotation_selfcheck(sources, seed, per_source=5, check="unchecked"):
    """Test the quotation checker on the real sources, by construction.

    For passages chosen by a fixed rule, take a stretch of the passage and
    ask the checker four questions whose right answers are known in advance:
      1. the stretch as written, citing the right passage  -> exact
      2. the same with doubled spaces and curly apostrophes -> exact after tidying
      3. the same with one long word spelled backwards      -> must NOT be located
      4. the stretch as written, citing a different passage -> found elsewhere
    """
    cases = []
    for sid in sorted(sources):
        src = sources[sid]
        body = [b for b in src["blocks"] if len(b["text"]) >= 260 and
                not fd._BACK_MATTER.match(b["section"])] or [b for b in src["blocks"] if len(b["text"]) >= 120]
        if not body:
            continue
        chosen = []
        for k in range(min(per_source, len(body))):
            b = body[_pick(seed, "%s|%d" % (sid, k), len(body))]
            if b["passage_id"] not in [c["passage_id"] for c in chosen]:
                chosen.append(b)
        for b in chosen:
            text = b["text"]
            start = _pick(seed, sid + b["passage_id"], max(1, len(text) - 150))
            quote = text[start:start + 140].strip()
            if len(quote) < 40:
                continue
            words = [w for w in quote.split() if len(w) >= 7 and w.isalpha() and w != w[::-1]]
            others = [o for o in src["blocks"] if o["passage_id"] != b["passage_id"] and quote not in o["text"]]
            tests = [("as_written", quote, b["passage_id"], ("exact",))]
            tidy = quote.replace(" ", "  ", 1).replace("'", "’")
            if tidy != quote:
                tests.append(("typography_changed", tidy, b["passage_id"], ("exact_after_normalization",)))
            if words:
                w = words[_pick(seed, quote, len(words))]
                altered = quote.replace(w, w[::-1], 1)
                tests.append(("one_word_altered", altered, b["passage_id"], ("not_found",)))
            if others:
                wrong = others[_pick(seed, "wrong" + quote, len(others))]["passage_id"]
                tests.append(("wrong_passage_cited", quote, wrong, ("found_in_other_passage",)))
            for name, q, pid, expected in tests:
                got = fd.check_quote(src, q, pid, check=check)
                cases.append({"source_id": sid, "test": name, "cited_passage": pid,
                              "expected": list(expected), "observed": got["outcome"],
                              "ok": got["outcome"] in expected, "locators": got["locators"]})
    by_test = {}
    for c in cases:
        t = by_test.setdefault(c["test"], {"cases": 0, "as_expected": 0})
        t["cases"] += 1
        t["as_expected"] += int(c["ok"])
    return {"seed": seed, "cases": len(cases), "as_expected": sum(c["ok"] for c in cases),
            "by_test": by_test, "failures": [c for c in cases if not c["ok"]], "all_cases": cases}


def cmd_foundations(args):
    config = load_config()
    run = Run("foundations", runs_dir=ROOT / config["runs_dir"], config=config)
    try:
        tests = run_unit_tests()
        run.write_output("test_results.json", tests)
        check = "constructed_tests" if tests["passed"] else "unchecked"
        full = fd.load_full_text_sources(ROOT / config["pilot_dir"])
        abstracts = fd.load_abstract_sources()
        sources = dict(full)
        sources.update(abstracts)
        for sid, s in sources.items():
            run.add_input("source:" + sid, s["text_sha256"])
        for name in ("selection_manifest.json", "source_lock.json"):
            run.add_input(name, sha256_file(ROOT / config["pilot_dir"] / name))
        for case, path in fd.ABSTRACT_PACKETS.items():
            run.add_input("abstract_packet:" + case, sha256_file(path))

        registry = []
        for sid in sorted(sources):
            entry = fd.registry_entry(sources[sid])
            entry["quality_notes"] = fd.source_quality_notes(sources[sid])
            registry.append(entry)
        run.write_output("source_registry.json", registry)

        studies = [c for sid in sorted(full) for c in fd.study_candidates(full[sid])]
        run.write_output("study_candidates.json", studies)

        # Every passage address must lead back to exactly the recorded text.
        integrity = {"sources": len(sources), "passages": 0, "addresses_resolved": 0, "failures": []}
        for sid, s in sources.items():
            for b in s["blocks"]:
                integrity["passages"] += 1
                try:
                    whole = fd.resolve_locator(fd.make_locator(sid, b["passage_id"]), sources)
                    part = fd.resolve_locator(fd.make_locator(sid, b["passage_id"], 0, len(b["text"])), sources) \
                        if b["text"] else ""
                    if whole == b["text"] and part == b["text"]:
                        integrity["addresses_resolved"] += 1
                    else:
                        integrity["failures"].append(sid + ":" + b["passage_id"])
                except Exception as exc:
                    integrity["failures"].append("%s:%s (%s)" % (sid, b["passage_id"], exc))
        run.write_output("integrity.json", fd.labeled(integrity, "recorded_or_computed", check, tool="locators"))

        seed = config["selfcheck_seed"]
        self_full = quotation_selfcheck(full, seed, per_source=5, check=check)
        picked = sorted(abstracts)
        picked = [picked[i] for i in sorted({_pick(seed, "abs%d" % k, len(picked)) for k in range(24)})]
        self_abs = quotation_selfcheck({k: abstracts[k] for k in picked}, seed, per_source=1, check=check)
        # A sentence from one paper must never be "found" in another.
        cross = []
        ids = sorted(full)
        for i, sid in enumerate(ids):
            other = full[ids[(i + 1) % len(ids)]]
            donor = [b for b in full[sid]["blocks"] if len(b["text"]) >= 260]
            b = donor[_pick(seed, "cross" + sid, len(donor))]
            q = b["text"][40:200]
            got = fd.check_quote(other, q, None, check=check)
            cross.append({"quote_from": sid, "checked_against": other["source_id"],
                          "expected": "not_found", "observed": got["outcome"], "ok": got["outcome"] == "not_found"})
        selfcheck = fd.labeled({
            "full_text": self_full, "abstracts": self_abs,
            "cross_document": {"cases": len(cross), "as_expected": sum(c["ok"] for c in cross), "all_cases": cross},
        }, "recorded_or_computed", check, tool="quotation_check")
        run.write_output("quotation_selfcheck.json", selfcheck)

        # Exercise the model-call record with a stand-in (not a model).
        line = "record-keeping check " + run.run_id
        rec = run.call_model(providers.StandInProvider(rule=lambda p: p.strip().splitlines()[-1]),
                             model="stand-in", prompt_id="selfcheck_echo_v1", variables={"line": line},
                             purpose="test that prompt, settings and raw reply are saved")
        record_ok = rec["output_text"] == line and (run.dir / "model_calls/0001.json").exists()

        n_fail = (self_full["cases"] - self_full["as_expected"]) + (self_abs["cases"] - self_abs["as_expected"]) \
            + sum(not c["ok"] for c in cross) + len(integrity["failures"])
        headed = [c for c in studies if c["status"] == "heading"]
        summary = [
            "# Concept Audit - step 1 foundations run",
            "",
            "Run: `%s`  " % run.run_id,
            "Harness version: %s. No model was called; the one recorded call went to a stand-in." % run.manifest["harness_version"],
            "",
            "Every figure below is recorded or computed. How far checked: %s, except the study boundaries, which are unchecked." % fd.CHECK[check],
            "",
            "| What | Result |",
            "|---|---|",
            "| Code tests (all steps built so far) | %d run, %d failed, %d errors, %d skipped |" % (tests["run"], tests["failed"], tests["errors"], tests["skipped"]),
            "| Full-text documents registered, text matching the locked version | %d of %d |" % (len(full), len(full)),
            "| Abstract-level records registered | %d |" % len(abstracts),
            "| Passages whose address leads back to the recorded text | %d of %d |" % (integrity["addresses_resolved"], integrity["passages"]),
            "| Documents with study headings found | %d of %d (%d candidate studies) |" % (len({c["source_id"] for c in headed}), len(full), len(headed)),
            "| Quotation checker, full texts: answers as expected | %d of %d |" % (self_full["as_expected"], self_full["cases"]),
            "| Quotation checker, abstracts: answers as expected | %d of %d |" % (self_abs["as_expected"], self_abs["cases"]),
            "| A sentence from one paper looked for in another: correctly not found | %d of %d |" % (sum(c["ok"] for c in cross), len(cross)),
            "| Model-call record written and complete (stand-in) | %s |" % ("yes" if record_ok else "NO"),
            "",
            "Quotation checker by kind of test (full texts):",
            "",
        ] + ["- %s: %d of %d as expected" % (k, v["as_expected"], v["cases"]) for k, v in sorted(self_full["by_test"].items())] + [
            "",
            "Study roles and claim records are built (the `roles` and `claims` commands). The constructed tests, the three-way comparison and the report: not implemented.",
            "Accuracy of the study boundaries: not assessed. They are pattern matches on headings.",
            "",
            "Files in this folder: `manifest.json` (the run record), `events.jsonl`, `source_registry.json`, "
            "`study_candidates.json`, `integrity.json`, `quotation_selfcheck.json`, `test_results.json`, `model_calls/0001.json`.",
        ]
        run.write_output("FOUNDATIONS_SUMMARY.md", "\n".join(summary) + "\n")
        ok = tests["passed"] and n_fail == 0 and record_ok
        run.finish("completed" if ok else "completed_with_failures")
        print("\n".join(summary))
        print("\nRun folder: %s" % run.dir.relative_to(ROOT))
        return 0 if ok else 1
    except Exception:
        run.finish("failed")
        raise


def cmd_check_quotes(args):
    config = load_config()
    items = json.loads(Path(args.file).read_text(encoding="utf-8"))
    run = Run("check-quotes", runs_dir=ROOT / config["runs_dir"], config=config)
    try:
        run.add_input("quotes_file", sha256_file(args.file))
        sources = fd.load_full_text_sources(ROOT / config["pilot_dir"])
        sources.update(fd.load_abstract_sources())
        results = []
        for item in items:
            if item["source_id"] not in sources:
                results.append(fd.labeled({"source_id": item["source_id"], "quote": item.get("quote"),
                                           "outcome": "unknown_source", "locators": []},
                                          "recorded_or_computed", "constructed_tests", tool="quotation_check"))
                continue
            r = fd.check_quote(sources[item["source_id"]], item.get("quote", ""), item.get("passage_id"),
                               check="constructed_tests")
            if item.get("id"):
                r["id"] = item["id"]
            results.append(r)
        summary = fd.quotation_summary(results)
        run.write_output("quotation_results.json", {"summary": {k: v for k, v in summary.items() if k != "mismatches"},
                                                    "results": results})
        run.finish("completed" if not summary["mismatches"] else "completed_with_failures")
        print("Quotations located in the passage cited: %d of %d" % (summary["located_in_cited_passage"], summary["total"]))
        for r in summary["mismatches"]:
            print("  MISMATCH %s %s %s: %s | %s" % (r.get("id", ""), r["source_id"], r.get("cited_passage"), r["outcome"],
                                                   (r.get("quote") or "")[:70]))
        print("Run folder: %s" % run.dir.relative_to(ROOT))
        return 0 if not summary["mismatches"] else 1
    except Exception:
        run.finish("failed")
        raise


def cmd_show(args):
    config = load_config()
    sources = fd.load_full_text_sources(ROOT / config["pilot_dir"])
    sources.update(fd.load_abstract_sources())
    loc = fd.parse_locator(args.locator)
    block = fd.get_block(sources[loc["source_id"]], loc["passage_id"])
    print("%s | %s | section label: %s" % (sources[loc["source_id"]]["title"], args.locator, block["section"]))
    print(fd.resolve_locator(args.locator, sources))
    return 0


def cmd_check_models(args):
    """Is each provider's key on file, is the service reachable, which models are offered?"""
    config = load_config()
    run = Run("check-models", runs_dir=ROOT / config["runs_dir"], config=config)
    report, ok = [], True
    for name in providers.CHECKED_PROVIDERS:
        local = name == "local"
        entry = {"provider": name, "key_on_file": None if local else False, "reachable": None, "models": [],
                 "problem": None}
        try:
            provider = providers.PROVIDERS[name]()
            entry["key_on_file"] = None if local else True
            entry["models"] = provider.list_models()
            entry["reachable"] = True
        except providers.ProviderError as exc:
            entry["problem"] = str(exc)
            if entry["key_on_file"] or local:
                entry["reachable"] = not str(exc).startswith("Could not reach")
            ok = ok and local  # a missing local model server is reported, not counted as a failure
        report.append(fd.labeled(entry, "recorded_or_computed", "unchecked", tool="check_models"))
        if entry["reachable"] and local:
            print("%-10s Ollama is running on this computer, %d models installed:" % (name, len(entry["models"])))
            for m in entry["models"]:
                print("             " + m)
        elif local:
            print("%-10s Ollama did not answer on this computer (%s)" % (name, entry["problem"]))
        elif entry["models"]:
            print("%-10s key on file, service reached, %d models offered:" % (name, len(entry["models"])))
            for m in entry["models"]:
                print("             " + m)
        elif not entry["key_on_file"]:
            print("%-10s no key on file" % name)
        else:
            print("%-10s key on file, but: %s" % (name, entry["problem"]))
    run.write_output("model_access.json", report)
    run.finish("completed" if ok else "completed_with_failures")
    print("Run folder: %s" % run.dir.relative_to(ROOT))
    return 0 if ok else 1


THINKING = {"default": None, "off": False, "on": True}


def _make_provider(args):
    """Build the provider a command asked for. --thinking applies to local models only."""
    if args.provider == "local":
        return providers.LocalProvider(think=THINKING[getattr(args, "thinking", "default")])
    return providers.PROVIDERS[args.provider]()


def _start_model_run(command, args):
    """Shared set-up for commands that call a model: run folder, replay cache, prices, sources."""
    config = load_config()
    run = Run(command, runs_dir=ROOT / config["runs_dir"], config=config)
    run.cache_dir = ROOT / config["cache_dir"]
    prices = {k: v for k, v in config.get("prices_usd_per_million_tokens", {}).items() if isinstance(v, dict)}
    definitions = ROOT / config["definitions"]["file"]
    run.add_input("definitions:" + definitions.name, sha256_file(definitions))
    sources = fd.load_full_text_sources(ROOT / config["pilot_dir"])
    sources.update(fd.load_abstract_sources())
    return config, run, prices, sources


def _roles_table(result):
    """A plain table of one document's role records for the summary file."""
    lines = ["| Record | Operation | Whose | Study | Roles | Evidence located | Status |", "|---|---|---|---|---|---|---|"]
    for r in result.get("records", []):
        parts = []
        for role in r["roles"]:
            label = str(role["role"])
            if role["role"] == "measured" and role.get("position"):
                label += " (%s)" % role["position"]
            if role["role"] == "insufficient_evidence" and role.get("reason"):
                label += " (%s)" % role["reason"]
            if role.get("strength") == "inferred":
                label += ", inferred"
            if not role["valid"]:
                label += " [not valid]"
            parts.append(label)
        n_ev = sum(len(role["evidence"]) for role in r["roles"])
        n_ok = sum(1 for role in r["roles"] for e in role["evidence"] if e["outcome"] in fd.QUOTE_OK)
        whose = r.get("whose") or ""
        if whose == "cited" and r.get("cited_work"):
            whose += ": " + str(r["cited_work"])
        name = str(r.get("operation") or "(unnamed)") + (" - " + str(r["variant"]) if r.get("variant") else "")
        lines.append("| %s | %s | %s | %s | %s | %d of %d | %s |" % (
            r["record_id"], name.replace("|", "/"), whose.replace("|", "/"), r.get("study_id") or "",
            "; ".join(parts) or "(none)", n_ok, n_ev, r["status"].replace("_", " ")))
    return lines


def cmd_roles(args):
    """Step 2. Ask a model for the role records of each document and check every reply."""
    config, run, prices, sources = _start_model_run("roles", args)
    try:
        provider = _make_provider(args)
    except providers.ProviderError as exc:
        run.note("Could not start: %s" % exc)
        run.finish("failed")
        print("Could not start: %s" % exc)
        return 1
    wanted = [d.strip() for d in args.docs.split(",")] if args.docs else config["sample"]["report"]
    unknown = [d for d in wanted if d not in sources]
    if unknown:
        run.finish("failed")
        print("Unknown document(s): %s" % ", ".join(unknown))
        return 1
    try:
        (run.dir / "roles").mkdir()
        totals = {"documents": 0, "read": 0, "records": 0, "usable": 0, "quotations": 0, "located": 0,
                  "near_misses": 0, "seconds": 0.0}
        md = ["# Study roles - run %s" % run.run_id, "",
              "Model: %s / %s (thinking: %s). Prompt: %s. Definitions: %s." % (
                  args.provider, args.model, args.thinking, roles_task.PROMPT_ID, config["definitions"]["file"]),
              "",
              "Every role record below is a model's reading and is unchecked. The counts of quotations located "
              "are recorded or computed. A record \"needs attention\" when it breaks a rule of the definitions "
              "or none of its quotations could be found where it said.", ""]
        for doc in wanted:
            print("reading %s ..." % doc, flush=True)
            result = roles_task.extract_roles(run, provider, args.model, sources[doc], prices=prices,
                                              ceiling=config.get("spend_ceiling_usd_per_run"), fresh=args.fresh)
            run.write_output("roles/%s.json" % doc, result)
            totals["documents"] += 1
            md += ["## %s - %s" % (doc, sources[doc]["title"]), ""]
            if result["problem"]:
                md += ["**No records.** %s" % result["problem"], ""]
                print("  %s" % result["problem"])
                continue
            c, call = result["counts"], result["call"]
            totals["read"] += 1
            totals["records"] += c["records"]; totals["usable"] += c["usable"]
            totals["quotations"] += c["quotations"]; totals["located"] += c["quotations_located"]
            totals["near_misses"] += c["near_misses"]; totals["seconds"] += call["seconds"] or 0
            line = ("%d records (%d from cited studies), %d usable; quotations: %d of %d located where cited "
                    "(%d of those differ only in spacing), %d elsewhere in the paper, %d not found; "
                    "%d named terms in no record; %s") % (
                c["records"], c["cited_records"], c["usable"], c["quotations_located"], c["quotations"],
                c["quotations_spacing_only"], c["quotations_elsewhere_in_document"],
                c["quotations"] - c["quotations_located"] - c["quotations_elsewhere_in_document"],
                len(result["possibly_left_out"]["in_no_record"]),
                "replayed from a saved reply" if call["replayed"] else "%.0f seconds" % (call["seconds"] or 0))
            print("  " + line)
            if result["reply_problems"]:
                print("  reply: " + "; ".join(result["reply_problems"]))
            md += [line + ".", "",
                   "Own studies, as the model read it: %s. Step 1 heading match: %s." % (
                       result["own_studies"], ", ".join(result["step1_study_candidates"]) or "none"), ""]
            md += _roles_table(result) + [""]
            d = result["derived"]
            if d["cited_only"]:
                md += ["Cited only: " + ", ".join(d["cited_only"]) + ".", ""]
            for item in d["several_roles"]:
                md += ["Same operation, more than one role: %s - %s." % (item["operation"], "; ".join(
                    "%s (%s)" % (x["role"], x["whose"]) for x in item["roles"])), ""]
            left = result["possibly_left_out"]["in_no_record"]
            if left:
                md += ["Named in the text and in no record (a word match that over-collects; a pointer for a "
                       "reader, recorded or computed, unchecked): " + "; ".join(
                           "%s (%s%s)" % (e["phrase"], e["first_passage"], ", %d times" % e["times"] if e["times"] > 1 else "")
                           for e in left) + ".", ""]
            flagged = [r for r in result["records"] if r["problems"]]
            if flagged or result["reply_problems"]:
                md += ["Problems found by the checks:", ""]
                md += ["- reply: %s" % p for p in result["reply_problems"]]
                md += ["- %s: %s" % (r["record_id"], "; ".join(r["problems"])) for r in flagged]
                md += [""]
        md[4:4] = ["| Documents asked | Documents read | Records | Usable | Quotations located | Near misses | Model time |",
                   "|---|---|---|---|---|---|---|",
                   "| %d | %d | %d | %d | %d of %d | %d | %.0f s |" % (
                       totals["documents"], totals["read"], totals["records"], totals["usable"], totals["located"],
                       totals["quotations"], totals["near_misses"], totals["seconds"]), ""]
        run.write_output("roles_summary.json", fd.labeled(totals, "recorded_or_computed", "unchecked", tool="study_roles"))
        run.write_output("ROLES_SUMMARY.md", "\n".join(md) + "\n")
        ok = totals["read"] == totals["documents"]
        run.finish("completed" if ok else "completed_with_failures")
        print("\n%d of %d documents read. %d records, %d usable. Quotations located: %d of %d (%d near misses)."
              % (totals["read"], totals["documents"], totals["records"], totals["usable"], totals["located"],
                 totals["quotations"], totals["near_misses"]))
        print("Read the tables in: %s/ROLES_SUMMARY.md" % run.dir.relative_to(ROOT))
        return 0 if ok else 1
    except Exception:
        run.finish("failed")
        raise


def _ev_count(evidence):
    evidence = [e for e in evidence if e]
    return "%d of %d" % (sum(e["outcome"] in fd.QUOTE_OK for e in evidence), len(evidence))


def _cell(text):
    return str(text if text not in (None, "") else "-").replace("|", "/").replace("\n", " ")


def _claim_block(r):
    """One claim record laid out stage by stage for the summary file."""
    out = ["### %s - %s" % (r["record_id"], _cell(r.get("proposition"))), "",
           "Stages occupied: %s. Status: %s.%s" % (", ".join(r.get("stages_occupied", [])) or "-",
                                                   r["status"].replace("_", " "),
                                                   " Reported second-hand: the test is another study's." if r.get("second_hand") else ""),
           "", "| Stage | Content | Quotations located |", "|---|---|---|"]
    if "whose" not in r:
        return out + ["", "Problems: " + "; ".join(r["problems"]), ""]
    se = r["statement_evidence"]
    out.append("| 1 Proposition | \"%s\" %s | %s |" % (_cell(se.get("quote")), _cell(se.get("passage_id")), _ev_count([se])))
    w = r["whose"]
    who = {"authors": "the authors", "cited": "cited: %s" % _cell(w.get("holder_name")),
           "not_attributed": "not attributed in the text"}.get(w["holder"], "-")
    out.append("| 2 Whose | %s; stance: %s | %s |" % (who, _cell((w["stance"] or "-").replace("_", " ")),
                                                      _ev_count([w["evidence"]])))
    t = r["test"]
    if t["status"] == "present":
        content = "%s%s: %s" % (t["whose"] or "-", ", " + ", ".join(t["study_ids"]) if t.get("study_ids") else "",
                                _cell(t.get("description")))
    else:
        content = (t["status"] or "-").replace("_", " ")
    out.append("| 3 Test | %s | %s |" % (content, _ev_count(t["evidence"])))
    res = r["results"]
    if res["status"] == "present":
        for item in res["items"]:
            out.append("| 4 Reported result%s | %s | %s |" % (" (%s)" % item["study_id"] if item.get("study_id") else "",
                                                             _cell(item.get("summary")), _ev_count(item["evidence"])))
    else:
        out.append("| 4 Reported result | %s | - |" % (res["status"] or "-").replace("_", " "))
    for key, label in (("interpretation", "5 Authors' interpretation"), ("unresolved", "6 Unresolved")):
        block = r[key]
        if block["status"] == "present":
            for item in block["items"]:
                mark = " (%s)" % item["hedging"] if item.get("hedging") else ""
                out.append("| %s%s | %s | %s |" % (label, mark, _cell(item.get("summary")), _ev_count(item["evidence"])))
        else:
            out.append("| %s | %s | - |" % (label, (block["status"] or "-").replace("_", " ")))
    out.append("")
    if r.get("flag"):
        out += ["Flag (a model's reading, not part of any stage): %s" % _cell(r["flag"]["text"]), ""]
    for shared in r.get("quotations_used_for_two_stages", []):
        out += ["One quotation serves two stages (%s): \"%s\"" % (" and ".join(shared["stages"]), _cell(shared["quote"][:160])), ""]
    if r["notes"]:
        out += ["Notes: " + "; ".join(r["notes"]) + ".", ""]
    if r["problems"]:
        out += ["Problems found by the checks: " + "; ".join(r["problems"]) + ".", ""]
    return out


def cmd_claims(args):
    """Step 3. Ask a model for the claim records of each document and check every reply."""
    config, run, prices, sources = _start_model_run("claims", args)
    try:
        provider = _make_provider(args)
    except providers.ProviderError as exc:
        run.note("Could not start: %s" % exc)
        run.finish("failed")
        print("Could not start: %s" % exc)
        return 1
    wanted = [d.strip() for d in args.docs.split(",")] if args.docs else config["sample"]["report"]
    unknown = [d for d in wanted if d not in sources]
    if unknown:
        run.finish("failed")
        print("Unknown document(s): %s" % ", ".join(unknown))
        return 1
    try:
        (run.dir / "claims").mkdir()
        keys = ("records", "usable", "tested_here", "second_hand", "no_test_in_document", "withheld", "flags",
                "quotations", "quotations_located", "near_misses")
        totals = dict.fromkeys(("documents", "read") + keys, 0)
        totals["seconds"] = 0.0
        md = ["# Claim records - run %s" % run.run_id, "",
              "Model: %s / %s (thinking: %s). Prompt: %s. Definitions: %s." % (
                  args.provider, args.model, args.thinking, claims_task.PROMPT_ID, config["definitions"]["file"]),
              "",
              "Every claim record below is a model's reading and is unchecked. The counts of quotations located are "
              "recorded or computed. The record does not judge whether a result supports a proposition; it sets the "
              "stages side by side. A hypothesis is never counted as a finding.", ""]
        for doc in wanted:
            print("reading %s ..." % doc, flush=True)
            concept = config["concepts"][sources[doc]["case"]]
            result = claims_task.extract_claims(run, provider, args.model, sources[doc], concept["name"],
                                                concept["neighbors"], prices=prices,
                                                ceiling=config.get("spend_ceiling_usd_per_run"), fresh=args.fresh)
            run.write_output("claims/%s.json" % doc, result)
            totals["documents"] += 1
            md += ["## %s - %s" % (doc, sources[doc]["title"]), ""]
            if result["problem"]:
                md += ["**No records.** %s" % result["problem"], ""]
                print("  %s" % result["problem"])
                continue
            c, call = result["counts"], result["call"]
            totals["read"] += 1
            for k in keys:
                totals[k] += c[k]
            totals["seconds"] += call["seconds"] or 0
            left = result["possibly_left_out"]["in_no_record"]
            line = ("%d propositions, %d usable; %d tested here, %d reported second-hand, %d with no test in the "
                    "document, %d withheld; %d flags; %d of %d quotations located; %d marked statements in no record; %s"
                    % (c["records"], c["usable"], c["tested_here"], c["second_hand"], c["no_test_in_document"],
                       c["withheld"], c["flags"], c["quotations_located"], c["quotations"], len(left),
                       "replayed from a saved reply" if call["replayed"] else "%.0f seconds" % (call["seconds"] or 0)))
            print("  " + line)
            md += [line + ".", ""]
            for r in result["records"]:
                md += _claim_block(r)
            if left:
                md += ["Marked as a proposition in the text and cited by no record (a wording match that "
                       "over-collects; a pointer for a reader, recorded or computed, unchecked):", ""]
                md += ["- %s: %s" % (e["passage_id"], _cell(e["text"])) for e in left] + [""]
            if result["reply_problems"]:
                md += ["Problems with the reply as a whole: " + "; ".join(result["reply_problems"]) + ".", ""]
        md[4:4] = ["| Documents read | Propositions | Usable | Tested here | Second-hand | No test in document | Withheld | Flags | Quotations located | Model time |",
                   "|---|---|---|---|---|---|---|---|---|---|",
                   "| %d of %d | %d | %d | %d | %d | %d | %d | %d | %d of %d | %.0f s |" % (
                       totals["read"], totals["documents"], totals["records"], totals["usable"], totals["tested_here"],
                       totals["second_hand"], totals["no_test_in_document"], totals["withheld"], totals["flags"],
                       totals["quotations_located"], totals["quotations"], totals["seconds"]), ""]
        run.write_output("claims_summary.json", fd.labeled(totals, "recorded_or_computed", "unchecked", tool="claim_records"))
        run.write_output("CLAIMS_SUMMARY.md", "\n".join(md) + "\n")
        ok = totals["read"] == totals["documents"]
        run.finish("completed" if ok else "completed_with_failures")
        print("\n%d of %d documents read. %d propositions, %d usable. Quotations located: %d of %d (%d near misses)."
              % (totals["read"], totals["documents"], totals["records"], totals["usable"], totals["quotations_located"],
                 totals["quotations"], totals["near_misses"]))
        print("Read the records in: %s/CLAIMS_SUMMARY.md" % run.dir.relative_to(ROOT))
        return 0 if ok else 1
    except Exception:
        run.finish("failed")
        raise


def _clean_reply(text):
    """Drop wrapping quotation marks and blank space a model may add around a copied sentence."""
    text = (text or "").strip()
    for left, right in (('"', '"'), ("\u201c", "\u201d"), ("'", "'")):
        if len(text) > 2 and text.startswith(left) and text.endswith(right):
            text = text[1:-1].strip()
    return text


def smoke_test(run, provider, model, sources, prices=None, ceiling=None, long_test=True):
    """Two live calls that test the plumbing, not the science.

    1. Short: copy one sentence from one passage. Shows that a call goes out,
       a reply comes back, the record is written and the quotation checker
       can judge what the model sent.
    2. Long: copy the final sentence of a whole paper. A model that was fed
       only the first part of the text cannot do this, so it shows whether
       the whole document was read.
    A wrong copy is a finding about the model. A failed call is a fault in
    the plumbing. The two are reported separately.
    """
    src = sources["E02"]
    results = []
    short = fd.get_block(src, "P0014")
    body = [b for b in src["blocks"] if b["passage_id"] <= "P0035"]
    # Allowances are generous because some models think at length before a one-line answer.
    tests = [("short_passage", "smoke_copy_sentence_v1", {"passage": short["text"]}, "P0014", 4000)]
    if long_test:
        tests.append(("whole_paper", "smoke_last_sentence_v1",
                      {"text": "\n\n".join(b["text"] for b in body)}, "P0035", 6000))
    for name, prompt_id, variables, passage_id, max_tokens in tests:
        entry = {"test": name, "source": "E02", "expected_passage": passage_id, "call_ok": False}
        try:
            rec = run.call_model(provider, model, prompt_id, variables, temperature=0, max_tokens=max_tokens,
                                 purpose="plumbing test: " + name, prices=prices, spend_ceiling_usd=ceiling)
        except RuntimeError as exc:
            entry["problem"] = str(exc)
            results.append(fd.labeled(entry, "recorded_or_computed", "unchecked", tool="smoke_test"))
            continue
        reply = _clean_reply(rec["output_text"])
        check = fd.check_quote(src, reply, passage_id)
        entry.update({"call_ok": True, "seconds": rec["seconds"], "tokens_in": (rec["usage"] or {}).get("input_tokens"),
                      "tokens_out": (rec["usage"] or {}).get("output_tokens"), "cost_usd": rec["cost_usd"],
                      "model_returned": rec["model_returned"], "reply": reply,
                      "quotation_outcome": check["outcome"], "locators": check["locators"],
                      "near_miss": check.get("near_miss")})
        if name == "whole_paper":
            last = fd.get_block(src, passage_id)["text"]
            ends = [fd.parse_locator(l)["end"] for l in check["locators"]
                    if check["outcome"] in fd.QUOTE_OK and fd.parse_locator(l)["passage_id"] == passage_id]
            entry["copied_the_final_sentence"] = bool(ends) and max(ends) >= len(last.rstrip()) - 1
        results.append(fd.labeled(entry, "recorded_or_computed", "unchecked", tool="smoke_test"))
    return results


def cmd_try_model(args):
    config = load_config()
    prices = {k: v for k, v in config.get("prices_usd_per_million_tokens", {}).items() if isinstance(v, dict)}
    run = Run("try-model", runs_dir=ROOT / config["runs_dir"], config=config)
    try:
        provider = _make_provider(args)
        sources = fd.load_full_text_sources(ROOT / config["pilot_dir"])
        results = smoke_test(run, provider, args.model, sources, prices=prices,
                             ceiling=config.get("spend_ceiling_usd_per_run"), long_test=not args.short_only)
        run.write_output("smoke_test.json", results)
        ok = all(r["call_ok"] for r in results)
        run.finish("completed" if ok else "completed_with_failures")
        for r in results:
            print("\n%s" % r["test"])
            if not r["call_ok"]:
                print("  PLUMBING FAULT: %s" % r["problem"])
                continue
            cost = "no charge" if r["cost_usd"] == 0 else ("$%.4f" % r["cost_usd"] if r["cost_usd"] is not None
                                                           else "cost not recorded (no price entered)")
            print("  call worked: %s, %.1f seconds, %s tokens read, %s written, %s"
                  % (r["model_returned"], r["seconds"], r["tokens_in"], r["tokens_out"], cost))
            print("  model replied: %s" % (r["reply"][:160] + ("..." if len(r["reply"]) > 160 else "")))
            print("  quotation check: %s" % r["quotation_outcome"])
            for d in (r.get("near_miss") or {}).get("differences", []):
                print("    near miss - the source has %r where the model wrote %r" % (d["source_has"], d["quote_has"]))
            if "copied_the_final_sentence" in r:
                print("  copied the paper's final sentence: %s" % ("yes" if r["copied_the_final_sentence"] else "no"))
        print("\nRun folder: %s" % run.dir.relative_to(ROOT))
        return 0 if ok else 1
    except providers.ProviderError as exc:
        run.note("Could not start: %s" % exc)
        run.finish("failed")
        print("Could not start: %s" % exc)
        return 1
    except Exception:
        run.finish("failed")
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("foundations")
    cq = sub.add_parser("check-quotes")
    cq.add_argument("--file", required=True)
    sh = sub.add_parser("show")
    sh.add_argument("locator")
    sub.add_parser("check-models")
    tm = sub.add_parser("try-model")
    tm.add_argument("--provider", required=True, choices=sorted(providers.CHECKED_PROVIDERS))
    tm.add_argument("--model", required=True)
    tm.add_argument("--short-only", action="store_true")
    tm.add_argument("--thinking", choices=sorted(THINKING), default="default",
                    help="local models only: let the model think first (on), not (off), or as it chooses (default)")
    ro = sub.add_parser("roles")
    ro.add_argument("--provider", required=True, choices=sorted(providers.CHECKED_PROVIDERS))
    ro.add_argument("--model", required=True)
    ro.add_argument("--docs", help="comma-separated document identifiers; default: the eight report documents")
    ro.add_argument("--fresh", action="store_true", help="ask the model again even if a saved reply exists")
    ro.add_argument("--thinking", choices=sorted(THINKING), default="default",
                    help="local models only: let the model think first (on), not (off), or as it chooses (default)")
    cl = sub.add_parser("claims")
    cl.add_argument("--provider", required=True, choices=sorted(providers.CHECKED_PROVIDERS))
    cl.add_argument("--model", required=True)
    cl.add_argument("--docs", help="comma-separated document identifiers; default: the eight report documents")
    cl.add_argument("--fresh", action="store_true", help="ask the model again even if a saved reply exists")
    cl.add_argument("--thinking", choices=sorted(THINKING), default="default",
                    help="local models only: let the model think first (on), not (off), or as it chooses (default)")
    for name in ("compare", "report"):
        sub.add_parser(name)
    args = ap.parse_args(argv)
    if args.command == "foundations":
        return cmd_foundations(args)
    if args.command == "check-quotes":
        return cmd_check_quotes(args)
    if args.command == "show":
        return cmd_show(args)
    if args.command == "check-models":
        return cmd_check_models(args)
    if args.command == "try-model":
        return cmd_try_model(args)
    if args.command == "roles":
        return cmd_roles(args)
    if args.command == "claims":
        return cmd_claims(args)
    print("'%s' is not implemented." % args.command)
    return 2


if __name__ == "__main__":
    sys.exit(main())
