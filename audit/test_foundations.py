"""Tests for step 1 of the Concept Audit: identifiers, addresses, quotation
checks, labels and the run record.

These are constructed cases with answers that follow from how each case was
built. Passing shows the tools behave as specified on those cases. It does
not show that any reading of a paper is correct.

Run from the Conceptual Coherence folder:
  python3 -m unittest discover -s code/audit -p 'test_*.py' -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

AUDIT_DIR = Path(__file__).resolve().parent
if str(AUDIT_DIR) not in sys.path:
    sys.path.insert(0, str(AUDIT_DIR))
import foundations as fd  # noqa: E402
import providers  # noqa: E402
import roles  # noqa: E402
import claims  # noqa: E402
from runlog import Run, PROMPT_DIR  # noqa: E402


def make_source(sid, passages):
    """Build a small source from (section, text) pairs."""
    blocks = [{"passage_id": "P%04d" % i, "section": sec, "text": text, "text_sha256": fd.rp.digest(text)}
              for i, (sec, text) in enumerate(passages, 1)]
    raw = "\n\n".join(b["text"] for b in blocks)
    return {"source_id": sid, "level": "full_text", "format": "test", "blocks": blocks, "raw_text": raw,
            "n_passages": len(blocks), "text_sha256": fd.rp.digest(raw)}


SRC = make_source("X01", [
    ("Method", "Participants crossed out every letter “e” on a page of text – the depleting task."),
    ("Results", "Persistence on the unsolvable puzzles was lower in the depletion condition."),
    ("Discussion", "We did not measure glucose. The authors’ account remains untested."),
])


class Labels(unittest.TestCase):
    def test_both_labels_are_required_and_must_be_known(self):
        rec = fd.labeled({"x": 1}, "recorded_or_computed", "unchecked")
        self.assertTrue(fd.require_labels(rec))
        with self.assertRaises(ValueError):
            fd.labeled({"x": 1}, "validated", "unchecked")
        with self.assertRaises(ValueError):
            fd.labeled({"x": 1}, "model_reading", "correct")
        with self.assertRaises(ValueError):
            fd.require_labels({"x": 1})

    def test_reserved_evaluation_material_cannot_be_opened(self):
        for name in fd.PROTECTED_NAMES:
            with self.assertRaises(PermissionError):
                fd.guard_path(Path("anywhere") / name)


class Locators(unittest.TestCase):
    def test_round_trip(self):
        loc = fd.make_locator("X01", "P0002", 0, 11)
        self.assertEqual(loc, "X01:P0002:0-11")
        self.assertEqual(fd.resolve_locator(loc, {"X01": SRC}), "Persistence")
        self.assertEqual(fd.resolve_locator("X01:P0002", {"X01": SRC}), SRC["blocks"][1]["text"])

    def test_bad_addresses_fail_loudly(self):
        for bad in ("X01", "X01:P2", "X01:P0002:5", "X01:P0002:9-3x", ""):
            with self.assertRaises(ValueError):
                fd.parse_locator(bad)
        with self.assertRaises(ValueError):
            fd.resolve_locator("X01:P0002:10-9999", {"X01": SRC})
        with self.assertRaises(KeyError):
            fd.resolve_locator("X01:P0009", {"X01": SRC})
        with self.assertRaises(KeyError):
            fd.resolve_locator("Z99:P0001", {"X01": SRC})

    def test_a_changed_passage_is_detected(self):
        changed = json.loads(json.dumps(SRC))
        changed["blocks"][1]["text"] = changed["blocks"][1]["text"].replace("lower", "higher")
        with self.assertRaises(ValueError):
            fd.resolve_locator("X01:P0002", {"X01": changed})


class Quotations(unittest.TestCase):
    def test_exact_quote_is_located_with_its_address(self):
        r = fd.check_quote(SRC, "lower in the depletion condition", "P0002")
        self.assertEqual(r["outcome"], "exact")
        self.assertEqual(fd.resolve_locator(r["locators"][0], {"X01": SRC}), "lower in the depletion condition")
        self.assertTrue(fd.require_labels(r))
        self.assertEqual(r["labels"]["basis"], "recorded_or_computed")

    def test_altered_quote_is_flagged(self):
        r = fd.check_quote(SRC, "higher in the depletion condition", "P0002")
        self.assertEqual(r["outcome"], "not_found")
        self.assertGreater(r["longest_shared_run_chars"], 10)

    def test_a_tidied_source_is_still_a_failed_quote_but_the_difference_is_shown(self):
        src = make_source("X05", [("Method", 'difficulty ("How difficult did you find the colornaming task?") on a scale')])
        r = fd.check_quote(src, 'difficulty ("How difficult did you find the color-naming task?") on a scale', "P0001")
        self.assertEqual(r["outcome"], "not_found")
        self.assertEqual(r["near_miss"]["differences"], [{"source_has": "", "quote_has": "-"}])
        self.assertEqual(fd.resolve_locator(r["near_miss"]["closest_source_text"], {"X05": src}),
                         src["blocks"][0]["text"])

    def test_an_invented_quote_has_no_near_miss(self):
        r = fd.check_quote(SRC, "Glucose levels fell sharply after the first task in every lab.", "P0002")
        self.assertEqual(r["outcome"], "not_found")
        self.assertNotIn("near_miss", r)

    def test_changed_capital_letter_is_not_accepted(self):
        self.assertEqual(fd.check_quote(SRC, "persistence on the unsolvable", "P0002")["outcome"], "not_found")

    def test_typography_is_tidied_but_reported_as_such(self):
        r = fd.check_quote(SRC, 'every letter "e" on a page of text - the depleting  task', "P0001")
        self.assertEqual(r["outcome"], "exact_after_normalization")
        found = fd.resolve_locator(r["locators"][0], {"X01": SRC})
        self.assertEqual(found, "every letter “e” on a page of text – the depleting task")
        r = fd.check_quote(SRC, "The authors' account remains untested.", "P0003")
        self.assertEqual(r["outcome"], "exact_after_normalization")

    def test_right_words_wrong_passage(self):
        r = fd.check_quote(SRC, "We did not measure glucose.", "P0002")
        self.assertEqual(r["outcome"], "found_in_other_passage")
        self.assertEqual(r["locators"], ["X01:P0003:0-27"])

    def test_no_passage_cited_is_never_exact(self):
        r = fd.check_quote(SRC, "We did not measure glucose.")
        self.assertEqual(r["outcome"], "found_in_other_passage")

    def test_elided_quote_needs_every_part_in_order(self):
        ok = fd.check_quote(SRC, "Participants crossed out [...] the depleting task.", "P0001")
        self.assertEqual(ok["outcome"], "exact_with_elision")
        self.assertEqual(len(ok["locators"]), 2)
        wrong_order = fd.check_quote(SRC, "the depleting task ... Participants crossed out", "P0001")
        self.assertEqual(wrong_order["outcome"], "not_found")
        invented_part = fd.check_quote(SRC, "Participants crossed out ... and reported fatigue", "P0001")
        self.assertEqual(invented_part["outcome"], "not_found")

    def test_spacing_from_a_pdf_is_forgiven_and_reported_but_no_character_may_differ(self):
        src = make_source("X06", [("PDF page 3", "hindrance stressors has a positive effect on employees \u2019 ego\ndepletion, and self-\nenhancing humor style")])
        r = fd.check_quote(src, "hindrance stressors has a positive effect on employees\u2019 ego depletion", "P0001")
        self.assertEqual(r["outcome"], "exact_ignoring_spacing")
        self.assertEqual(fd.resolve_locator(r["locators"][0], {"X06": src}),
                         "hindrance stressors has a positive effect on employees \u2019 ego\ndepletion")
        self.assertEqual(fd.check_quote(src, "ego depletion, and self-enhancing humor style", "P0001")["outcome"],
                         "exact_ignoring_spacing")
        self.assertEqual(fd.check_quote(src, "ego depletion, and self enhancing humor style", "P0001")["outcome"], "not_found")
        self.assertEqual(fd.check_quote(src, "a negative effect on employees\u2019 ego depletion", "P0001")["outcome"], "not_found")
        self.assertEqual(fd.check_quote(src, "ego\ndepletion", "P0001")["outcome"], "exact")      # still exact when it is
        self.assertEqual(fd.check_quote(src, "egodepletion", "P0001")["outcome"], "not_found")     # too short to forgive

    def test_a_quote_trailing_off_with_dots_is_checked_up_to_the_dots(self):
        r = fd.check_quote(SRC, "Persistence on the unsolvable puzzles was lower...", "P0002")
        self.assertEqual(r["outcome"], "exact_with_elision")
        self.assertEqual(fd.resolve_locator(r["locators"][0], {"X01": SRC}), "Persistence on the unsolvable puzzles was lower")
        self.assertEqual(fd.check_quote(SRC, "Persistence on the solvable puzzles was lower...", "P0002")["outcome"], "not_found")

    def test_stitching_across_passages_is_not_accepted(self):
        r = fd.check_quote(SRC, "lower in the depletion condition. We did not measure glucose.", "P0002")
        self.assertEqual(r["outcome"], "not_found")

    def test_empty_missing_and_short(self):
        self.assertEqual(fd.check_quote(SRC, "   ", "P0001")["outcome"], "empty_quote")
        r = fd.check_quote(SRC, "We did not measure glucose.", "P0042")
        self.assertFalse(r["cited_passage_exists"])
        self.assertEqual(r["outcome"], "found_in_other_passage")
        self.assertTrue(fd.check_quote(SRC, "glucose", "P0003")["short_quote"])

    def test_summary_counts(self):
        rs = [fd.check_quote(SRC, "We did not measure glucose.", "P0003"),
              fd.check_quote(SRC, "We measured glucose.", "P0003")]
        s = fd.quotation_summary(rs)
        self.assertEqual((s["total"], s["located_in_cited_passage"], len(s["mismatches"])), (2, 1, 1))


class Studies(unittest.TestCase):
    def test_headings_make_candidates_and_mentions_do_not(self):
        src = make_source("X02", [
            ("Introduction", "In Study 4 of Smith (2001), participants resisted cookies."),
            ("STUDY 1: FEEDBACK", "We tested the prediction."),
            ("Method", "Participants completed the Stroop task."),
            ("STUDY 2: MONITORING", "The goal of Study 2 was replication."),
            ("Results", "The effect held."),
            ("General Discussion", "General Discussion"),
            ("General Discussion", "Both studies agree."),
            ("References", "Smith (2001). Study 9 of resistance."),
        ])
        got = {c["study_id"]: c for c in fd.study_candidates(src)}
        self.assertEqual(got["X02.S1"]["status"], "heading")
        self.assertEqual((got["X02.S1"]["first_passage"], got["X02.S1"]["last_passage"]), ("P0002", "P0003"))
        self.assertEqual((got["X02.S2"]["first_passage"], got["X02.S2"]["last_passage"]), ("P0004", "P0005"))
        self.assertEqual(got["X02.S4"]["status"], "mention_only")
        self.assertNotIn("X02.S9", got)  # reference list is ignored
        for c in got.values():
            self.assertEqual(c["labels"], {"basis": "recorded_or_computed", "checked": "unchecked",
                                           "tool": "study_candidates"})

    def test_roman_numeral_heading_is_the_same_study_as_its_arabic_mentions(self):
        src = make_source("X03", [
            ("INTRO", "Study I: Daily Diary Study"),
            ("INTRO", "The purpose of Study 1 is to test the model within individuals."),
            ("INTRO", "Study 2: Questionnaire Study"),
            ("INTRO", "Study 2 used two waves."),
        ])
        got = {c["study_id"]: c["status"] for c in fd.study_candidates(src)}
        self.assertEqual(got, {"X03.S1": "heading", "X03.S2": "heading"})

    def test_no_heading_gives_whole_document_entry(self):
        got = fd.study_candidates(SRC)
        self.assertEqual([(c["study_id"], c["status"]) for c in got], [("X01.S0", "whole_document")])


class FakeModel:
    """Behaves like a real provider (is_model = True) without any network."""
    name = "fake_provider"
    is_model = True

    def __init__(self, fail=False):
        self.fail = fail

    def complete(self, model, system, prompt, temperature, max_tokens):
        if self.fail:
            raise providers.ProviderError("HTTP 500 from test", body="server said no")
        return {"text": "ok", "model_returned": model + "-20990101",
                "usage": {"input_tokens": 1000, "output_tokens": 500}, "raw": {"id": "abc", "content": "ok"}}


class RunRecord(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.prompts = self.base / "prompts"
        self.prompts.mkdir()
        (self.prompts / "p_v1.md").write_text("Read this: {{passage}}", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_call_record_keeps_prompt_model_settings_and_raw_reply(self):
        run = Run("test", runs_dir=self.base / "runs", argv=["test"])
        rec = run.call_model(FakeModel(), "model-x", "p_v1", {"passage": "We measured X."},
                             temperature=0, max_tokens=50, prompt_dir=self.prompts,
                             prices={"model-x": {"input_per_million": 2.0, "output_per_million": 10.0}})
        saved = json.loads((run.dir / "model_calls/0001.json").read_text())
        self.assertEqual(saved["prompt_as_sent"], "Read this: We measured X.")
        self.assertEqual(saved["model_requested"], "model-x")
        self.assertEqual(saved["model_returned"], "model-x-20990101")
        self.assertEqual(saved["raw_response"], {"id": "abc", "content": "ok"})
        self.assertEqual(saved["settings"], {"temperature": 0, "max_tokens": 50, "provider_options": None})
        self.assertAlmostEqual(saved["cost_usd"], 0.007)
        run.finish()
        manifest = json.loads((run.dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["model_calls"], 1)
        self.assertEqual(manifest["models"]["fake_provider/model-x"]["models_returned"], ["model-x-20990101"])
        self.assertEqual(manifest["prompts"]["p_v1"]["sha256"], rec["prompt_template_sha256"])
        self.assertIn("code/audit/foundations.py", manifest["code_sha256"])

    def test_cost_is_never_guessed(self):
        run = Run("test", runs_dir=self.base / "runs", argv=["test"])
        rec = run.call_model(FakeModel(), "model-x", "p_v1", {"passage": "x"}, prompt_dir=self.prompts)
        self.assertIsNone(rec["cost_usd"])
        self.assertIn("not recorded", rec["cost_note"])
        self.assertIn("incomplete", json.loads((run.dir / "manifest.json").read_text())["cost_note"])

    def test_changing_the_prompt_changes_its_recorded_version(self):
        run = Run("test", runs_dir=self.base / "runs", argv=["test"])
        a = run.call_model(FakeModel(), "m", "p_v1", {"passage": "x"}, prompt_dir=self.prompts)
        (self.prompts / "p_v1.md").write_text("Read this carefully: {{passage}}", encoding="utf-8")
        b = run.call_model(FakeModel(), "m", "p_v1", {"passage": "x"}, prompt_dir=self.prompts)
        self.assertNotEqual(a["prompt_template_sha256"], b["prompt_template_sha256"])

    def test_missing_prompt_value_is_an_error_not_a_blank(self):
        run = Run("test", runs_dir=self.base / "runs", argv=["test"])
        with self.assertRaises(KeyError):
            run.call_model(FakeModel(), "m", "p_v1", {}, prompt_dir=self.prompts)

    def test_failed_call_is_recorded(self):
        run = Run("test", runs_dir=self.base / "runs", argv=["test"])
        with self.assertRaises(RuntimeError):
            run.call_model(FakeModel(fail=True), "m", "p_v1", {"passage": "x"}, prompt_dir=self.prompts)
        saved = json.loads((run.dir / "model_calls/0001.json").read_text())
        self.assertEqual(saved["error"]["body"], "server said no")
        self.assertIsNone(saved["output_text"])

    def test_stand_in_is_never_recorded_as_a_model(self):
        run = Run("test", runs_dir=self.base / "runs", argv=["test"])
        rec = run.call_model(providers.StandInProvider(replies=["prepared"]), "stand-in", "p_v1",
                             {"passage": "x"}, prompt_dir=self.prompts)
        self.assertFalse(rec["is_model"])
        self.assertIn("not a model", rec["cost_note"])

    def test_spend_ceiling_stops_further_calls(self):
        run = Run("test", runs_dir=self.base / "runs", argv=["test"])
        prices = {"m": {"input_per_million": 2.0, "output_per_million": 10.0}}
        run.call_model(FakeModel(), "m", "p_v1", {"passage": "x"}, prompt_dir=self.prompts, prices=prices,
                       spend_ceiling_usd=0.005)
        with self.assertRaises(RuntimeError):
            run.call_model(FakeModel(), "m", "p_v1", {"passage": "x"}, prompt_dir=self.prompts, prices=prices,
                           spend_ceiling_usd=0.005)
        self.assertFalse((run.dir / "model_calls/0002.json").exists())

    def test_keys_never_enter_the_record(self):
        run = Run("test", runs_dir=self.base / "runs", argv=["test"])
        provider = providers.AnthropicProvider(key="sk-test-SECRET-VALUE")
        provider.complete = FakeModel().complete  # no network in tests
        run.call_model(provider, "m", "p_v1", {"passage": "x"}, prompt_dir=self.prompts)
        run.finish()
        everything = "".join(p.read_text() for p in run.dir.rglob("*") if p.is_file())
        self.assertNotIn("SECRET-VALUE", everything)

    def test_google_reply_is_read_and_thinking_tokens_count_as_output(self):
        provider = providers.GoogleProvider(key="test-SECRET-VALUE")
        sent = {}

        def fake_post(url, headers, body, timeout=300):
            sent.update({"url": url, "headers": headers, "body": body})
            return {"candidates": [{"content": {"parts": [{"text": "hidden", "thought": True}, {"text": "answer"}]}}],
                    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 4, "thoughtsTokenCount": 6},
                    "modelVersion": "model-g-001"}
        original, providers._post = providers._post, fake_post
        try:
            reply = provider.complete("model-g", "be exact", "Read this.", 0, 100)
        finally:
            providers._post = original
        self.assertEqual(reply["text"], "answer")
        self.assertEqual(reply["usage"], {"input_tokens": 10, "output_tokens": 10})
        self.assertEqual(reply["model_returned"], "model-g-001")
        self.assertNotIn("SECRET-VALUE", sent["url"])
        self.assertEqual(sent["body"]["systemInstruction"], {"parts": [{"text": "be exact"}]})

    def _local(self, reply, tags=None):
        provider = providers.LocalProvider(host="http://localhost:11434")
        sent = {}

        def fake_post(url, headers, body, timeout=300):
            sent.update({"url": url, "body": body})
            return reply

        def fake_get(url, headers, timeout=30):
            return tags or {"models": [{"name": "open-model:9b", "digest": "abcdef0123456789ffff"}]}
        return provider, sent, fake_post, fake_get

    def _with_fakes(self, fake_post, fake_get, call):
        saved = providers._post, providers._get
        providers._post, providers._get = fake_post, fake_get
        try:
            return call()
        finally:
            providers._post, providers._get = saved

    def test_local_model_gets_a_window_big_enough_and_its_weights_are_recorded(self):
        prompt = "word " * 20000  # 100,000 characters: far beyond Ollama's default window
        provider, sent, fake_post, fake_get = self._local(
            {"model": "open-model:9b", "message": {"content": "answer"}, "done_reason": "stop",
             "prompt_eval_count": 21000, "eval_count": 300})
        reply = self._with_fakes(fake_post, fake_get, lambda: provider.complete("open-model:9b", None, prompt, 0, 1000))
        self.assertGreaterEqual(sent["body"]["options"]["num_ctx"], 100000 // 3 + 1000)
        self.assertFalse(sent["body"]["stream"])
        self.assertEqual(reply["model_returned"], "open-model:9b@abcdef012345")
        self.assertEqual(reply["usage"], {"input_tokens": 21000, "output_tokens": 300})
        self.assertEqual(reply["raw"]["context_check"]["verdict"], "ok")

    def test_local_model_that_read_only_part_of_the_prompt_is_rejected(self):
        prompt = "word " * 20000
        provider, sent, fake_post, fake_get = self._local(
            {"model": "open-model:9b", "message": {"content": "confident answer about half a paper"},
             "done_reason": "stop", "prompt_eval_count": 4096, "eval_count": 300})
        with self.assertRaises(providers.ProviderError):
            self._with_fakes(fake_post, fake_get, lambda: provider.complete("open-model:9b", None, prompt, 0, 1000))

    def test_local_model_that_filled_its_window_is_rejected(self):
        window = providers.LocalProvider.window_for("x" * 30000, None, 600)
        provider, sent, fake_post, fake_get = self._local(
            {"model": "open-model:9b", "message": {"content": "cut off mid"}, "done_reason": "length",
             "prompt_eval_count": window - 600, "eval_count": 600})
        with self.assertRaises(providers.ProviderError):
            self._with_fakes(fake_post, fake_get, lambda: provider.complete("open-model:9b", None, "x" * 30000, 0, 600))

    def test_thinking_is_only_mentioned_to_the_model_when_set_and_changes_the_saved_reply(self):
        reply = {"model": "open-model:9b", "message": {"content": "ok"}, "done_reason": "stop",
                 "prompt_eval_count": 12, "eval_count": 3}
        provider, sent, fake_post, fake_get = self._local(reply)
        self._with_fakes(fake_post, fake_get, lambda: provider.complete("open-model:9b", None, "x" * 100, 0, 50))
        self.assertNotIn("think", sent["body"])
        quiet = providers.LocalProvider(host="http://localhost:11434", think=False)
        self._with_fakes(fake_post, fake_get, lambda: quiet.complete("open-model:9b", None, "x" * 100, 0, 50))
        self.assertIs(sent["body"]["think"], False)
        run = Run("test", runs_dir=self.base / "runs", argv=["test"])
        run.cache_dir = self.base / "cache"
        a = run._cache_file(provider, "m", None, "p", 0, 50)
        b = run._cache_file(quiet, "m", None, "p", 0, 50)
        self.assertNotEqual(a, b)

    def test_a_model_that_only_thought_is_a_failed_call_with_a_plain_reason(self):
        provider, sent, fake_post, fake_get = self._local(
            {"model": "open-model:9b", "message": {"content": "", "thinking": "Let me consider..."},
             "done_reason": "length", "prompt_eval_count": 148, "eval_count": 400})
        with self.assertRaises(providers.ProviderError) as caught:
            self._with_fakes(fake_post, fake_get, lambda: provider.complete("open-model:9b", None, "x" * 600, 0, 400))
        self.assertIn("--thinking off", str(caught.exception))

    def test_local_model_costs_nothing_and_the_record_says_why(self):
        run = Run("test", runs_dir=self.base / "runs", argv=["test"])
        provider, sent, fake_post, fake_get = self._local(
            {"model": "open-model:9b", "message": {"content": "ok"}, "done_reason": "stop",
             "prompt_eval_count": 12, "eval_count": 3})
        rec = self._with_fakes(fake_post, fake_get, lambda: run.call_model(
            provider, "open-model:9b", "p_v1", {"passage": "We measured X."}, prompt_dir=self.prompts))
        self.assertEqual(rec["cost_usd"], 0.0)
        self.assertIn("no charge", rec["cost_note"])
        self.assertTrue(rec["is_model"])
        self.assertEqual(rec["raw_response"]["weights_digest"], "abcdef0123456789ffff")

    def test_run_folders_are_never_reused(self):
        a = Run("test", runs_dir=self.base / "runs", argv=["test"])
        b = Run("test", runs_dir=self.base / "runs", argv=["test"])
        self.assertNotEqual(a.dir, b.dir)


@unittest.skipUnless((fd.PILOT_DIR / "source_lock.json").exists(), "Prepared pilot documents required")
class PreparedDocuments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = fd.load_full_text_sources()
        cls.abstracts = fd.load_abstract_sources()

    def test_twelve_documents_and_120_abstracts(self):
        self.assertEqual(len(self.full), 12)
        self.assertEqual(sorted(s["case"] for s in self.full.values()).count("ego"), 8)
        self.assertEqual(len(self.abstracts), 120)
        self.assertTrue(all(s["level"] == "abstract" for s in self.abstracts.values()))

    def test_every_passage_address_resolves(self):
        for sid, s in self.full.items():
            for b in s["blocks"]:
                self.assertEqual(fd.resolve_locator(fd.make_locator(sid, b["passage_id"]), self.full), b["text"])

    def test_every_document_gets_at_least_one_study_entry(self):
        for s in self.full.values():
            got = fd.study_candidates(s)
            self.assertTrue(got)
            self.assertTrue(any(c["status"] in ("heading", "whole_document") for c in got))

    def test_registry_holds_identifiers_but_no_text(self):
        for s in self.full.values():
            entry = fd.registry_entry(s)
            self.assertNotIn("raw_text", entry)
            self.assertNotIn("blocks", entry)
            self.assertEqual(len(entry["text_sha256"]), 64)
            self.assertTrue(fd.require_labels(entry))

    def test_a_real_sentence_is_found_and_an_altered_one_is_not(self):
        s = self.full["E02"]
        b = next(b for b in s["blocks"] if len(b["text"]) > 400)
        quote = b["text"][100:260]
        self.assertEqual(fd.check_quote(s, quote, b["passage_id"])["outcome"], "exact")
        self.assertNotIn(fd.check_quote(s, quote[:80] + " not " + quote[80:], b["passage_id"])["outcome"], fd.QUOTE_OK)


class CountingModel(FakeModel):
    """A fake paid model that counts how often it is really called."""

    def __init__(self, fail=False):
        FakeModel.__init__(self, fail)
        self.calls = 0

    def complete(self, model, system, prompt, temperature, max_tokens):
        self.calls += 1
        return FakeModel.complete(self, model, system, prompt, temperature, max_tokens)


class Replay(unittest.TestCase):
    """A saved reply is reused for the same prompt, model and settings, and for nothing else."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.prompts = self.base / "prompts"
        self.prompts.mkdir()
        (self.prompts / "p_v1.md").write_text("Read this: {{passage}}", encoding="utf-8")
        self.prices = {"m": {"input_per_million": 2.0, "output_per_million": 10.0}}

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self):
        run = Run("test", runs_dir=self.base / "runs", argv=["test"])
        run.cache_dir = self.base / "cache"
        return run

    def test_same_prompt_is_replayed_and_costs_nothing_the_second_time(self):
        model = CountingModel()
        first = self._run().call_model(model, "m", "p_v1", {"passage": "x"}, prompt_dir=self.prompts, prices=self.prices)
        run2 = self._run()
        second = run2.call_model(model, "m", "p_v1", {"passage": "x"}, prompt_dir=self.prompts, prices=self.prices)
        self.assertEqual(model.calls, 1)
        self.assertEqual(second["output_text"], first["output_text"])
        self.assertEqual(second["raw_response"], first["raw_response"])
        self.assertEqual(second["cost_usd"], 0.0)
        self.assertEqual(second["replayed_from"]["run_id"], first["run_id"])
        self.assertIn("$0.0070", second["cost_note"])
        manifest = json.loads((run2.dir / "manifest.json").read_text())
        self.assertEqual((manifest["replayed_calls"], manifest["tokens"]), (1, {"input": 0, "output": 0}))

    def test_anything_different_is_asked_afresh(self):
        model = CountingModel()
        run = self._run()
        run.call_model(model, "m", "p_v1", {"passage": "x"}, prompt_dir=self.prompts)
        run.call_model(model, "m", "p_v1", {"passage": "y"}, prompt_dir=self.prompts)       # different text
        run.call_model(model, "m2", "p_v1", {"passage": "x"}, prompt_dir=self.prompts)      # different model
        run.call_model(model, "m", "p_v1", {"passage": "x"}, prompt_dir=self.prompts, temperature=1)  # different setting
        run.call_model(model, "m", "p_v1", {"passage": "x"}, prompt_dir=self.prompts, fresh=True)     # asked to
        self.assertEqual(model.calls, 5)

    def test_failed_calls_and_stand_ins_are_never_saved(self):
        run = self._run()
        with self.assertRaises(RuntimeError):
            run.call_model(CountingModel(fail=True), "m", "p_v1", {"passage": "x"}, prompt_dir=self.prompts)
        run.call_model(providers.StandInProvider(replies=["a"]), "s", "p_v1", {"passage": "x"}, prompt_dir=self.prompts)
        self.assertFalse((self.base / "cache").exists() and list((self.base / "cache").iterdir()))


ROLE_SRC = make_source("X09", [
    ("INTRO", "In Smith (2001), the Stroop task was the outcome measure."),
    ("INTRO", "Participants in the depletion condition crossed out letters under a hard rule; controls used an easy rule."),
    ("DISCUSS", "Persistence on unsolvable puzzles was the dependent variable."),
    ("INTRO", "All participants then completed a mood scale."),
    ("References", "Smith (2001). The Stroop task as an outcome measure."),
])


def role_reply(operations, own="one", studies=None):
    return json.dumps({"own_studies": own, "studies": studies or [], "operations": operations})


def op(name, whose, role_list, mention, study="0", cited_work=None):
    return {"operation": name, "variant": None, "scored": None, "study": study if whose == "own" else None,
            "whose": whose, "cited_work": cited_work, "mention": {"passage_id": mention[0], "quote": mention[1]},
            "roles": role_list}


def role(name, evidence, position=None, strength="stated", reason=None):
    return {"role": name, "position": position, "strength": strength, "reason": reason,
            "evidence": [{"passage_id": p, "quote": q} for p, q in evidence]}


class StudyRoles(unittest.TestCase):
    def _validate(self, operations, **kw):
        data, problem = roles.parse_reply(role_reply(operations, **kw))
        self.assertIsNone(problem)
        _, sent, _ = roles.render_document(ROLE_SRC)
        return roles.validate(data, ROLE_SRC, sent)

    def test_reply_is_found_inside_fences_and_chatter_and_bad_replies_are_named(self):
        good = '{"own_studies": "none", "studies": [], "operations": []}'
        for text in (good, "```json\n" + good + "\n```", "Here is the record:\n" + good + "\nDone."):
            data, problem = roles.parse_reply(text)
            self.assertIsNone(problem)
            self.assertEqual(data["own_studies"], "none")
        for text in ("", "I could not find any operations.", '{"own_studies": "one", "operations": [', "[1, 2]"):
            self.assertIsNotNone(roles.parse_reply(text)[1])

    def test_document_is_shown_without_back_matter_and_can_have_parts_removed(self):
        text, sent, left_out = roles.render_document(ROLE_SRC)
        self.assertEqual((sent, left_out), (["P0001", "P0002", "P0003", "P0004"], ["P0005"]))
        self.assertIn("[P0003 | archive label: DISCUSS]", text)
        _, sent, left_out = roles.render_document(ROLE_SRC, keep=lambda b: b["section"] != "DISCUSS")
        self.assertEqual((sent, left_out), (["P0001", "P0002", "P0004"], ["P0003", "P0005"]))

    def test_a_sound_record_is_usable_and_labeled_as_a_models_reading(self):
        out = self._validate([op("letter-crossing task", "own",
                                 [role("manipulated", [("P0002", "crossed out letters under a hard rule")])],
                                 ("P0002", "crossed out letters"))])
        rec = out["records"][0]
        self.assertEqual((rec["status"], rec["study_id"], rec["problems"]), ("usable", "X09.S0", []))
        self.assertEqual(rec["labels"], {"basis": "model_reading", "checked": "unchecked", "tool": "study_roles"})
        self.assertEqual(rec["roles"][0]["evidence"][0]["outcome"], "exact")

    def test_role_in_the_discussion_counts_and_a_section_label_is_not_evidence(self):
        out = self._validate([
            op("unsolvable puzzles", "own", [role("outcome", [("P0003", "was the dependent variable")])],
               ("P0003", "unsolvable puzzles")),
            op("mood scale", "own", [role("outcome", [("P0004", "archive label: INTRO")])], ("P0004", "a mood scale")),
        ])
        self.assertEqual(out["records"][0]["status"], "usable")
        self.assertEqual(out["records"][1]["status"], "needs_attention")

    def test_an_altered_quotation_does_not_support_a_role_and_the_alteration_is_shown(self):
        out = self._validate([op("unsolvable puzzles", "own",
                                 [role("outcome", [("P0003", "Persistence on unsolvable puzzles was the dependant variable.")])],
                                 ("P0003", "unsolvable puzzles"))])
        rec = out["records"][0]
        self.assertEqual(rec["status"], "needs_attention")
        self.assertEqual(rec["roles"][0]["evidence"][0]["near_miss"]["differences"], [{"source_has": "e", "quote_has": "a"}])

    def test_rules_of_the_definitions_are_enforced(self):
        out = self._validate([
            op("task A", "own", [role("focal measure", [("P0004", "completed a mood scale")])], ("P0004", "mood scale")),
            op("task B", "own", [role("outcome", [])], ("P0004", "mood scale")),
            op("task C", "ours", [role("covariate", [("P0004", "completed a mood scale")], position="mediator")],
               ("P0004", "mood scale")),
            op("task D", "cited", [role("outcome", [("P0001", "was the outcome measure")])], ("P0001", "the Stroop task")),
            op("task E", "own", [role("measured", [("P0004", "completed a mood scale")])], ("P0004", "mood scale")),
            op("task F", "own", [role("outcome", [("P0005", "The Stroop task as an outcome measure.")])],
               ("P0001", "the Stroop task")),
        ])
        r = {x["operation"]: x for x in out["records"]}
        self.assertEqual(r["task A"]["status"], "needs_attention")                       # unknown role
        self.assertTrue(any("unknown role" in p for p in r["task A"]["problems"]))
        self.assertEqual(r["task B"]["status"], "needs_attention")                       # role without evidence
        self.assertEqual(r["task C"]["whose"], "unclear")                                # invalid owner
        self.assertIsNone(r["task C"]["roles"][0]["position"])                           # position only for measured
        self.assertTrue(any("cited work is not named" in p for p in r["task D"]["problems"]))
        self.assertEqual(r["task E"]["roles"][0]["position"], "not_stated")
        self.assertEqual(r["task F"]["roles"][0]["evidence"][0]["outcome"], "passage_was_not_shown_to_the_model")
        self.assertEqual(r["task F"]["status"], "needs_attention")

    def test_withholding_needs_a_reason_and_a_located_mention(self):
        out = self._validate([
            op("mood scale", "own", [role("insufficient_evidence", [], strength=None, reason="text_silent")],
               ("P0004", "a mood scale")),
            op("mood scale 2", "own", [role("insufficient_evidence", [], strength=None)], ("P0004", "a mood scale")),
            op("mood scale 3", "own", [role("insufficient_evidence", [], strength=None, reason="text_missing")],
               ("P0004", "a sadness scale")),
        ])
        self.assertEqual([r["status"] for r in out["records"]], ["usable", "needs_attention", "needs_attention"])

    def test_same_task_two_roles_two_owners_and_cited_only(self):
        out = self._validate([
            op("Stroop task", "cited", [role("outcome", [("P0001", "was the outcome measure")])],
               ("P0001", "the Stroop task"), cited_work="Smith (2001)"),
            op("Stroop Task", "own", [role("manipulated", [("P0002", "under a hard rule")])], ("P0002", "crossed out letters")),
            op("mood scale", "cited", [role("outcome", [("P0001", "was the outcome measure")])],
               ("P0001", "the Stroop task"), cited_work="Smith (2001)"),
        ])
        d = roles.derive(out["records"])
        self.assertEqual(d["cited_only"], ["mood scale"])
        self.assertEqual(d["several_roles"][0]["operation"], "stroop task")
        self.assertEqual(d["several_roles"][0]["roles"],
                         [{"role": "manipulated", "whose": "own"}, {"role": "outcome", "whose": "cited"}])
        self.assertEqual(d["labels"]["basis"], "recorded_or_computed")

    def test_a_role_given_twice_is_pointed_out_but_not_rejected(self):
        out = self._validate([op("mood scale", "own", [role("covariate", [("P0004", "completed a mood scale")]),
                                                       role("covariate", [("P0004", "All participants then")])],
                                 ("P0004", "a mood scale"))])
        rec = out["records"][0]
        self.assertEqual(rec["status"], "usable")
        self.assertTrue(any("more than once" in p for p in rec["problems"]))

    def test_named_tasks_that_no_record_covers_are_listed(self):
        src = make_source("X10", [
            ("INTRO", "Earlier work used a letter-crossing task followed by a multisource interference task."),
            ("METHODS", "Participants completed the Stroop task. A t test compared groups. We did not test moderators."),
            ("References", "Jones (1999). The hand-grip task."),
        ])
        _, sent, _ = roles.render_document(src)
        records = [{"operation": "Stroop task", "variant": None}]
        out = roles.unrecorded_mentions(src, sent, records)
        self.assertEqual(sorted(e["phrase"] for e in out["in_no_record"]),
                         ["letter-crossing task", "multisource interference task"])
        self.assertEqual(out["labels"], {"basis": "recorded_or_computed", "checked": "unchecked",
                                         "tool": "unrecorded_mentions"})
        covered = roles.unrecorded_mentions(src, sent, records + [{"operation": "letter-crossing task", "variant": None},
                                                                  {"operation": "Multisource Interference Task", "variant": None}])
        self.assertEqual(covered["in_no_record"], [])

    def test_numbered_studies_must_be_declared(self):
        ops = [op("mood scale", "own", [role("covariate", [("P0004", "completed a mood scale")])],
                  ("P0004", "a mood scale"), study="2")]
        undeclared = self._validate(ops, own="several")
        self.assertTrue(any("not in the reply's list of studies" in p for p in undeclared["records"][0]["problems"]))
        self.assertTrue(any("several studies, but 0 are listed" in p for p in undeclared["problems"]))
        two = [{"number": "1", "label": "Study 1", "passage_id": "P0002", "quote": "Participants in the depletion condition"},
               {"number": "2", "label": "Study 2", "passage_id": "P0004", "quote": "All participants"}]
        declared = self._validate(ops, own="several", studies=two)
        self.assertEqual((declared["records"][0]["study_id"], declared["records"][0]["problems"], declared["problems"]),
                         ("X09.S2", [], []))
        unnumbered = self._validate([op("mood scale", "own", [role("covariate", [("P0004", "completed a mood scale")])],
                                        ("P0004", "a mood scale"), study="0")], own="several", studies=two)
        self.assertTrue(any("no study number" in p for p in unnumbered["records"][0]["problems"]))

    def test_one_document_end_to_end_with_a_stand_in(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            run = Run("test", runs_dir=Path(tmp.name) / "runs", argv=["test"])
            good = role_reply([op("unsolvable puzzles", "own", [role("outcome", [("P0003", "was the dependent variable")])],
                                  ("P0003", "unsolvable puzzles"))])
            stand_in = providers.StandInProvider(replies=[good, "Sorry, I cannot help with that."])
            first = roles.extract_roles(run, stand_in, "stand-in", ROLE_SRC)
            self.assertEqual(first["counts"]["usable"], 1)
            self.assertEqual((first["counts"]["quotations"], first["counts"]["quotations_located"]), (2, 2))
            sent = json.loads((run.dir / "model_calls/0001.json").read_text())["prompt_as_sent"]
            self.assertIn("[P0003 | archive label: DISCUSS]", sent)
            self.assertNotIn("Smith (2001). The Stroop task as an outcome measure.", sent)   # reference list not sent
            second = roles.extract_roles(run, stand_in, "stand-in", ROLE_SRC)
            self.assertIn("could not be read", second["problem"])
            third = roles.extract_roles(run, stand_in, "stand-in", ROLE_SRC)                 # stand-in has run dry
            self.assertIn("model call failed", third["problem"])
        finally:
            tmp.cleanup()

    def test_the_prompt_names_every_role_and_has_both_slots(self):
        text = (PROMPT_DIR / (roles.PROMPT_ID + ".md")).read_text(encoding="utf-8")
        for word in roles.ROLES + roles.WHOSE + roles.STRENGTH + roles.POSITION + roles.WITHHOLD:
            self.assertIn(word, text)
        self.assertIn("{{coverage}}", text)
        self.assertIn("{{document}}", text)


CLAIM_SRC = make_source("X11", [
    ("INTRO", "According to the strength model (Baum, 2007), self-control draws on a limited resource."),
    ("INTRO", "Hypothesis 1: Prior exertion lowers persistence."),
    ("INTRO", "Participants in the depletion condition crossed out letters; persistence on puzzles was then timed."),
    ("RESULTS", "Persistence was lower after exertion, d = 0.30, p = .04."),
    ("DISCUSS", "These results suggest that prior exertion may impair persistence."),
    ("DISCUSS", "We did not measure the resource itself."),
    ("INTRO", "We argue that willpower beliefs matter for persistence."),
])


def ev(pid, quote):
    return {"passage_id": pid, "quote": quote}


def claim(**changes):
    """A sound six-stage record for Hypothesis 1; tests alter one thing at a time."""
    base = {
        "proposition": "Prior exertion lowers persistence.",
        "statement_evidence": ev("P0002", "Hypothesis 1: Prior exertion lowers persistence."),
        "holder": "authors", "holder_name": None, "stance": "assert", "whose_evidence": None,
        "test": {"status": "present", "whose": "own", "studies": ["0"], "description": "Two conditions compared on persistence.",
                 "evidence": [ev("P0003", "persistence on puzzles was then timed")]},
        "results": {"status": "present", "items": [{"study": "0", "summary": "Lower persistence, d = 0.30.",
                                                    "evidence": [ev("P0004", "d = 0.30, p = .04")]}]},
        "interpretation": {"status": "present", "items": [{"summary": "May impair persistence.", "hedging": "hedged",
                                                           "evidence": [ev("P0005", "may impair persistence")]}]},
        "unresolved": {"status": "present", "items": [{"summary": "The resource was not measured.",
                                                       "evidence": [ev("P0006", "We did not measure the resource itself.")]}]},
        "flag": None,
    }
    base.update(changes)
    return base


THEORY = claim(proposition="Self-control draws on a limited resource.",
               statement_evidence=ev("P0001", "self-control draws on a limited resource"),
               holder="cited", holder_name="Baum (2007)", stance="report",
               whose_evidence=ev("P0001", "According to the strength model (Baum, 2007)"),
               test={"status": "none_in_document", "whose": None, "studies": [], "description": None, "evidence": []},
               results={"status": "none_in_document", "items": []},
               interpretation={"status": "none_stated", "items": []},
               unresolved={"status": "present", "items": [{"summary": "The resource was not measured.",
                                                           "evidence": [ev("P0006", "We did not measure the resource itself.")]}]})


class ClaimRecords(unittest.TestCase):
    def _validate(self, propositions, keep=None):
        _, sent, _ = roles.render_document(CLAIM_SRC, keep)
        return claims.validate({"propositions": propositions}, CLAIM_SRC, sent, full_text=keep is None)

    def test_a_sound_record_keeps_six_stages_and_is_labeled_a_models_reading(self):
        rec = self._validate([claim()])["records"][0]
        self.assertEqual((rec["status"], rec["problems"]), ("usable", []))
        self.assertEqual(rec["stages_occupied"], ["proposed", "tested", "result reported", "interpreted"])
        self.assertEqual(rec["test"]["study_ids"], ["X11.S0"])
        two = claim(test=dict(claim()["test"], studies=["1", "2"]))
        self.assertEqual(self._validate([two])["records"][0]["test"]["study_ids"], ["X11.S1", "X11.S2"])
        self.assertFalse(rec["second_hand"])
        self.assertEqual(rec["labels"], {"basis": "model_reading", "checked": "unchecked", "tool": "claim_records"})

    def test_a_cited_theory_that_is_not_tested_stays_a_proposal(self):
        rec = self._validate([THEORY])["records"][0]
        self.assertEqual((rec["status"], rec["stages_occupied"]), ("usable", ["proposed"]))
        self.assertEqual(rec["whose"]["holder_name"], "Baum (2007)")
        self.assertEqual(rec["results"]["status"], "none_in_document")

    def test_a_hypothesis_is_never_counted_as_a_finding(self):
        merged = dict(THEORY, results={"status": "present", "items": [
            {"study": "0", "summary": "Confirmed.", "evidence": [ev("P0004", "d = 0.30, p = .04")]}]})
        rec = self._validate([merged])["records"][0]
        self.assertEqual(rec["status"], "needs_attention")
        self.assertTrue(any("no test recorded" in p for p in rec["problems"]))

    def test_with_results_removed_the_proposal_stands_and_the_result_is_withheld(self):
        no_results = lambda b: b["section"] != "RESULTS"
        withheld = claim(results={"status": "insufficient_evidence", "items": []})
        rec = self._validate([withheld], keep=no_results)["records"][0]
        self.assertEqual((rec["status"], rec["stages_occupied"]), ("usable", ["proposed", "tested", "interpreted"]))
        guessed = self._validate([claim()], keep=no_results)["records"][0]          # quotes a passage it was not shown
        self.assertEqual(guessed["status"], "needs_attention")
        overreach = claim(results={"status": "none_in_document", "items": []})
        rec = self._validate([overreach], keep=no_results)["records"][0]
        self.assertTrue(any("without the full text" in p for p in rec["problems"]))
        self.assertEqual(rec["status"], "needs_attention")

    def test_an_altered_number_in_a_result_is_caught(self):
        wrong = claim(results={"status": "present", "items": [
            {"study": "0", "summary": "d = 0.80", "evidence": [ev("P0004", "d = 0.80, p = .04")]}]})
        rec = self._validate([wrong])["records"][0]
        self.assertEqual(rec["status"], "needs_attention")
        self.assertEqual(rec["results"]["items"][0]["evidence"][0]["near_miss"]["differences"],
                         [{"source_has": "3", "quote_has": "8"}])

    def test_one_quotation_used_as_result_and_interpretation_is_pointed_out(self):
        same = claim(interpretation={"status": "present", "items": [
            {"summary": "Lower.", "hedging": "asserted", "evidence": [ev("P0004", "d = 0.30, p = .04")]}]})
        rec = self._validate([same])["records"][0]
        self.assertEqual(rec["quotations_used_for_two_stages"], [{"quote": "d = 0.30, p = .04",
                                                                  "stages": ["interpretation", "result"]}])
        self.assertEqual(rec["status"], "usable")

    def test_attribution_must_be_shown(self):
        unnamed = dict(THEORY, holder_name=None, whose_evidence=None)
        problems = self._validate([unnamed])["records"][0]["problems"]
        self.assertTrue(any("not named" in p for p in problems))
        self.assertTrue(any("no quotation showing the attribution" in p for p in problems))

    def test_a_flag_is_kept_apart_from_the_stages(self):
        flagged = claim(flag="The interpretation is stronger than the result.")
        rec = self._validate([flagged])["records"][0]
        plain = self._validate([claim()])["records"][0]
        self.assertEqual(rec["flag"]["labels"]["basis"], "model_reading")
        for stage in ("test", "results", "interpretation", "unresolved"):
            self.assertEqual(rec[stage], plain[stage])

    def test_another_teams_test_is_marked_second_hand(self):
        cited = claim(holder="cited", holder_name="Baum (2007)", stance="report",
                      whose_evidence=ev("P0001", "According to the strength model (Baum, 2007)"),
                      test={"status": "present", "whose": "cited", "studies": [], "description": "Reported from Baum.",
                            "evidence": [ev("P0003", "persistence on puzzles was then timed")]})
        rec = self._validate([cited])["records"][0]
        self.assertTrue(rec["second_hand"])
        self.assertEqual(rec["test"]["study_ids"], [])

    def test_marked_statements_that_no_record_cites_are_listed(self):
        out = claims.unrecorded_marked_statements(CLAIM_SRC, ["P%04d" % i for i in range(1, 8)],
                                                  self._validate([claim()])["records"])
        self.assertEqual([e["passage_id"] for e in out["in_no_record"]], ["P0007"])
        self.assertEqual(out["labels"]["basis"], "recorded_or_computed")

    def test_one_document_end_to_end_with_a_stand_in(self):
        tmp = tempfile.TemporaryDirectory()
        try:
            run = Run("test", runs_dir=Path(tmp.name) / "runs", argv=["test"])
            stand_in = providers.StandInProvider(replies=[json.dumps({"propositions": [claim(), THEORY]}), "no"])
            out = claims.extract_claims(run, stand_in, "stand-in", CLAIM_SRC, "ego depletion", ["self-control"])
            c = out["counts"]
            self.assertEqual((c["records"], c["usable"], c["tested_here"], c["no_test_in_document"]), (2, 2, 1, 1))
            self.assertEqual(c["quotations"], c["quotations_located"])
            sent = json.loads((run.dir / "model_calls/0001.json").read_text())["prompt_as_sent"]
            self.assertIn("ego depletion", sent)
            self.assertIn("[P0004 | archive label: RESULTS]", sent)
            self.assertIn("could not be read", claims.extract_claims(
                run, stand_in, "stand-in", CLAIM_SRC, "ego depletion", ["self-control"])["problem"])
        finally:
            tmp.cleanup()

    def test_the_prompt_names_every_allowed_value_and_has_its_slots(self):
        text = (PROMPT_DIR / (claims.PROMPT_ID + ".md")).read_text(encoding="utf-8")
        for word in claims.HOLDER + claims.STANCE + claims.TEST_STATUS + claims.SAID_STATUS + claims.HEDGING:
            self.assertIn(word, text)
        for slot in ("{{concept}}", "{{neighbors}}", "{{coverage}}", "{{document}}"):
            self.assertIn(slot, text)


@unittest.skipUnless((fd.PILOT_DIR / "source_lock.json").exists(), "Prepared pilot documents required")
class SmokeTest(unittest.TestCase):
    """The plumbing test must tell a model that read the whole paper from one that did not."""

    def setUp(self):
        import audit
        self.audit = audit
        self.tmp = tempfile.TemporaryDirectory()
        self.sources = fd.load_full_text_sources()
        self.last = fd.get_block(self.sources["E02"], "P0035")["text"]
        self.final_sentence = "However, our results pave the way" + self.last.split("However, our results pave the way")[1]

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, replies):
        run = Run("test", runs_dir=Path(self.tmp.name) / "runs", argv=["test"])
        return self.audit.smoke_test(run, providers.StandInProvider(replies=replies), "stand-in", self.sources)

    def test_a_faithful_copy_passes_both_tests(self):
        first = fd.get_block(self.sources["E02"], "P0014")["text"].split(") on a")[0] + ")"
        short, whole = self._run(['"' + first[:90] + '"', self.final_sentence])
        self.assertTrue(short["call_ok"] and whole["call_ok"])
        self.assertEqual(short["quotation_outcome"], "exact")
        self.assertTrue(whole["copied_the_final_sentence"])

    def test_a_model_that_saw_only_the_start_is_caught(self):
        early = fd.get_block(self.sources["E02"], "P0002")["text"].split(". ")[0] + "."
        short, whole = self._run(["An invented sentence that is not in the passage.", early])
        self.assertEqual(short["quotation_outcome"], "not_found")
        self.assertEqual(whole["quotation_outcome"], "found_in_other_passage")
        self.assertFalse(whole["copied_the_final_sentence"])

    def test_a_failed_call_is_reported_as_a_plumbing_fault(self):
        results = self._run([])  # the stand-in has nothing to say, so both calls fail
        self.assertTrue(all(not r["call_ok"] and r["problem"] for r in results))


if __name__ == "__main__":
    unittest.main()
