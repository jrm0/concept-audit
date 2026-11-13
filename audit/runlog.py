"""Concept Audit - run records and the model-call record.

Every run of the audit gets its own folder that is never overwritten:

  manifest.json      what was run, when, with which code and which inputs
  events.jsonl       a line for each thing that happened, in order
  model_calls/       one file per model call: the prompt as sent, the model
                     asked for, the model that answered, the settings, the
                     complete raw reply, token counts, time taken, and cost
                     when a price has been entered
  (outputs)          whatever the command produced

Nothing in this file decides anything about a paper. It keeps the record.
"""
import datetime
import hashlib
import json
import platform
import re
import sys
import time
import uuid
from pathlib import Path

HARNESS_VERSION = "0.1.0-step1"
AUDIT_DIR = Path(__file__).resolve().parent
CODE_DIR = AUDIT_DIR.parent
ROOT = CODE_DIR.parent
DEFAULT_RUNS = ROOT / "revision_work/2026-11-12/concept_audit/runs"
PROMPT_DIR = AUDIT_DIR / "prompts"


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class Run:
    """One execution of the audit, with its folder and its record."""

    def __init__(self, command, runs_dir=None, config=None, argv=None):
        runs_dir = Path(runs_dir or DEFAULT_RUNS)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = "%s_%s_%s" % (stamp, command, uuid.uuid4().hex[:6])
        self.dir = runs_dir / self.run_id
        self.dir.mkdir(parents=True, exist_ok=False)  # never reuse a run folder
        (self.dir / "model_calls").mkdir()
        self.cache_dir = None  # set by a command to switch on replay of saved replies
        self._calls = 0
        self._started = time.time()
        code = {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(AUDIT_DIR.glob("*.py"))}
        pilot = CODE_DIR / "reading_pilot.py"
        if pilot.exists():
            code[str(pilot.relative_to(ROOT))] = sha256_file(pilot)
        self.manifest = {
            "run_id": self.run_id, "command": command, "argv": list(argv if argv is not None else sys.argv),
            "harness_version": HARNESS_VERSION, "started": now(), "finished": None, "status": "running",
            "python": sys.version.split()[0], "platform": platform.platform(),
            "code_sha256": code, "config": config or {}, "inputs_sha256": {},
            "prompts": {}, "models": {}, "model_calls": 0, "replayed_calls": 0,
            "tokens": {"input": 0, "output": 0},
            "cost_usd": None, "cost_note": "no model calls", "outputs": [], "notes": [],
        }
        self._save()
        self.event("run_started", command=command)

    # -- record keeping -----------------------------------------------------
    def _save(self):
        write_json(self.dir / "manifest.json", self.manifest)

    def event(self, kind, **details):
        line = {"time": now(), "event": kind}
        line.update(details)
        with (self.dir / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    def add_input(self, name, sha256):
        self.manifest["inputs_sha256"][name] = sha256

    def note(self, text):
        self.manifest["notes"].append(text)
        self.event("note", text=text)

    def write_output(self, name, data):
        """Save an output file in the run folder and list it in the manifest."""
        path = self.dir / name
        if isinstance(data, str):
            path.write_text(data, encoding="utf-8")
        else:
            write_json(path, data)
        self.manifest["outputs"].append({"file": name, "sha256": sha256_file(path)})
        self.event("output_written", file=name)
        self._save()
        return path

    def finish(self, status="completed"):
        self.manifest["finished"] = now()
        self.manifest["status"] = status
        self.manifest["seconds"] = round(time.time() - self._started, 2)
        self._save()
        self.event("run_finished", status=status)

    # -- prompts -------------------------------------------------------------
    def load_prompt(self, prompt_id, prompt_dir=None):
        """Read a prompt file and record its exact version (a hash of its text).

        A prompt lives in code/audit/prompts/<prompt_id>.md. Changing one
        character changes the hash, so a run always shows which wording it used.
        """
        path = Path(prompt_dir or PROMPT_DIR) / (prompt_id + ".md")
        text = path.read_text(encoding="utf-8")
        try:
            shown = str(path.resolve().relative_to(ROOT))
        except ValueError:
            shown = str(path)
        self.manifest["prompts"][prompt_id] = {"file": shown, "sha256": sha256_text(text)}
        self._save()
        return text

    @staticmethod
    def render(template, variables):
        """Fill {{name}} slots. A slot with no value is an error, never a blank."""
        def sub(match):
            key = match.group(1)
            if key not in variables:
                raise KeyError("Prompt needs a value for {{%s}}" % key)
            return str(variables[key])
        return re.sub(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}", sub, template)

    # -- model calls ---------------------------------------------------------
    def _cache_file(self, provider, model, system, prompt, temperature, max_tokens):
        options = getattr(provider, "options", None)
        fields = [provider.name, model, system, prompt, temperature, max_tokens]
        if options and any(v is not None for v in options.values()):
            fields.append(options)  # e.g. thinking switched off: a different setting, so a different saved reply
        key = sha256_text(json.dumps(fields, sort_keys=True))
        return Path(self.cache_dir) / (key + ".json")

    def call_model(self, provider, model, prompt_id, variables=None, system=None,
                   temperature=0, max_tokens=2000, purpose=None, prompt_dir=None, prices=None,
                   spend_ceiling_usd=None, fresh=False):
        """Call a model through a provider and save the complete record.

        provider   an object with .name, .is_model and
                   .complete(model, system, prompt, temperature, max_tokens)
                   returning {text, model_returned, usage, raw}
        prices     {model: {"input_per_million": x, "output_per_million": y}}
                   in US dollars. If the model has no price entered, cost is
                   recorded as "not recorded" - never guessed.
        The record is written whether the call succeeds or fails.

        Replay. When the run has a cache folder, a successful reply is saved
        there. The next time exactly the same prompt goes to the same model
        with the same settings, the saved reply is used: no call is made and
        nothing is charged. The record says it was replayed and from where.
        Pass fresh=True to ask the model again regardless.
        """
        template = self.load_prompt(prompt_id, prompt_dir)
        prompt = self.render(template, variables or {})
        if spend_ceiling_usd is not None and (self.manifest["cost_usd"] or 0) >= spend_ceiling_usd:
            self.event("model_call_refused", reason="spend ceiling reached", ceiling=spend_ceiling_usd)
            raise RuntimeError("Spend ceiling reached; no further model calls in this run")
        self._calls += 1
        seq = self._calls
        record = {
            "seq": seq, "run_id": self.run_id, "purpose": purpose,
            "provider": provider.name, "is_model": bool(getattr(provider, "is_model", True)),
            "model_requested": model, "model_returned": None,
            "prompt_id": prompt_id, "prompt_template_sha256": sha256_text(template),
            "prompt_variables_sha256": {k: sha256_text(str(v)) for k, v in (variables or {}).items()},
            "system": system, "prompt_as_sent": prompt, "prompt_as_sent_sha256": sha256_text(prompt),
            "settings": {"temperature": temperature, "max_tokens": max_tokens,
                         "provider_options": getattr(provider, "options", None)},
            "started": now(), "finished": None, "seconds": None,
            "output_text": None, "raw_response": None, "usage": None,
            "cost_usd": None, "cost_note": None, "error": None,
        }
        t0 = time.time()
        use_cache = self.cache_dir is not None and record["is_model"]
        cache_file = self._cache_file(provider, model, system, prompt, temperature, max_tokens) if use_cache else None
        saved = None
        if use_cache and not fresh and cache_file.exists():
            saved = json.loads(cache_file.read_text(encoding="utf-8"))
        if saved is not None:
            reply = saved["reply"]
            record.update({"output_text": reply.get("text"), "raw_response": reply.get("raw"),
                           "model_returned": reply.get("model_returned"), "usage": reply.get("usage"),
                           "replayed_from": saved["first"]})
        else:
            try:
                reply = provider.complete(model=model, system=system, prompt=prompt,
                                          temperature=temperature, max_tokens=max_tokens)
                record.update({"output_text": reply.get("text"), "raw_response": reply.get("raw"),
                               "model_returned": reply.get("model_returned"), "usage": reply.get("usage")})
            except Exception as exc:  # the failed call is part of the record
                record["error"] = {"type": type(exc).__name__, "message": str(exc),
                                   "body": getattr(exc, "body", None)}
        record["finished"] = now()
        record["seconds"] = round(time.time() - t0, 3)
        usage = record["usage"] or {}
        price = (prices or {}).get(model)
        if saved is not None:
            record["cost_usd"] = 0.0
            first = saved["first"]
            record["cost_note"] = "replayed from %s call %s; nothing charged now (the original call cost %s)" % (
                first["run_id"], first["seq"],
                "$%.4f" % first["cost_usd"] if first.get("cost_usd") is not None else "an unrecorded amount")
        elif not record["is_model"]:
            record["cost_note"] = "stand-in provider; not a model; no cost"
        elif getattr(provider, "free", False):
            record["cost_usd"] = 0.0
            record["cost_note"] = "model runs on this computer; no charge"
        elif price and usage.get("input_tokens") is not None and usage.get("output_tokens") is not None:
            record["cost_usd"] = round(usage["input_tokens"] * price["input_per_million"] / 1e6
                                       + usage["output_tokens"] * price["output_per_million"] / 1e6, 6)
            record["cost_note"] = "tokens x the price entered in the configuration"
        else:
            record["cost_note"] = "not recorded: no price entered for this model, or no token counts returned"
        if use_cache and saved is None and record["error"] is None:
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            write_json(cache_file, {
                "reply": {"text": record["output_text"], "raw": record["raw_response"],
                          "model_returned": record["model_returned"], "usage": record["usage"]},
                "first": {"run_id": self.run_id, "seq": seq, "cost_usd": record["cost_usd"],
                          "seconds": record["seconds"], "finished": record["finished"]},
                "provider": provider.name, "model_requested": model, "prompt_id": prompt_id,
                "prompt_as_sent_sha256": record["prompt_as_sent_sha256"],
                "settings": record["settings"]})
        write_json(self.dir / "model_calls" / ("%04d.json" % seq), record)
        m = self.manifest
        if saved is not None:
            m["replayed_calls"] += 1
        m["model_calls"] = seq
        key = "%s/%s" % (provider.name, model)
        entry = m["models"].setdefault(key, {"provider": provider.name, "model_requested": model,
                                             "is_model": record["is_model"], "models_returned": [], "calls": 0})
        entry["calls"] += 1
        if record["model_returned"] and record["model_returned"] not in entry["models_returned"]:
            entry["models_returned"].append(record["model_returned"])
        if saved is None:
            m["tokens"]["input"] += usage.get("input_tokens") or 0
            m["tokens"]["output"] += usage.get("output_tokens") or 0
        if saved is not None:
            if m["cost_usd"] is None:
                m["cost_usd"] = 0.0
                m["cost_note"] = "replayed from saved replies; nothing charged in this run"
        elif record["cost_usd"] is not None:
            m["cost_usd"] = round((m["cost_usd"] or 0) + record["cost_usd"], 6)
            if not getattr(provider, "free", False):
                self._priced = True
            if "incomplete" not in (m["cost_note"] or ""):
                m["cost_note"] = ("sum of calls with a price entered" if getattr(self, "_priced", False)
                                  else "models ran on this computer; no charge")
        elif record["is_model"]:
            m["cost_note"] = "incomplete: at least one call has no recorded cost"
        elif m["cost_usd"] is None:
            m["cost_note"] = "stand-in provider only; no model was called"
        self._save()
        self.event("model_call", seq=seq, provider=provider.name, model=model, prompt_id=prompt_id,
                   ok=record["error"] is None, seconds=record["seconds"])
        if record["error"]:
            raise RuntimeError("Model call %d failed: %s" % (seq, record["error"]["message"]))
        return record
