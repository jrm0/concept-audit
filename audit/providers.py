"""Concept Audit - ways of reaching a model.

Five providers, all with the same small interface:

  StandInProvider   not a model. Returns replies written in advance, so the
                    record-keeping can be tested without spending anything.
                    Every record it produces says is_model = false.
  AnthropicProvider Anthropic's Messages API.
  OpenAIProvider    OpenAI's Chat Completions API.
  GoogleProvider    Google's Gemini API (generateContent).
  LocalProvider     An open-weights model running on this computer through
                    Ollama. No key, no charge, and nothing leaves the machine.

The three real providers use only Python's standard library, so nothing has
to be installed. THEY HAVE NOT BEEN RUN AGAINST A LIVE API: no key was
available when this was written (2026-11-12). The first live call is their
test. `python3 code/audit/audit.py check-models` is the free first step: it
asks each provider which models the key can use, and spends nothing.

Keys are read from code/.secrets/api_keys.yaml (entries named `anthropic`,
`openai` and `google`) or from the environment (ANTHROPIC_API_KEY,
OPENAI_API_KEY, GOOGLE_API_KEY).
A key is never written to a run record or printed.
"""
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

SECRETS = Path(__file__).resolve().parent.parent / ".secrets" / "api_keys.yaml"


class ProviderError(RuntimeError):
    def __init__(self, message, body=None):
        super().__init__(message)
        self.body = body


def read_key(name, env):
    """Find a key by name without ever printing it. Returns None if absent."""
    if os.environ.get(env):
        return os.environ[env]
    if SECRETS.exists():
        for line in SECRETS.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\s*%s\s*:\s*['\"]?([^'\"#\s]+)" % re.escape(name), line)
            if m:
                return m.group(1)
    return None


class StandInProvider:
    """Not a model. Hands back prepared replies in order, or by a rule."""
    name = "stand_in"
    is_model = False

    def __init__(self, replies=None, rule=None):
        self.replies = list(replies or [])
        self.rule = rule

    def complete(self, model, system, prompt, temperature, max_tokens):
        if self.rule is not None:
            text = self.rule(prompt)
        elif self.replies:
            text = self.replies.pop(0)
        else:
            raise ProviderError("Stand-in provider has no reply left")
        return {"text": text, "model_returned": "stand-in (not a model)",
                "usage": {"input_tokens": None, "output_tokens": None},
                "raw": {"stand_in": True, "text": text}}


def _post(url, headers, body, timeout=300):
    return _send(url, headers, body, "POST", timeout)


def _get(url, headers, timeout=30):
    return _send(url, headers, None, "GET", timeout)


def _send(url, headers, body, method, timeout):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", "replace")
        raise ProviderError("HTTP %s from %s" % (exc.code, url), body=text[:4000])
    except urllib.error.URLError as exc:
        raise ProviderError("Could not reach %s: %s" % (url, exc.reason))


class AnthropicProvider:
    name = "anthropic"
    is_model = True
    url = "https://api.anthropic.com/v1/messages"

    def __init__(self, key=None):
        self._key = key or read_key("anthropic", "ANTHROPIC_API_KEY")
        if not self._key:
            raise ProviderError("No Anthropic key found (api_keys.yaml entry `anthropic` or ANTHROPIC_API_KEY)")

    def complete(self, model, system, prompt, temperature, max_tokens):
        body = {"model": model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}]}
        if system:
            body["system"] = system
        raw = _post(self.url, {"x-api-key": self._key, "anthropic-version": "2023-06-01",
                               "content-type": "application/json"}, body)
        text = "".join(part.get("text", "") for part in raw.get("content", []) if part.get("type") == "text")
        usage = raw.get("usage") or {}
        return {"text": text, "model_returned": raw.get("model"),
                "usage": {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")},
                "raw": raw}

    def list_models(self):
        raw = _get("https://api.anthropic.com/v1/models?limit=100",
                   {"x-api-key": self._key, "anthropic-version": "2023-06-01"})
        return sorted(m.get("id") for m in raw.get("data", []) if m.get("id"))


class OpenAIProvider:
    name = "openai"
    is_model = True
    url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, key=None):
        self._key = key or read_key("openai", "OPENAI_API_KEY")
        if not self._key:
            raise ProviderError("No OpenAI key found (api_keys.yaml entry `openai` or OPENAI_API_KEY)")

    def complete(self, model, system, prompt, temperature, max_tokens):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_completion_tokens": max_tokens}
        raw = _post(self.url, {"Authorization": "Bearer " + self._key, "content-type": "application/json"}, body)
        choice = (raw.get("choices") or [{}])[0]
        usage = raw.get("usage") or {}
        return {"text": (choice.get("message") or {}).get("content"), "model_returned": raw.get("model"),
                "usage": {"input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens")},
                "raw": raw}

    def list_models(self):
        raw = _get("https://api.openai.com/v1/models", {"Authorization": "Bearer " + self._key})
        return sorted(m.get("id") for m in raw.get("data", []) if m.get("id"))


class GoogleProvider:
    name = "google"
    is_model = True
    base = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, key=None):
        self._key = key or read_key("google", "GOOGLE_API_KEY")
        if not self._key:
            raise ProviderError("No Google key found (api_keys.yaml entry `google` or GOOGLE_API_KEY)")

    def complete(self, model, system, prompt, temperature, max_tokens):
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        # The key travels in a header, never in the address, so it cannot leak into an error message.
        raw = _post("%s/%s:generateContent" % (self.base, model),
                    {"x-goog-api-key": self._key, "content-type": "application/json"}, body)
        candidate = (raw.get("candidates") or [{}])[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        usage = raw.get("usageMetadata") or {}
        output = usage.get("candidatesTokenCount")
        if output is not None and usage.get("thoughtsTokenCount"):
            output += usage["thoughtsTokenCount"]  # tokens the model spent thinking are billed as output
        return {"text": "".join(p.get("text", "") for p in parts if not p.get("thought")),
                "model_returned": raw.get("modelVersion"),
                "usage": {"input_tokens": usage.get("promptTokenCount"), "output_tokens": output},
                "raw": raw}

    def list_models(self):
        raw = _get(self.base + "?pageSize=200", {"x-goog-api-key": self._key})
        return sorted(m.get("name", "").replace("models/", "") for m in raw.get("models", [])
                      if "generateContent" in (m.get("supportedGenerationMethods") or []))


class LocalProvider:
    """An open-weights model served by Ollama on this computer.

    Two things matter here that do not arise with the paid services.

    1. Ollama reads only as much of a prompt as its "context window" allows,
       and its default window is small. A long paper would be cut off without
       any error, and the model would answer about the part it saw. So every
       call asks for a window big enough for the prompt plus the reply, and
       the reply is checked: if the model reports reading far fewer tokens
       than the prompt holds, or reports that it ran out of room, the call
       fails loudly instead of returning a reading of half a paper.
    2. The weights have an exact fingerprint. It is recorded with every call,
       so a run can be repeated against exactly the same model.
    """
    name = "local"
    is_model = True
    free = True

    def __init__(self, host=None, think=None):
        # think: None leaves the model's own default; False asks a "thinking" model to answer
        # without working through the problem first (faster, usually less careful); True asks it to.
        self.think = think
        self.options = {"think": think}
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self._digests = None

    @staticmethod
    def window_for(prompt, system, max_tokens):
        """Tokens of context to ask for: a cautious estimate, rounded up."""
        chars = len(prompt) + len(system or "")
        needed = int(chars / 3.0) + max_tokens + 512  # English runs near 4 characters a token; 3 leaves a margin
        return max(8192, ((needed // 4096) + 1) * 4096)

    def _installed(self):
        if self._digests is None:
            raw = _get(self.host + "/api/tags", {})
            self._digests = {m.get("name"): m.get("digest") for m in raw.get("models", [])}
        return self._digests

    def list_models(self):
        return sorted(k for k in self._installed() if k)

    def complete(self, model, system, prompt, temperature, max_tokens):
        window = self.window_for(prompt, system, max_tokens)
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": model, "messages": messages, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens, "num_ctx": window}}
        if self.think is not None:
            body["think"] = self.think
        raw = _post(self.host + "/api/chat", {"content-type": "application/json"}, body, timeout=3600)
        read = raw.get("prompt_eval_count")
        written = raw.get("eval_count")
        expected_low = int((len(prompt) + len(system or "")) / 8.0)  # no English text is this sparse in tokens
        check = {"window_requested": window, "prompt_tokens_reported": read, "done_reason": raw.get("done_reason"),
                 "least_plausible_prompt_tokens": expected_low, "verdict": "ok"}
        if read is not None and read < expected_low:
            check["verdict"] = "prompt probably cut off: far fewer tokens read than the prompt holds"
        elif read is not None and written is not None and read + written >= window:
            check["verdict"] = "window filled: the prompt or the reply may have been cut off"
        elif read is None:
            check["verdict"] = "not verified: the server did not report how many prompt tokens it read"
        if check["verdict"] != "ok" and not check["verdict"].startswith("not verified"):
            raise ProviderError("Local model call rejected - " + check["verdict"], body=json.dumps(check))
        text = (raw.get("message") or {}).get("content")
        if not (text or "").strip():
            thought = (raw.get("message") or {}).get("thinking")
            spent = ("the model used its whole allowance of %d tokens thinking before it answered; "
                     "allow more tokens or run with --thinking off" % max_tokens) if thought else "no text came back"
            check["thinking_chars"] = len(thought or "")
            raise ProviderError("Local model call returned an empty reply (%s; stopped because: %s)"
                                % (spent, raw.get("done_reason")), body=json.dumps(check))
        check["thinking_chars"] = len((raw.get("message") or {}).get("thinking") or "")
        digest = (self._installed().get(model) or self._installed().get(raw.get("model")) or "")
        version = "%s@%s" % (raw.get("model") or model, digest[:12]) if digest else (raw.get("model") or model)
        return {"text": text, "model_returned": version,
                "usage": {"input_tokens": read, "output_tokens": written},
                "raw": {"response": raw, "context_check": check, "weights_digest": digest or None}}


PROVIDERS = {"anthropic": AnthropicProvider, "openai": OpenAIProvider, "google": GoogleProvider,
             "local": LocalProvider, "stand_in": StandInProvider}
REAL_PROVIDERS = ("anthropic", "openai", "google")
CHECKED_PROVIDERS = REAL_PROVIDERS + ("local",)
