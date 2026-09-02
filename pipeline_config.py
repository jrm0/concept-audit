"""
pipeline_config.py — One config file per concept; every lineage script reads it.

Why this exists (2026-09-02, protocol refactor). Before this, the concept
name, family file, control terms, period floor, event dates, term pairs,
venue-category regexes, citation anchors and even result numbers in figure
titles were literals spread across 16/17/18/20/22/23/23b. Running a second
construct meant editing six scripts. Now: one YAML in configs/, and

    python3 18_lineage_analysis.py --config configs/ptsd.yaml

The default config is configs/ego_depletion.yaml, so existing commands keep
working unchanged.

Also provides:
  * shared helpers that were duplicated across scripts (slug, canon,
    term_pattern, period_of, venue_cat)
  * a RUN MANIFEST written next to every script's outputs
    (results/<slug>/manifest_<script>.json + append to run_log.jsonl):
    timestamp, git commit, config path + content hash, parameters, library
    versions. Outputs without a manifest are the reason lineage_tree.json
    went stale without anyone noticing.

Requires: pip install pyyaml
"""

import datetime as _dt
import hashlib
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    raise SystemExit("pipeline_config needs PyYAML:  pip install pyyaml")

DEFAULT_CONFIG = "configs/ego_depletion.yaml"


class Config(dict):
    """dict with attribute access and a few typed conveniences."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def path(self, key):
        return Path(self[key])

    @property
    def results(self):
        p = Path(self["results_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def controls(self):
        return list(self.get("control_terms", []))


def load(argv=None):
    """Return (cfg, remaining_argv). Pulls --config PATH out of argv."""
    argv = list(sys.argv[1:] if argv is None else argv)
    path = DEFAULT_CONFIG
    if "--config" in argv:
        i = argv.index("--config")
        if i + 1 >= len(argv):
            raise SystemExit("--config needs a path")
        path = argv[i + 1]
        del argv[i:i + 2]
    else:
        for i, a in enumerate(argv):
            if a.startswith("--config="):
                path = a.split("=", 1)[1]
                del argv[i]
                break
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Config not found: {p}  (cwd={Path.cwd()})")
    with open(p, encoding="utf-8") as f:
        raw = f.read()
    cfg = Config(yaml.safe_load(raw) or {})
    cfg["_config_path"] = str(p)
    cfg["_config_sha1"] = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    for key in ("concept", "slug", "docs_path", "emb_dir", "results_dir"):
        if key not in cfg:
            raise SystemExit(f"Config {p} is missing required key '{key}'")
    return cfg, argv


# ------------------------------------------------------------ shared helpers

def slug(t):
    return re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_")


def canon(term):
    """Hyphens and spaces are the SAME separator (see 16's docstring)."""
    return re.sub(r"[\s\-]+", " ", (term or "").strip().lower())


def term_pattern(term):
    """Regex matching either separator, so one canonical term captures all
    surface variants."""
    body = re.escape(term).replace(r"\ ", "[-\\s]+").replace(r"\-", "[-\\s]+")
    return re.compile(r"\b" + body + r"\b", re.I)


def period_of(year, cfg):
    per = cfg["periods"]
    y0min, ymax, w = int(per["year_min"]), int(per["year_max"]), int(per["width"])
    if year is None or year < y0min:
        return None
    y0 = y0min + ((year - y0min) // w) * w
    return f"{y0}-{min(y0 + w - 1, ymax)}"


def venue_categorizer(cfg):
    """Returns f(journal_name) -> category name, first regex match wins."""
    cats = [(c["name"], re.compile(c["pattern"], re.I))
            for c in cfg.get("venue_categories", [])]
    core = cfg.get("core_label", "core/other psych")

    def cat(j):
        j = (j or "")
        for name, pat in cats:
            if pat.search(j):
                return name
        return core
    cat.names = [n for n, _ in cats] + [core]
    cat.core = core
    return cat


def events(cfg, short=False):
    src = cfg.get("event_short_labels" if short else "events", {}) or {}
    return {int(k): str(v) for k, v in src.items()}


# ------------------------------------------------------------ run manifest

import os as _os
# GIT_OPTIONAL_LOCKS=0: read-only queries must never create .git/index.lock
# (a leftover lock from a sandboxed run once blocked every commit).
_GIT_ENV = dict(_os.environ, GIT_OPTIONAL_LOCKS="0")


def _git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent, env=_GIT_ENV).decode().strip()
    except Exception:
        return None


def _git_dirty():
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "*.py", "configs"],
            stderr=subprocess.DEVNULL, cwd=Path(__file__).parent,
            env=_GIT_ENV).decode()
        return bool(out.strip())
    except Exception:
        return None


def _versions():
    v = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "sklearn", "diptest", "torch", "transformers"):
        try:
            m = __import__(mod)
            v[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return v


def write_manifest(cfg, script, params=None, outputs=None, notes=None):
    """Write results/<slug>/manifest_<script>.json and append to run_log.jsonl."""
    name = Path(script).stem
    man = {
        "script": name,
        "concept": cfg["concept"],
        "config": cfg["_config_path"],
        "config_sha1": cfg["_config_sha1"],
        "timestamp": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git_hash(),
        "git_dirty": _git_dirty(),
        "argv": sys.argv[1:],
        "params": params or {},
        "outputs": [str(o) for o in (outputs or [])],
        "versions": _versions(),
    }
    if notes:
        man["notes"] = notes
    out = cfg.results / f"manifest_{name}.json"
    out.write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")
    with open(cfg.results / "run_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(man, default=str) + "\n")
    print(f"[manifest] {out}")
    return man
