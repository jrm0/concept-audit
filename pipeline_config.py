"""
pipeline_config.py — One config file per concept; every lineage script reads it.

Why this exists (2026-09-08, protocol refactor). Before this, the concept
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


# Prefixes that English writes solid, hyphenated or spaced with no change of
# meaning: "posttraumatic" = "post-traumatic" = "post traumatic". canon() joins
# them; term_pattern() matches all three. Found 2026-09-13: the family note
# claimed canon() covered "post-traumatic", it did not — 15,879 PTSD documents
# in the hyphenated spelling were invisible to the PTSD pattern and their
# mentions leaked into "traumatic stress".
_PREFIXES = ("post", "pre", "non", "anti", "inter", "intra", "multi", "sub",
             "counter", "semi", "over", "under", "meta", "hyper", "hypo",
             "psycho", "neuro", "socio", "bio")   # no 2-letter prefixes: "co" splits "control"
_PREFIX_JOIN = re.compile(r"\b(" + "|".join(_PREFIXES) + r")[\s\-]+(?=[a-z]{3,})", re.I)
_PREFIX_WORD = re.compile(r"^(" + "|".join(_PREFIXES) + r")([a-z]{3,})$", re.I)

# Unicode dashes. Publisher-derived abstracts (Wiley especially) write
# "post‐traumatic" with U+2010 HYPHEN, not U+002D. Found 2026-09-14: 1,484 of
# the "traumatic stress" leak sentences had U+2010, 86 had ASCII "-". A
# non-ASCII dash is a non-word character, so "\btraumatic" matched right
# after it while no separator class or hyphen rule recognised it. All text is
# normalised at ingest (16) and every separator class also accepts them.
_DASHES = "‐‑‒–—⁃−­﹣－"
_DASH_RE = re.compile("[" + _DASHES + "]")
_SEP = "[-\\s" + _DASHES + "]"          # one separator: hyphen (any), space


def normalize_text(s):
    """ASCII hyphen for every dash variant; ordinary space for NBSP/thin
    spaces. Applied to titles and abstracts in 16 before anything is matched."""
    if not s:
        return s
    s = _DASH_RE.sub("-", s)
    return re.sub("[     ]", " ", s)


def canon(term):
    """Hyphens and spaces are the SAME separator (see 16's docstring), and a
    prefix + separator + word is the solid word ("post-traumatic" ->
    "posttraumatic")."""
    t = normalize_text((term or "").strip().lower())
    t = _PREFIX_JOIN.sub(r"\1", t)
    return re.sub(r"[\s\-]+", " ", t)


def term_pattern(term):
    """Regex matching either separator between words, and an optional
    separator after a recognised prefix inside a word, so one canonical term
    captures all surface variants."""
    words = []
    for w in canon(term).split(" "):
        m = _PREFIX_WORD.match(w)
        if m:
            words.append(re.escape(m.group(1)) + _SEP + "?" + re.escape(m.group(2)))
        else:
            words.append(re.escape(w))
    body = (_SEP + "+").join(words)
    return re.compile(r"\b" + body + r"\b", re.I)


# ------------------------------------------------------- mention extraction
#
# 2026-09-13 (Session 29). The first PTSD/trauma run produced 49 "supported"
# within-period splits, and reading showed the four focal terms' splits were
# SYNTAX, not sense:
#   * "ptsd": documents where PTSD occurs only inside "(PTSD)" after the long
#     form vs documents that use it as a word (Jaccard 1.00, silhouette .52).
#   * "traumatic stress": 89% of one cluster was "post-traumatic stress
#     disorder" — the hyphen is a word boundary, so the hyphenated PTSD
#     spelling leaked into the shorter term.
#   * "posttraumatic stress disorder": definitional sentence
#     ("... disorder (PTSD) is ...") vs use.
#   * "trauma": the trauma-informed literature, matched INSIDE "trauma-informed".
# Stability metrics endorse any consistent partition, and in masked-context
# embeddings the most consistent partitions are syntactic. Three rules, all
# config-switchable under `mentions:` so the old behaviour is a sensitivity run:
#   exclude_nested (default True): a match of term T that lies inside a match
#     of a LONGER family term does not count as a mention of T. The parent's
#     shard becomes bare uses only; compounds count for the compound.
#   strip_acronym_parentheticals (default True): "(PTSD)" immediately after a
#     family-term match is removed from the context sentence; a parenthetical-
#     only occurrence "(PTSD)" is never a mention of "ptsd".
# The syntax guard in 18 reports what is left (title-sentence, definitional,
# parenthetical, attributive rates per cluster) and flags splits that differ
# by more than 0.5 on any of them.

#   exclude_hyphen_prefixed (default True): a match that is the TAIL of a
#     hyphenated word ("post-traumatic stress symptoms", "non-trauma",
#     "re-trauma") is a different lexeme. The solid spelling never matched
#     ("posttraumatic" has no boundary), so without this rule the hyphenated
#     spelling alone leaked — 42% of "traumatic stress" sentences after the
#     nesting rule were still "post-traumatic stress ...".
DEFAULT_MENTION_RULES = {"exclude_nested": True,
                         "strip_acronym_parentheticals": True,
                         "exclude_hyphen_prefixed": True}


def mention_rules(cfg):
    r = dict(DEFAULT_MENTION_RULES)
    r.update(cfg.get("mentions") or {})
    return r


def family_patterns(cfg):
    """{canonical family term: compiled pattern} from the config's family CSV."""
    import csv
    pats = {}
    with open(cfg["family_csv"], encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = canon(row.get("term"))
            if t and t not in pats:
                pats[t] = term_pattern(t)
    return pats


_PAREN_AFTER = re.compile(r"\s*\(\s*([A-Za-z][A-Za-z\-]{0,11})\s*\)")


def strip_acronym_parentheticals(sentence, fam_pats):
    """'posttraumatic stress disorder (PTSD) is' -> 'posttraumatic stress disorder is'.
    Removes a short parenthetical that immediately follows a family-term match
    when it is an acronym (all caps), a family term, or a short mixed-case
    token. The parenthetical is a definitional device, not a use of either
    term — and it was the strongest 'cluster' in the first PTSD run."""
    cuts = []
    for pat in fam_pats.values():
        for m in pat.finditer(sentence):
            pm = _PAREN_AFTER.match(sentence, m.end())
            if not pm:
                continue
            inner = pm.group(1)
            if (inner.isupper() or canon(inner) in fam_pats
                    or (len(inner) <= 6 and any(c.isupper() for c in inner))):
                cuts.append((pm.start(), pm.end()))
    if not cuts:
        return sentence, 0
    cuts.sort()
    out, pos, n = [], 0, 0
    for a, b in cuts:
        if a < pos:
            continue
        out.append(sentence[pos:a])
        pos = b
        n += 1
    out.append(sentence[pos:])
    return "".join(out), n


def all_spans(sentence, pats):
    """{term: [(start, end), ...]} for every pattern that matches."""
    out = {}
    for t, pat in pats.items():
        sp = [(m.start(), m.end()) for m in pat.finditer(sentence)]
        if sp:
            out[t] = sp
    return out


def _parenthetical(sentence, span):
    a, b = span
    i = a - 1
    while i >= 0 and sentence[i] == " ":
        i -= 1
    j = b
    while j < len(sentence) and sentence[j] == " ":
        j += 1
    return i >= 0 and sentence[i] == "(" and j < len(sentence) and sentence[j] == ")"


def valid_spans(term, spans_by_term, sentence, rules):
    """Spans of `term` that count as mentions under the rules (see above).
    `spans_by_term` is all_spans(sentence, patterns) for the same sentence."""
    spans = list(spans_by_term.get(term, []))
    if not spans:
        return []
    if rules.get("exclude_nested", True):
        containers = [c for t2, sp in spans_by_term.items()
                      if t2 != term and len(t2) > len(term) for c in sp]
        spans = [s for s in spans
                 if not any(c[0] <= s[0] and s[1] <= c[1] and c != s
                            for c in containers)]
    if rules.get("exclude_hyphen_prefixed", True):
        spans = [s for s in spans
                 if not (s[0] >= 2 and sentence[s[0] - 1] in "-" + _DASHES
                         and sentence[s[0] - 2].isalnum())]
    spans = [s for s in spans if not _parenthetical(sentence, s)]
    return spans


# Syntax-guard features of one mention (computed on the cleaned sentence).
_DEFN_AFTER = re.compile(r"^[\s,]*(is|are|was|were|refers?|denotes?|describes?|"
                         r"can be|has been|have been|is defined|are defined)\b", re.I)
_DEFN_BEFORE = re.compile(r"(defined as|referred to as|known as|termed|called|"
                          r"the term|the concept of|the construct of)\s*$", re.I)


# Grammatical FRAME of the mention (Session 30). With the lexical artifacts
# removed, the clean splits that remained followed the term's syntactic role:
# "posttraumatic stress SYMPTOMS" (modifier) vs "... and posttraumatic stress
# in ..." (head); "depression AND PTSD AND ..." (coordinated list) vs "of/with
# PTSD" (head with preposition); "CHILDHOOD trauma" (premodified head) vs
# "trauma EXPOSURE" (modifier). Masked-token BERT vectors carry syntactic role
# strongly. These three features let the guard NAME a frame split.
_FUNCTION_WORDS = set("""a an the this that these those of in on at to for with by from as into
onto over under about among between within without during after before since until
and or nor but yet so is are was were be been being has have had do does did not no
can could may might must shall should will would than then there their its it he she
they we you i who whom which what when where how if because while although though
such very more most less least many much some any each every all both either neither
one two three several few own same other another per via versus vs against toward
towards including regarding following related associated based due""".split())
_NEXT_TOKEN = re.compile(r"^[\s]*([A-Za-z][A-Za-z\-]*)")
_PREV_TOKEN = re.compile(r"([A-Za-z][A-Za-z\-]*)[\s]*$")
_COORD_NEXT = re.compile(r"^\s*(,|;|/|\band\b|\bor\b|&)", re.I)
_COORD_PREV = re.compile(r"(,|;|/|\band\b|\bor\b|&)\s*$", re.I)


def syntax_features(sentence, span, title=None):
    """Per-mention 0/1 features, in SYNTAX_FEATURE_NAMES order:
    title       — the mention sentence is the title
    defn        — definitional frame ("X is a ...", "defined as X")
    paren       — the term sits inside parentheses
    attributive — hyphenated attributive ("trauma-focused")
    modifier    — followed directly by a content word ("PTSD symptoms")
    premodified — preceded directly by a content word ("childhood trauma")
    coordinated — adjacent to a list coordinator ("depression and PTSD, ...")"""
    a, b = span
    is_title = 0
    if title:
        t = title.strip().rstrip(".").lower()[:60]
        is_title = int(sentence.strip().rstrip(".").lower()[:60] == t)
    defn = int(bool(_DEFN_AFTER.match(sentence[b:b + 40]))
               or bool(_DEFN_BEFORE.search(sentence[max(0, a - 30):a])))
    paren = int(_parenthetical(sentence, span))
    after, before = sentence[b:], sentence[:a]
    attributive = int(after[:1] == "-" and after[1:2].isalpha())
    # Crude POS: a following/preceding word ending in -ed is (almost always) a
    # verb, not the noun that would make this a modifier/premodified frame
    # ("PTSD showed", "assessed PTSD"); -ing/-ly on the left are verb/adverb.
    mn = _NEXT_TOKEN.match(after)
    nxt = mn.group(1).lower() if mn else ""
    modifier = int(bool(mn) and not attributive
                   and nxt not in _FUNCTION_WORDS and not nxt.endswith("ed")
                   and after[:len(mn.group(0))].strip() == mn.group(1))   # no punctuation between
    mp = _PREV_TOKEN.search(before)
    prv = mp.group(1).lower() if mp else ""
    premodified = int(bool(mp) and prv not in _FUNCTION_WORDS
                      and not prv.endswith(("ed", "ing", "ly"))
                      and not before.rstrip().endswith("-"))
    coordinated = int(bool(_COORD_NEXT.match(after)) or bool(_COORD_PREV.search(before)))
    return [is_title, defn, paren, attributive, modifier, premodified, coordinated]


SYNTAX_FEATURE_NAMES = ["title", "defn", "paren", "attributive",
                        "modifier", "premodified", "coordinated"]
FRAME_FEATURES = ["attributive", "modifier", "premodified", "coordinated"]


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
