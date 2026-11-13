"""Concept Audit - step 1 foundations.

Source and study identifiers, passage locators and exact-quotation checks,
built on the reading pilot (code/reading_pilot.py), which is imported and
never edited.

Nothing here calls a model. Everything here is "recorded or computed".

Plain-language map of this file:
  labels      - the two labels every output must carry
  sources     - the 12 full-text documents and the 120 abstracts, each with
                a stable identifier and a hash of its text
  studies     - candidate study boundaries found from headings (candidates,
                not facts)
  locators    - a short address for any stretch of text: E02:P0006:120-245
  quotations  - is this quotation really in the source, and where?
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
import reading_pilot as rp  # noqa: E402  (reused, not modified)

ROOT = rp.ROOT
PILOT_DIR = ROOT / "revision_work/2026-10-21/reading_pilot"
ABSTRACT_PACKETS = {
    "ego": ROOT / "code/results/ego_depletion_full_v2_revision_20261019/source_review_packet.jsonl",
    "trauma": ROOT / "code/results/ptsd_trauma_v2_revision_20261019/source_review_packet.jsonl",
}
# Files this code must never open. The names are checked in guard_path().
PROTECTED_NAMES = ("reserved_evaluation_pool.jsonl", "identity_catalog.jsonl")

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------
BASIS = {
    "recorded_or_computed": "recorded or computed",
    "model_reading": "a model's reading",
    "researcher_judgment": "a researcher's judgment",
}
CHECK = {
    "unchecked": "unchecked",
    "constructed_tests": "constructed tests",
    "source_reviewed": "source-reviewed",
    "held_out": "held-out evaluation",
}
NOT_IMPLEMENTED = "not implemented"
NOT_ASSESSED = "not assessed"


def labeled(record, basis, check, tool=None):
    """Attach the two required labels to an output record and return it."""
    if basis not in BASIS:
        raise ValueError("Unknown basis label: %r" % (basis,))
    if check not in CHECK:
        raise ValueError("Unknown check label: %r" % (check,))
    out = dict(record)
    out["labels"] = {"basis": basis, "checked": check}
    if tool:
        out["labels"]["tool"] = tool
    return out


def require_labels(record):
    """Raise unless a record carries a valid basis and a valid check label."""
    labels = record.get("labels") if isinstance(record, dict) else None
    if not labels or labels.get("basis") not in BASIS or labels.get("checked") not in CHECK:
        raise ValueError("Output record without the two required labels")
    return True


def guard_path(path):
    """Refuse to open reserved evaluation material."""
    if Path(path).name in PROTECTED_NAMES:
        raise PermissionError("Reserved evaluation material must not be opened: %s" % Path(path).name)
    return Path(path)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------
def load_full_text_sources(pilot_dir=PILOT_DIR):
    """Load the 12 prepared documents and verify them against the pilot's lock.

    Returns {source_id: source}. Each source keeps the pilot's passages
    (blocks) and gains a registry entry. A document whose text no longer
    matches the locked hash raises an error: the audit never runs on a
    silently changed source.
    """
    pilot_dir = Path(pilot_dir)
    selection = json.loads(guard_path(pilot_dir / "selection_manifest.json").read_text(encoding="utf-8"))
    lock = json.loads(guard_path(pilot_dir / "source_lock.json").read_text(encoding="utf-8"))
    sources = {}
    for spec in selection["documents"]:
        sid = spec["pilot_id"]
        path = guard_path(pilot_dir / "packets" / sid / "source.json")
        saved = json.loads(path.read_text(encoding="utf-8"))
        text_hash = rp.digest(saved["raw_text"])
        if text_hash != lock[sid]["source_sha256"]:
            raise ValueError("Source text differs from the locked version: " + sid)
        for block in saved["blocks"]:
            if rp.digest(block["text"]) != block["text_sha256"]:
                raise ValueError("Passage text differs from its recorded hash: %s %s" % (sid, block["passage_id"]))
        meta = spec["metadata"]
        prov = saved["source_provenance"]
        sources[sid] = {
            "source_id": sid,
            "level": "full_text",
            "case": spec["case"],
            "doi": spec["doi"],
            "title": meta.get("title"),
            "year": meta.get("year"),
            "journal": meta.get("journal"),
            "format": prov.get("kind"),
            "text_sha256": text_hash,
            "source_version": text_hash[:12],
            "n_passages": len(saved["blocks"]),
            "n_chars": len(saved["raw_text"]),
            "path": str(path.relative_to(ROOT)),
            "provenance": prov,
            "coverage_warning": saved.get("coverage_warning"),
            "blocks": saved["blocks"],
            "raw_text": saved["raw_text"],
        }
    return sources


def load_abstract_sources(packets=None):
    """Load the 120 abstract-level review packets as one-passage sources.

    These are abstracts. They are for testing extraction from abstracts and
    whether a system withholds an answer when the evidence is missing. The
    blank review fields in the packets are ignored.
    """
    sources = {}
    for case, path in (packets or ABSTRACT_PACKETS).items():
        for row in rp.rows(guard_path(path)):
            sid = row["review_id"]
            text = row.get("passage") or ""
            block = {"passage_id": "P0001", "section": "Abstract-level packet text", "text": text,
                     "text_sha256": rp.digest(text),
                     "locator": {"kind": "packet_passage_field", "review_id": sid}}
            sources[sid] = {
                "source_id": sid, "level": "abstract", "case": case,
                "doi": rp.doi_norm(row.get("doi")) or None, "title": row.get("title"),
                "year": row.get("year"), "journal": row.get("journal"),
                "format": "review_packet_passage", "text_sha256": rp.digest(text),
                "source_version": rp.digest(text)[:12], "n_passages": 1, "n_chars": len(text),
                "path": str(Path(path).relative_to(ROOT)), "provenance": {"document_id": row.get("document_id")},
                "coverage_warning": "Abstract-level text only; methods and results are not available.",
                "blocks": [block], "raw_text": text,
            }
    return sources


def registry_entry(source):
    """The registry record for a source: identifiers and hashes, no text."""
    entry = {k: v for k, v in source.items() if k not in ("blocks", "raw_text")}
    return labeled(entry, "recorded_or_computed", "unchecked", tool="source_registry")


def source_quality_notes(source):
    """Computed warnings about how far the passage labels can be trusted."""
    notes = []
    sections = [b["section"] for b in source["blocks"]]
    distinct = sorted(set(sections))
    if source["level"] == "abstract":
        return ["Abstract-level text only."]
    if source["format"] == "pdf_pages":
        notes.append("Passages are page positions from a PDF; there are no section labels.")
    if source["format"] in ("pmc", "bioc"):
        notes.append("Section labels are the archive's coarse types (INTRO, METHODS, ...); they can be wrong for a given passage.")
    if source["format"] == "s2orc":
        notes.append("Section labels are headings detected by the export; a heading can swallow unrelated text that follows it.")
    if source["n_passages"] < 20:
        notes.append("Short document: fewer than 20 passages.")
    if not any(re.search(r"method|procedure|participants", s, re.I) for s in distinct) and \
       not any(len(b["text"]) < 90 and re.match(r"\s*(method|methods|participants|procedure)\b", b["text"], re.I)
               for b in source["blocks"]):
        notes.append("No Methods heading or label found; this may be a non-empirical paper or an unlabeled export.")
    return notes


# --------------------------------------------------------------------------
# Studies
# --------------------------------------------------------------------------
_STUDY_HEAD = re.compile(r"^\s*(?:pilot\s+)?(study|experiment)\s+(\d+[a-z]?|[IVX]{1,4})\b", re.I)
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}


def _number(match):
    """'2' -> '2', 'II' -> '2'. Headings sometimes use Roman numerals."""
    raw = match.group(2).lower()
    return _ROMAN.get(raw, raw)
_STUDY_MENTION = re.compile(r"\b(Study|Experiment)\s+(\d+[a-z]?)\b")
_BACK_MATTER = re.compile(r"^(?:ref|bib|acknowledg|funding|author|auth_cont|comp_int|conflict|declaration|suppl)", re.I)
_GENERAL_DISCUSSION = re.compile(r"^\s*general\s+discussion\b", re.I)


def study_id(source_id, number):
    """E04 + '2' -> 'E04.S2'. 'E04.S0' means the document taken as a whole."""
    return "%s.S%s" % (source_id, number)


def study_candidates(source):
    """Find candidate studies in a document from its headings.

    This is a pattern match on headings and on phrases such as "Study 2".
    It proposes; it does not decide. A paper can say "Study 1" about someone
    else's study, and many papers number nothing. Every candidate is labeled
    recorded-or-computed and unchecked. Three kinds of entry:
      heading         a heading such as "STUDY 2" or "Experiment 1" was found
      mention_only    the text says "Study 4" somewhere but no heading was
                      found; it may be another paper's study
      whole_document  no study heading was found, so the document gets the
                      entry S0 and is treated as a whole
    """
    sid = source["source_id"]
    blocks = source["blocks"]
    body = [b for b in blocks if not _BACK_MATTER.match(b["section"])]
    heads = {}  # number -> (index in blocks, how it was found)
    for i, b in enumerate(blocks):
        if _BACK_MATTER.match(b["section"]):
            continue
        m = _STUDY_HEAD.match(b["section"])
        if m and _number(m) not in heads:
            heads[_number(m)] = (i, "section label: " + b["section"][:80])
            continue
        text = b["text"].strip()
        m = _STUDY_HEAD.match(text)
        if m and len(text) < 90 and not text.endswith(".") and _number(m) not in heads \
                and not re.match(r"^(?:table|fig)", b["section"], re.I):
            heads[_number(m)] = (i, "heading text: " + text[:80])
    mentioned = {}
    for b in body:
        for m in _STUDY_MENTION.finditer(b["text"]):
            mentioned.setdefault(m.group(2).lower(), b["passage_id"])
    stop = next((i for i, b in enumerate(blocks) if _GENERAL_DISCUSSION.match(b["text"]) or
                 _GENERAL_DISCUSSION.match(b["section"])), None)
    out = []
    ordered = sorted(heads.items(), key=lambda kv: kv[1][0])
    for n, (number, (start, how)) in enumerate(ordered):
        following = [s for _, (s, _h) in ordered[n + 1:]]
        end = following[0] if following else (stop if stop is not None and stop > start else None)
        if end is None:
            last_body = max((i for i, b in enumerate(blocks) if not _BACK_MATTER.match(b["section"])), default=start)
            end, end_note = last_body + 1, "no later study heading or General Discussion found; runs to the end of the body text"
        else:
            end_note = "ends where the next study heading or the General Discussion begins"
        out.append({"study_id": study_id(sid, number.upper()), "source_id": sid, "number": number.upper(),
                    "found_by": how, "first_passage": blocks[start]["passage_id"],
                    "last_passage": blocks[end - 1]["passage_id"], "boundary_note": end_note,
                    "status": "heading"})
    for number, pid in sorted(mentioned.items()):
        if number not in heads:
            out.append({"study_id": study_id(sid, number.upper()), "source_id": sid, "number": number.upper(),
                        "found_by": "mentioned in the text, first at %s; no heading found" % pid,
                        "first_passage": None, "last_passage": None,
                        "boundary_note": "boundary not found; the mention may refer to another paper's study",
                        "status": "mention_only"})
    if not heads:
        out.append({"study_id": study_id(sid, 0), "source_id": sid, "number": "0",
                    "found_by": "no numbered study or experiment heading found",
                    "first_passage": None, "last_passage": None,
                    "boundary_note": "the document is treated as a whole; it may report one study, a multi-site study, or no study at all - not determined",
                    "status": "whole_document"})
    return [labeled(c, "recorded_or_computed", "unchecked", tool="study_candidates") for c in out]


# --------------------------------------------------------------------------
# Locators
# --------------------------------------------------------------------------
_LOCATOR = re.compile(r"^([A-Za-z0-9_\-]+):(P\d{4})(?::(\d+)-(\d+))?$")


def make_locator(source_id, passage_id, start=None, end=None):
    """E02 + P0006 + 120 + 245 -> 'E02:P0006:120-245' (characters within the passage)."""
    if start is None:
        return "%s:%s" % (source_id, passage_id)
    return "%s:%s:%d-%d" % (source_id, passage_id, start, end)


def parse_locator(locator):
    m = _LOCATOR.match(locator or "")
    if not m:
        raise ValueError("Not a locator: %r" % (locator,))
    sid, pid, a, b = m.groups()
    return {"source_id": sid, "passage_id": pid,
            "start": int(a) if a is not None else None, "end": int(b) if b is not None else None}


def get_block(source, passage_id):
    for b in source["blocks"]:
        if b["passage_id"] == passage_id:
            return b
    raise KeyError("No passage %s in %s" % (passage_id, source["source_id"]))


def resolve_locator(locator, sources):
    """Return the text a locator points to, after re-checking the passage hash."""
    loc = parse_locator(locator)
    if loc["source_id"] not in sources:
        raise KeyError("Unknown source: " + loc["source_id"])
    block = get_block(sources[loc["source_id"]], loc["passage_id"])
    if rp.digest(block["text"]) != block["text_sha256"]:
        raise ValueError("Passage changed since it was recorded: " + locator)
    if loc["start"] is None:
        return block["text"]
    if not 0 <= loc["start"] < loc["end"] <= len(block["text"]):
        raise ValueError("Locator range is outside the passage: " + locator)
    return block["text"][loc["start"]:loc["end"]]


# --------------------------------------------------------------------------
# Quotation checks
# --------------------------------------------------------------------------
_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ", " ": " ", " ": " ",
}
_DROP = {"­", "​", "‌", "‍", "﻿"}
_ELLIPSIS = re.compile(r"\s*(?:\[\s*(?:\.\.\.|…)\s*\]|\(\s*(?:\.\.\.|…)\s*\)|\.\.\.|…)\s*")
SHORT_QUOTE_CHARS = 20
MIN_CHARS_IGNORING_SPACING = 15


def _normalize(text):
    """Tidy typography only: quote marks, dashes, spaces. Returns (text, map).

    The map gives, for each character of the tidied text, its position in the
    original, so a match can be reported as an address in the original.
    Letters, digits and case are never changed.
    """
    out, index = [], []
    pending_space = False
    for i, ch in enumerate(text):
        if ch in _DROP:
            continue
        ch = _CHAR_MAP.get(ch, ch)
        for c in unicodedata.normalize("NFKC", ch):
            c = _CHAR_MAP.get(c, c)
            if c.isspace():
                pending_space = True
                continue
            if pending_space and out:
                out.append(" "); index.append(i)
            pending_space = False
            out.append(c); index.append(i)
    return "".join(out), index


def _find(quote, text):
    """Look for a quote in one passage. Returns (how, start, end) or None."""
    pos = text.find(quote)
    if pos >= 0:
        return "exact", pos, pos + len(quote)
    nq, _ = _normalize(quote)
    nt, index = _normalize(text)
    if nq:
        pos = nt.find(nq)
        if pos >= 0:
            return "exact_after_normalization", index[pos], index[pos + len(nq) - 1] + 1
    # Last resort: the same characters in the same order, with only spaces and line breaks
    # differing. Text taken from PDFs breaks lines mid-sentence and puts stray spaces around
    # punctuation; other exports drop a space here and there. Every letter, digit and
    # punctuation mark must still match.
    sq = nq.replace(" ", "")
    if len(sq) >= MIN_CHARS_IGNORING_SPACING:
        kept = [(c, index[i]) for i, c in enumerate(nt) if c != " "]
        pos = "".join(c for c, _ in kept).find(sq)
        if pos >= 0:
            return "exact_ignoring_spacing", kept[pos][1], kept[pos + len(sq) - 1][1] + 1
    return None


def _longest_shared_run(quote, text):
    from difflib import SequenceMatcher
    nq, _ = _normalize(quote)
    nt, _ = _normalize(text)
    m = SequenceMatcher(None, nq, nt, autojunk=False).find_longest_match(0, len(nq), 0, len(nt))
    return m.size, nq[m.a:m.a + m.size]


def _near_miss(quote, text, source_id, passage_id, threshold=0.9):
    """If a quote is almost in a passage, say exactly where it departs from the source.

    This never turns a failed quotation into a located one. It tells a
    reviewer what kind of failure it was: a model that tidied a typo or a
    symbol in the source leaves a one-character difference; an invented
    quotation leaves no near miss at all.
    """
    from difflib import SequenceMatcher
    quote = quote.strip()
    if not quote:
        return None
    blocks = [b for b in SequenceMatcher(None, text, quote, autojunk=False).get_matching_blocks() if b.size]
    if not blocks:
        return None
    shared = sum(b.size for b in blocks)
    if shared / float(len(quote)) < threshold:
        return None
    start, end = blocks[0].a, blocks[-1].a + blocks[-1].size
    span = text[start:end]
    core = quote[blocks[0].b:blocks[-1].b + blocks[-1].size]
    diffs = [{"source_has": span[i1:i2], "quote_has": core[j1:j2]}
             for tag, i1, i2, j1, j2 in SequenceMatcher(None, span, core, autojunk=False).get_opcodes()
             if tag != "equal"]
    extra = {"before": quote[:blocks[0].b], "after": quote[blocks[-1].b + blocks[-1].size:]}
    return {"share_of_quote_found_in_source": round(shared / float(len(quote)), 4),
            "closest_source_text": make_locator(source_id, passage_id, start, end),
            "differences": diffs,
            "quote_text_outside_the_match": {k: v for k, v in extra.items() if v}}


def check_quote(source, quote, passage_id=None, check="unchecked"):
    """Is this quotation in the source, and where?

    Outcomes, strongest first:
      exact                      the words appear, character for character,
                                 in the passage cited
      exact_after_normalization  the same, once curly quotes, dashes and
                                 spacing are tidied (letters never change)
      exact_ignoring_spacing     every letter, digit and punctuation mark
                                 matches in order; only spaces or line
                                 breaks differ (common in text from PDFs)
      exact_with_elision         the quote contains "..." (in the middle or
                                 at an end) and every part appears, in
                                 order, in the passage cited
      found_in_other_passage     the words are in the document but not in
                                 the passage cited (or no passage was cited)
      not_found                  the words are not in this document
    Anything other than the first four should be treated as a failed
    quotation until a person has looked.
    """
    sid = source["source_id"]
    result = {"source_id": sid, "quote": quote, "cited_passage": passage_id,
              "outcome": "not_found", "locators": [], "short_quote": len(quote.strip()) < SHORT_QUOTE_CHARS}
    if not quote or not quote.strip():
        result["outcome"] = "empty_quote"
        return labeled(result, "recorded_or_computed", check, tool="quotation_check")
    cited = None
    if passage_id is not None:
        try:
            cited = get_block(source, passage_id)
        except KeyError:
            result["cited_passage_exists"] = False
    if cited is not None:
        hit = _find(quote, cited["text"])
        if hit:
            result["outcome"] = hit[0]
            result["locators"] = [make_locator(sid, passage_id, hit[1], hit[2])]
            return labeled(result, "recorded_or_computed", check, tool="quotation_check")
        parts = [p for p in _ELLIPSIS.split(quote) if p.strip()]
        # More than one part, or one part with "..." trimmed from an end of it.
        if len(parts) > 1 or (len(parts) == 1 and parts[0].strip() != quote.strip()):
            cursor, locs, ok = 0, [], True
            for part in parts:
                hit = _find(part, cited["text"][cursor:])
                if not hit:
                    ok = False
                    break
                locs.append(make_locator(sid, passage_id, cursor + hit[1], cursor + hit[2]))
                cursor += hit[2]
            if ok:
                result["outcome"] = "exact_with_elision"
                result["locators"] = locs
                return labeled(result, "recorded_or_computed", check, tool="quotation_check")
    elsewhere = []
    for b in source["blocks"]:
        if cited is not None and b["passage_id"] == passage_id:
            continue
        hit = _find(quote, b["text"])
        if hit:
            elsewhere.append((hit[0], make_locator(sid, b["passage_id"], hit[1], hit[2])))
    if elsewhere:
        result["outcome"] = "found_in_other_passage"
        result["match_kind"] = sorted({h for h, _ in elsewhere})
        result["locators"] = [loc for _, loc in elsewhere]
    else:
        scope = cited["text"] if cited is not None else source["raw_text"]
        size, run = _longest_shared_run(quote, scope)
        result["longest_shared_run_chars"] = size
        result["longest_shared_run"] = run[:200]
        result["quote_chars_after_normalization"] = len(_normalize(quote)[0])
        candidates = [cited] if cited is not None else source["blocks"]
        misses = [m for m in (_near_miss(quote, b["text"], sid, b["passage_id"]) for b in candidates) if m]
        if misses:
            result["near_miss"] = max(misses, key=lambda m: m["share_of_quote_found_in_source"])
    return labeled(result, "recorded_or_computed", check, tool="quotation_check")


QUOTE_OK = ("exact", "exact_after_normalization", "exact_ignoring_spacing", "exact_with_elision")


def quotation_summary(results):
    """Counts for the report line 'Quotations located in source: __ of __'."""
    counts = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    located = sum(counts.get(k, 0) for k in QUOTE_OK)
    return {"total": len(results), "located_in_cited_passage": located, "by_outcome": counts,
            "mismatches": [r for r in results if r["outcome"] not in QUOTE_OK]}
