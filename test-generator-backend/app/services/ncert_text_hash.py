"""
app/services/ncert_text_hash.py — the ONE normalize() / content_hash() for NCERT question text.

Pinned definition, HASH_VERSION "v2" (the frontend copy must give the same result on
scripts/fixtures/ncert_normalize_fixtures.json):
  NFC unicode → lowercase → strip $, \\(, \\), \\[, \\] → collapse all whitespace to one space
  → strip leading prefixes until none is left: v2 "Example 1 :" / "Q1." label and question numbering
  ("Q1.", "1)", "(i)"), in any order → strip trailing punctuation → trim.
content_hash = SHA-256 hex of normalize(question_text).
Empty after normalize (e.g. a bare "Example 1 :") means junk, never a duplicate.

v1 (G1a audit) = the same without the label step.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

HASH_VERSION = "v2"

_MATH_DELIMS = re.compile(r"\$|\\\(|\\\)|\\\[|\\\]")
_WHITESPACE = re.compile(r"\s+")
_LEADING_LABEL = re.compile(r"^\s*(example|q)\s*\d+\s*[\.:\)]\s*", re.I)
# "q1." / "q 12." / "1)" / "1." (not "1.5") / "(i)" / "(a)" / "(3)"
_LEADING_NUMBERING = re.compile(
    r"^(?:q\s*\.?\s*\d+[a-z]?\s*[.):]?|\d+\s*[.)](?=\s|$)|\(\s*(?:[ivxlcdm]+|[a-z]|\d+)\s*\))\s*"
)
_TRAILING_PUNCT = re.compile(r"[\s.,;:!?…।]+$")


def _normalize(text: str | None, strip_label: bool) -> str:
    t = unicodedata.normalize("NFC", text or "").lower()
    t = _MATH_DELIMS.sub("", t)
    t = _WHITESPACE.sub(" ", t).strip()
    # Prefixes stack in any order ("Q 12. (ii) Prove …", "(i) Example 2 : …"): strip until nothing changes
    while True:
        stripped = _LEADING_LABEL.sub("", t, count=1) if strip_label else t
        stripped = _LEADING_NUMBERING.sub("", stripped, count=1)
        if stripped == t:
            break
        t = stripped
    t = _TRAILING_PUNCT.sub("", t)
    return t.strip()


def normalize(text: str | None) -> str:
    return _normalize(text, strip_label=True)


def content_hash(text: str | None) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()
