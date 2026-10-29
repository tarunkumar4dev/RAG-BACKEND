"""
app/services/ncert_options.py — the ONE place NCERT MCQ options are decoded and their labels cleaned.

The frontend copy (src/lib/ncert/optionLabel.ts) must give the same result on
scripts/fixtures/ncert_option_label_fixtures.json (tests/unit/test_ncert_options.py runs it here).
  normalize_option_label: "(A) (A) text" / "(A)(a) text" / "(1) (1) text" → one leading label (OCR doubled it).
  strip_option_labels:    "(A) 60°" → "60°" when the label is that option's own letter — the card and the PDF
                          print A/B/C/D themselves ("A (A) 60°" otherwise).
"""
from __future__ import annotations

import json
import re

# "(A)" / "(a)" / "(1)" / "(अ)" — ASCII digits only, so Python and JavaScript agree
_BRACKET_LABEL = r"\(\s*([A-Za-z]|[0-9]{1,2}|[ऀ-ॿ]{1,3})\s*\)"
_FIRST_LABEL = re.compile(r"^\s*" + _BRACKET_LABEL)
_NEXT_LABEL = re.compile(r"\s*" + _BRACKET_LABEL)
_MAX_REPEATS = 3
# own-letter label: "(A)", "A.", "A)" (any case)
_OWN_LABEL = re.compile(r"^\s*(?:\(\s*([A-Fa-f])\s*\)|([A-Fa-f])\s*[.)])\s*")


def decode_options(raw) -> list:
    """options is jsonb, but most rows hold a JSON *string* of the list ("[\\"(A) 60°\\", …]"). Accept both."""
    value = raw
    for _ in range(3):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except Exception:
            return []
    return [str(o) for o in value] if isinstance(value, list) else []


def normalize_option_label(opt: str) -> str:
    """Collapse a repeated leading label into the first one; anything else is returned unchanged."""
    first = _FIRST_LABEL.match(opt)
    if not first:
        return opt
    label = first.group(1).lower()
    pos = first.end()
    for _ in range(_MAX_REPEATS):
        nxt = _NEXT_LABEL.match(opt, pos)
        if not nxt or nxt.group(1).lower() != label:
            break
        pos = nxt.end()
    if pos == first.end():
        return opt
    head = opt[:first.end()].strip()
    rest = opt[pos:].lstrip()
    return f"{head} {rest}" if rest else head


def strip_option_labels(options: list) -> list:
    out = []
    for i, opt in enumerate(options):
        opt = normalize_option_label(opt)
        m = _OWN_LABEL.match(opt)
        letter = (m.group(1) or m.group(2)).upper() if m else None
        rest = opt[m.end():] if m else opt
        out.append(rest if letter == chr(65 + i) and rest.strip() else opt)
    return out
