"""Name / section normalisation for entity resolution (English + Hinglish)."""
from __future__ import annotations

import re

_HONORIFICS = r"(?:sir|ma'am|ma’am|maam|madam|mam|ji|mr|mrs|ms|miss)\.?"
_HONORIFIC_RE = re.compile(rf"(?<![\w'’]){_HONORIFICS}(?![\w'’])", re.IGNORECASE)
_TRAILING_PARTICLE_RE = re.compile(r"\s+(?:ki|ka|ke|ko|ne)$", re.IGNORECASE)

_ROMAN = {"VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10", "XI": "11", "XII": "12"}


def strip_honorifics(s: str) -> str:
    """'Sharma ma'am ki' -> 'Sharma'; 'Mr. Verma' -> 'Verma'. Only honorific tokens + a trailing particle."""
    out = _HONORIFIC_RE.sub(" ", s or "")
    out = re.sub(r"\s+", " ", out).strip()
    prev = None
    while prev != out:
        prev = out
        out = _TRAILING_PARTICLE_RE.sub("", out).strip()
    return out


def normalize_section(s: str) -> str:
    """'10th A', '10-A', '10 a', 'X-A', 'Xa', 'class 10A' -> '10A'; '10th' -> '10'; 'IX' -> '9'."""
    t = (s or "").strip()
    t = re.sub(r"^(?:class|cls|std|grade|section|sec)\.?\s*", "", t, flags=re.IGNORECASE)
    m = re.fullmatch(r"(?i)(XII|XI|X|IX|VIII|VII|VI)(?:th)?[\s\-_/]*([A-Z])?", t)
    if m:
        num = _ROMAN[m.group(1).upper()]
        return num + (m.group(2) or "").upper()
    m = re.fullmatch(r"(?i)(\d{1,2})(?:st|nd|rd|th)?[\s\-_/]*([A-Z])?", t)
    if m:
        return m.group(1) + (m.group(2) or "").upper()
    return t
