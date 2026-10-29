"""Hide copies of the same NCERT question at read time, for rows the quality pass has not merged yet.

Class 12 Accountancy (and every class except Class 10 Maths) was imported several times: "Illustration 1"
of a chapter can be stored 6 times in 3 slightly different extractions. Until a data pass marks them
`duplicate`, the list and the stats show one row per question:

- rows with the same chapter and the same "Illustration N" / "Example N" are one question (NCERT numbers
  these once per chapter; the stored section is not trusted — copies sit in different sections);
- any other rows are one question when chapter, text and options are the same once case, spaces and
  punctuation are ignored.

Kept row: lowest quality_rank, then not visibly garbled (full-width ＋－＝ or "sin )" — a PDF parse that
lost the operands, "tanθ− θ+="), then the extraction most copies agree on (a bad parse is usually the odd one out),
then the lowest id. Order of the input is kept.
"""

import re
from collections import Counter

_NUMBERED = re.compile(r"^\s*(illustration|example)\s*[-.:]?\s*(\d+[a-z]?(?:\s*\(\s*[ivx]+\s*\))?)\s*$", re.I)
_NOT_WORD = re.compile(r"[^0-9a-zऀ-ॿ]+")
_GARBLED = re.compile(r"[＋－＝]|\b(?:sin|cos|tan|cot|sec|cosec)\s*\)|θ[−+]\s*θ|[−+]=")


def _norm(value) -> str:
    if isinstance(value, (list, tuple)):
        value = "|".join(str(v) for v in value)
    return _NOT_WORD.sub("", str(value or "").lower())


def _content(row: dict) -> str:
    return _norm(row.get("question_text")) + "#" + _norm(row.get("options"))


def _group_key(row: dict) -> tuple:
    chapter = (row.get("chapter") or "").strip().lower()
    m = _NUMBERED.match(row.get("question_number") or "")
    if m:
        return ("numbered", chapter, m.group(1).lower(), re.sub(r"\s+", "", m.group(2).lower()))
    return ("text", chapter, _content(row))


def dedupe_ncert_rows(rows: list) -> list:
    """One row per question; see the module docstring. Rows need id, chapter, question_number,
    question_text (options and quality_rank optional)."""
    groups: dict = {}
    for row in rows:
        groups.setdefault(_group_key(row), []).append(row)

    keep = set()
    for members in groups.values():
        votes = Counter(_content(r) for r in members)
        best = min(members, key=lambda r: (
            r.get("quality_rank") or 0,
            bool(_GARBLED.search(r.get("question_text") or "")),
            -votes[_content(r)],
            r.get("id") or 0,
        ))
        keep.add(id(best))
    return [r for r in rows if id(r) in keep]
