"""
app/services/ncert_subjects.py — one subject filter for every NCERT table query.

Subjects are matched EXACTLY (case-insensitive), never as a substring: "%Science%" also matched
"Political Science", so Class 9/10 Science dropdowns and retrieval pulled civics chapters.
A few subjects are stored under several names (e.g. "Physics", "Physics-I", "Physics-II");
those map to their whole family.
"""

from typing import List, Optional

_FAMILIES = {
    "science": ["Science"],
    "physics": ["Physics", "Physics-I", "Physics-II"],
    "physics-i": ["Physics", "Physics-I", "Physics-II"],
    "physics-ii": ["Physics", "Physics-I", "Physics-II"],
    "chemistry": ["Chemistry", "Chemistry-I", "Chemistry-II"],
    "chemistry-i": ["Chemistry", "Chemistry-I", "Chemistry-II"],
    "chemistry-ii": ["Chemistry", "Chemistry-I", "Chemistry-II"],
    "political science": ["Political Science", "Political_Science"],
    "political_science": ["Political Science", "Political_Science"],
    "pol science": ["Political Science", "Political_Science"],
}


def ncert_subject_names(subject: str) -> Optional[List[str]]:
    """Exact DB subject names for a subject family, or None for a single case-insensitive match."""
    return _FAMILIES.get((subject or "").strip().lower())


def fetch_all_rows(make_query, page: int = 1000) -> list:
    """All rows of a PostgREST query (one request returns at most 1000). `make_query` builds a
    fresh, ordered query each call."""
    rows, start = [], 0
    while True:
        batch = make_query().range(start, start + page - 1).execute().data or []
        rows.extend(batch)
        if len(batch) < page:
            return rows
        start += page


# ── Chapter names ──────────────────────────────────────────────────────────
# Many chapters are stored in CAPITALS ("LAWS OF MOTION"). The dropdown shows a title-cased label;
# requests keep the exact DB name so `.eq("chapter", …)` in retrieval still matches.
_SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "of",
                "on", "or", "the", "to", "vs", "with"}
_ROMAN = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}


def chapter_display_name(name: str) -> str:
    name = (name or "").strip()
    if not name or name != name.upper():
        return name                      # already mixed case → keep as stored
    out = []
    for i, word in enumerate(name.lower().split()):
        core = word.strip("()[],.:;")
        if core in _ROMAN:
            out.append(word.upper())
        elif i > 0 and core in _SMALL_WORDS:
            out.append(word)
        else:
            out.append("-".join(p[:1].upper() + p[1:] for p in word.split("-")))
    return " ".join(out)


# Back-matter of a textbook that was ingested like a chapter. Never offered as a chapter.
_JUNK_CHAPTERS = {"answers", "answer", "appendices", "appendix", "index", "glossary",
                  "bibliography", "contents", "table of contents", "preface", "foreword",
                  "references", "acknowledgements", "acknowledgments"}


def is_junk_chapter(name: str) -> bool:
    key = chapter_key(name)
    return key in _JUNK_CHAPTERS or key.startswith(("appendix ", "answers to "))


def chapter_key(name: str) -> str:
    """Case/punctuation-insensitive identity, to merge "Electric Charges and Fields" (Physics)
    with "ELECTRIC CHARGES AND FIELDS" (Physics-I)."""
    return " ".join("".join(ch for ch in (name or "").lower() if ch.isalnum() or ch.isspace()).split())


def apply_ncert_subject_filter(query, subject: str):
    names = ncert_subject_names(subject)
    if names:
        return query.in_("subject", names)
    # ILIKE without wildcards = case-insensitive equality; escape LIKE metacharacters
    clean = (subject or "").strip().replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    return query.ilike("subject", clean)
