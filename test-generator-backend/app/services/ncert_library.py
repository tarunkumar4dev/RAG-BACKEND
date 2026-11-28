"""Filters, sort and facets for the Test Builder question library (GET /ncert-questions, /ncert-question-stats).

The library holds NCERT textbook questions and, since migration 37, CBSE previous-year questions (PYQs) in the
same `ncert_questions` table:

    source      'ncert' | 'pyq'           (NOT NULL, default 'ncert')
    language    'en' | 'hi' | NULL        (NULL = untagged textbook row = English)
    pyq_year    2026 …                    (PYQs only)
    pyq_exam    'CBSE'                    (PYQs only)
    pyq_papers  {'67/5/1','67/5/2'}       (every Q.P. code the question appeared in)
    pair_id     the same value on the English and the Hindi version of one PYQ

Every new filter has a default that reproduces the old list exactly (source=all, language=en, the rest off), so
a client that sends none of them sees what it saw before — except that Hindi rows stay hidden from it.

If the backend runs before migration 37 is applied, the new columns do not exist: `columns_ready=False` drops
the source / language / year filters (an old library has only English NCERT rows, so the defaults still hold)
and a request that can only match PYQ or Hindi rows gets an empty list instead of a database error.
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional

SOURCES = ("all", "ncert", "pyq")
LANGUAGES = ("en", "hi", "all")
DIFFICULTIES = ("easy", "medium", "hard")
SORTS = ("default", "chapter", "marks_asc", "marks_desc", "difficulty_asc", "difficulty_desc", "year_desc")
PYQ_COLUMNS = ("source", "language", "pyq_year", "pyq_exam", "pyq_papers", "pair_id")

_DIFF_RANK = {"easy": 0, "medium": 1, "hard": 2}
_LANG_RANK = {"en": 0, "hi": 1}
_DIGITS = re.compile(r"(\d+)")


@dataclass(frozen=True)
class LibraryFilters:
    source: str = "all"
    language: str = "en"
    year: Optional[int] = None
    difficulty: Optional[str] = None
    marks: Optional[int] = None
    sort: str = "default"

    def cache_key(self) -> list:
        return [self.source, self.language, self.year, self.difficulty, self.marks, self.sort]


def _choice(value, allowed: tuple, default):
    v = str(value or "").strip().lower()
    return v if v in allowed else default


def normalize_filters(source=None, language=None, year=None, difficulty=None, marks=None, sort=None) -> LibraryFilters:
    """Unknown values fall back to the default instead of failing: an old or newer client never breaks the list."""
    try:
        year = int(year) if year not in (None, "") else None
    except (TypeError, ValueError):
        year = None
    if year is not None and not 1990 <= year <= 2100:
        year = None
    try:
        marks = int(marks) if marks not in (None, "") else None
    except (TypeError, ValueError):
        marks = None
    if marks is not None and not 1 <= marks <= 20:
        marks = None
    return LibraryFilters(
        source=_choice(source, SOURCES, "all"),
        language=_choice(language, LANGUAGES, "en"),
        year=year,
        difficulty=_choice(difficulty, DIFFICULTIES, None),
        marks=marks,
        sort=_choice(sort, SORTS, "default"),
    )


def needs_pyq_columns(f: LibraryFilters) -> bool:
    """True when the request can only match rows that exist after migration 37 (PYQ or Hindi rows)."""
    return f.source == "pyq" or f.language == "hi" or f.year is not None


def apply_query_filters(query, f: LibraryFilters, columns_ready: bool):
    """Add the new filters to a supabase-py query. Returns None when nothing can match (see module doc)."""
    if not columns_ready:
        if needs_pyq_columns(f):
            return None
    else:
        if f.source != "all":
            query = query.eq("source", f.source)
        if f.language == "en":
            query = query.or_("language.is.null,language.eq.en")
        elif f.language == "hi":
            query = query.eq("language", "hi")
        if f.year is not None:
            query = query.eq("pyq_year", f.year)
    if f.difficulty:
        query = query.eq("difficulty", f.difficulty)
    if f.marks is not None:
        query = query.eq("marks", f.marks)
    return query


def row_source(row: dict) -> str:
    return row.get("source") or "ncert"


def row_language(row: dict) -> str:
    return row.get("language") or "en"


def _in_language(row: dict, language: str) -> bool:
    return language == "all" or row_language(row) == language


def _in_source(row: dict, source: str) -> bool:
    return source == "all" or row_source(row) == source


def natural_key(value) -> tuple:
    """'7(a)' < '10' < '10(b)': digits compare as numbers."""
    return tuple((0, int(p), "") if p.isdigit() else (1, 0, p.lower()) for p in _DIGITS.split(str(value or "")) if p)


def sort_rows(rows: list, sort: str) -> list:
    """PYQs keep the database order's slots but are put in paper order (newest year, Q1 → Q34, English before
    Hindi), because question_number is text and '10' sorts before '2'. Then the requested sort, stable, so ties
    keep that order. NCERT rows keep the order the database returned."""
    rows = list(rows)
    slots = [i for i, r in enumerate(rows) if row_source(r) == "pyq"]
    if slots:
        ordered = sorted((rows[i] for i in slots), key=lambda r: (
            -(r.get("pyq_year") or 0), natural_key(r.get("question_number")), _LANG_RANK.get(row_language(r), 9)))
        for i, r in zip(slots, ordered):
            rows[i] = r

    if sort == "chapter":
        rows.sort(key=lambda r: (r.get("chapter") or "").lower())
    elif sort == "marks_asc":
        rows.sort(key=lambda r: r.get("marks") or 0)
    elif sort == "marks_desc":
        rows.sort(key=lambda r: -(r.get("marks") or 0))
    elif sort == "difficulty_asc":
        rows.sort(key=lambda r: _DIFF_RANK.get((r.get("difficulty") or "").lower(), 9))
    elif sort == "difficulty_desc":
        rows.sort(key=lambda r: -_DIFF_RANK.get((r.get("difficulty") or "").lower(), -1))
    elif sort == "year_desc":
        rows.sort(key=lambda r: -(r.get("pyq_year") or 0))
    return rows


def _count_list(counter: dict, key_fn=None) -> list:
    keys = sorted(counter, key=key_fn) if key_fn else sorted(counter)
    return [{"name": k, "count": counter[k]} for k in keys]


def build_stats(rows: list, f: LibraryFilters, chapter_number: Callable[[dict], Optional[int]],
                with_facets: bool) -> dict:
    """Chapter sidebar counts + facets from already-deduplicated rows of one class + subject (every source and
    language). `rows` need chapter, section, question_type; source, language, pyq_year, difficulty, marks when
    with_facets.

    - chapters: rows in the source + language slice (difficulty / marks / year do not narrow the sidebar);
      a chapter with no row in the slice is left out. Each chapter carries `pyq`: its PYQ rows in the language.
    - facets.sources / languages: the whole class + subject, so the toggles can show what exists.
    - facets.years / difficulties / marks: within the source + language slice.
    """
    chapter_stats: dict = {}
    pyq_per_chapter: dict = {}
    sources: dict = {}
    languages: dict = {}
    years: dict = {}
    difficulties: dict = {}
    marks: dict = {}

    for row in rows:
        src, lang = row_source(row), row_language(row)
        sources[src] = sources.get(src, 0) + 1
        languages[lang] = languages.get(lang, 0) + 1
        if not _in_language(row, f.language):
            continue
        ch = row.get("chapter")
        if src == "pyq":
            pyq_per_chapter[ch] = pyq_per_chapter.get(ch, 0) + 1
        if not _in_source(row, f.source):
            continue

        sec = row.get("section") or "Uncategorized"
        qtype = row.get("question_type") or "exercise"
        stat = chapter_stats.setdefault(ch, {"chapter": ch, "total": 0, "sections": {}, "types": {}})
        stat["total"] += 1
        stat["sections"][sec] = stat["sections"].get(sec, 0) + 1
        stat["types"][qtype] = stat["types"].get(qtype, 0) + 1

        if row.get("pyq_year"):
            years[row["pyq_year"]] = years.get(row["pyq_year"], 0) + 1
        diff = (row.get("difficulty") or "").lower()
        if diff in _DIFF_RANK:
            difficulties[diff] = difficulties.get(diff, 0) + 1
        if row.get("marks"):
            marks[row["marks"]] = marks.get(row["marks"], 0) + 1

    stats_list = sorted(chapter_stats.values(), key=lambda x: x["chapter"] or "")
    for stat in stats_list:
        stat["chapter_no"] = chapter_number(stat["sections"])
        stat["sections"] = [{"name": k, "count": v} for k, v in sorted(stat["sections"].items())]
        stat["types"] = [{"name": k, "count": v} for k, v in sorted(stat["types"].items())]
        if with_facets:
            stat["pyq"] = pyq_per_chapter.get(stat["chapter"], 0)

    out = {
        "chapters": stats_list,
        "totalQuestions": sum(s["total"] for s in stats_list),
        "totalChapters": len(stats_list),
    }
    if with_facets:
        out["facets"] = {
            "sources": [{"name": s, "count": sources.get(s, 0)} for s in ("ncert", "pyq")],
            "languages": [{"name": lang, "count": languages.get(lang, 0)} for lang in ("en", "hi")],
            "years": _count_list(years, key_fn=lambda y: -y),
            "difficulties": [{"name": d, "count": difficulties[d]} for d in DIFFICULTIES if d in difficulties],
            "marks": _count_list(marks),
        }
    return out
