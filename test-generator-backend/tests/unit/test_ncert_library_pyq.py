"""PYQ filters, sort and facets for the Test Builder library (app/services/ncert_library.py) and the loader merge."""

import json
import sys
from pathlib import Path

import pytest

from app.services import ncert_library as lib
from app.services.ncert_dedupe import dedupe_ncert_rows

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import pyq_load  # noqa: E402


class FakeQuery:
    """Records the supabase-py calls a filter makes."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*args):
            self.calls.append((name, *args))
            return self
        return call


def test_defaults_reproduce_the_old_list():
    f = lib.normalize_filters()
    assert f == lib.LibraryFilters(source="all", language="en", year=None, difficulty=None, marks=None,
                                   sort="default")
    q = lib.apply_query_filters(FakeQuery(), f, columns_ready=True)
    # only the English filter: untagged textbook rows (NULL) still match
    assert q.calls == [("or_", "language.is.null,language.eq.en")]


def test_unknown_values_fall_back_to_defaults():
    f = lib.normalize_filters(source="PYQ ", language="fr", year="20x6", difficulty="tough", marks="99", sort="x")
    assert (f.source, f.language, f.year, f.difficulty, f.marks, f.sort) == ("pyq", "en", None, None, None, "default")


def test_every_filter_is_applied():
    f = lib.normalize_filters("pyq", "hi", "2026", "Hard", "6", "marks_desc")
    q = lib.apply_query_filters(FakeQuery(), f, columns_ready=True)
    assert q.calls == [("eq", "source", "pyq"), ("eq", "language", "hi"), ("eq", "pyq_year", 2026),
                       ("eq", "difficulty", "hard"), ("eq", "marks", 6)]


def test_before_migration_defaults_still_work_and_pyq_requests_are_empty():
    q = lib.apply_query_filters(FakeQuery(), lib.normalize_filters(difficulty="easy"), columns_ready=False)
    assert q.calls == [("eq", "difficulty", "easy")]  # no unknown column touched
    for f in (lib.normalize_filters(source="pyq"), lib.normalize_filters(language="hi"),
              lib.normalize_filters(year=2026)):
        assert lib.apply_query_filters(FakeQuery(), f, columns_ready=False) is None
    # ncert / all / language=all only ever had English textbook rows: no filter needed
    assert lib.apply_query_filters(FakeQuery(), lib.normalize_filters("ncert", "all"), False).calls == []


def _row(i, **kw):
    base = {"id": i, "chapter": "Ch", "section": "S", "question_type": "exercise", "question_number": str(i),
            "question_text": f"question {i}", "options": "[]"}
    base.update(kw)
    return base


def test_pyq_rows_take_paper_order_in_their_own_slots():
    rows = [
        _row(1),
        _row(2, source="pyq", pyq_year=2026, question_number="10", language="en"),
        _row(3),
        _row(4, source="pyq", pyq_year=2026, question_number="2", language="hi"),
        _row(5, source="pyq", pyq_year=2026, question_number="2", language="en"),
        _row(6, source="pyq", pyq_year=2025, question_number="1", language="en"),
    ]
    out = [r["id"] for r in lib.sort_rows(rows, "default")]
    # textbook rows stay at positions 0 and 2; PYQs: newest year, Q2 (en, hi), Q10, then 2025
    assert out == [1, 5, 3, 4, 2, 6]


@pytest.mark.parametrize("sort,expected", [
    ("marks_asc", [3, 1, 2]),
    ("marks_desc", [2, 1, 3]),
    ("difficulty_asc", [2, 3, 1]),
    ("difficulty_desc", [1, 3, 2]),
    ("year_desc", [2, 1, 3]),
    ("chapter", [3, 1, 2]),
])
def test_sorts(sort, expected):
    rows = [
        _row(1, marks=3, difficulty="hard", pyq_year=2025, source="pyq", chapter="B"),
        _row(2, marks=6, difficulty="easy", pyq_year=2026, source="pyq", chapter="C"),
        _row(3, marks=1, difficulty="medium", chapter="A"),
    ]
    assert [r["id"] for r in lib.sort_rows(rows, sort)] == expected


def test_dedupe_never_merges_pyq_with_textbook_or_english_with_hindi():
    text = {"question_text": "Same words", "options": "[]", "chapter": "Ch", "question_number": "1"}
    rows = [
        dict(text, id=1),
        dict(text, id=2, source="pyq", language="en"),
        dict(text, id=3, source="pyq", language="hi"),
        dict(text, id=4, source="pyq", language="en"),  # a real copy
    ]
    assert [r["id"] for r in dedupe_ncert_rows(rows)] == [1, 2, 3]


def _stats(rows, **filters):
    return lib.build_stats(rows, lib.normalize_filters(**filters), lambda sections: None, with_facets=True)


ROWS = [
    _row(1, chapter="Goodwill", difficulty="easy", marks=1),
    _row(2, chapter="Goodwill", difficulty="hard", marks=6, source="pyq", language="en", pyq_year=2026),
    _row(3, chapter="Goodwill", difficulty="hard", marks=6, source="pyq", language="hi", pyq_year=2026),
    _row(4, chapter="Computerised Accounting", difficulty="easy", marks=1, source="pyq", language="en",
         pyq_year=2026),
    _row(5, chapter="Computerised Accounting", difficulty="easy", marks=1, source="pyq", language="hi",
         pyq_year=2026),
]


def test_stats_all_sources_english():
    s = _stats(ROWS)
    assert [(c["chapter"], c["total"], c["pyq"]) for c in s["chapters"]] == [
        ("Computerised Accounting", 1, 1), ("Goodwill", 2, 1)]
    assert s["facets"]["sources"] == [{"name": "ncert", "count": 1}, {"name": "pyq", "count": 4}]
    assert s["facets"]["languages"] == [{"name": "en", "count": 3}, {"name": "hi", "count": 2}]
    assert s["facets"]["years"] == [{"name": 2026, "count": 2}]
    assert s["facets"]["marks"] == [{"name": 1, "count": 2}, {"name": 6, "count": 1}]
    assert s["totalQuestions"] == 3


def test_stats_ncert_only_drops_pyq_only_chapters_but_keeps_pyq_chip_count():
    s = _stats(ROWS, source="ncert")
    assert [(c["chapter"], c["total"], c["pyq"]) for c in s["chapters"]] == [("Goodwill", 1, 1)]
    assert s["facets"]["years"] == []


def test_stats_hindi_pyq():
    s = _stats(ROWS, source="pyq", language="hi")
    assert [(c["chapter"], c["total"]) for c in s["chapters"]] == [("Computerised Accounting", 1), ("Goodwill", 1)]
    assert s["facets"]["difficulties"] == [{"name": "easy", "count": 1}, {"name": "hard", "count": 1}]


def test_stats_without_migration_have_no_facets():
    s = lib.build_stats([_row(1)], lib.normalize_filters(), lambda sections: None, with_facets=False)
    assert "facets" not in s and "pyq" not in s["chapters"][0]


def _paper(code, questions):
    return {"exam": "CBSE", "year": 2026, "subject": "Accountancy", "class_grade": "12", "paper": code,
            "_file": code.replace("/", "_") + ".json", "questions": questions}


def _q(number, text, **kw):
    q = {"question_number": number, "part": "A", "chapter": "Accounting Ratios", "question_type": "mcq",
         "marks": 1, "difficulty": "easy", "answer": "B: because",
         "en": {"text": text, "options": ["(A) one", "(B) two", "(C) three", "(D) four"]},
         "hi": {"text": "हिंदी प्रश्न " + text, "options": ["एक", "दो", "तीन", "चार"]}}
    q.update(kw)
    return q


def test_loader_merges_sets_and_pairs_languages():
    papers = [_paper("67/5/2", [_q("8", "Same question?"), _q("9", "Only in set two?")]),
              _paper("67/5/1", [_q("13", "Same  question ?")])]
    papers.sort(key=lambda p: pyq_load.paper_sort_key(p["paper"]))
    rows, errors = pyq_load.build_rows(papers)
    assert errors == []
    en = [r for r in rows if r["language"] == "en"]
    assert len(rows) == 4 and len(en) == 2
    same = next(r for r in en if r["question_text"].startswith("Same"))
    assert same["pyq_papers"] == ["67/5/1", "67/5/2"] and same["question_number"] == "13"
    hi = next(r for r in rows if r["language"] == "hi" and r["pair_id"] == same["pair_id"])
    assert json.loads(hi["options"])[0] == "A. एक"
    assert json.loads(same["options"]) == ["A. one", "B. two", "C. three", "D. four"]
    assert same["source"] == "pyq" and same["pyq_year"] == 2026 and same["section"] == "CBSE 2026"


def test_loader_reports_bad_questions():
    bad = _q("1", "Bad?", chapter="Not a chapter", answer="maybe")
    bad["hi"] = {"text": "english only text", "options": []}
    _, errors = pyq_load.build_rows([_paper("67/1/1", [bad])])
    text = "\n".join(errors)
    assert "unknown chapter" in text and "answer should start" in text
    assert "hi mcq has 0 options" in text and "no Devanagari" in text


def test_loader_rejects_a_number_twice_in_one_paper():
    _, errors = pyq_load.build_rows([_paper("67/1/1", [_q("27", "First one?", part="B-I"),
                                                       _q("27", "Second one?", part="B-I"),
                                                       _q("27", "Third one?", part="B-II")])])
    assert errors == ["67_1_1.json: Q27 (B-I) appears 2 times"]


def test_extractor_keeps_only_the_asked_range_once():
    import pyq_extract
    seen = set()
    got = [{"question_number": n, "part": p} for n, p in
           [("23(a)", "A"), ("23(b)", "A"), ("27", "B-I"), ("27", "B-II"), ("23(a)", "A")]]
    assert [q["question_number"] for q in pyq_extract.in_range(got, 23, 26, seen)] == ["23(a)", "23(b)"]
    assert len(pyq_extract.in_range(got, 27, 34, seen)) == 2


# ── the endpoints, with a fake Supabase client ───────────────────────────────

class FakeTable:
    """Enough of supabase-py for _ncert_questions_impl / the stats fallback: filters are recorded, rows are
    served from memory and narrowed by the eq() filters (the rest is covered by the pure tests above)."""

    def __init__(self, rows, log):
        self.rows, self.log, self.eqs = rows, log, []

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.log.append((name, *args))
            if name == "eq" and args[0] in ("source", "language", "pyq_year", "difficulty", "marks"):
                self.eqs.append(args)
            return self
        return call

    def range(self, start, end):
        self.slice = (start, end)
        return self

    def execute(self):
        rows = [r for r in self.rows if all(r.get(k) == v for k, v in self.eqs)]
        start, end = getattr(self, "slice", (0, len(rows)))
        return type("R", (), {"data": [dict(r) for r in rows[start:end + 1]]})()


@pytest.fixture
def endpoint(monkeypatch):
    import asyncio
    from app.api.v1.endpoints import test_generator as tg

    log = []
    rows = [
        _row(1, chapter="Accounting Ratios", subject="Accountancy", marks=1, difficulty="easy"),
        _row(2, chapter="Accounting Ratios", subject="Accountancy", marks=6, difficulty="hard", source="pyq",
             language="en", pyq_year=2026, question_number="10"),
        _row(3, chapter="Accounting Ratios", subject="Accountancy", marks=3, difficulty="medium", source="pyq",
             language="en", pyq_year=2026, question_number="2"),
    ]
    client = type("C", (), {"table": lambda self, name: FakeTable(rows, log)})()
    monkeypatch.setattr(tg, "get_supabase", lambda: client)
    monkeypatch.setitem(tg._PYQ_COLUMNS_STATE, "ready", True)
    return tg, log, lambda coro: asyncio.run(coro)


def test_list_endpoint_filters_and_sorts(endpoint):
    tg, log, run = endpoint
    out = run(tg._ncert_questions_impl("Accountancy", "12", None, None, None, None, 50, 0, None,
                                       lib.normalize_filters(source="pyq")))
    assert out["ok"] and [q["id"] for q in out["questions"]] == [3, 2]  # Q2 before Q10
    assert ("eq", "source", "pyq") in log and ("or_", "language.is.null,language.eq.en") in log

    out = run(tg._ncert_questions_impl("Accountancy", "12", None, None, None, None, 50, 0, None,
                                       lib.normalize_filters(sort="marks_desc")))
    assert [q["id"] for q in out["questions"]] == [2, 3, 1]


def test_list_endpoint_before_migration(endpoint, monkeypatch):
    tg, log, run = endpoint
    monkeypatch.setitem(tg._PYQ_COLUMNS_STATE, "ready", False)
    monkeypatch.setattr(tg, "_pyq_columns_ready", lambda: False)
    out = run(tg._ncert_questions_impl("Accountancy", "12", None, None, None, None, 50, 0, None,
                                       lib.normalize_filters(source="pyq")))
    assert out == {"ok": True, "subject": "Accountancy", "classGrade": "12", "chapter": None, "questions": [],
                   "total": 0, "limit": 50, "offset": 0, "hasMore": False}
    log.clear()
    out = run(tg._ncert_questions_impl("Accountancy", "12", None, None, None, None, 50, 0, None, None))
    assert out["ok"] and not any(c[1] in ("source", "language") for c in log if c[0] == "eq")
    assert ("or_", "language.is.null,language.eq.en") not in log


def test_stats_endpoint_facets(endpoint, monkeypatch):
    tg, log, run = endpoint
    import app.core.db_pool as db_pool
    monkeypatch.setattr(db_pool, "get_db_connection", lambda: None)  # take the Supabase fallback
    out = run(tg._ncert_question_stats_impl("Accountancy", "12", lib.normalize_filters(source="pyq")))
    assert out["ok"] and out["chapters"][0]["total"] == 2 and out["chapters"][0]["pyq"] == 2
    assert out["facets"]["sources"] == [{"name": "ncert", "count": 1}, {"name": "pyq", "count": 2}]
    selected = next(c for c in log if c[0] == "select")[1]
    assert "source" in selected and "pyq_year" in selected


def test_pyq_column_probe_retries_after_a_minute(monkeypatch):
    from app.api.v1.endpoints import test_generator as tg
    calls = []

    def broken():
        calls.append(1)
        raise RuntimeError("column ncert_questions.source does not exist")

    monkeypatch.setattr(tg, "get_supabase", broken)
    monkeypatch.setattr(tg, "_PYQ_COLUMNS_STATE", {"ready": False, "checked_at": None})
    clock = [1000.0]
    monkeypatch.setattr(tg.time, "monotonic", lambda: clock[0])
    assert tg._pyq_columns_ready() is False and tg._pyq_columns_ready() is False and len(calls) == 1
    clock[0] += 61
    assert tg._pyq_columns_ready() is False and len(calls) == 2


def test_loader_merges_shuffled_options_and_flags_disagreeing_sets():
    a = _q("8", "Same question here?")
    b = _q("13", "Same question here?", answer="C: other",
           en={"text": "Same question here?", "options": ["(A) four", "(B) three", "(C) two", "(D) one"]})
    rows, errors = pyq_load.build_rows([_paper("67/5/1", [a]), _paper("67/5/2", [b])])
    assert len(rows) == 2 and rows[0]["pyq_papers"] == ["67/5/1", "67/5/2"]
    assert errors == []  # B in one set and C in the other both pick "two"

    c = dict(b, answer="A: four", marks=2)
    _, errors = pyq_load.build_rows([_paper("67/5/1", [a]), _paper("67/5/2", [c])])
    assert any("marks 2 but 67/5/1 Q8" in e for e in errors)
    assert any("picks a different option" in e for e in errors)


def test_loader_warns_about_near_identical_questions():
    papers = [_paper("67/5/1", [_q("8", "Calculate interest on capital at 10% p.a. for Ravi and Sunil.")]),
              _paper("67/5/2", [_q("9", "Calculate interest on capitals at 10% p.a. for Ravi and Sunil.")]),
              _paper("67/5/3", [_q("9", "Prepare the Realisation Account of the firm.")])]
    assert pyq_load.near_duplicates(papers) == ["67/5/1 Q8 ≈ 67/5/2 Q9 (99% alike, not merged)"]


def test_loader_clears_a_hidden_status_on_reload():
    rows, _ = pyq_load.build_rows([_paper("67/1/1", [_q("1", "Some question?")])])
    assert all(r["quality_status"] is None for r in rows)
