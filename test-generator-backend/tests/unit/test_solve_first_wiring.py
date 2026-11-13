"""Phase 1B wiring: _solve_first_numeric returns solver-verified GeneratedQuestions
for numeric Accountancy MCQs, and None (legacy path) otherwise. No network."""
import re
from fractions import Fraction

from app.models.test_generator import ChapterSection, DifficultyLevel, QuestionFormat
from app.services import test_generator_service as T


class _Req:
    subject = "Accountancy"
    class_grade = "12"


def _chapter(chapter, topic=None, fmt="mcq", qty=10, diff="hard", marks=1):
    return ChapterSection(
        chapter=chapter, topic=topic, subtopics=[topic] if topic else [],
        quantity=qty, difficulty=DifficultyLevel(diff),
        format=QuestionFormat(fmt), marks_per_question=marks,
    )


def _val(opt):
    m = re.search(r"Rs\.\s*([\d,]+)", opt)
    return Fraction(int(m.group(1).replace(",", "")))


def test_fundamentals_rotation_builds_solver_mcqs():
    qs = T._solve_first_numeric(_chapter("Fundamentals of Partnership", qty=15),
                                _Req(), 15, client=None)
    assert qs is not None and len(qs) == 15
    for q in qs:
        assert q.format == QuestionFormat.MCQ
        assert len(q.options) == 4
        vals = [_val(o) for o in q.options]
        assert len(set(vals)) == 4
        assert q.correct_answer in q.options
        assert sum(1 for v in vals if v == _val(q.correct_answer)) == 1
        assert q.validation_status == "verified"


def test_subtopic_routes_to_single_family():
    qs = T._solve_first_numeric(
        _chapter("Fundamentals of Partnership", topic="Interest on Drawings", qty=5),
        _Req(), 5, client=None)
    assert qs is not None and len(qs) == 5
    assert all("Interest on Drawings" == q.topic for q in qs)


def test_non_numeric_chapter_returns_none():
    # Accounting Ratios is not a numeric Fundamentals family -> legacy path.
    assert T._solve_first_numeric(_chapter("Accounting Ratios"), _Req(), 5, client=None) is None


def test_non_mcq_returns_none():
    assert T._solve_first_numeric(
        _chapter("Fundamentals of Partnership", fmt="short_answer"),
        _Req(), 5, client=None) is None
