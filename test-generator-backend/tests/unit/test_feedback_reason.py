"""Checker bundle #1: every below-full-marks answer's feedback starts with the weakest point's reason.
Deterministic: the G1 fixture copy with recorded (mock) Gemini replies; no network, no paid call."""
from __future__ import annotations

import re

import pytest

from app.services import smart_assignment_service as svc
from app.services.answer_checker.copy_check import grade_copy
from tests.conftest import FIXTURES

PRAISE_ONLY = re.compile(r"^(good|great|excellent|well done|nice|correct|very good|good attempt|good answer)"
                         r"[\s.!,]*(answer|work|attempt|job)?[\s.!]*$", re.I)


def _leaves(result: dict) -> list[dict]:
    """Every answer that has its own marking points (top-level questions and sub-parts)."""
    out = []

    def walk(x):
        if isinstance(x, dict):
            if x.get("points") and isinstance(x["points"], list) and "marks_awarded" in x:  # composite parents have none
                out.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(result.get("graded_questions") or [])
    return out


@pytest.fixture(scope="module")
def graded():
    rubric = svc.build_rubric(str(FIXTURES / "question_paper.pdf"), str(FIXTURES / "answer_key.pdf"))
    runs = {}
    for flag in (False, True):
        svc._mock_seen.clear()
        old = svc.FEEDBACK_REASON
        svc.FEEDBACK_REASON = flag
        try:
            runs[flag] = grade_copy([str(FIXTURES / "answer_sheet_messy.pdf")], rubric)[0]
        finally:
            svc.FEEDBACK_REASON = old
    return runs


def test_marks_do_not_change(graded):
    off, on = graded[False], graded[True]
    assert on["total_marks_awarded"] == off["total_marks_awarded"]
    assert [q["marks_awarded"] for q in _leaves(on)] == [q["marks_awarded"] for q in _leaves(off)]


def test_every_deduction_states_the_reason(graded):
    below = [q for q in _leaves(graded[True]) if not q["not_attempted"] and q["marks_awarded"] < q["max_marks"]]
    assert below, "fixture must contain below-full answers"
    for q in below:
        fb = q["feedback"]
        reason = svc._weakest_reason(q["points"])
        assert fb.strip(), q
        assert reason and reason.lower() in fb.lower(), (reason, fb)
        assert not PRAISE_ONLY.match(fb.strip()), fb


def test_full_marks_and_unattempted_untouched(graded):
    on, off = _leaves(graded[True]), _leaves(graded[False])
    for a, b in zip(on, off):
        if a["not_attempted"] or a["marks_awarded"] >= a["max_marks"]:
            assert a["feedback"] == b["feedback"]


def test_weakest_reason_unit():
    pts = [{"verdict": "full", "marks_awarded": 1, "max_marks": 1, "reason": "ok"},
           {"verdict": "partial", "marks_awarded": 0.5, "max_marks": 1, "reason": "Second law  not\nstated."},
           {"verdict": "mostly", "marks_awarded": 0.5, "max_marks": 0.5, "reason": "minor"}]
    assert svc._weakest_reason(pts) == "Second law not stated."
    assert svc._weakest_reason([{"verdict": "none", "marks_awarded": 0, "max_marks": 2, "reason": "",
                                 "text": "Define refraction"}]) == "Missing: Define refraction"
    fb = svc._feedback_with_reason("Good answer.", pts, 2.0, 2.5, True)
    assert fb == "Second law not stated.\nGood answer."
    assert svc._feedback_with_reason("Second law not stated. Add it.", pts, 2.0, 2.5, True) == \
        "Second law not stated. Add it."                       # already there: unchanged
    assert svc._feedback_with_reason("Great.", pts, 2.5, 2.5, True) == "Great."
