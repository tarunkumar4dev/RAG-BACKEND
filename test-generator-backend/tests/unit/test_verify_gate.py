"""Phase 1C/1D: the blocking verify gate and honest validation_status.

No network — the checker is injected as a fake `verifier`. Covers the blocking
semantics for MCQ/AR and the honest-status semantics for free-text.
"""
import json

import pytest

from app.core.config import settings
from app.models.test_generator import (
    ChapterSection, DifficultyLevel, QuestionFormat, GeneratedQuestion,
)
from app.services import test_generator_service as T


class _Req:
    subject = "Science"
    class_grade = "10"


def _q(qid, fmt="mcq", correct="B) Water", options=None):
    if options is None and fmt in ("mcq", "assertion_reason"):
        options = ["A) Air", "B) Water", "C) Fire", "D) Earth"]
    return GeneratedQuestion(
        id=qid, text=f"Question {qid}?", options=options,
        correct_answer=correct, explanation="because.", marks=1,
        difficulty=DifficultyLevel.MEDIUM, chapter="Ch",
        format=QuestionFormat(fmt), validation_status="needs_review",
    )


def _verifier(results, checked_ids):
    def fake(dicts, subject, class_grade):
        return {"results": results, "checked_ids": checked_ids,
                "checked": len(checked_ids), "fixed": 0, "flagged": 0, "model": "fake"}
    return fake


@pytest.fixture(autouse=True)
def _strict(monkeypatch):
    monkeypatch.setattr(settings, "STRICT_PIPELINE", True)


# ── MCQ / AR: blocking ────────────────────────────────────────────────────

def test_mcq_ok_is_verified_and_kept():
    qs = [_q("m1")]
    out = T._verify_gate(qs, _Req(), verifier=_verifier([], ["m1"]))
    assert len(out) == 1 and out[0].validation_status == "verified"


def test_mcq_reject_is_dropped():
    qs = [_q("m1")]
    out = T._verify_gate(qs, _Req(),
                         verifier=_verifier([{"id": "m1", "verdict": "reject", "issue": "two correct"}], ["m1"]))
    assert out == []


def test_mcq_fix_is_dropped_not_shipped():
    """Generator and checker disagreed on the key -> drop, do NOT ship a single-model fix."""
    qs = [_q("m1")]
    out = T._verify_gate(qs, _Req(),
                         verifier=_verifier([{"id": "m1", "verdict": "fix",
                                              "correctAnswer": "C) Fire"}], ["m1"]))
    assert out == []


def test_mcq_unchecked_is_dropped():
    """Checker could not check it -> an unverified MCQ must not ship."""
    qs = [_q("m1")]
    out = T._verify_gate(qs, _Req(), verifier=_verifier([], []))  # not in checked_ids
    assert out == []


def test_assertion_reason_unchecked_dropped():
    qs = [_q("a1", fmt="assertion_reason",
             correct="A) Both A and R are true and R explains A")]
    out = T._verify_gate(qs, _Req(), verifier=_verifier([], []))
    assert out == []


# ── Free-text: honest status, no hard block except reject ─────────────────

def test_short_answer_ok_is_verified():
    qs = [_q("s1", fmt="short_answer", correct="A clear two-step answer.", options=None)]
    out = T._verify_gate(qs, _Req(), verifier=_verifier([], ["s1"]))
    assert len(out) == 1 and out[0].validation_status == "verified"


def test_short_answer_fix_applies_correction_and_flags_review():
    qs = [_q("s1", fmt="short_answer", correct="wrong answer", options=None)]
    out = T._verify_gate(qs, _Req(),
                         verifier=_verifier([{"id": "s1", "verdict": "fix",
                                              "correctAnswer": "corrected answer",
                                              "explanation": "new expl", "issue": "key off"}], ["s1"]))
    assert len(out) == 1
    assert out[0].validation_status == "needs_review"
    assert out[0].correct_answer == "corrected answer"
    assert out[0].explanation == "new expl"


def test_short_answer_reject_is_dropped():
    qs = [_q("s1", fmt="short_answer", correct="x", options=None)]
    out = T._verify_gate(qs, _Req(),
                         verifier=_verifier([{"id": "s1", "verdict": "reject", "issue": "ambiguous"}], ["s1"]))
    assert out == []


def test_short_answer_unchecked_kept_as_needs_review():
    qs = [_q("s1", fmt="short_answer", correct="x", options=None)]
    out = T._verify_gate(qs, _Req(), verifier=_verifier([], []))
    assert len(out) == 1 and out[0].validation_status == "needs_review"


# ── checker unavailable ───────────────────────────────────────────────────

def test_checker_failure_drops_mcq_flags_freetext():
    def boom(dicts, subject, class_grade):
        raise RuntimeError("checker down")
    qs = [_q("m1"), _q("s1", fmt="short_answer", correct="x", options=None)]
    out = T._verify_gate(qs, _Req(), verifier=boom)
    ids = {q.id for q in out}
    assert "m1" not in ids                       # MCQ dropped (blocking)
    assert "s1" in ids                           # free-text kept
    assert next(q for q in out if q.id == "s1").validation_status == "needs_review"


# ── flag OFF = no-op ──────────────────────────────────────────────────────

def test_gate_is_noop_when_strict_off(monkeypatch):
    monkeypatch.setattr(settings, "STRICT_PIPELINE", False)
    qs = [_q("m1")]
    # Even a verifier that would reject must be ignored (gate returns early).
    out = T._verify_gate(qs, _Req(),
                         verifier=_verifier([{"id": "m1", "verdict": "reject"}], ["m1"]))
    assert len(out) == 1


# ── 1D: honest parse status ───────────────────────────────────────────────

def _chapter():
    return ChapterSection(chapter="Light", topic=None, subtopics=[], quantity=5,
                          difficulty=DifficultyLevel.MEDIUM, format=QuestionFormat.MCQ,
                          marks_per_question=1)


_RAW = json.dumps({"questions": [{
    "text": "Which lens converges light to a focal point for distant objects?",
    "format": "mcq",
    "options": ["A) Concave", "B) Convex", "C) Plane", "D) Cylindrical"],
    "correct_answer": "B) Convex",
    "explanation": "A convex lens converges parallel rays to its focus.",
    "marks": 1, "difficulty": "medium", "bloom_level": "understand",
    "topic": "Lenses",
}]})


def test_parse_marks_needs_review_under_strict(monkeypatch):
    monkeypatch.setattr(settings, "STRICT_PIPELINE", True)
    qs = T._parse_batch(_RAW, _chapter(), _Req())
    assert qs and all(q.validation_status == "needs_review" for q in qs)


def test_parse_keeps_verified_when_strict_off(monkeypatch):
    monkeypatch.setattr(settings, "STRICT_PIPELINE", False)
    qs = T._parse_batch(_RAW, _chapter(), _Req())
    assert qs and all(q.validation_status == "verified" for q in qs)
