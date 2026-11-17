"""Assertion-reason answers that name no option are dropped, never defaulted to A (fix/p0-quality, task 2)."""
import json
import logging

from app.models.test_generator import ChapterSection, TestGenerationRequest
from app.services import test_generator_service as g

CH = ChapterSection(chapter="Light", quantity=3, difficulty="medium", format="assertion_reason", marks_per_question=1)
REQ = TestGenerationRequest(exam_title="T", subject="Science", class_grade="10", teacher_id="u", chapters=[CH])
A, B, C, D = g.ASSERTION_REASON_OPTIONS


def _ar(n, answer, options=None):
    return {"text": f"Assertion (A): Statement number {n} about mirrors is long enough. Reason (R): Because of reason {n}.",
            "format": "assertion_reason", "options": options, "correct_answer": answer,
            "explanation": "A long enough explanation of the assertion and the reason."}


def _parse(*qs):
    return g._parse_batch(json.dumps({"questions": list(qs)}), CH, REQ)


def test_unmatchable_answer_is_dropped_not_shipped_as_A(caplog):
    with caplog.at_level(logging.WARNING, logger=g.logger.name):
        out = _parse(_ar(1, "both are true"), _ar(2, "C"))
    assert [q.correct_answer for q in out] == [C]
    assert "AR_ANSWER_UNMATCHED" in caplog.text


def test_letter_and_text_forms_match():
    for answer, want in [("B", B), ("(d)", D), ("Option C", C), ("c)", C), (A, A),
                         ("A is true but R is false", C), ("D) A is false but R is true", D)]:
        assert g._match_ar_answer(answer, g.ASSERTION_REASON_OPTIONS) == want, answer


def test_contradictory_or_empty_answer_matches_nothing():
    assert g._match_ar_answer("C) Both A and R are true and R is the correct explanation of A", g.ASSERTION_REASON_OPTIONS) is None
    assert g._match_ar_answer("", g.ASSERTION_REASON_OPTIONS) is None
    assert g._match_ar_answer("Both statements are correct", g.ASSERTION_REASON_OPTIONS) is None


def test_letter_maps_by_position_when_options_have_no_letters():
    opts = ["Both true, R explains A", "Both true, R does not explain A", "A true, R false", "A false, R true"]
    out = _parse(_ar(1, "B", options=opts))
    assert out[0].correct_answer == opts[1]
