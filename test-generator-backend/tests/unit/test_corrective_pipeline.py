"""Phase 2C: corrective re-prompt, JSON repair, verify-gate reason capture.
No network — LLM calls are injected."""
import json

import pytest

from app.core.config import settings
from app.models.test_generator import DifficultyLevel, QuestionFormat, GeneratedQuestion
from app.services import test_generator_service as T


# ── corrective note ───────────────────────────────────────────────────────

def test_corrective_note_empty_when_no_reasons():
    assert T._corrective_note({}) == ""


def test_corrective_note_lists_top_reasons():
    note = T._corrective_note({"wrong_answer": 5, "duplicate": 2, "malformed": 1, "low_quality": 1})
    assert "Fix these specifically" in note
    assert "EXACTLY ONE option" in note          # wrong_answer (most frequent) included
    # capped at 3 bullet lines
    assert note.count("\n- ") <= 3


def test_corrective_note_ignores_unknown_reason():
    assert T._corrective_note({"nonsense": 3}) == ""


# ── JSON repair ─────────────────────────────────────────────────────────────

_GOOD = json.dumps({"questions": [{"text": "x" * 20, "format": "mcq",
                                   "options": ["A", "B", "C", "D"],
                                   "correct_answer": "A", "explanation": "because it is"}]})


def test_repair_json_recovers_valid_json():
    # LLM returns clean JSON given the broken input.
    out = T._repair_json("{questions: [broken", lambda p: _GOOD)
    assert out == _GOOD


def test_repair_json_returns_none_when_still_invalid():
    assert T._repair_json("garbage", lambda p: "still not json {") is None


def test_repair_json_handles_llm_error():
    def boom(p):
        raise RuntimeError("down")
    assert T._repair_json("garbage", boom) is None


def test_repair_json_empty_input():
    assert T._repair_json("", lambda p: _GOOD) is None


# ── verify-gate reason capture ───────────────────────────────────────────────

def _mcq(qid):
    return GeneratedQuestion(id=qid, text=f"Q{qid}?", options=["A) a", "B) b", "C) c", "D) d"],
                             correct_answer="A) a", explanation="x", marks=1,
                             difficulty=DifficultyLevel.MEDIUM, chapter="C",
                             format=QuestionFormat.MCQ, validation_status="needs_review")


class _Req:
    subject = "Science"
    class_grade = "10"


def test_verify_gate_records_reasons(monkeypatch):
    monkeypatch.setattr(settings, "STRICT_PIPELINE", True)

    def verifier(dicts, subject, grade):
        return {"results": [{"id": "m1", "verdict": "reject", "issue": "two correct"}],
                "checked_ids": ["m1", "m2"]}   # m2 ok, m1 reject; m3 unchecked

    reasons = []
    out = T._verify_gate([_mcq("m1"), _mcq("m2"), _mcq("m3")], _Req(),
                         verifier=verifier, reasons_out=reasons)
    ids = {q.id for q in out}
    assert ids == {"m2"}                       # m1 rejected, m3 unverified -> dropped
    assert "wrong_answer" in reasons           # from the reject
    assert "unverified" in reasons             # from m3 not in checked_ids


def test_verify_gate_checker_down_records_unverified(monkeypatch):
    monkeypatch.setattr(settings, "STRICT_PIPELINE", True)

    def boom(dicts, subject, grade):
        raise RuntimeError("checker down")

    reasons = []
    out = T._verify_gate([_mcq("m1")], _Req(), verifier=boom, reasons_out=reasons)
    assert out == []                           # MCQ dropped
    assert reasons == ["unverified"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
