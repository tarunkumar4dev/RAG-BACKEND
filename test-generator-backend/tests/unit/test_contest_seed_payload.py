"""scripts/contest_seed/contest_payload.json (the Education Beast contest, posted by post_contest.py) is a valid
POST /contests body: 50 MCQs, 50 marks, max_warnings 5, every answer a bare label of its own options."""

import json

import pytest

from app.schemas.contest import CreateContestRequest
from tests.conftest import ROOT

PAYLOAD = ROOT / "scripts" / "contest_seed" / "contest_payload.json"
pytestmark = pytest.mark.skipif(not PAYLOAD.exists(), reason=f"{PAYLOAD.name} not in scripts/contest_seed/")


@pytest.fixture(scope="module")
def body():
    return CreateContestRequest(**json.loads(PAYLOAD.read_text(encoding="utf-8-sig")))


def test_payload_is_accepted_by_create_contest(body):
    assert len(body.questions) == 50
    assert sum(q.marks for q in body.questions) == 50
    assert body.max_warnings == 5
    assert 5 <= body.duration_minutes <= 180


def test_every_answer_is_a_label_of_its_options(body):
    for i, q in enumerate(body.questions, start=1):
        assert q.question_type == "MCQ", i
        labels = [o["label"] if isinstance(o, dict) else "ABCDEF"[k] for k, o in enumerate(q.options or [])]
        assert len(labels) >= 2, i
        assert q.correct_answer in labels, (i, q.correct_answer, labels)
