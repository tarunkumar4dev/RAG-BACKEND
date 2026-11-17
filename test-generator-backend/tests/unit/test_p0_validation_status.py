"""Honest validation_status (fix/p0-quality, task 5): the generator says needs_review; only a checker "ok"
verdict can make a question verified (applied by the client); /save stores the status the client holds."""
import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.api.v1.endpoints import test_generator as ep
from app.core.auth import AuthUser, require_user
from app.models.test_generator import ChapterSection, TestGenerationRequest
from app.services import test_generator_service as g

TEST_ID = "00000000-0000-0000-0000-0000000000aa"
CH = ChapterSection(chapter="Light", quantity=2, difficulty="medium", format="mcq", marks_per_question=1)
REQ = TestGenerationRequest(exam_title="T", subject="Science", class_grade="10", teacher_id="u", chapters=[CH])


def test_generated_questions_are_needs_review():
    raw = json.dumps({"questions": [{
        "text": "Which mirror always forms a virtual, erect and diminished image?", "format": "mcq",
        "options": ["A) Plane mirror", "B) Concave mirror", "C) Convex mirror", "D) None of these"],
        "correct_answer": "C) Convex mirror", "explanation": "A convex mirror always forms such an image."}]})
    out = g._parse_batch(raw, CH, REQ)
    assert [q.validation_status for q in out] == ["needs_review"]
    assert ep._transform_backend_to_frontend(out, ep.FrontendGenerateRequest(
        simpleData=[ep.FrontendChapterRow(topic="Light", quantity=1)])).questions[0].validationStatus == "needs_review"


def test_only_ok_verdicts_can_upgrade(monkeypatch):
    """The route reports "ok" explicitly; fix / reject / not-checked questions get no "ok" row to upgrade from."""
    opts = ["A) 1", "B) 2", "C) 3", "D) 4"]
    qs = [{"id": f"q{i}", "format": "mcq", "text": f"Question {i}?", "options": opts, "correctAnswer": "A) 1",
           "explanation": "x"} for i in range(1, 5)]
    reply = json.dumps({"results": [{"id": "q1", "verdict": "ok"},
                                    {"id": "q2", "verdict": "fix", "correct_answer": "B) 2"},
                                    {"id": "q3", "verdict": "reject", "issue": "ambiguous"}]})   # q4: no verdict
    res = g.verify_answer_key(qs, "Maths", "10", call=lambda prompt: reply, model="m", include_ok=True)
    assert [r["id"] for r in res["results"] if r["verdict"] == "ok"] == ["q1"]


class _FakeTable:
    def __init__(self, db, name):
        self.db, self.name, self.op, self.payload = db, name, None, None

    def select(self, *a, **k):
        self.op = "select"; return self

    def insert(self, payload):
        self.op, self.payload = "insert", payload; return self

    def update(self, payload):
        self.op, self.payload = "update", payload; return self

    def delete(self):
        self.op = "delete"; return self

    def eq(self, *a):
        return self

    def execute(self):
        self.db.ops.append((self.name, self.op, self.payload))
        data = [{"id": TEST_ID, "teacher_id": "teacher-1", "exam_title": "T"}] if self.op == "select" else []
        return type("R", (), {"data": data})()


class _FakeDb:
    def __init__(self):
        self.ops = []

    def table(self, name):
        return _FakeTable(self, name)


@pytest.fixture
def db(monkeypatch):
    fake = _FakeDb()
    monkeypatch.setattr(ep, "get_supabase_admin", lambda: fake)
    main._rate_store.clear()
    main.app.dependency_overrides[require_user] = lambda: AuthUser(id="teacher-1", email=None, is_anonymous=False,
                                                                   role="authenticated")
    yield fake
    main.app.dependency_overrides.clear()


def test_save_persists_the_current_status(db):
    qs = [{"id": "00000000-0000-0000-0000-000000000001", "text": "a", "validationStatus": "verified"},
          {"id": "00000000-0000-0000-0000-000000000002", "text": "b", "validationStatus": "needs_review"},
          {"id": "00000000-0000-0000-0000-000000000003", "text": "c"},                                   # old client
          {"id": "00000000-0000-0000-0000-000000000004", "text": "d", "validationStatus": "anything"},
          {"id": "00000000-0000-0000-0000-000000000005", "text": "e", "isManual": True, "validationStatus": "verified"}]
    r = TestClient(main.app).post("/api/v1/test-generator/save", json={"test_id": TEST_ID, "questions": qs})
    assert r.status_code == 200
    rows = next(p for name, op, p in db.ops if name == "questions" and op == "insert")
    assert [row["validation_status"] for row in rows] == ["verified", "needs_review", "needs_review", "needs_review", "manual"]
