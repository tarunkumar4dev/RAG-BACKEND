"""POST /test-generator/verify-answers (fix/p0-quality, task 1): auth, checker choice, verdict mapping,
checker failure -> 200 {"available": false}, one paid pass per paper. Mocked checker, no network."""
import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.api.v1.endpoints import test_generator as ep
from app.core.auth import AuthUser, require_user
from app.core.config import settings
from app.services import test_generator_service as g

from audit_support import no_network  # noqa: F401

URL = "/api/v1/test-generator/verify-answers"
OPTS = ["A) 30 deg", "B) 45 deg", "C) 60 deg", "D) 90 deg"]
QS = [
    {"id": "q1", "format": "mcq", "text": "Angle one?", "options": OPTS, "correctAnswer": "A) 30 deg", "explanation": "x"},
    {"id": "q2", "format": "mcq", "text": "Angle two?", "options": OPTS, "correctAnswer": "A) 30 deg", "explanation": "x"},
    {"id": "q3", "format": "mcq", "text": "Angle three?", "options": OPTS, "correctAnswer": "A) 30 deg", "explanation": "x"},
    {"id": "q4", "format": "mcq", "text": "Angle four?", "options": OPTS, "correctAnswer": "A) 30 deg", "explanation": "x"},
]
REPLY = json.dumps({"results": [
    {"id": "q1", "verdict": "ok", "issue": ""},
    {"id": "q2", "verdict": "fix", "issue": "key is wrong", "correct_answer": "C) 60 deg", "explanation": "because"},
    {"id": "q3", "verdict": "reject", "issue": "two correct options"},
    {"id": "q4", "verdict": "fix", "issue": "key is wrong", "correct_answer": "120 deg"},   # not an option
]})


@pytest.fixture
def client(monkeypatch, no_network):  # noqa: F811
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "ds-key")
    monkeypatch.setattr(ep, "log_usage", lambda *a, **k: {})
    ep.api_cache.clear()
    main._rate_store.clear()
    main.app.dependency_overrides[require_user] = lambda: AuthUser(id="teacher-1", email=None, is_anonymous=False,
                                                                   role="authenticated")
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()
    ep.api_cache.clear()


def _body(**extra):
    return {"subject": "Maths", "classGrade": "Class 10", "questions": QS, **extra}


def test_requires_login(monkeypatch, no_network):  # noqa: F811
    main.app.dependency_overrides.clear()
    main._rate_store.clear()
    assert TestClient(main.app).post(URL, json=_body()).status_code == 401


def test_verdicts_from_independent_checker(client, monkeypatch):
    calls = []
    monkeypatch.setattr(g, "_call_deepseek", lambda prompt, temperature=None: calls.append(prompt) or REPLY)
    monkeypatch.setattr(g, "_call_gemini_model", lambda *a, **k: pytest.fail("Gemini must not check its own paper"))
    r = client.post(URL, json=_body())
    assert r.status_code == 200
    out = r.json()
    assert out["available"] is True and out["checker_independent"] is True
    assert out["model"] == settings.DEEPSEEK_MODEL
    by = {x["id"]: x for x in out["results"]}
    assert by["q1"]["verdict"] == "ok"
    assert by["q2"]["verdict"] == "fix" and by["q2"]["correctAnswer"] == by["q2"]["correct_answer"] == "C) 60 deg"
    assert by["q2"]["explanation"] == "because"
    assert by["q3"] == {"id": "q3", "verdict": "reject", "issue": "two correct options"}
    assert by["q4"]["verdict"] == "reject"          # a "fix" that names no option is never auto-applied
    assert (out["checked"], out["fixed"], out["flagged"], out["unchecked"]) == (4, 1, 2, 0)
    assert len(calls) == 1                          # 4 questions = one batch = one paid call


def test_no_deepseek_key_is_not_independent(client, monkeypatch):
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "")
    seen = []
    monkeypatch.setattr(g, "_get_gemini_client", lambda: object())
    monkeypatch.setattr(g, "_call_gemini_model",
                        lambda client, prompt, model, temperature=None: seen.append(model) or REPLY)
    out = client.post(URL, json=_body()).json()
    assert out["available"] is True and out["checker_independent"] is False
    assert seen == [settings.GEMINI_VAL_MODEL] and out["model"] == settings.GEMINI_VAL_MODEL


def test_checker_failure_is_200_unavailable(client, monkeypatch):
    def boom(prompt, temperature=None):
        raise g.GenerationError("deepseek: HTTP 500", 502)
    monkeypatch.setattr(g, "_call_deepseek", boom)
    monkeypatch.setattr(g, "_call_gemini_model", lambda *a, **k: pytest.fail("no fallback to a self-check"))
    r = client.post(URL, json=_body())
    assert r.status_code == 200 and r.json() == {"available": False}


def test_unreadable_reply_is_unavailable(client, monkeypatch):
    monkeypatch.setattr(g, "_call_deepseek", lambda prompt, temperature=None: '{"something": "else"}')
    assert client.post(URL, json=_body()).json() == {"available": False}


def test_one_paid_pass_per_paper(client, monkeypatch):
    calls = []
    monkeypatch.setattr(g, "_call_deepseek", lambda prompt, temperature=None: calls.append(1) or REPLY)
    first = client.post(URL, json=_body(testId="paper-1")).json()
    again = client.post(URL, json=_body(testId="paper-1")).json()
    assert first == again and len(calls) == 1
    client.post(URL, json=_body(testId="paper-2"))
    assert len(calls) == 2


def test_route_is_rate_limited():
    from app.core import rate_limit
    assert rate_limit.ROUTE_LIMITS["POST /api/v1/test-generator/verify-answers"] <= 10


def test_offline_eval_contract_unchanged(monkeypatch):
    """scripts/eval_generation.py: no "ok" rows unless asked for."""
    monkeypatch.setattr(g, "_call_checker", lambda prompt: REPLY)
    res = g.verify_answer_key(QS, "Maths", "10")
    assert sorted(r["verdict"] for r in res["results"]) == ["fix", "reject", "reject"]
