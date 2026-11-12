"""Gate 2 fix #1: /worksheet/generate-direct and /modules/{id}/generate-worksheet need a verified login, are
rate-limited on the verified user id, count against the plan quota and never send more than
WORKSHEET_MAX_INPUT_CHARS to Gemini. Mocked auth / DB / Gemini, no network."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import modules as ep
from app.core import auth
from app.main import app
from audit_support import no_network  # noqa: F401

USER = "11111111-2222-3333-4444-555555555555"
OTHER = "99999999-2222-3333-4444-555555555555"


@pytest.fixture
def ws(monkeypatch, no_network):  # noqa: F811
    prompts, usage = [], {"checks": 0, "records": 0, "allowed": True}

    async def verify(token):
        return auth.AuthUser(id=USER, email=None, is_anonymous=False, role="authenticated") if token == "good" else None
    monkeypatch.setattr(auth, "verify_token", verify)
    def reserve(uid, action, key):
        usage["checks"] += 1
        if not usage["allowed"]:
            raise ep.HTTPException(403, "limit")
        return SimpleNamespace(open=True)
    monkeypatch.setattr(ep, "reserve", reserve)
    monkeypatch.setattr(ep, "consume", lambda r: usage.__setitem__("records", usage["records"] + 1) or {})
    monkeypatch.setattr(ep, "release", lambda r: usage.__setitem__("released", usage.get("released", 0) + 1))
    import google.genai as genai_mod
    from app.services import module_service as ms

    class Client:
        def __init__(self, **_):
            self.models = SimpleNamespace(generate_content=lambda model, contents, config: (
                prompts.append(contents) or SimpleNamespace(text=json.dumps({"questions": [{"q_no": 1}]}))))
    monkeypatch.setattr(genai_mod, "Client", Client)
    monkeypatch.setattr(ms, "get_supabase", lambda: SimpleNamespace(storage=SimpleNamespace(
        from_=lambda b: SimpleNamespace(download=lambda p: b"%PDF-fake"))))
    monkeypatch.setattr(ms.ModuleService, "_extract_pdf", staticmethod(lambda path: ("Z" * 500_000, 1, False)))
    ep._worksheet_limiter._hits.clear()
    c = TestClient(app)
    body = {"teacher_id": OTHER, "storage_path": f"{USER}/worksheets/a.pdf", "subject": "Science", "num_questions": 5}
    return c, body, prompts, usage


def _post(c, body, token="good"):
    h = {"Authorization": f"Bearer {token}"} if token else {}
    return c.post("/api/v1/worksheet/generate-direct", json=body, headers=h)


def test_no_token_is_401_and_no_gemini_call(ws):
    c, body, prompts, usage = ws
    assert _post(c, body, token=None).status_code == 401
    assert _post(c, body, token="forged").status_code == 401
    assert prompts == [] and usage["checks"] == 0


def test_ok_counts_quota_and_caps_input(ws):
    c, body, prompts, usage = ws
    r = _post(c, body)
    assert r.status_code == 200, r.text
    assert usage == {"checks": 1, "records": 1, "allowed": True}
    assert len(prompts) == 1
    assert prompts[0].count("Z") == ep.WORKSHEET_MAX_INPUT_CHARS == 320_000


def test_quota_exhausted_blocks_before_gemini(ws):
    c, body, prompts, usage = ws
    usage["allowed"] = False
    assert _post(c, body).status_code == 403
    assert prompts == [] and usage["records"] == 0


def test_other_users_folder_is_refused(ws):
    c, body, prompts, usage = ws
    assert _post(c, {**body, "storage_path": f"{OTHER}/worksheets/a.pdf"}).status_code == 403
    assert _post(c, {**body, "storage_path": f"{USER}/../{OTHER}/a.pdf"}).status_code == 403
    assert prompts == [] and usage.get("released") == 2 and usage["records"] == 0


def test_rate_limit_is_on_verified_user_not_body_id(ws):
    c, body, prompts, _ = ws
    for i in range(ep.WORKSHEET_PER_USER_PER_MIN):
        assert _post(c, {**body, "teacher_id": f"{i:08d}-0000-0000-0000-000000000000"}).status_code == 200
    r = _post(c, {**body, "teacher_id": OTHER})
    assert r.status_code == 429 and r.headers.get("retry-after") == "60"
    assert len(prompts) == ep.WORKSHEET_PER_USER_PER_MIN


def test_module_worksheet_route_needs_token(ws):
    c, *_ = ws
    r = c.post(f"/api/v1/modules/{USER}/generate-worksheet", json={"teacher_id": USER})
    assert r.status_code == 401
