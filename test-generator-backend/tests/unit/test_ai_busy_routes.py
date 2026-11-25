"""Gemini busy after the one retry → HTTP 503 with error.code AI_BUSY (copy checker + smart-assignment router)."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.core.api_errors import AI_BUSY, AI_BUSY_MSG
from app.routers import copy_checker_router as ccr
from app.routers import smart_assignment_router as sar
from app.services.smart_assignment_service import GeminiBusy

SHEET = "00000000-0000-0000-0000-000000000001"
ASG = "00000000-0000-0000-0000-000000000002"


@pytest.fixture
def client(monkeypatch):
    caller = SimpleNamespace(user=SimpleNamespace(id="teacher-1"), db=None, token="t")
    main.app.dependency_overrides[sar.require_caller] = lambda: caller
    monkeypatch.setattr(ccr, "_checks_today", lambda uid: 0)
    monkeypatch.setattr(sar, "_checks_today", lambda uid: 0)
    main._rate_store.clear()
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def _busy(*a, **k):
    raise GeminiBusy(429)


def _assert_ai_busy(r):
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == {"code": AI_BUSY, "message": AI_BUSY_MSG}
    assert body["detail"] == AI_BUSY_MSG          # old frontends show `detail`
    assert r.headers.get("retry-after") == "60"


def test_copy_checker_sheet(client, monkeypatch):
    monkeypatch.setattr(ccr, "_check_sheet", _busy)
    _assert_ai_busy(client.post(f"/api/v1/copy-checker/sheets/{SHEET}/check"))


def test_copy_checker_rubric(client, monkeypatch):
    monkeypatch.setattr(ccr, "_build_session_rubric", _busy)
    _assert_ai_busy(client.post(f"/api/v1/copy-checker/sessions/{SHEET}/rubric", json={}))


def test_smart_assignment_submission(client, monkeypatch):
    monkeypatch.setattr(sar, "_check_submission", _busy)
    _assert_ai_busy(client.post(f"/api/v1/smart-assignment-checker/assignments/{ASG}/submissions/{SHEET}/check"))


def test_smart_assignment_rubric(client, monkeypatch):
    monkeypatch.setattr(sar, "_ensure_rubric", _busy)
    _assert_ai_busy(client.post(f"/api/v1/smart-assignment-checker/assignments/{ASG}/rubric", json={}))


# ── Gemini call timeout on the direct copy-checker routes (17 Nov: a 40-mark paper's rubric call needed more
#    than the global 60 s; the session failed with the generic "Could not prepare the marking scheme") ──────────

class ReadTimeout(Exception):
    """Stands in for requests.exceptions.ReadTimeout (what google-genai 0.8.0 raises on a call timeout)."""


def _timed_out(*a, **k):
    raise ReadTimeout("read timed out")


def test_copy_checker_rubric_timeout_is_ai_busy(client, monkeypatch):
    monkeypatch.setattr(ccr, "_build_session_rubric", _timed_out)
    _assert_ai_busy(client.post(f"/api/v1/copy-checker/sessions/{SHEET}/rubric", json={}))


def test_copy_checker_sheet_timeout_is_ai_busy(client, monkeypatch):
    monkeypatch.setattr(ccr, "_check_sheet", _timed_out)
    _assert_ai_busy(client.post(f"/api/v1/copy-checker/sheets/{SHEET}/check"))


def _rubric_session(monkeypatch, build):
    """_build_session_rubric for an uploaded paper with the DB / storage stubbed; returns the session updates."""
    from app.services import smart_assignment_service as svc
    updates = []
    monkeypatch.setattr(ccr, "_get_session", lambda c, sid: {"source": "upload", "qp_paths": ["u/s/paper/1.pdf"]})
    monkeypatch.setattr(ccr, "_update_session", lambda c, sid, fields: updates.append(dict(fields)) or fields)
    monkeypatch.setattr(ccr, "_download_own", lambda c, p, prefix: "/tmp/x.pdf")
    monkeypatch.setattr(ccr, "_cleanup", lambda *paths: None)
    monkeypatch.setattr(ccr, "build_rubric", lambda qp, key: build(svc))
    return updates


def test_copy_checker_rubric_call_gets_the_stage_timeout(monkeypatch):
    seen = []

    def build(svc):
        seen.append(svc._call_timeout.get())
        return {"questions": [], "total_marks": 40}

    updates = _rubric_session(monkeypatch, build)
    ccr._build_session_rubric(None, SHEET, False)
    assert seen == [ccr.RUBRIC_CALL_TIMEOUT_S] and ccr.RUBRIC_CALL_TIMEOUT_S > 60
    assert updates[-1]["rubric_status"] == "ready"


def test_copy_checker_rubric_timeout_saves_the_busy_message(monkeypatch):
    updates = _rubric_session(monkeypatch, lambda svc: _timed_out())
    with pytest.raises(ReadTimeout):
        ccr._build_session_rubric(None, SHEET, False)
    assert updates[-1] == {"rubric_status": "failed", "rubric_error": AI_BUSY_MSG}
