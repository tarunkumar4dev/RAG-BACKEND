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
