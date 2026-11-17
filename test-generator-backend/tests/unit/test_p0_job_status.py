"""tests.status while a paper is generated: generating -> draft | failed, and GET /status/{test_id}
(fix/p0-quality, task 7). Fake database, no LLM, no network."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import main
from app.api.v1.endpoints import test_generator as ep
from app.core import quota
from app.core.auth import AuthUser, require_user
from app.services.test_generator_service import GenerationError, MSG_PAPER_FAILED

from audit_support import no_network  # noqa: F401

USER = "11111111-2222-3333-4444-555555555555"
OTHER = "99999999-2222-3333-4444-555555555555"


class FakeDB:
    """tests table with insert / update / select and the eq filters the endpoint uses."""

    def __init__(self):
        self.rows, self.history, self.fail_insert = {}, [], False

    def table(self, name):
        assert name == "tests"
        db = self

        class Q:
            def __init__(self):
                self.op, self.payload, self.filters = None, None, {}

            def insert(self, row):
                self.op, self.payload = "insert", row
                return self

            def update(self, fields):
                self.op, self.payload = "update", fields
                return self

            def select(self, *_a):
                self.op = "select"
                return self

            def eq(self, col, val):
                self.filters[col] = val
                return self

            def limit(self, _n):
                return self

            def execute(self):
                hit = [r for r in db.rows.values() if all(r.get(c) == v for c, v in self.filters.items())]
                if self.op == "insert":
                    if db.fail_insert:
                        raise RuntimeError("db down")
                    db.rows[self.payload["id"]] = dict(self.payload)
                    db.history.append(("insert", self.payload["status"]))
                    return SimpleNamespace(data=[self.payload])
                if self.op == "update":
                    for r in hit:
                        r.update(self.payload)
                        db.history.append(("update", r["status"]))
                    return SimpleNamespace(data=hit)
                return SimpleNamespace(data=hit)
        return Q()


@pytest.fixture
def db(monkeypatch, no_network):  # noqa: F811
    fake = FakeDB()
    seen = {}
    monkeypatch.setattr(ep, "get_supabase_admin", lambda: fake)
    monkeypatch.setattr(ep, "reserve", lambda *a, **k: SimpleNamespace(usage={}))
    monkeypatch.setattr(ep, "consume", lambda rsv: {"used": 1, "limit": 10, "remaining": 9})
    monkeypatch.setattr(ep, "release", lambda rsv: None)
    monkeypatch.setattr(ep, "log_usage", lambda *a, **k: None)
    monkeypatch.setattr(ep, "retrieve_context", lambda *a, **k: [{"chapter": "Light", "content": "x"}])

    def generate(*a, **k):
        seen["status_during_generation"] = [r["status"] for r in fake.rows.values()]
        return []
    monkeypatch.setattr(ep, "generate_test", generate)
    fake.seen = seen
    return fake


def _req():
    return ep.FrontendGenerateRequest(subject="Science", cbsePattern=False,
                                      simpleData=[ep.FrontendChapterRow(topic="Light", quantity=5)])


def _call():
    user = AuthUser(id=USER, email=None, is_anonymous=False, role="authenticated")
    return asyncio.run(ep.generate_from_frontend(_req(), user, "k-1"))


def test_success_goes_generating_then_draft_under_one_id(db):
    resp = _call()
    assert db.seen["status_during_generation"] == ["generating"]       # row exists before the LLM is called
    assert db.history == [("insert", "generating"), ("update", "draft")]
    assert list(db.rows) == [resp.testId]
    row = db.rows[resp.testId]
    assert (row["status"], row["teacher_id"], row["total_questions"]) == ("draft", USER, 0)


@pytest.mark.parametrize("error, http_status", [
    (RuntimeError("provider down"), 500),
    (GenerationError(MSG_PAPER_FAILED, 503), 503),
    (ValueError("bad chapter"), 422),
    (HTTPException(status_code=404, detail="Content not available"), 404),
])
def test_failure_path_writes_failed(db, monkeypatch, error, http_status):
    def boom(*a, **k):
        raise error
    monkeypatch.setattr(ep, "generate_test", boom)
    with pytest.raises(HTTPException) as e:
        _call()
    assert e.value.status_code == http_status
    assert db.history == [("insert", "generating"), ("update", "failed")]
    assert [r["status"] for r in db.rows.values()] == ["failed"]


def test_no_row_when_the_request_is_refused_before_generation(db, monkeypatch):
    def no_quota(*a, **k):
        raise HTTPException(status_code=403, detail="limit reached")
    monkeypatch.setattr(ep, "reserve", no_quota)
    with pytest.raises(HTTPException):
        _call()
    assert db.rows == {} and db.history == []


def test_status_row_failure_never_stops_the_paper(db):
    db.fail_insert = True                     # "generating" insert fails -> old behaviour: one insert at the end
    with pytest.raises(RuntimeError):         # (the final insert fails too here; it is logged, the paper is returned)
        db.table("tests").insert({"id": "x", "status": "draft"}).execute()
    resp = _call()
    assert resp.ok is True and db.rows == {}


def _client(user_id):
    main._rate_store.clear()
    main.app.dependency_overrides[require_user] = lambda: AuthUser(id=user_id, email=None, is_anonymous=False,
                                                                   role="authenticated")
    return TestClient(main.app)


def test_status_endpoint_is_owner_only(db):
    resp = _call()
    url = f"/api/v1/test-generator/status/{resp.testId}"
    try:
        assert _client(USER).get(url).json() == {"status": "draft"}
        assert _client(OTHER).get(url).status_code == 404
        assert _client(USER).get("/api/v1/test-generator/status/not-a-uuid").status_code == 400
        main.app.dependency_overrides.clear()
        assert TestClient(main.app).get(url).status_code == 401
    finally:
        main.app.dependency_overrides.clear()


def test_stale_generating_row_is_reported_failed(db):
    now = datetime.now(timezone.utc)
    fresh, stale = "00000000-0000-0000-0000-00000000000a", "00000000-0000-0000-0000-00000000000b"
    db.rows[fresh] = {"id": fresh, "teacher_id": USER, "status": "generating", "created_at": now.isoformat()}
    db.rows[stale] = {"id": stale, "teacher_id": USER, "status": "generating",
                      "created_at": (now - timedelta(hours=1)).isoformat()}
    try:
        client = _client(USER)
        assert client.get(f"/api/v1/test-generator/status/{fresh}").json() == {"status": "generating"}
        assert client.get(f"/api/v1/test-generator/status/{stale}").json() == {"status": "failed"}
    finally:
        main.app.dependency_overrides.clear()
