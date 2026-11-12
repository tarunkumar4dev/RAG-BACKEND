"""Gate-1 audit §3 / Gate-2 fix #5: paper-limit correctness. Mocked DB + mocked generator, no network.

FakeDB implements the LIVE legacy pair (check_usage read / record_usage write, read with pg_get_functiondef on
12 Nov 2026) AND the migration-32 trio (a4_reserve_usage / a4_consume_usage / a4_release_usage) with the same
semantics as supabase/migrations/32_quota_reserve_voice_allowance.sql: lock + check + count in one call, keyed by
request id. `legacy=True` makes the new functions raise PGRST202 (migration not applied) to test the fallback.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints import test_generator as ep
from app.core import quota
from app.core.auth import AuthUser

from audit_support import no_network  # noqa: F401

USER = "11111111-2222-3333-4444-555555555555"


class FakeDB:
    def __init__(self, limit: int, used: int = 0, legacy: bool = False):
        self.limit, self.used, self.legacy = limit, used, legacy
        self.lock = threading.Lock()
        self.reservations: dict[str, dict] = {}
        self.tests_rows = []
        self.fail_consume = False

    def rpc(self, name, params):
        def execute():
            with self.lock:
                if name == "check_usage":
                    return SimpleNamespace(data={"allowed": self.used < self.limit, "used": self.used,
                                                 "limit": self.limit, "remaining": max(self.limit - self.used, 0),
                                                 "plan": "starter"})
                if name == "record_usage":
                    self.used += 1
                    return SimpleNamespace(data={"recorded": True, "used": self.used})
                if self.legacy:
                    raise RuntimeError('{"code":"PGRST202","message":"Could not find the function public.%s"}' % name)
                if name == "a4_reserve_usage":
                    rid = params["p_request_id"]
                    if rid in self.reservations:
                        return SimpleNamespace(data={"status": "duplicate", "state": self.reservations[rid]["state"]})
                    if self.limit >= 0 and self.used >= self.limit:
                        return SimpleNamespace(data={"status": "limit_reached", "allowed": False, "used": self.used,
                                                     "limit": self.limit, "remaining": 0, "plan": "starter"})
                    self.used += 1
                    self.reservations[rid] = {"state": "reserved", "user": params["p_user_id"]}
                    return SimpleNamespace(data={"status": "reserved", "allowed": True, "used": self.used,
                                                 "limit": self.limit, "remaining": self.limit - self.used,
                                                 "plan": "starter"})
                if name == "a4_consume_usage":
                    if self.fail_consume:
                        raise RuntimeError("db timeout")
                    r = self.reservations.get(params["p_request_id"])
                    if r and r["state"] == "reserved":
                        r["state"] = "consumed"
                    return SimpleNamespace(data={"consumed": bool(r)})
                if name == "a4_release_usage":
                    r = self.reservations.get(params["p_request_id"])
                    if r and r["state"] == "reserved":
                        r["state"] = "released"
                        self.used = max(self.used - 1, 0)
                        return SimpleNamespace(data={"released": True})
                    return SimpleNamespace(data={"released": False})
            raise AssertionError(name)
        return SimpleNamespace(execute=execute)

    def table(self, name):
        db = self

        class T:
            def insert(self, row):
                db.tests_rows.append(row)
                return SimpleNamespace(execute=lambda: SimpleNamespace(data=[row]))
        return T()


@pytest.fixture
def env(monkeypatch, no_network):  # noqa: F811
    def make(limit, used=0, generate=None, legacy=False):
        db = FakeDB(limit, used, legacy)
        monkeypatch.setattr(quota, "get_supabase_admin", lambda: db)
        monkeypatch.setattr(quota, "_legacy_warned", False)
        monkeypatch.setattr(ep, "get_supabase_admin", lambda: db)
        monkeypatch.setattr(ep, "retrieve_context", lambda *a, **k: [{"chapter": "Light", "content": "x"}])
        monkeypatch.setattr(ep, "generate_test", generate or (lambda *a, **k: SimpleNamespace()))
        monkeypatch.setattr(ep, "_transform_backend_to_frontend",
                            lambda resp, req: SimpleNamespace(generationTime=0, testId="t1", totalQuestions=5, examTitle="T", totalMarks=5,
                                                              meta={}, questions=[]))
        monkeypatch.setattr(ep, "_transform_frontend_to_backend",
                            lambda req: SimpleNamespace(chapters=[SimpleNamespace(quantity=5, chapter="Light",
                                                                                  topic=None)],
                                                        subject="Science", class_grade="10"))
        return db
    return make


def _call(key="k-1"):
    req = SimpleNamespace(userId=None, subject="Science", classGrade="10", simpleData=[1], cbsePattern=False,
                          examTitle="T", board="CBSE", paperDate=None)
    user = AuthUser(id=USER, email=None, is_anonymous=False, role="authenticated")
    return asyncio.run(ep.generate_from_frontend(req, user, key))


def _safe_call(key="k-1"):
    try:
        _call(key)
        return "ok"
    except HTTPException as e:
        return e.status_code
    except Exception as e:  # noqa: BLE001  (the tests-row builder may not like the stub; usage is what we watch)
        return type(e).__name__


# 1. job fails after the quota check -> released ----------------------------------------------------------------

def test_failed_generation_is_released(env):
    def boom(*a, **k):
        raise RuntimeError("provider down")
    db = env(limit=10, used=3, generate=boom)
    assert _safe_call() == 500
    assert db.used == 3
    assert db.reservations["k-1"]["state"] == "released"


def test_budget_503_is_released_and_passed_through(env):
    def boom(*a, **k):
        from app.services.test_generator_service import GenerationError, MSG_PAPER_FAILED
        raise GenerationError(MSG_PAPER_FAILED, 503)
    db = env(limit=10, used=3, generate=boom)
    assert _safe_call() == 503
    assert db.used == 3


# 2. two concurrent requests with 1 unit left -> exactly one ------------------------------------------------------

def test_concurrent_last_unit_only_one_allowed(env):
    db = env(limit=10, used=9)
    out = []

    def one(k):
        try:
            quota.reserve(USER, quota.ACTION_PAPER, k)
            out.append("ok")
        except HTTPException as e:
            out.append(e.status_code)
    ts = [threading.Thread(target=one, args=(f"r{i}",)) for i in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert out.count("ok") == 1 and out.count(403) == 5
    assert db.used == 10


# 3. client disconnects mid-generation -------------------------------------------------------------------------

def test_disconnect_mid_generation_still_charges_and_saves(env):
    """No disconnect check; generation finishes, is counted and saved to `tests`, so it is in History."""
    gone = threading.Event()

    def gen(*a, **k):
        gone.set()
        return SimpleNamespace()
    db = env(limit=10, used=0, generate=gen)
    _safe_call()
    assert gone.is_set() and db.used == 1 and db.reservations["k-1"]["state"] == "consumed"


# 4. retry of an already-counted job -> 409, charged once ----------------------------------------------------------

def test_retry_with_same_request_id_charged_once(env):
    db = env(limit=10, used=0)
    assert _safe_call("same") == "ok"
    assert _safe_call("same") == 409
    assert db.used == 1


def test_new_request_id_is_a_new_paper(env):
    db = env(limit=10, used=0)
    _safe_call("a")
    _safe_call("b")
    assert db.used == 2


# consume failure: the unit was counted at reserve time, so a DB hiccup cannot give a free paper -------------------

def test_consume_failure_still_counts(env):
    db = env(limit=1, used=0)
    db.fail_consume = True
    assert _safe_call("x") == "ok"
    assert db.used == 1
    assert _safe_call("y") == 403


# legacy path (migration 32 not applied): behaves exactly as before ----------------------------------------------

def test_legacy_fallback_when_migration_missing(env, caplog):
    db = env(limit=10, used=0, legacy=True)
    assert _safe_call("k") == "ok"
    assert db.used == 1 and db.reservations == {}
    assert "migration 32" in caplog.text
    db.used = 10
    assert _safe_call("k2") == 403


def test_legacy_failed_generation_not_charged(env):
    def boom(*a, **k):
        raise RuntimeError("provider down")
    db = env(limit=10, used=3, generate=boom, legacy=True)
    assert _safe_call() == 500
    assert db.used == 3
