"""Gate-1 audit §3: paper-limit correctness (docs/COST_MAP.md §3). Mocked DB + mocked generator, no network.

FakeDB mirrors the LIVE SQL (read with pg_get_functiondef on 12 Nov 2026):
  check_usage(u)  -> allowed = sum(count this month) < plan.test_limit        (read only)
  record_usage(u) -> upsert count + 1, NO limit check                           (write)
The endpoint runs check -> generate (tens of seconds) -> record. Tests named *_today pin current behaviour;
xfail(strict=True) tests state the correct behaviour and flip when a fix lands.
"""
from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints import test_generator as ep
from app.core.auth import AuthUser

from audit_support import no_network  # noqa: F401

USER = "11111111-2222-3333-4444-555555555555"


class FakeDB:
    def __init__(self, limit: int, used: int = 0):
        self.limit, self.used = limit, used
        self.lock = threading.Lock()
        self.tests_rows = []

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
    def make(limit, used=0, generate=None):
        db = FakeDB(limit, used)
        monkeypatch.setattr(ep, "get_supabase_admin", lambda: db)
        monkeypatch.setattr(ep, "retrieve_context", lambda *a, **k: [{"chapter": "Light", "content": "x"}])
        monkeypatch.setattr(ep, "generate_test", generate or (lambda *a, **k: SimpleNamespace()))
        monkeypatch.setattr(ep, "_transform_backend_to_frontend",
                            lambda resp, req: SimpleNamespace(generationTime=0, testId="t1", totalQuestions=5,
                                                              meta={}, questions=[]))
        monkeypatch.setattr(ep, "_transform_frontend_to_backend",
                            lambda req: SimpleNamespace(chapters=[SimpleNamespace(quantity=5, chapter="Light",
                                                                                  topic=None)],
                                                        subject="Science", class_grade="10"))
        return db
    return make


def _call(req_overrides=None):
    req = SimpleNamespace(userId=None, subject="Science", classGrade="10", simpleData=[1], cbsePattern=False,
                          examTitle="T", board="CBSE", **(req_overrides or {}))
    user = AuthUser(id=USER, email=None, is_anonymous=False, role="authenticated")
    return asyncio.run(ep.generate_from_frontend(req, user))


def _safe_call():
    try:
        _call()
        return "ok"
    except HTTPException as e:
        return e.status_code
    except Exception as e:  # noqa: BLE001  (the tests-row builder may not like the stub; usage is what we watch)
        return type(e).__name__


# 1. job fails after the quota check ----------------------------------------------------------------------------

def test_failed_generation_is_not_charged(env):
    def boom(*a, **k):
        raise RuntimeError("provider down")
    db = env(limit=10, used=3, generate=boom)
    assert _safe_call() == 500
    assert db.used == 3   # correct: nothing recorded


# 2. two concurrent requests with 1 unit left --------------------------------------------------------------------

def _two_instances_race(db):
    """Two serverless instances: both run check_usage before either records (generation takes 20-60 s)."""
    a = ep.check_usage(USER)
    b = ep.check_usage(USER)
    for _ in (a, b):
        ep.record_usage(USER)
    return a["allowed"], b["allowed"]


def test_concurrent_last_unit_today_overruns_limit(env):
    db = env(limit=10, used=9)
    assert _two_instances_race(db) == (True, True)
    assert db.used == 11          # 11/10: the plan limit is exceeded (we pay for the extra paper)


@pytest.mark.xfail(strict=True, reason="AUDIT: check and record are separate; record_usage has no limit check")
def test_concurrent_last_unit_only_one_allowed(env):
    db = env(limit=10, used=9)
    _two_instances_race(db)
    assert db.used <= 10


# 3. client disconnects mid-generation ---------------------------------------------------------------------------

def test_disconnect_mid_generation_still_charges_and_saves(env):
    """The endpoint never checks request.is_disconnected(); generation is synchronous inside `async def`, so a
    closed tab cannot stop it. The paper is counted AND saved to `tests`, so the teacher finds it in history."""
    gone = threading.Event()

    def gen(*a, **k):
        gone.set()        # the browser tab closes here
        return SimpleNamespace()
    db = env(limit=10, used=0, generate=gen)
    _safe_call()
    assert gone.is_set()
    assert db.used == 1


# 4. retry of an already-counted job -----------------------------------------------------------------------------

def test_retry_of_counted_request_today_charges_twice(env):
    """No idempotency key: a resubmit after a network drop (response lost, paper already counted) is a new paper."""
    db = env(limit=10, used=0)
    _safe_call()
    _safe_call()
    assert db.used == 2


@pytest.mark.xfail(strict=True, reason="AUDIT: /generate-frontend has no idempotency key")
def test_retry_with_same_request_id_charged_once(env):
    db = env(limit=10, used=0)
    _safe_call()
    _safe_call()
    assert db.used == 1


# record_usage failure is swallowed ------------------------------------------------------------------------------

def test_record_failure_gives_free_paper_today(env, monkeypatch):
    db = env(limit=1, used=0)
    real_rpc = db.rpc

    def rpc(name, params):
        if name == "record_usage":
            raise RuntimeError("db timeout")
        return real_rpc(name, params)
    db.rpc = rpc
    _safe_call()
    _safe_call()
    assert db.used == 0           # two papers, zero counted ("Record usage failed (non-fatal)")
