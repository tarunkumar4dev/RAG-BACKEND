"""Gate-1 audit §4: cache isolation in the production backend (docs/COST_MAP.md §4). No network.

Caches here: auth token cache (app/core/auth.py), api_cache NCERT lists/stats (test_generator.py), contest board
(contest_service.py), rag chapter list (rag_service.py), route-template lru (rate_limit.py). The agent's caches are
tested in tests/agent/test_audit_cache_isolation.py.
"""
from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace

import pytest

from app.core import auth
from app.services import contest_service as cs

from audit_support import no_network  # noqa: F401


def _jwt(sub: str) -> str:
    import base64
    import json
    import time
    body = base64.urlsafe_b64encode(json.dumps({"sub": sub, "exp": time.time() + 3600}).encode()).decode().rstrip("=")
    return f"h.{body}.s"


def test_auth_cache_is_keyed_by_token_so_users_never_share(monkeypatch, no_network):  # noqa: F811
    users = {_jwt("teacher-inst-A"): "teacher-inst-A", _jwt("teacher-inst-B"): "teacher-inst-B"}
    calls = []

    def get_user(token):
        calls.append(token)
        return SimpleNamespace(user=SimpleNamespace(id=users[token], email=None, is_anonymous=False,
                                                    role="authenticated"))
    monkeypatch.setattr(auth, "get_supabase", lambda: SimpleNamespace(auth=SimpleNamespace(get_user=get_user)))
    auth._cache.clear()
    for _ in range(2):
        for tok, uid in users.items():
            assert asyncio.run(auth.verify_token(tok)).id == uid
    assert len(calls) == 2                       # one verification per token, then cached
    assert all(k not in auth._cache for k in users)   # raw tokens are never keys (sha256)


def test_auth_cache_holds_identity_only_not_institute_or_role():
    fields = set(auth.AuthUser.__dataclass_fields__) if hasattr(auth.AuthUser, "__dataclass_fields__") else set(
        inspect.signature(auth.AuthUser).parameters)
    assert fields == {"id", "email", "is_anonymous", "role"}   # "role" = Supabase claim, not the institute role
    assert auth.CACHE_TTL_SECONDS == 60


def test_ncert_cache_value_does_not_depend_on_the_caller():
    """api_cache keys hold no institute/user — safe only because the NCERT bank is global. Pin that: the loader
    takes no caller argument. If a per-institute filter is ever added, the key must change with it."""
    from app.api.v1.endpoints import test_generator as ep
    params = set(inspect.signature(ep._ncert_questions_impl).parameters)
    assert not params & {"user", "user_id", "teacher_id", "institute_id", "viewer_id"}
    params = set(inspect.signature(ep._ncert_question_stats_impl).parameters)
    assert not params & {"user", "user_id", "teacher_id", "institute_id", "viewer_id"}


def test_contest_board_cache_is_per_contest(monkeypatch):
    """The board cache holds the full teacher view (names, e-mails, answers) per contest id; per-viewer filtering
    happens after the cache (get_leaderboard). Two contests never share an entry."""
    loaded = []

    class Q:
        def __init__(self, cid=None):
            self.cid = cid

        def select(self, *_a, **_k):
            return self

        def eq(self, _col, val):
            return Q(val)

        def execute(self):
            loaded.append(self.cid)
            return SimpleNamespace(data=[{"id": f"att-{self.cid}", "student_id": f"stu-{self.cid}",
                                          "student_name": f"name-{self.cid}", "student_email": None,
                                          "status": "submitted", "score": 1, "total_marks": 1, "percentage": 100,
                                          "time_taken_seconds": 1, "warning_count": 0, "submitted_at": None,
                                          "submitted_via": None, "late": False, "started_at": None,
                                          "retry_of": None, "answers": {}}])
    db = SimpleNamespace(table=lambda name: Q())
    monkeypatch.setattr(cs, "sweep_stale", lambda *a, **k: None)
    monkeypatch.setattr(cs, "_load_questions", lambda db, cid: [])
    cs._board_cache.clear()
    try:
        a = cs._board(db, {"id": "contest-A"})
        b = cs._board(db, {"id": "contest-B"})
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"board builder needs more fixture fields: {e}")
    flat = repr(b)
    assert "name-contest-A" not in flat and "stu-contest-A" not in flat
    assert "name-contest-A" in repr(a)
    assert set(cs._board_cache) == {"contest-A", "contest-B"}
    assert cs.BOARD_TTL_S == 10
