"""Phase 3 tests: store parity (SQLite + Postgres), identity cache invalidation, refusal never cached."""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from agent.store import MemoryStore, Owner, auto_title, normalize_query


_INST = "aaaaaaaa-0000-0000-0000-000000000001"
_USER = "bbbbbbbb-0000-0000-0000-000000000001"
_O = Owner(_USER, _INST)

# ── store parity suite (runs on SQLite MemoryStore; parametrize with Postgres in CI) ──

class TestStoreParity:
    """Every behaviour that SQLite and Postgres must agree on. Run on MemoryStore by default.
    When AGENT_TEST_STORE_URL is set, also run on PostgresStore (tests/agent/conftest.py could do this)."""

    @pytest.fixture()
    def store(self):
        return MemoryStore()

    def test_create_and_get_session(self, store):
        sid = str(uuid.uuid4())
        s = store.create_session(_O, sid, "admin", "hello")
        assert s["id"] == sid and s["title"] == "hello" and s["message_count"] == 0

    def test_list_sessions_ordering(self, store):
        s1 = str(uuid.uuid4())
        s2 = str(uuid.uuid4())
        store.create_session(_O, s1, "admin", "first")
        time.sleep(0.01)
        store.create_session(_O, s2, "admin", "second")
        items, _ = store.list_sessions(_O)
        assert items[0]["id"] == s2  # newest first

    def test_patch_session_title(self, store):
        sid = str(uuid.uuid4())
        store.create_session(_O, sid, "admin", "old")
        s = store.patch_session(_O, sid, title="new")
        assert s["title"] == "new" and s["title_source"] == "user"

    def test_set_auto_title_does_not_overwrite_user(self, store):
        sid = str(uuid.uuid4())
        store.create_session(_O, sid, "admin", "auto title")
        store.patch_session(_O, sid, title="user title")
        store.set_auto_title(_O, sid, "should not take")
        s = store.get_session(_O, sid)
        assert s["title"] == "user title"

    def test_delete_session(self, store):
        sid = str(uuid.uuid4())
        store.create_session(_O, sid, "admin", "del")
        assert store.delete_session(_O, sid)
        assert store.get_session(_O, sid) is None

    def test_add_and_list_messages(self, store):
        sid = str(uuid.uuid4())
        store.create_session(_O, sid, "admin", "msg test")
        store.add_message(_O, sid, "user", "hello")
        store.add_message(_O, sid, "assistant", "hi", events=[{"type": "text_delta", "text": "hi"}])
        msgs, _ = store.list_messages(_O, sid)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant"

    def test_truncate(self, store):
        sid = str(uuid.uuid4())
        store.create_session(_O, sid, "admin", "trunc")
        store.add_message(_O, sid, "user", "a")
        store.add_message(_O, sid, "assistant", "b")
        store.add_message(_O, sid, "user", "c")
        n = store.truncate(_O, sid, 2)
        assert n == 2
        msgs, _ = store.list_messages(_O, sid)
        assert len(msgs) == 1

    def test_record_and_list_queries(self, store):
        store.create_session(_O, str(uuid.uuid4()), "admin", "q")
        store.record_query(_O, "attendance ECE-2")
        store.record_query(_O, "attendance ECE-2")  # duplicate
        qs = store.list_queries(_O)
        assert len(qs) == 1 and qs[0]["use_count"] == 2

    def test_feedback(self, store):
        sid = str(uuid.uuid4())
        store.create_session(_O, sid, "admin", "fb")
        store.add_message(_O, sid, "user", "q")
        store.add_message(_O, sid, "assistant", "a")
        ok = store.set_feedback(_O, sid, 2, "up", "good")
        assert ok
        fb = store.get_feedback(_O, sid)
        assert len(fb) == 1 and fb[0]["rating"] == "up"

    def test_owner_isolation(self, store):
        """Another user cannot see the session."""
        sid = str(uuid.uuid4())
        store.create_session(_O, sid, "admin", "mine")
        other = Owner("other-user", _INST)
        assert store.get_session(other, sid) is None

    def test_approval_persistence(self, store):
        sid = str(uuid.uuid4())
        aid = str(uuid.uuid4())
        store.create_session(_O, sid, "admin", "approval")
        events = [{"type": "pending_approval", "approval_id": aid, "action": "finalize_test",
                   "summary": "fin", "args_sha256": "x", "expires_at": "2099-01-01T00:00:00+00:00",
                   "status": "pending", "result_ref": None, "decided_at": None, "seq": 0, "request_id": "r"}]
        store.add_message(_O, sid, "assistant", None, events=events)
        assert store.update_approval_in_events(_O, sid, aid, "approved", result_ref="/agent/drafts/1/pdf")
        msgs = store.all_messages(_O, sid)
        pa = msgs[0]["events"][0]
        assert pa["status"] == "approved" and pa["result_ref"] == "/agent/drafts/1/pdf"

    def test_expired_approval_on_replay(self, store):
        sid = str(uuid.uuid4())
        aid = str(uuid.uuid4())
        past = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat(timespec="seconds")
        store.create_session(_O, sid, "admin", "expired")
        events = [{"type": "pending_approval", "approval_id": aid, "action": "finalize_test",
                   "summary": "fin", "args_sha256": "x", "expires_at": past,
                   "status": "pending", "result_ref": None, "decided_at": None, "seq": 0, "request_id": "r"}]
        store.add_message(_O, sid, "assistant", None, events=events)
        msgs = store.all_messages(_O, sid)
        assert msgs[0]["events"][0]["status"] == "expired"

    def test_pinned_filter(self, store):
        s1 = str(uuid.uuid4())
        s2 = str(uuid.uuid4())
        store.create_session(_O, s1, "admin", "pinned")
        store.create_session(_O, s2, "admin", "not pinned")
        store.patch_session(_O, s1, pinned=True)
        items, _ = store.list_sessions(_O, pinned=True)
        assert all(i["pinned"] for i in items) and any(i["id"] == s1 for i in items)


# ── Postgres store parity (only when AGENT_TEST_STORE_URL is set) ──

@pytest.fixture()
def pg_store():
    url = os.getenv("AGENT_TEST_STORE_URL")
    if not url:
        pytest.skip("AGENT_TEST_STORE_URL not set; Postgres store parity tests skipped")
    from agent.store_pg import PostgresStore
    s = PostgresStore(url, pool_min=1, pool_max=2)
    yield s
    s.close()


class TestPostgresStoreParity(TestStoreParity):
    """Same suite, but on PostgresStore."""

    @pytest.fixture()
    def store(self, pg_store):
        return pg_store


# ── identity cache ────────────────────────────────────────────────────────────

class TestIdentityCache:
    def test_invalidate_specific_user(self):
        from agent.auth import _IDENTITY, cached_identity, invalidate_identity
        _IDENTITY.clear()
        from agent.schemas import SessionCtx
        ctx = SessionCtx(user_id=uuid.UUID(_USER), institute_id=uuid.UUID(_INST),
                         teacher_id=uuid.UUID(_USER), role="admin", data_source="demo")
        key = (_USER, _INST, "demo")
        _IDENTITY[key] = (time.monotonic(), ctx)
        n = invalidate_identity(_USER)
        assert n == 1
        assert key not in _IDENTITY

    def test_invalidate_all(self):
        from agent.auth import _IDENTITY, invalidate_identity
        _IDENTITY["a"] = (0, None)
        _IDENTITY["b"] = (0, None)
        n = invalidate_identity()
        assert n >= 2 and len(_IDENTITY) == 0

    def test_refusal_never_cached(self, monkeypatch):
        """If build() raises (401/403), nothing is cached."""
        from agent.auth import _IDENTITY, cached_identity
        _IDENTITY.clear()
        monkeypatch.setenv("AGENT_IDENTITY_CACHE_S", "60")
        key = ("test-refusal",)
        with pytest.raises(ValueError):
            cached_identity(key, lambda: (_ for _ in ()).throw(ValueError("forbidden")))
        assert key not in _IDENTITY
