"""Phase 2 tests: cancel abort, retries, approval persistence, DB-down error."""
from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from agent import config
from agent.core import Agent, Cancelled, DataUnavailable, ProviderFailed
from agent.llm import LLMAborted, LLMResponse, ToolCall
from agent.schemas import AgentEventAdapter, SessionCtx
from agent.store import MemoryStore, Owner


# ── fixtures ────────────────────────────────────────────────────────────────────

_INST = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
_USER = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000001")
CTX = SessionCtx(user_id=_USER, institute_id=_INST, teacher_id=_USER, role="admin",
                 data_source="demo")


class Scripted:
    model = "test-model"

    def __init__(self, responses, delay=0):
        self._r = list(responses)
        self._i = 0
        self.delay = delay

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False, cancel=None):
        if self.delay:
            time.sleep(self.delay)
        if cancel is not None and cancel.is_set():
            raise LLMAborted(self.model, 0, max_tokens)
        r = self._r[self._i % len(self._r)]
        self._i += 1
        return r


def _run(agent, msg):
    return list(agent.run(msg))


# ── cancel abort ──────────────────────────────────────────────────────────────

class TestCancelAbort:
    def test_cancel_before_llm_stops_immediately(self):
        """When the cancel event is already set, the agent yields a final with cancelled=True without calling LLM."""
        cancel = threading.Event()
        cancel.set()  # pre-set
        prov = Scripted([LLMResponse(text="should not reach", model="test-model")])
        agent = Agent(prov, CTX, str(uuid.uuid4()), cancel_event=cancel)
        evs = _run(agent, "hello")
        final = evs[-1]
        assert final["type"] == "final"
        assert final["cancelled"] is True

    def test_cancel_during_llm_aborts(self):
        """A provider that raises LLMAborted when cancel is set → cancelled final."""
        cancel = threading.Event()

        class AbortingProvider:
            model = "test-model"
            supports_cancel = True

            def complete(self, system, messages, tools, max_tokens=2048, json_mode=False, cancel=None):
                raise LLMAborted(self.model, 10, max_tokens)

        agent = Agent(AbortingProvider(), CTX, str(uuid.uuid4()), cancel_event=cancel)
        cancel.set()
        evs = _run(agent, "hello")
        assert evs[-1]["type"] == "final"
        assert evs[-1]["cancelled"] is True


# ── retries ────────────────────────────────────────────────────────────────────

class TestRetries:
    def test_retry_on_429(self, monkeypatch):
        """A 429 followed by a success should produce a normal response (not an error event)."""
        monkeypatch.setattr(config, "LLM_RETRIES", 1)
        monkeypatch.setattr(config, "LLM_BACKOFF_BASE_S", 0.01)

        call_count = 0

        class RetryProvider:
            model = "test-model"

            def complete(self, system, messages, tools, max_tokens=2048, json_mode=False, cancel=None):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Simulate a 429
                    from anthropic import RateLimitError
                    raise RateLimitError(
                        message="rate limited",
                        response=MagicMock(status_code=429, headers={"retry-after": "0"}),
                        body={"error": {"message": "rate limited", "type": "rate_limit_error"}}
                    )
                return LLMResponse(text="hello back", model="test-model", tokens_in=10, tokens_out=5)

        try:
            from anthropic import RateLimitError
        except ImportError:
            pytest.skip("anthropic not installed")

        agent = Agent(RetryProvider(), CTX, str(uuid.uuid4()))
        evs = _run(agent, "hi")
        # Should succeed on retry
        texts = [e for e in evs if e["type"] == "text_delta"]
        assert any("hello" in e["text"] for e in texts)
        final = evs[-1]
        assert final["type"] == "final"
        assert final["usage"]["retries"] >= 1

    def test_exhausted_retries_give_error_event(self, monkeypatch):
        """When all retries fail, a provider_error event with retryable=True is emitted."""
        monkeypatch.setattr(config, "LLM_RETRIES", 1)
        monkeypatch.setattr(config, "LLM_BACKOFF_BASE_S", 0.01)

        class FailProvider:
            model = "test-model"

            def complete(self, system, messages, tools, max_tokens=2048, json_mode=False, cancel=None):
                raise ConnectionError("network down")

        agent = Agent(FailProvider(), CTX, str(uuid.uuid4()))
        evs = _run(agent, "hi")
        errors = [e for e in evs if e["type"] == "error"]
        assert len(errors) == 1
        assert errors[0]["code"] == "provider_error"
        assert evs[-1]["type"] == "final"


# ── per-tool timeout ──────────────────────────────────────────────────────────

class TestToolTimeout:
    def test_tool_timeout_gives_retryable_error(self, monkeypatch):
        """A tool that exceeds its timeout produces a retryable error event."""
        monkeypatch.setenv("AGENT_TOOL_TIMEOUT_S", "0.05")  # 50 ms timeout

        class SlowToolProvider:
            model = "test-model"

            def __init__(self):
                self._i = 0

            def complete(self, system, messages, tools, max_tokens=2048, json_mode=False, cancel=None):
                self._i += 1
                if self._i == 1:
                    return LLMResponse(
                        tool_calls=[ToolCall("t1", "attendance_stats", {"batch_id": "xxx"})],
                        model="test-model")
                return LLMResponse(text="done", model="test-model")

        agent = Agent(SlowToolProvider(), CTX, str(uuid.uuid4()))

        # Monkey-patch _call to be slow
        orig_call = agent._call

        def slow_call(name, args, emit):
            time.sleep(0.2)  # exceeds the 50 ms timeout
            return orig_call(name, args, emit)

        agent._call = slow_call
        evs = _run(agent, "attendance stats")
        errors = [e for e in evs if e["type"] == "error"]
        assert len(errors) == 1
        assert errors[0]["code"] == "tool_timeout"
        assert errors[0].get("retryable") is True


# ── DB down ────────────────────────────────────────────────────────────────────

class TestDBDown:
    def test_db_unavailable_gives_retryable_error(self):
        """When a tool returns db_unavailable status, the agent emits a retryable error (no crash).
        We patch _call to return the db_unavailable status directly (as the real _call does when it catches
        OperationalError)."""

        class DBFailProvider:
            model = "test-model"

            def __init__(self):
                self._i = 0

            def complete(self, system, messages, tools, max_tokens=2048, json_mode=False, cancel=None):
                self._i += 1
                if self._i == 1:
                    return LLMResponse(
                        tool_calls=[ToolCall("t1", "attendance_stats", {"batch_id": "xxx"})],
                        model="test-model")
                return LLMResponse(text="done", model="test-model")

        agent = Agent(DBFailProvider(), CTX, str(uuid.uuid4()))

        def fail_call(name, args, emit):
            # Simulate what _call does when psycopg.OperationalError is raised
            return {"status": "db_unavailable"}

        agent._call = fail_call
        evs = _run(agent, "show attendance")
        errors = [e for e in evs if e["type"] == "error"]
        assert len(errors) == 1
        assert errors[0]["code"] == "db_unavailable"
        assert errors[0].get("retryable") is True
        assert evs[-1]["type"] == "final"

    def test_operational_error_caught_in_call(self):
        """psycopg.OperationalError inside _call → {"status": "db_unavailable"} (not an exception)."""
        import psycopg
        from agent.core import Agent
        prov = Scripted([LLMResponse(tool_calls=[ToolCall("t1", "my_scope", {})], model="test-model")])
        agent = Agent(prov, CTX, str(uuid.uuid4()))
        spec = agent.tools.get("my_scope")
        if spec is not None:
            orig_fn = spec.fn
            object.__setattr__(spec, "fn",
                               lambda ctx, args: (_ for _ in ()).throw(psycopg.OperationalError("conn refused")))
            res = agent._call("my_scope", {}, lambda x: None)
            assert res["status"] == "db_unavailable"
            object.__setattr__(spec, "fn", orig_fn)


# ── approval persistence ──────────────────────────────────────────────────────

class TestApprovalPersistence:
    def _setup(self):
        store = MemoryStore()
        o = Owner(str(_USER), str(_INST))
        sid = str(uuid.uuid4())
        store.create_session(o, sid, "admin", "test session")
        return store, o, sid

    def test_update_approval_approved(self):
        store, o, sid = self._setup()
        aid = str(uuid.uuid4())
        events = [
            {"type": "text_delta", "text": "here's a paper", "seq": 0},
            {"type": "pending_approval", "approval_id": aid, "action": "finalize_test",
             "summary": "finalize", "args_sha256": "abc", "expires_at": "2099-01-01T00:00:00+00:00",
             "status": "pending", "result_ref": None, "decided_at": None, "seq": 1, "request_id": "r1"},
        ]
        store.add_message(o, sid, "assistant", "paper ready", events=events)
        ok = store.update_approval_in_events(o, sid, aid, "approved", result_ref="/agent/drafts/x/pdf")
        assert ok is True
        msgs = store.all_messages(o, sid)
        stored_events = msgs[0]["events"]
        pa = next(e for e in stored_events if e["type"] == "pending_approval")
        assert pa["status"] == "approved"
        assert pa["result_ref"] == "/agent/drafts/x/pdf"
        assert pa["decided_at"] is not None

    def test_update_approval_cancelled(self):
        store, o, sid = self._setup()
        aid = str(uuid.uuid4())
        events = [
            {"type": "pending_approval", "approval_id": aid, "action": "finalize_test",
             "summary": "finalize", "args_sha256": "abc", "expires_at": "2099-01-01T00:00:00+00:00",
             "status": "pending", "result_ref": None, "decided_at": None, "seq": 0, "request_id": "r1"},
        ]
        store.add_message(o, sid, "assistant", None, events=events)
        ok = store.update_approval_in_events(o, sid, aid, "cancelled")
        assert ok is True
        msgs = store.all_messages(o, sid)
        pa = msgs[0]["events"][0]
        assert pa["status"] == "cancelled"

    def test_expired_approval_on_replay(self):
        store, o, sid = self._setup()
        aid = str(uuid.uuid4())
        past = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat(timespec="seconds")
        events = [
            {"type": "pending_approval", "approval_id": aid, "action": "finalize_test",
             "summary": "finalize", "args_sha256": "abc", "expires_at": past,
             "status": "pending", "result_ref": None, "decided_at": None, "seq": 0, "request_id": "r1"},
        ]
        store.add_message(o, sid, "assistant", None, events=events)
        msgs = store.all_messages(o, sid)
        pa = msgs[0]["events"][0]
        assert pa["status"] == "expired"

    def test_not_expired_stays_pending(self):
        store, o, sid = self._setup()
        aid = str(uuid.uuid4())
        future = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat(timespec="seconds")
        events = [
            {"type": "pending_approval", "approval_id": aid, "action": "finalize_test",
             "summary": "finalize", "args_sha256": "abc", "expires_at": future,
             "status": "pending", "result_ref": None, "decided_at": None, "seq": 0, "request_id": "r1"},
        ]
        store.add_message(o, sid, "assistant", None, events=events)
        msgs = store.all_messages(o, sid)
        pa = msgs[0]["events"][0]
        assert pa["status"] == "pending"

    def test_update_nonexistent_approval_returns_false(self):
        store, o, sid = self._setup()
        store.add_message(o, sid, "assistant", "no approvals", events=[{"type": "text_delta", "text": "hi", "seq": 0}])
        assert store.update_approval_in_events(o, sid, "nonexistent", "approved") is False
