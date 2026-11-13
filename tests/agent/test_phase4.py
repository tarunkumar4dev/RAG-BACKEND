"""Phase 4 tests: rate limits, metrics (no PII), structured logging."""
from __future__ import annotations

import json
import logging
import time
import uuid

import pytest

from agent import features
from agent.metrics import METRICS
from agent.rate_limit import (
    RateLimitHit, check_institute_budget, check_user_rate, record_tokens, reset,
)


@pytest.fixture(autouse=True)
def _clean():
    reset()
    METRICS.reset()
    yield
    reset()
    METRICS.reset()


# ── per-user rate limit ────────────────────────────────────────────────────────

class TestUserRate:
    def test_under_limit_ok(self, monkeypatch):
        monkeypatch.setenv("AGENT_USER_TURNS_PER_10MIN", "5")
        for _ in range(5):
            assert check_user_rate("u1") is None

    def test_over_limit_blocked(self, monkeypatch):
        monkeypatch.setenv("AGENT_USER_TURNS_PER_10MIN", "3")
        for _ in range(3):
            assert check_user_rate("u1") is None
        hit = check_user_rate("u1")
        assert hit is not None
        assert hit.scope == "user" and hit.code == "rate_limited"
        assert hit.retry_after_s > 0

    def test_different_users_independent(self, monkeypatch):
        monkeypatch.setenv("AGENT_USER_TURNS_PER_10MIN", "2")
        assert check_user_rate("u1") is None
        assert check_user_rate("u1") is None
        assert check_user_rate("u2") is None  # u2 is fresh
        assert check_user_rate("u1") is not None  # u1 is blocked

    def test_disabled_when_zero(self, monkeypatch):
        monkeypatch.setenv("AGENT_USER_TURNS_PER_10MIN", "0")
        for _ in range(100):
            assert check_user_rate("u1") is None


# ── per-institute daily budget ─────────────────────────────────────────────────

class TestInstituteBudget:
    def test_no_budget_always_ok(self, monkeypatch):
        monkeypatch.delenv("AGENT_INSTITUTE_DAILY_TOKENS", raising=False)
        record_tokens("inst1", 999999)
        assert check_institute_budget("inst1") is None

    def test_under_budget_ok(self, monkeypatch):
        monkeypatch.setenv("AGENT_INSTITUTE_DAILY_TOKENS", "10000")
        record_tokens("inst1", 5000)
        assert check_institute_budget("inst1") is None

    def test_over_budget_blocked(self, monkeypatch):
        monkeypatch.setenv("AGENT_INSTITUTE_DAILY_TOKENS", "1000")
        record_tokens("inst1", 1500)
        hit = check_institute_budget("inst1")
        assert hit is not None
        assert hit.scope == "institute"
        assert hit.retry_after_s > 0


# ── metrics ────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_render_contains_help_and_type(self):
        METRICS.inc("agent_chat_turns_total", {"route": "fast", "outcome": "ok"})
        text = METRICS.render()
        assert "# HELP agent_chat_turns_total" in text
        assert "# TYPE agent_chat_turns_total counter" in text
        assert 'agent_chat_turns_total{outcome="ok",route="fast"} 1' in text

    def test_histogram_buckets(self):
        METRICS.observe("agent_tool_seconds", {"tool": "resolve_entity"}, 0.03)
        text = METRICS.render()
        assert 'agent_tool_seconds_bucket{le="0.05",tool="resolve_entity"} 1' in text
        assert 'agent_tool_seconds_count{tool="resolve_entity"} 1' in text

    def test_no_pii_in_metrics(self):
        """Metric labels must come from fixed vocabularies only. No user ids, names, or text."""
        METRICS.tool("attendance_stats", 0.1, True, "ok")
        METRICS.llm("claude-sonnet-5-5", 1.0, 100, 50, 0, 0, 0.001)
        METRICS.route("fast", "insight")
        METRICS.turn("fast", 1.5, "ok")
        METRICS.error("budget_exceeded")
        METRICS.request("POST", "/agent/chat", 200, 1.5)
        text = METRICS.render()
        # No UUIDs, no names, no text fragments
        import re
        assert not re.search(r"[0-9a-f]{8}-[0-9a-f]{4}-", text), "UUID found in metrics"
        # Only known label values
        for line in text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            assert "user_id" not in line and "student" not in line.lower()

    def test_unknown_tool_sanitized(self):
        METRICS.tool("some_injected_name", 0.1, True, "ok")
        text = METRICS.render()
        assert 'tool="unknown"' in text

    def test_rate_limit_hits_counted(self, monkeypatch):
        monkeypatch.setenv("AGENT_USER_TURNS_PER_10MIN", "1")
        check_user_rate("u_test")
        check_user_rate("u_test")  # this one should be blocked
        val = METRICS.value("agent_rate_limit_hits_total", {"scope": "user"})
        assert val >= 1


# ── JSON logging ──────────────────────────────────────────────────────────────

class TestJSONLogging:
    def test_json_formatter(self):
        from agent.logging_config import JSONFormatter
        fmt = JSONFormatter()
        record = logging.LogRecord("agent", logging.INFO, "", 0, "test message", (), None)
        record.request_id = "abc123"
        line = fmt.format(record)
        d = json.loads(line)
        assert d["message"] == "test message"
        assert d["request_id"] == "abc123"
        assert d["level"] == "INFO"
        # No PII fields
        assert "user" not in d and "institute" not in d

    def test_no_request_id_omitted(self):
        from agent.logging_config import JSONFormatter
        fmt = JSONFormatter()
        record = logging.LogRecord("agent", logging.INFO, "", 0, "hello", (), None)
        d = json.loads(fmt.format(record))
        assert "request_id" not in d
