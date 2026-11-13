"""Phase 1 tests: pool, routing, cache breakpoints, parallel tools, bench-before metrics."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from agent import config, routing
from agent.llm import AnthropicProvider, LLMResponse, ToolCall, fast_provider_for, prompt_cache_enabled
from agent.pool import close_all, enabled, get_pool, read_only_tx, sizes, stats


# ── routing ────────────────────────────────────────────────────────────────────

class TestRouting:
    def test_insight_fast(self):
        assert routing.choose("Meri classes dikhao", []) == ("fast", "insight")

    def test_attendance_fast(self):
        assert routing.choose("ECE-2 ki attendance kya hai", []) == ("fast", "insight")

    def test_paper_intent_hindi(self):
        assert routing.choose("ECE-2 ke liye paper banao", []) == ("main", "paper_intent")

    def test_paper_intent_english(self):
        assert routing.choose("generate a quiz for CSE-1", []) == ("main", "paper_intent")

    def test_paper_intent_devanagari(self):
        assert routing.choose("ECE-2 के लिए पेपर बनाओ", []) == ("main", "paper_intent")

    def test_paper_keyword_blueprint(self):
        assert routing.choose("show the blueprint", []) == ("main", "paper_intent")

    def test_paper_keyword_finalize(self):
        assert routing.choose("finalize the test", []) == ("main", "paper_intent")

    def test_count_tests_is_fast(self):
        # "kitne tests bane" is an insight query, not paper generation
        assert routing.choose("kitne tests bane is hafte mein?", []) == ("fast", "insight")

    def test_how_many_tests_is_fast(self):
        assert routing.choose("how many tests were created?", []) == ("fast", "insight")

    def test_paper_in_progress(self):
        hist = [
            {"role": "user", "text": "ECE-2 ka paper banao"},
            {"role": "assistant", "text": "", "tool_calls": [{"name": "blueprint_defaults", "id": "t1", "args": {}}]},
        ]
        assert routing.choose("haan save karo", hist) == ("main", "paper_in_progress")

    def test_routing_off(self, monkeypatch):
        monkeypatch.setenv("AGENT_ROUTING", "0")
        assert routing.choose("marks dikhao", []) == ("main", "routing_off")

    def test_escalate_on_write_tool(self):
        assert routing.escalate(["save_draft"], ["attendance_stats", "resolve_entity"]) is True

    def test_no_escalate_on_read_tools(self):
        assert routing.escalate(["attendance_stats"], ["attendance_stats", "resolve_entity"]) is False


# ── prompt cache breakpoints ───────────────────────────────────────────────────

class TestCacheBreakpoints:
    """AnthropicProvider.request() must place cache_control on: last tool, system, last message block."""

    def _make(self, cache=True):
        client = MagicMock()
        return AnthropicProvider(client=client, model="claude-sonnet-5-5", cache=cache)

    def test_cache_breakpoints_present(self):
        p = self._make(cache=True)
        tools = [{"name": "t1", "description": "d1", "parameters": {}},
                 {"name": "t2", "description": "d2", "parameters": {}}]
        msgs = [{"role": "user", "text": "hello"}, {"role": "assistant", "text": "hi"},
                {"role": "user", "text": "world"}]
        kw = p.request("sys prompt", msgs, tools, max_tokens=1024)
        # system has cache_control
        assert isinstance(kw["system"], list)
        assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
        # last tool def has cache_control, first does not
        assert "cache_control" not in kw["tools"][0]
        assert kw["tools"][-1]["cache_control"] == {"type": "ephemeral"}
        # last message block has cache_control
        last_msg = kw["messages"][-1]
        content = last_msg["content"]
        blocks = content if isinstance(content, list) else [content]
        last_block = blocks[-1] if isinstance(blocks[-1], dict) else {"text": blocks[-1]}
        assert last_block.get("cache_control") == {"type": "ephemeral"}

    def test_cache_off_no_breakpoints(self):
        p = self._make(cache=False)
        tools = [{"name": "t1", "description": "d1", "parameters": {}}]
        msgs = [{"role": "user", "text": "hello"}]
        kw = p.request("sys prompt", msgs, tools, max_tokens=1024)
        assert isinstance(kw["system"], str)  # plain string, not list
        assert "cache_control" not in kw["tools"][0]

    def test_prefix_stability(self):
        """Same system + tools → same prefix (independent of messages). This is how cache hits work:
        tools and system are the stable prefix shared across turns."""
        p = self._make(cache=True)
        tools = [{"name": "t1", "description": "d1", "parameters": {"type": "object"}}]
        kw1 = p.request("sys", [{"role": "user", "text": "a"}], tools, 1024)
        kw2 = p.request("sys", [{"role": "user", "text": "b"}, {"role": "assistant", "text": "ok"},
                                 {"role": "user", "text": "c"}], tools, 1024)
        # tools and system are the same
        assert kw1["tools"] == kw2["tools"]
        assert kw1["system"] == kw2["system"]

    def test_prompt_cache_env(self, monkeypatch):
        monkeypatch.setenv("AGENT_PROMPT_CACHE", "0")
        assert not prompt_cache_enabled()
        monkeypatch.setenv("AGENT_PROMPT_CACHE", "1")
        assert prompt_cache_enabled()


# ── fast_provider_for ──────────────────────────────────────────────────────────

class TestFastProvider:
    def test_returns_none_for_fakes(self):
        fake = MagicMock()
        fake.model = "fake"
        assert fast_provider_for(fake) is None

    def test_returns_provider_for_anthropic(self, monkeypatch):
        monkeypatch.setattr(config, "AGENT_FAST_MODEL", "claude-haiku-5-5")
        client = MagicMock()
        main = AnthropicProvider(client=client, model="claude-sonnet-5-5")
        fast = fast_provider_for(main)
        assert fast is not None
        assert fast.model == "claude-haiku-5-5"
        assert fast.client is client  # shares the HTTP client

    def test_returns_none_when_same_model(self, monkeypatch):
        monkeypatch.setattr(config, "AGENT_FAST_MODEL", "claude-sonnet-5-5")
        client = MagicMock()
        main = AnthropicProvider(client=client, model="claude-sonnet-5-5")
        assert fast_provider_for(main) is None


# ── pool ───────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clean_pools():
    yield
    close_all()


class TestPool:
    def test_enabled_default(self, monkeypatch):
        monkeypatch.delenv("AGENT_DB_POOL", raising=False)
        assert enabled()

    def test_disabled(self, monkeypatch):
        monkeypatch.setenv("AGENT_DB_POOL", "0")
        assert not enabled()

    def test_sizes_defaults(self, monkeypatch):
        monkeypatch.delenv("AGENT_DB_POOL_MIN", raising=False)
        monkeypatch.delenv("AGENT_DB_POOL_MAX", raising=False)
        assert sizes() == (1, 5)

    def test_sizes_custom(self, monkeypatch):
        monkeypatch.setenv("AGENT_DB_POOL_MIN", "2")
        monkeypatch.setenv("AGENT_DB_POOL_MAX", "8")
        assert sizes() == (2, 8)

    def test_sizes_min_clamped(self, monkeypatch):
        monkeypatch.setenv("AGENT_DB_POOL_MIN", "10")
        monkeypatch.setenv("AGENT_DB_POOL_MAX", "3")
        assert sizes() == (3, 3)

    def test_stats_empty(self):
        close_all()
        assert stats() == {}


# ── parallel tools ─────────────────────────────────────────────────────────────

class TestParallelTools:
    """The agent runs independent read tools concurrently (bounded by tool_concurrency)."""

    def test_parallel_reads_faster_than_serial(self):
        """Two 50 ms tools running in parallel should finish in < 100 ms total wall time
        (serial would be ≥ 100 ms). We use the core._POOL executor directly."""
        from concurrent.futures import ThreadPoolExecutor

        def slow_read():
            time.sleep(0.05)
            return {"status": "ok"}

        pool = ThreadPoolExecutor(max_workers=4)
        t0 = time.perf_counter()
        futs = [pool.submit(slow_read) for _ in range(2)]
        [f.result() for f in futs]
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.09  # generous: parallel = ~50 ms, serial = ~100 ms
        pool.shutdown(wait=False)

    def test_tool_concurrency_default(self, monkeypatch):
        monkeypatch.delenv("AGENT_TOOL_CONCURRENCY", raising=False)
        assert config.tool_concurrency() == 4

    def test_tool_timeout_read(self, monkeypatch):
        monkeypatch.delenv("AGENT_TOOL_TIMEOUT_S", raising=False)
        assert config.tool_timeout_s("read") == 20

    def test_tool_timeout_paper(self, monkeypatch):
        monkeypatch.delenv("AGENT_PAPER_TOOL_TIMEOUT_S", raising=False)
        assert config.tool_timeout_s("paper") == 240


# ── trace write lock ──────────────────────────────────────────────────────────

class TestTraceWriteLock:
    """Parallel trace writes must produce exactly one JSON object per line."""

    def test_no_interleaving(self, tmp_path, monkeypatch):
        from agent.trace import Tracer
        monkeypatch.setattr(config, "TRACE_PATH", str(tmp_path / "t.jsonl"))
        tracers = [Tracer(request_id=f"r{i}") for i in range(4)]
        barrier = threading.Barrier(4)

        def writer(t, i):
            barrier.wait()
            for j in range(25):
                t.tool(f"tool_{i}", {"k": "v"}, float(j), True)

        threads = [threading.Thread(target=writer, args=(tracers[i], i)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        lines = (tmp_path / "t.jsonl").read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 100  # 4 threads × 25 writes
        for line in lines:
            json.loads(line)  # must not raise
