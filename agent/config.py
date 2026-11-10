"""Agent config. Key VALUES are never stored here, only env var NAMES."""
import os
import tempfile

AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")            # paper generation, verifier, paper flows
AGENT_FAST_MODEL = os.getenv("AGENT_FAST_MODEL", "claude-haiku-5-5")   # insight-only turns (agent/routing.py)
GEMINI_MODEL = os.getenv("AGENT_GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FAST_MODEL = os.getenv("AGENT_GEMINI_FAST_MODEL", GEMINI_MODEL)
ANTHROPIC_KEY_ENV = os.getenv("AGENT_ANTHROPIC_KEY_ENV", "A4AI_ANTHROPIC_API_KEY")
GEMINI_KEY_ENV = os.getenv("AGENT_GEMINI_KEY_ENV", "GEMINI_API_KEY")
PROVIDER = os.getenv("AGENT_PROVIDER", "anthropic")  # anthropic | gemini

MAX_LLM_STEPS = 10
MAX_TOOL_CALLS = 40
MAX_WALL_S = 150
APPROVAL_TTL_S = 15 * 60
MAX_VERIFY_ROUNDS = 3
TRACE_PATH = os.getenv("AGENT_TRACE_PATH", os.path.join(tempfile.gettempdir(), "a4ai_agent_trace.jsonl"))  # /var/tmp does not exist on Windows

# USD per 1M tokens (in, out) — for trace cost estimates only. Cache reads cost CACHE_READ_X of the input price,
# cache writes CACHE_WRITE_X (Anthropic 5-minute cache). Gemini implicit cache reads: 0.25 x input.
PRICES = {"claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50), "gemini-2.5-flash": (0.30, 2.50)}
CACHE_READ_X = {"anthropic": 0.10, "gemini": 0.25}
CACHE_WRITE_X = 1.25


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def tool_timeout_s(kind: str) -> float:
    """Per-tool wall-clock limit: read tools (DB queries, 8 s statement timeout inside) vs paper tools (LLM work)."""
    return _float("AGENT_TOOL_TIMEOUT_S", 20.0) if kind == "read" else _float("AGENT_PAPER_TOOL_TIMEOUT_S", 240.0)


def tool_concurrency() -> int:
    """Independent read tools requested in ONE model response run concurrently, at most this many at a time."""
    return max(int(_float("AGENT_TOOL_CONCURRENCY", 4)), 1)


LLM_RETRIES = 2            # retries on 429 / 5xx / overloaded / connection errors (then a retryable error event)
LLM_BACKOFF_BASE_S = 0.6   # delay = base * 2**attempt + uniform(0, base), capped by LLM_BACKOFF_MAX_S
LLM_BACKOFF_MAX_S = 8.0
