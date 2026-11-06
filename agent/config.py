"""Agent config. Key VALUES are never stored here, only env var NAMES."""
import os
import tempfile

AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-sonnet-5-5")
GEMINI_MODEL = os.getenv("AGENT_GEMINI_MODEL", "gemini-2.5-flash")
ANTHROPIC_KEY_ENV = os.getenv("AGENT_ANTHROPIC_KEY_ENV", "A4AI_ANTHROPIC_API_KEY")
GEMINI_KEY_ENV = os.getenv("AGENT_GEMINI_KEY_ENV", "GEMINI_API_KEY")
PROVIDER = os.getenv("AGENT_PROVIDER", "anthropic")  # anthropic | gemini

MAX_LLM_STEPS = 10
MAX_TOOL_CALLS = 40
MAX_WALL_S = 150
APPROVAL_TTL_S = 15 * 60
MAX_VERIFY_ROUNDS = 3
TRACE_PATH = os.getenv("AGENT_TRACE_PATH", os.path.join(tempfile.gettempdir(), "a4ai_agent_trace.jsonl"))  # /var/tmp does not exist on Windows

# USD per 1M tokens (in, out) — for trace cost estimates only.
PRICES = {"claude-sonnet-5-5": (3.0, 15.0), "gemini-2.5-flash": (0.30, 2.50)}
