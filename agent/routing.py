"""Deterministic model routing (no LLM call): which model answers this turn.

- "main" (AGENT_MODEL, default claude-sonnet-5-5): paper generation (blueprint / generate / verify / save /
  finalize / assign) and anything that continues a paper flow. The question generator and the verifier always use
  the main model, whatever the route.
- "fast" (AGENT_FAST_MODEL, default claude-haiku-5-5): everything else = insight turns (attendance, marks, scope,
  comparisons, exports), which only call read tools.
- Escalation inside a turn: if the fast model asks for a non-read tool, the rest of the turn runs on the main model.

Decided from the message text + the turn history (tool names only). Logged by id only: the trace records
{route, reason, model}, never the message. AGENT_ROUTING=0 sends every turn to the main model.
"""
from __future__ import annotations

import os
import re
from typing import Iterable

PAPER_TOOLS = frozenset({"blueprint_defaults", "generate_questions", "verify_paper", "regenerate_failed", "save_draft",
                         "finalize_test", "assign_to_section", "export_pdf"})

_MAKE = (r"banao|bana\s*do|banaiye|banana|bana\s*ke|banake|bnao|generate|create|make|prepare|draft|set\s+(?:a|the)?|"
         r"taiyar|tayyar|likho|बनाओ|बनाइए|बना\s*दो|तैयार")
_PAPER = (r"paper|test|tests|exam|quiz|worksheet|questions?|mcqs?|sawal|savaal|prashn\w*|प्रश्न|पेपर|टेस्ट|सवाल")
_PAPER_INTENT = re.compile(rf"(?:\b(?:{_PAPER})\b.{{0,60}}(?:{_MAKE}))|(?:(?:{_MAKE}).{{0,60}}\b(?:{_PAPER})\b)",
                           re.IGNORECASE | re.DOTALL)
_PAPER_WORDS = re.compile(r"\b(blueprint|question\s*paper|marking\s*scheme|regenerate|finali[sz]e|"
                          r"assign\s+(?:it|the\s+test|test))\b", re.IGNORECASE)
# "kitne tests bane / how many tests were created" are insight questions (tests_created_by_teacher)
_COUNT_TESTS = re.compile(r"\b(kitne|kitna|how\s+many|count|number\s+of|tests?\s+created|bane\b|banaye\b)",
                          re.IGNORECASE)


def enabled() -> bool:
    return os.getenv("AGENT_ROUTING", "1") != "0"


def paper_in_progress(history: Iterable[dict], last_turns: int = 2) -> bool:
    """A paper tool ran in one of the last `last_turns` user turns (follow-ups like 'haan save karo')."""
    turns = 0
    for m in reversed(list(history)):
        if m.get("role") == "user":
            turns += 1
            if turns > last_turns:
                break
        for tc in m.get("tool_calls") or []:
            if tc.get("name") in PAPER_TOOLS:
                return True
    return False


def choose(message: str, history: list[dict]) -> tuple[str, str]:
    """-> (route, reason): route is "fast" or "main"."""
    if not enabled():
        return "main", "routing_off"
    text = message or ""
    if _PAPER_WORDS.search(text):
        return "main", "paper_intent"
    if _PAPER_INTENT.search(text) and not _COUNT_TESTS.search(text):
        return "main", "paper_intent"
    if paper_in_progress(history):
        return "main", "paper_in_progress"
    return "fast", "insight"


def escalate(tool_names: Iterable[str], read_tools: Iterable[str]) -> bool:
    """True if the fast model asked for anything that is not a read tool."""
    reads = set(read_tools)
    return any(n not in reads for n in tool_names)
