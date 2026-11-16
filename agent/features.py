"""Trial switches read from env at call time (so tests and a redeploy can change them without code changes).

AGENT_ENABLED_INSTITUTES   comma list of institute ids that may use the agent; unset/empty = every institute.
                           Callers from other institutes get 403 on every agent endpoint except /agent/meta,
                           which answers enabled=false so the UI can hide the agent.
AGENT_USER_TURNS_PER_10MIN chat turns one user may start in any 10-minute window (default 20).
AGENT_INSTITUTE_DAILY_TOKENS  LLM tokens (in + out) one institute may use per IST day; unset = no budget.
"""
from __future__ import annotations

import os
from typing import Optional


def enabled_institutes(env=os.environ) -> Optional[set[str]]:
    raw = (env.get("AGENT_ENABLED_INSTITUTES") or "").strip()
    if not raw:
        return None
    return {x.strip().lower() for x in raw.split(",") if x.strip()}


def institute_enabled(institute_id, env=os.environ) -> bool:
    allowed = enabled_institutes(env)
    return allowed is None or str(institute_id).lower() in allowed


def _int(name: str, default: Optional[int], env=os.environ) -> Optional[int]:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), 0)
    except ValueError:
        return default


def user_turns_per_10min(env=os.environ) -> int:
    return _int("AGENT_USER_TURNS_PER_10MIN", 20, env) or 0


def institute_daily_tokens(env=os.environ) -> Optional[int]:
    return _int("AGENT_INSTITUTE_DAILY_TOKENS", None, env)


NOT_ENABLED_MESSAGE = "Agent abhi aapke institute ke liye on nahi hai. Admin se baat karein."
