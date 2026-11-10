"""Rate limiting: per-user turn rate + per-institute daily token budget.

Tracking is in-process (a dict behind a lock). It resets on restart, which is fine for a trial: the limits are
generous enough that a restart doesn't help an abuser. Production: move counters into the agent store (Postgres).

Why in-process: no extra table, no migration, no DB round trip on the hot path. The downside is that two workers
each allow the full quota — acceptable for WEB_CONCURRENCY=2 (2× the nominal limit, still bounded).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from . import features
from .metrics import METRICS

IST = ZoneInfo("Asia/Kolkata")

_lock = threading.Lock()
_user_turns: dict[str, list[float]] = {}   # user_id -> list of timestamps (last 10 min)
_inst_tokens: dict[str, tuple[str, int]] = {}  # institute_id -> (date_ist, token_count)


@dataclass
class RateLimitHit:
    scope: str            # "user" or "institute"
    code: str             # "rate_limited"
    message: str
    retry_after_s: int


def _prune(ts_list: list[float], window: float, now: float) -> list[float]:
    cutoff = now - window
    return [t for t in ts_list if t > cutoff]


def check_user_rate(user_id: str) -> Optional[RateLimitHit]:
    """Check and record a turn for the per-user rate limit. Returns None if OK, or a RateLimitHit."""
    limit = features.user_turns_per_10min()
    if limit <= 0:
        return None
    now = time.time()
    window = 600.0  # 10 minutes
    with _lock:
        ts = _prune(_user_turns.get(user_id, []), window, now)
        if len(ts) >= limit:
            oldest = min(ts) if ts else now
            retry = int(oldest + window - now) + 1
            METRICS.inc("agent_rate_limit_hits_total", {"scope": "user"})
            return RateLimitHit(
                scope="user", code="rate_limited",
                message=f"Aap ne 10 minute mein {limit} se zyada sawaal poochhe. "
                        f"Thodi der ({retry}s) baad dobara try karein.",
                retry_after_s=max(retry, 1),
            )
        ts.append(now)
        _user_turns[user_id] = ts
    return None


def check_institute_budget(institute_id: str) -> Optional[RateLimitHit]:
    """Check the per-institute daily token budget. Returns None if OK or if no budget is set."""
    budget = features.institute_daily_tokens()
    if budget is None:
        return None
    today = datetime.now(IST).strftime("%Y-%m-%d")
    with _lock:
        entry = _inst_tokens.get(institute_id)
        if entry is None or entry[0] != today:
            return None  # no usage yet today, or new day
        if entry[1] >= budget:
            # Midnight IST reset
            midnight = datetime.now(IST).replace(hour=0, minute=0, second=0, microsecond=0)
            from datetime import timedelta
            next_midnight = midnight + timedelta(days=1)
            retry = int((next_midnight - datetime.now(IST)).total_seconds()) + 1
            METRICS.inc("agent_rate_limit_hits_total", {"scope": "institute"})
            return RateLimitHit(
                scope="institute", code="rate_limited",
                message=f"Aaj ka token budget khatam ho gaya hai. Kal subah ({next_midnight.strftime('%d %b')}) "
                        f"se phir use kar sakte hain.",
                retry_after_s=max(retry, 1),
            )
    return None


def record_tokens(institute_id: str, tokens: int) -> None:
    """Record tokens used by an institute (called after a turn completes)."""
    today = datetime.now(IST).strftime("%Y-%m-%d")
    with _lock:
        entry = _inst_tokens.get(institute_id)
        if entry is None or entry[0] != today:
            _inst_tokens[institute_id] = (today, tokens)
        else:
            _inst_tokens[institute_id] = (today, entry[1] + tokens)


def reset() -> None:
    """For tests."""
    with _lock:
        _user_turns.clear()
        _inst_tokens.clear()
