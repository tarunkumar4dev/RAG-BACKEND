"""Monthly per-plan voice allowance (Gate 2 #5).

Live data (ctx.data_source == "live"): plans.voice_monthly_limit through a4_voice_allowance(uid) — migration 32
in the product repo. NULL / unset / function missing => 0 => "voice is not in your plan" (403). -1 = unlimited.
Use is counted in usage_records (action 'voice_query') via record_usage(uid, 'voice_query') on a read-write
connection; if that write is refused (read-only role) the count falls back to this process and a warning is logged
once, so the monthly cap still holds per worker.
Demo data: VOICE_DEMO_MONTHLY_LIMIT (default 100), counted in this process.

Only successful, billed requests are counted (transcribe that reached STT, speak that produced audio). The
10-minute / daily in-process limits in agent/voice/limits.py stay as the burst guard.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional
from zoneinfo import ZoneInfo

from ..schemas import SessionCtx

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
ACTION = "voice_query"
DEMO_MONTHLY_DEFAULT = 100
CACHE_S = 30.0
MSG_NOT_IN_PLAN = ("Voice aapke plan mein shamil nahi hai. Likh kar poochhiye. / "
                   "Voice is not included in your plan. Please type your question.")
MSG_MONTH_DONE = ("Is mahine ki voice limit poori ho gayi ({used}/{limit}). Likh kar poochhiye. / "
                  "This month's voice allowance is used up ({used}/{limit}). Please type your question.")

_lock = threading.Lock()
_local_used: dict[str, tuple[str, int]] = {}            # user_id -> (YYYY-MM, count)  demo + live fallback
_cache: dict[str, tuple[float, "Allowance"]] = {}
_write_warned = False


@dataclass(frozen=True)
class Allowance:
    limit: int          # -1 = unlimited, 0 = not in plan
    used: int
    source: str         # "demo" | "live" | "live-local"

    @property
    def remaining(self) -> int:
        return -1 if self.limit < 0 else max(self.limit - self.used, 0)

    def blocked(self) -> Optional[str]:
        if self.limit == 0:
            return MSG_NOT_IN_PLAN
        if 0 < self.limit <= self.used:
            return MSG_MONTH_DONE.format(used=self.used, limit=self.limit)
        return None


def _month(now: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if now is None else now, IST).strftime("%Y-%m")


def _local(uid: str) -> int:
    d = _local_used.get(uid)
    return d[1] if d and d[0] == _month() else 0


def _bump_local(uid: str) -> int:
    n = _local(uid) + 1
    _local_used[uid] = (_month(), n)
    return n


def demo_limit(env=os.environ) -> int:
    try:
        return int(env.get("VOICE_DEMO_MONTHLY_LIMIT", str(DEMO_MONTHLY_DEFAULT)))
    except ValueError:
        return DEMO_MONTHLY_DEFAULT


# ---- live DB access (replaceable in tests) ----------------------------------------------------------------------
def _live_fetch(uid: str) -> Optional[dict]:
    """{'limit': int, 'used': int} from a4_voice_allowance, or None when the function does not exist yet."""
    import psycopg
    from .. import db_real
    try:
        with db_real.connect_real() as conn:
            row = conn.execute("select public.a4_voice_allowance(%s::uuid) as a", (uid,)).fetchone()
        a = (row or {}).get("a") or {}
        return {"limit": int(a.get("limit", 0) or 0), "used": int(a.get("used", 0) or 0)}
    except psycopg.errors.UndefinedFunction:
        return None
    except Exception as e:  # noqa: BLE001  (DB down: fail closed, but say why in the log)
        logger.warning("voice allowance read failed: %s", type(e).__name__)
        return {"limit": 0, "used": 0}


def _live_consume(uid: str) -> bool:
    """record_usage(uid, 'voice_query') on the read-write connection; False if the write was refused."""
    global _write_warned
    try:
        from .. import db
        with db.connect(readonly=False) as conn:
            conn.execute("select public.record_usage(%s::uuid, %s)", (uid, ACTION))
        return True
    except Exception as e:  # noqa: BLE001
        if not _write_warned:
            logger.warning("voice use could not be written to usage_records (%s); counting in-process until a "
                           "read-write role is configured", type(e).__name__)
            _write_warned = True
        return False


live_fetch: Callable[[str], Optional[dict]] = _live_fetch
live_consume: Callable[[str], bool] = _live_consume


# ---- public API --------------------------------------------------------------------------------------------------
def get(ctx: SessionCtx) -> Allowance:
    uid = str(ctx.user_id)
    if ctx.data_source != "live":
        with _lock:
            return Allowance(demo_limit(), _local(uid), "demo")
    now = time.monotonic()
    with _lock:
        hit = _cache.get(uid)
        if hit and now - hit[0] < CACHE_S:
            return hit[1]
    a = live_fetch(uid)
    with _lock:
        if a is None:
            out = Allowance(0, 0, "live")              # migration 32 not applied: nothing is included
        else:
            out = Allowance(a["limit"], max(a["used"], _local(uid)), "live")
        _cache[uid] = (now, out)
        return out


def check(ctx: SessionCtx) -> Optional[str]:
    """Message when the user may not use voice this month, else None."""
    return get(ctx).blocked()


def consume(ctx: SessionCtx) -> None:
    """One successful, billed voice request."""
    uid = str(ctx.user_id)
    if ctx.data_source == "live" and live_consume(uid):
        with _lock:
            _cache.pop(uid, None)
        return
    with _lock:
        _bump_local(uid)
        _cache.pop(uid, None)


def reset() -> None:
    global _write_warned
    with _lock:
        _local_used.clear()
        _cache.clear()
        _write_warned = False
