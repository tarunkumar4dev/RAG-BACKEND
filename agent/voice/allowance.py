"""Voice usage counting and the plan allowance (Gate 2 #5, voice upgrade phase 3).

Counts live in the agent store (agent/store.py `voice_usage`: SQLite / Postgres), keyed by institute_id + user_id +
period, so they are shared by every server process and survive restarts. Two periods are kept per request: the IST
day (burst / fairness cap, VOICE_USER_DAILY) and the month (the plan's allowance).

Check BEFORE processing (voice_paid_gate), increment AFTER success (transcribe that reached STT, speak that
produced audio, the first streamed audio chunk of a voice chat). Two processes can each let the last unit
through — an overrun of one, by design; nothing is held or released.

The MONTHLY LIMIT comes from the plan:
  live data  -> plans.voice_monthly_limit through a4_voice_allowance(uid) (migration 32 in the product repo).
                NULL / unset / function missing => 0 => "voice is not in your plan" (403). -1 = unlimited.
  demo data  -> VOICE_DEMO_MONTHLY_LIMIT (default 100).
The limit is cached 30 s per user; the counts are read from the store on every check (one indexed query).
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
from ..store import Owner, get_store
from . import config as vc

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")
DEMO_MONTHLY_DEFAULT = 100
LIMIT_CACHE_S = 30.0
MSG_NOT_IN_PLAN = ("Voice aapke plan mein shamil nahi hai. Likh kar poochhiye. / "
                   "Voice is not included in your plan. Please type your question.")
MSG_MONTH_DONE = ("Is mahine ki voice limit poori ho gayi ({used}/{limit}). Likh kar poochhiye. / "
                  "This month's voice allowance is used up ({used}/{limit}). Please type your question.")
MSG_DAY_DONE = ("Aaj ki voice limit ({limit}) poori ho gayi. Kal phir se, ya abhi likh kar poochhiye. / "
                "Today's voice limit ({limit}) is used up. Try again tomorrow, or type your question.")

_lock = threading.Lock()
_limit_cache: dict[str, tuple[float, int]] = {}


@dataclass(frozen=True)
class Allowance:
    limit: int          # monthly: -1 = unlimited, 0 = not in plan
    used: int           # this month
    used_day: int
    day_limit: int      # 0 = no daily cap
    source: str         # "demo" | "live"

    @property
    def remaining(self) -> int:
        return -1 if self.limit < 0 else max(self.limit - self.used, 0)

    def status(self) -> Optional[tuple[str, str]]:
        """("plan" | "month" | "day", message) when voice is not allowed now, else None."""
        if self.limit == 0:
            return "plan", MSG_NOT_IN_PLAN
        if 0 < self.limit <= self.used:
            return "month", MSG_MONTH_DONE.format(used=self.used, limit=self.limit)
        if 0 < self.day_limit <= self.used_day:
            return "day", MSG_DAY_DONE.format(limit=self.day_limit)
        return None

    def blocked(self) -> Optional[str]:
        st = self.status()
        return st[1] if st else None


def month_period(now: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if now is None else now, IST).strftime("%Y-%m")


def day_period(now: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if now is None else now, IST).strftime("%Y-%m-%d")


def demo_limit(env=os.environ) -> int:
    try:
        return int(env.get("VOICE_DEMO_MONTHLY_LIMIT", str(DEMO_MONTHLY_DEFAULT)))
    except ValueError:
        return DEMO_MONTHLY_DEFAULT


# ---- live plan limit (replaceable in tests) -----------------------------------------------------------------------
def _live_fetch(uid: str) -> Optional[dict]:
    """{'limit': int} from a4_voice_allowance, or None when the function does not exist yet."""
    import psycopg
    from .. import db_real
    try:
        with db_real.connect_real() as conn:
            row = conn.execute("select public.a4_voice_allowance(%s::uuid) as a", (uid,)).fetchone()
        a = (row or {}).get("a") or {}
        return {"limit": int(a.get("limit", 0) or 0)}
    except psycopg.errors.UndefinedFunction:
        return None
    except Exception as e:  # noqa: BLE001  (DB down: fail closed, but say why in the log)
        logger.warning("voice allowance read failed: %s", type(e).__name__)
        return {"limit": 0}


live_fetch: Callable[[str], Optional[dict]] = _live_fetch


def _monthly_limit(ctx: SessionCtx) -> int:
    if ctx.data_source != "live":
        return demo_limit()
    uid = str(ctx.user_id)
    now = time.monotonic()
    with _lock:
        hit = _limit_cache.get(uid)
        if hit and now - hit[0] < LIMIT_CACHE_S:
            return hit[1]
    a = live_fetch(uid)
    limit = 0 if a is None else int(a.get("limit", 0))   # migration 32 not applied: nothing is included
    with _lock:
        _limit_cache[uid] = (now, limit)
    return limit


# ---- public API --------------------------------------------------------------------------------------------------
def get(ctx: SessionCtx, now: Optional[float] = None) -> Allowance:
    o = Owner.of(ctx)
    day, month = day_period(now), month_period(now)
    counts = get_store().voice_usage_get(o, [day, month])
    return Allowance(limit=_monthly_limit(ctx), used=counts.get(month, 0), used_day=counts.get(day, 0),
                     day_limit=vc.user_daily(), source=ctx.data_source)


def check(ctx: SessionCtx) -> Optional[str]:
    """Message when the user may not use voice now, else None. Reads the store; stores nothing."""
    return get(ctx).blocked()


def seconds_to_midnight_ist(now: Optional[float] = None) -> int:
    from datetime import timedelta
    t = datetime.fromtimestamp(time.time() if now is None else now, IST)
    midnight = t.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return max(int((midnight - t).total_seconds()) + 1, 1)


def consume(ctx: SessionCtx, now: Optional[float] = None) -> None:
    """One successful, billed voice request: +1 on today and on this month, in the shared store."""
    get_store().voice_usage_add(Owner.of(ctx), [day_period(now), month_period(now)], 1)


def reset() -> None:
    with _lock:
        _limit_cache.clear()
