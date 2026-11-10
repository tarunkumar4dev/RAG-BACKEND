"""Voice limits (in-process, like agent/rate_limit.py): per-user requests per 10 minutes, per-user daily cap (IST),
and an Idempotency-Key cache so a retried upload (patchy 4G) is not transcribed — or billed — twice.

Why in-process: the trial runs 1–2 workers; a restart resets the counters, which only helps an abuser by one window.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

from . import config as vc

IST = ZoneInfo("Asia/Kolkata")
_lock = threading.Lock()
_recent: dict[str, list[float]] = {}            # user_id -> request timestamps (last 10 min)
_daily: dict[str, tuple[str, int]] = {}          # user_id -> (ist date, count)
_idem: dict[tuple[str, str], tuple[float, Any]] = {}  # (user_id, key) -> (expires_at, response)
_IDEM_MAX = 2000


@dataclass
class Hit:
    scope: str          # "user_10min" | "user_daily"
    message: str
    retry_after_s: int


def check_user(user_id: str, now: Optional[float] = None) -> Optional[Hit]:
    """Count one voice request; None = allowed."""
    now = time.time() if now is None else now
    per10, daily = vc.user_per_10min(), vc.user_daily()
    today = datetime.fromtimestamp(now, IST).strftime("%Y-%m-%d")
    with _lock:
        ts = [t for t in _recent.get(user_id, []) if t > now - 600]
        if per10 and len(ts) >= per10:
            retry = int(min(ts) + 600 - now) + 1
            return Hit("user_10min", f"Thodi der ruk kar boliye ({retry} second). Tab tak likh kar poochh sakte hain.",
                       max(retry, 1))
        d = _daily.get(user_id)
        used = d[1] if d and d[0] == today else 0
        if daily and used >= daily:
            midnight = datetime.fromtimestamp(now, IST).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            retry = int((midnight - datetime.fromtimestamp(now, IST)).total_seconds()) + 1
            return Hit("user_daily", f"Aaj ki voice limit ({daily}) poori ho gayi. Kal phir se, ya abhi likh kar poochhiye.",
                       max(retry, 1))
        ts.append(now)
        _recent[user_id] = ts
        _daily[user_id] = (today, used + 1)
    return None


def idem_get(user_id: str, key: Optional[str], now: Optional[float] = None) -> Optional[Any]:
    if not key:
        return None
    now = time.time() if now is None else now
    with _lock:
        hit = _idem.get((user_id, key))
        if hit and hit[0] > now:
            return hit[1]
        if hit:
            _idem.pop((user_id, key), None)
    return None


def idem_put(user_id: str, key: Optional[str], value: Any, now: Optional[float] = None) -> None:
    if not key:
        return
    now = time.time() if now is None else now
    with _lock:
        if len(_idem) >= _IDEM_MAX:
            expired = [k for k, v in _idem.items() if v[0] <= now]
            for k in expired:
                _idem.pop(k, None)
            if len(_idem) >= _IDEM_MAX:
                _idem.clear()
        _idem[(user_id, key)] = (now + vc.IDEMPOTENCY_TTL_S, value)


def reset() -> None:
    with _lock:
        _recent.clear()
        _daily.clear()
        _idem.clear()
