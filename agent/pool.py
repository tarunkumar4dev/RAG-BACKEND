"""Connection pools (psycopg_pool) for the agent's READ paths: real Supabase (agent_ro) and the seed/dev demo DB.

Why: a fresh connection per tool call costs a TCP + TLS + auth round trip (≈75 ms against the local embedded server,
far more to a remote pooler). A small pool keeps a few warm connections per worker process.

Safety is unchanged and applied on EVERY checkout, not once per connection:
  BEGIN READ ONLY; SET LOCAL statement_timeout; SET LOCAL search_path ... ROLLBACK
plus `default_transaction_read_only = on` set once when the pool opens a connection. A connection that fails its
health check (ConnectionPool.check_connection on checkout) is replaced; one returned in a bad state is discarded.

Sizing: AGENT_DB_POOL_MIN (1) .. AGENT_DB_POOL_MAX (5) per process. agent_ro has `connection limit 10`
(agent/sql/create_agent_ro_role.sql), so WEB_CONCURRENCY x AGENT_DB_POOL_MAX must stay below 10 (deploy: 2 x 4).
AGENT_DB_POOL_TIMEOUT (5 s): waiting longer for a free connection raises PoolTimeout (an OperationalError, which the
agent reports as db_unavailable). AGENT_DB_POOL=0 switches pooling off (one connection per call, the v2 behaviour).
"""
from __future__ import annotations

import atexit
import logging
import os
import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import psycopg
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()
_POOLS: dict[tuple[str, str], ConnectionPool] = {}


def enabled() -> bool:
    return os.getenv("AGENT_DB_POOL", "1") != "0"


def _int(name: str, default: int) -> int:
    try:
        return max(int(os.getenv(name, str(default))), 0)
    except ValueError:
        return default


def sizes() -> tuple[int, int]:
    mx = max(_int("AGENT_DB_POOL_MAX", 5), 1)
    return min(_int("AGENT_DB_POOL_MIN", 1), mx), mx


def get_pool(name: str, conninfo: str, kwargs: dict, configure: Optional[Callable] = None) -> ConnectionPool:
    """One pool per (name, conninfo) per process; created lazily, opened without blocking startup."""
    key = (name, conninfo)
    with _LOCK:
        p = _POOLS.get(key)
        if p is None or p.closed:
            mn, mx = sizes()
            try:
                timeout = float(os.getenv("AGENT_DB_POOL_TIMEOUT", "5"))
            except ValueError:
                timeout = 5.0
            p = ConnectionPool(conninfo, kwargs=kwargs, min_size=mn, max_size=mx, configure=configure,
                               check=ConnectionPool.check_connection, timeout=timeout, max_idle=300,
                               max_lifetime=1800, name=name, open=False)
            p.open(wait=False)
            _POOLS[key] = p
        return p


@contextmanager
def read_only_tx(conn: psycopg.Connection, statement_timeout: str, search_path: str) -> Iterator[psycopg.Connection]:
    conn.execute("BEGIN READ ONLY")
    try:
        conn.execute(f"SET LOCAL statement_timeout = '{statement_timeout}'")
        conn.execute(f"SET LOCAL search_path = {search_path}")
        yield conn
    finally:
        try:
            conn.execute("ROLLBACK")
        except psycopg.Error:  # broken connection: the pool discards it on return
            pass


def stats() -> dict[str, dict]:
    """Pool counters for /agent/metrics (no conninfo: names only)."""
    out: dict[str, dict] = {}
    with _LOCK:
        pools = list(_POOLS.items())
    for (name, _), p in pools:
        if not p.closed:
            s = p.get_stats()
            agg = out.setdefault(name, {"size": 0, "available": 0, "waiting": 0, "timeouts": 0})
            agg["size"] += s.get("pool_size", 0)
            agg["available"] += s.get("pool_available", 0)
            agg["waiting"] += s.get("requests_waiting", 0)
            agg["timeouts"] += s.get("requests_errors", 0)
    return out


def close_all() -> None:
    with _LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for p in pools:
        try:
            p.close(timeout=2)
        except Exception:  # noqa: BLE001  shutdown must not fail
            pass


atexit.register(close_all)
