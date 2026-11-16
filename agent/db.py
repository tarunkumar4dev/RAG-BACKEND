"""Local demo DB access for the a4ai Agent (never points at production by default)."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

_DEFAULT_DSN = "postgresql://postgres@/a4agent?host=/var/tmp&port=54329"
_DEFAULT_RO_DSN = "postgresql://agent_ro@/a4agent?host=/var/tmp&port=54329"

DSN = os.getenv("AGENT_DATABASE_URL", _DEFAULT_DSN)
RO_DSN = os.getenv("AGENT_DATABASE_RO_URL", _DEFAULT_RO_DSN)

SEED_DIR = Path(__file__).resolve().parent / "seed"


_RO_KWARGS = dict(row_factory=dict_row, autocommit=True)


def _configure_ro(conn: psycopg.Connection) -> None:
    conn.execute("SET default_transaction_read_only = on")
    conn.execute("SET search_path = agent_demo")


def connect(readonly: bool):
    """Read-only = agent_ro role + read-only transactions (pooled, see agent/pool.py; use as `with connect(True) as c`).
    Read-write = autocommit, one connection (demo writes: sessions/drafts/approvals). DSNs are read at call time, so
    agent.embedded_pg / agent.dev_server can point them at a local server."""
    from . import pool
    if readonly and pool.enabled():
        return _pooled_ro()
    if readonly:
        conn = psycopg.connect(os.getenv("AGENT_DATABASE_RO_URL", RO_DSN), row_factory=dict_row, autocommit=True)
        conn.execute("SET default_transaction_read_only = on")
    else:
        conn = psycopg.connect(os.getenv("AGENT_DATABASE_URL", DSN), row_factory=dict_row, autocommit=True)
    conn.execute("SET search_path = agent_demo")
    return conn


@contextmanager
def _pooled_ro():
    from . import pool
    url = os.getenv("AGENT_DATABASE_RO_URL", RO_DSN)
    with pool.get_pool("seed_ro", url, _RO_KWARGS, _configure_ro).connection() as conn:
        with pool.read_only_tx(conn, "8s", "agent_demo") as c:
            yield c


def reset_and_seed() -> None:
    with connect(readonly=False) as conn:
        for name in ("schema.sql", "seed.sql"):
            conn.execute((SEED_DIR / name).read_text(encoding="utf-8"))
