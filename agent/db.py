"""Local demo DB access for the a4ai Agent (never points at production by default)."""
from __future__ import annotations

import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

_DEFAULT_DSN = "postgresql://postgres@/a4agent?host=/var/tmp&port=54329"
_DEFAULT_RO_DSN = "postgresql://agent_ro@/a4agent?host=/var/tmp&port=54329"

DSN = os.getenv("AGENT_DATABASE_URL", _DEFAULT_DSN)
RO_DSN = os.getenv("AGENT_DATABASE_RO_URL", _DEFAULT_RO_DSN)

SEED_DIR = Path(__file__).resolve().parent / "seed"


def connect(readonly: bool) -> psycopg.Connection:
    """Read-only = agent_ro role + read-only transactions. Read-write = autocommit."""
    if readonly:
        conn = psycopg.connect(RO_DSN, row_factory=dict_row, autocommit=True)
        conn.execute("SET default_transaction_read_only = on")
    else:
        conn = psycopg.connect(DSN, row_factory=dict_row, autocommit=True)
    conn.execute("SET search_path = agent_demo")
    return conn


def reset_and_seed() -> None:
    with connect(readonly=False) as conn:
        for name in ("schema.sql", "seed.sql"):
            conn.execute((SEED_DIR / name).read_text(encoding="utf-8"))
