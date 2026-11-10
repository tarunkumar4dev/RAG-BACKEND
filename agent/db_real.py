"""Real a4ai data (Supabase Postgres) through a dedicated READ-ONLY role (agent/sql/create_agent_ro_role.sql).

- URL: AGENT_REAL_DB_URL, e.g. the Supabase *Session pooler* (IPv4) postgresql://agent_ro.<project-ref>:<pw>@
  aws-0-<region>.pooler.supabase.com:5432/postgres?sslmode=require. Never printed or logged.
- Every checkout runs inside BEGIN READ ONLY with SET LOCAL statement_timeout = 8s (works with transaction poolers
  too), and the transaction is always rolled back. Each connection also gets default_transaction_read_only = on.
- Connections come from a small psycopg_pool pool (agent/pool.py; health-checked on checkout). AGENT_DB_POOL=0 = one
  connection per call.
- startup_check(): refuses to start if the role is superuser / can create roles or DBs / bypasses RLS, or holds
  INSERT/UPDATE/DELETE/TRUNCATE on any table, or can read auth.* / storage.*.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row

STATEMENT_TIMEOUT = "8s"


def _url() -> str:
    url = os.environ.get("AGENT_REAL_DB_URL")
    if not url:
        raise RuntimeError("AGENT_REAL_DB_URL is not set")
    return url


_KWARGS = dict(row_factory=dict_row, autocommit=True, prepare_threshold=None, connect_timeout=10,
               application_name="a4ai-agent-ro")


def _configure(conn: psycopg.Connection) -> None:
    conn.execute("SET default_transaction_read_only = on")


@contextmanager
def connect_real() -> Iterator[psycopg.Connection]:
    """Read-only transaction on the real DB. Callers only SELECT; nothing is ever committed."""
    from . import pool
    if pool.enabled():
        with pool.get_pool("real_ro", _url(), _KWARGS, _configure).connection() as conn:
            with pool.read_only_tx(conn, STATEMENT_TIMEOUT, "public") as c:
                yield c
        return
    conn = psycopg.connect(_url(), **_KWARGS)
    try:
        conn.execute("SET default_transaction_read_only = on")
        conn.execute("BEGIN READ ONLY")
        conn.execute(f"SET LOCAL statement_timeout = '{STATEMENT_TIMEOUT}'")
        conn.execute("SET LOCAL search_path = public")
        yield conn
    finally:
        try:
            conn.execute("ROLLBACK")
        finally:
            conn.close()


class UnsafeRoleError(RuntimeError):
    pass


PRIVILEGE_PROBE = """
select n.nspname || '.' || c.relname as rel, p.priv
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
cross join (values ('INSERT'), ('UPDATE'), ('DELETE'), ('TRUNCATE')) p(priv)
where c.relkind in ('r', 'p', 'v', 'm', 'f')
    and n.nspname not in ('pg_catalog', 'information_schema', 'pg_toast', 'net', 'cron', 'supabase_functions', 'extensions', 'graphql', 'graphql_public', 'realtime', 'pgsodium')
  and has_table_privilege(c.oid, p.priv)
order by 1, 2
limit 10
"""

SENSITIVE_READ_PROBE = """
select n.nspname || '.' || c.relname as rel
from pg_class c join pg_namespace n on n.oid = c.relnamespace
where c.relkind in ('r', 'p', 'v', 'm')
    and n.nspname in ('auth', 'storage', 'vault')
  and has_table_privilege(c.oid, 'SELECT')
order by 1 limit 10
"""


def startup_check() -> dict:
    """Refuse to start unless the connected role is strictly read-only. Returns a safe summary (no URL)."""
    with connect_real() as c:
        r = c.execute("select current_user as u, rolsuper, rolcreaterole, rolcreatedb, rolbypassrls "
                      "from pg_roles where rolname = current_user").fetchone()
        if r is None or r["rolsuper"] or r["rolcreaterole"] or r["rolcreatedb"] or r["rolbypassrls"]:
            raise UnsafeRoleError("AGENT_REAL_DB_URL role is superuser / createrole / createdb / bypassrls — "
                                  "use the agent_ro role (agent/sql/create_agent_ro_role.sql)")
        writable = c.execute(PRIVILEGE_PROBE).fetchall()
        if writable:
            sample = ", ".join(f"{w['priv']} {w['rel']}" for w in writable[:5])
            raise UnsafeRoleError(f"agent DB role can write ({sample}) — refusing to start")
        sensitive = c.execute(SENSITIVE_READ_PROBE).fetchall()
        if sensitive:
            raise UnsafeRoleError("agent DB role can read " + ", ".join(s["rel"] for s in sensitive[:5]) +
                                  " — refusing to start")
        tx_ro = c.execute("show transaction_read_only").fetchone()["transaction_read_only"]
        if tx_ro != "on":
            raise UnsafeRoleError("read-only transaction not in effect — refusing to start")
        return {"role": r["u"], "read_only": True}
