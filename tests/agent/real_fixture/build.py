"""Build the schema-faithful real-data fixture DB (`a4real`) on the local test server and apply the REAL
agent/sql/create_agent_ro_role.sql to it. Never touches production: the server is the local seed DSN's server."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import psycopg

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOAD_ORDER = [
    "schema/00_supabase_stub.sql", "schema/01_baseline_from_audit.sql", "schema/02_tests_table_from_supabase_migration.sql",
    "schema/m_01_access_helpers.sql", "schema/m_02_rls_policies.sql", "schema/m_05_student_portal.sql",
    "schema/m_06a_students_phone.sql", "schema/m_10_smart_assignment_checks.sql", "schema/m_11_assignment_ai_check.sql",
    "schema/m_12_copy_checker.sql", "data.sql",
]
FIXTURE_DB = "a4real"
FIXTURE_PASSWORD = "fixture-only-password-not-a-secret-0123"


def _with_db(dsn: str, db: str, user: str | None = None) -> str:
    u = urlparse(dsn)
    netloc = u.netloc
    if user is not None:
        host = netloc.split("@", 1)[-1]
        netloc = f"{user}:{FIXTURE_PASSWORD}@{host}" if user == "agent_ro" else f"{user}@{host}"
    return urlunparse(u._replace(path=f"/{db}", netloc=netloc))


def build() -> tuple[str, str]:
    """-> (owner_dsn, agent_ro_dsn) of a freshly built fixture DB."""
    base = os.environ.get("AGENT_DATABASE_URL", "postgresql://postgres@/a4agent?host=/var/tmp&port=54329")
    admin = _with_db(base, "postgres")
    with psycopg.connect(admin, autocommit=True) as c:
        c.execute(f"drop database if exists {FIXTURE_DB} with (force)")
        c.execute(f"create database {FIXTURE_DB}")
    owner = _with_db(base, FIXTURE_DB)
    with psycopg.connect(owner, autocommit=True) as c:
        for rel in LOAD_ORDER:
            c.execute((HERE / rel).read_text(encoding="utf-8"))
        role_sql = (ROOT / "agent" / "sql" / "create_agent_ro_role.sql").read_text(encoding="utf-8")
        script = role_sql.split("-- ─── VERIFY")[0].replace("<SET-A-LONG-RANDOM-PASSWORD>", FIXTURE_PASSWORD)
        c.execute(script)
    return owner, _with_db(base, FIXTURE_DB, user="agent_ro")


def verify_queries() -> list[str]:
    role_sql = (ROOT / "agent" / "sql" / "create_agent_ro_role.sql").read_text(encoding="utf-8")
    block = role_sql.split("-- ─── VERIFY")[1].split("\n", 1)[1]
    out = []
    for q in block.split(";"):
        body = "\n".join(l for l in q.splitlines() if not l.strip().startswith("--")).strip()
        if body.lower().startswith("select"):
            out.append(body)
    return out
