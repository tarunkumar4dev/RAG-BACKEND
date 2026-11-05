"""Embedded Postgres for local dev/tests with NO external install: `pgserver` (pip wheel that bundles the
Postgres binaries; Windows/macOS/Linux, Python 3.9-3.12). Used when AGENT_DB=embedded.

Why not SQLite: the seed and the insight tools use Postgres features (md5()::uuid, generate_series, jsonb,
SELECT ... FOR UPDATE, a read-only role). One SQL dialect keeps the golden numbers identical everywhere.
"""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse, urlunparse

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / ".devdata" / "pg"   # gitignored (agent/.gitignore)
_server = None


def _with_user(uri: str, user: str) -> str:
    u = urlparse(uri)
    host = u.netloc.split("@", 1)[-1]
    return urlunparse(u._replace(netloc=f"{user}@{host}"))


def start(data_dir: Path | str | None = None, database: str = "a4agent") -> tuple[str, str]:
    """Start (or reuse) the embedded server; return (rw_dsn, ro_dsn). Both are local (socket or localhost)."""
    global _server
    try:
        import pgserver  # type: ignore
    except ImportError as e:  # pragma: no cover - message for the user
        raise RuntimeError("AGENT_DB=embedded needs `pip install pgserver` (wheels exist for Python 3.9-3.12)") from e
    d = Path(data_dir or os.getenv("AGENT_PGDATA") or DEFAULT_DATA_DIR)
    d.mkdir(parents=True, exist_ok=True)
    if _server is None:
        _server = pgserver.get_server(str(d), cleanup_mode="stop")
    if database not in _server.psql(f"select datname from pg_database where datname = '{database}';"):
        _server.psql(f"create database {database};")
    rw = _server.get_uri(database)
    return rw, _with_user(rw, "agent_ro")


def ensure_env(data_dir: Path | str | None = None) -> None:
    """Point agent.db at the embedded server unless an explicit AGENT_DATABASE_URL is set."""
    if os.getenv("AGENT_DATABASE_URL"):
        return
    rw, ro = start(data_dir)
    os.environ["AGENT_DATABASE_URL"] = rw
    os.environ["AGENT_DATABASE_RO_URL"] = ro
