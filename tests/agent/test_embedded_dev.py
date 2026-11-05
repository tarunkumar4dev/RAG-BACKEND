"""Embedded-Postgres dev path (AGENT_DB=embedded, pgserver) + dev_server production guard."""
import os
from datetime import date

import pytest

from _agent_ids import SHARMA
from agent import dev_server
from agent.auth import _is_local_url

EMBEDDED = os.getenv("AGENT_DB") == "embedded"


@pytest.mark.skipif(not EMBEDDED, reason="run with AGENT_DB=embedded (pgserver, Python <= 3.12)")
def test_embedded_server_is_local_seeded_and_read_only():
    from agent.db import connect
    from agent.tools.insights import AttendanceStatsArgs, attendance_stats
    from agent.schemas import SessionCtx
    from _agent_ids import ADMIN, MAIN
    rw, ro = os.environ["AGENT_DATABASE_URL"], os.environ["AGENT_DATABASE_RO_URL"]
    assert _is_local_url(rw) and _is_local_url(ro) and "agent_ro@" in ro
    with connect(readonly=False) as c:
        assert c.execute("select count(*) n from students where institute_id = %s", (str(MAIN),)).fetchone()["n"] == 60
    ctx = SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN)
    r = attendance_stats(ctx, AttendanceStatsArgs(scope="teacher", entity_id=SHARMA, date_from=date(2026, 11, 1),
                                                  date_to=date(2026, 11, 5)))
    assert (r["pct"], r["marked"]) == (83.8, 160)  # golden #1, same as on the system Postgres
    with connect(readonly=True) as c, pytest.raises(Exception):
        c.execute("insert into institutes(id, name) values (gen_random_uuid(), 'x')")


@pytest.mark.parametrize("var,url", [
    ("AGENT_DATABASE_URL", "postgresql://postgres:x@db.abcd.supabase.co:5432/postgres"),
    ("SUPABASE_URL", "https://abcd.supabase.co"),
    ("DATABASE_URL", "postgresql://u:p@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"),
])
def test_dev_server_refuses_production_db_before_starting_anything(var, url, monkeypatch):
    started = []
    monkeypatch.setattr("agent.embedded_pg.start", lambda *a, **k: started.append(1))
    env = {var: url}
    with pytest.raises(RuntimeError) as e:
        dev_server.prepare_env(env)
    assert var in str(e.value) and url not in str(e.value) and not started
    assert env["AGENT_DEV_MODE"] == "1"


def test_dev_server_defaults():
    env = {}
    dev_server.prepare_env(env)
    assert env["AGENT_DB"] == "embedded"
    assert env["AGENT_CORS_ORIGINS"] == "http://localhost:5173,http://localhost:8080"


def test_windows_style_embedded_uri_counts_as_local():
    assert _is_local_url("postgresql://postgres:@localhost:5432/a4agent")
    assert _is_local_url("postgresql://postgres:@127.0.0.1:61234/a4agent")
    assert not _is_local_url("postgresql://postgres:@10.0.0.5:5432/a4agent")
