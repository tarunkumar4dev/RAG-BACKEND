from datetime import date

import psycopg
import pytest

from agent.db import connect
from agent.tools.insights import (AttendanceStatsArgs, LowAttendanceArgs, MarksStatsArgs, ResolveEntityArgs,
                                  attendance_stats, low_attendance_list, marks_stats, resolve_entity)
from _agent_ids import B10A, SHARMA

D0, D1 = date(2026, 8, 1), date(2026, 11, 5)


def _student_id(ctx, name):
    r = resolve_entity(ctx, ResolveEntityArgs(kind="student", name=name))
    assert r["status"] == "ok", r
    return r["matches"][0]["id"]


def test_decoy_cannot_see_main(decoy_ctx, admin_ctx):
    for kind, name in [("student", "Aarav Mehta"), ("section", "10B"), ("section", "9A"), ("teacher", "Priya Sharma"),
                       ("teacher", "Verma")]:
        assert resolve_entity(decoy_ctx, ResolveEntityArgs(kind=kind, name=name))["status"] == "not_found"
    aarav = _student_id(admin_ctx, "Aarav Mehta")
    assert attendance_stats(decoy_ctx, AttendanceStatsArgs(scope="student", entity_id=aarav, date_from=D0, date_to=D1)) == {"status": "not_found"}
    assert attendance_stats(decoy_ctx, AttendanceStatsArgs(scope="section", entity_id=B10A, date_from=D0, date_to=D1)) == {"status": "not_found"}
    assert attendance_stats(decoy_ctx, AttendanceStatsArgs(scope="teacher", entity_id=SHARMA, date_from=D0, date_to=D1)) == {"status": "not_found"}
    assert marks_stats(decoy_ctx, MarksStatsArgs(scope="section", entity_id=B10A)) == {"status": "not_found"}
    names = {r["name"] for r in low_attendance_list(decoy_ctx, LowAttendanceArgs(threshold=100, date_from=D0, date_to=D1))["rows"]}
    assert "Aarav Mehta" not in names and len(names) <= 5


def test_main_admin_cannot_resolve_decoy_teacher(admin_ctx):
    assert resolve_entity(admin_ctx, ResolveEntityArgs(kind="teacher", name="Anil Sharma"))["status"] == "not_found"
    # decoy "Rahul Kumar" never shows up for MAIN
    r = resolve_entity(admin_ctx, ResolveEntityArgs(kind="student", name="Rahul Kumar"))
    assert len(r["matches"]) == 2


def test_verma_scope(verma_ctx, admin_ctx):
    assert resolve_entity(verma_ctx, ResolveEntityArgs(kind="section", name="10A"))["status"] == "not_found"
    aarav_10a = _student_id(admin_ctx, "Aarav Mehta")
    assert attendance_stats(verma_ctx, AttendanceStatsArgs(scope="student", entity_id=aarav_10a, date_from=D0, date_to=D1)) == {"status": "not_found"}
    assert attendance_stats(verma_ctx, AttendanceStatsArgs(scope="section", entity_id=B10A, date_from=D0, date_to=D1)) == {"status": "not_found"}
    assert attendance_stats(verma_ctx, AttendanceStatsArgs(scope="teacher", entity_id=SHARMA, date_from=D0, date_to=D1)) == {"status": "forbidden"}
    assert marks_stats(verma_ctx, MarksStatsArgs(scope="student", entity_id=aarav_10a)) == {"status": "not_found"}
    rows = low_attendance_list(verma_ctx, LowAttendanceArgs(threshold=100, date_from=D0, date_to=D1))["rows"]
    assert rows and {r["section"] for r in rows} <= {"10B", "9A"}


def test_readonly_connection_rejects_writes():
    with connect(readonly=True) as conn:
        with pytest.raises(psycopg.Error):
            conn.execute("insert into agent_demo.institutes values (gen_random_uuid(), 'x')")
