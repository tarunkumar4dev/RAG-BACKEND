"""Golden values computed ONCE by hand with independent SQL in psql (see evals/cases.yaml notes):

G1 Sharma pct 2026-11-01..05 (marked 160):
  select count(*), round(100.0*sum(case when ar.status in ('present','late') then 1 else 0 end)/count(*),1)
  from attendance_records ar join class_sessions cs on cs.id=ar.session_id
  where cs.teacher_id='aaaaaaaa-...-0002' and cs.session_date between '2026-11-01' and '2026-11-05';   -> 160 | 83.8
G2 10A pct October 2026: same, cs.batch_id = 10A, '2026-10-01'..'2026-10-31'                         -> 540 | 80.7
G3 10A students < 75 % for 2026-09-07..2026-11-05 (per-student pct, count where p < 75)              -> 4
G4 test-2 (10A Half Yearly Mathematics, max 80): round(avg(marks_obtained)*100/80,1)                -> 60.8 (2 absent)
G5 Aarav Mehta marks: Unit Test 1 - Real Numbers 16/40, Half Yearly Mathematics 41/80
"""
from datetime import date

from agent.tools.insights import (AttendanceStatsArgs, LowAttendanceArgs, MarksStatsArgs, ResolveEntityArgs,
                                  attendance_stats, low_attendance_list, marks_stats, resolve_entity)
from _agent_ids import B10A, SHARMA


def test_g1_sharma_week(admin_ctx, sharma_ctx):
    for ctx in (admin_ctx, sharma_ctx):
        r = attendance_stats(ctx, AttendanceStatsArgs(scope="teacher", entity_id=SHARMA,
                                                      date_from=date(2026, 11, 1), date_to=date(2026, 11, 5)))
        assert r["status"] == "ok" and r["marked"] == 160 and r["pct"] == 83.8


def test_g2_10a_october(admin_ctx):
    r = attendance_stats(admin_ctx, AttendanceStatsArgs(scope="section", entity_id=B10A,
                                                        date_from=date(2026, 10, 1), date_to=date(2026, 10, 31)))
    assert r["marked"] == 540 and r["pct"] == 80.7 and r["entity"] == "10A"


def test_g3_10a_below_75(admin_ctx):
    r = low_attendance_list(admin_ctx, LowAttendanceArgs(threshold=75, section_id=B10A,
                                                         date_from=date(2026, 9, 7), date_to=date(2026, 11, 5)))
    assert len(r["rows"]) == 4
    pcts = [x["pct"] for x in r["rows"]]
    assert pcts == sorted(pcts) and all(p < 75 for p in pcts)


def test_g4_half_yearly_avg(admin_ctx):
    r = marks_stats(admin_ctx, MarksStatsArgs(scope="section", entity_id=B10A))
    t = next(x for x in r["tests"] if x["test"] == "Half Yearly Mathematics")
    assert t["avg_pct"] == 60.8 and t["absent"] == 2 and t["max_marks"] == 80


def test_g5_student_marks(admin_ctx):
    sid = resolve_entity(admin_ctx, ResolveEntityArgs(kind="student", name="Aarav Mehta"))["matches"][0]["id"]
    r = marks_stats(admin_ctx, MarksStatsArgs(scope="student", entity_id=sid))
    assert [(t["test"], t["marks"]) for t in r["tests"]] == [("Unit Test 1 - Real Numbers", 16.0), ("Half Yearly Mathematics", 41.0)]
