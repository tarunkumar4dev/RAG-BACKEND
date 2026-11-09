"""Read-only insight tools for the a4ai Agent.

Identity/scope comes ONLY from SessionCtx (server side). Tool args never carry SQL, filters or
institute ids. Every query filters institute_id = ctx.institute_id; teachers are limited to the
batches in their teaching_assignments and to their own class_sessions / tests.
Out-of-scope or other-institute entities -> {'status': 'not_found'} (existence is never leaked).
"""
from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent.db import connect
from agent.schemas import SessionCtx
from agent.text import normalize_section, strip_honorifics

MAX_RANGE_DAYS = 400
MAX_MATCHES = 5


# ---- argument models ------------------------------------------------------
class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _DateRange(_Args):
    date_from: date
    date_to: date

    @model_validator(mode="after")
    def _check_range(self):
        if self.date_from > self.date_to:
            raise ValueError("date_from must be on or before date_to")
        if (self.date_to - self.date_from).days > MAX_RANGE_DAYS:
            raise ValueError(f"date range longer than {MAX_RANGE_DAYS} days")
        return self


class ResolveEntityArgs(_Args):
    kind: Literal["teacher", "student", "section"]
    name: str = Field(min_length=1, max_length=80, pattern=r"^[\w\s.'\-]+$")


class AttendanceStatsArgs(_DateRange):
    scope: Literal["teacher", "section", "student"]
    entity_id: UUID


class LowAttendanceArgs(_DateRange):
    threshold: float = Field(ge=0, le=100)
    section_id: Optional[UUID] = None


class MarksStatsArgs(_Args):
    scope: Literal["section", "student"]
    entity_id: UUID
    subject_id: Optional[UUID] = None


class AttendanceTrendArgs(_DateRange):
    scope: Literal["teacher", "section", "student"]
    entity_id: UUID
    granularity: Literal["day", "week", "month"] = "week"


class TestsCreatedArgs(_DateRange):
    teacher_id: Optional[UUID] = None  # None = me (teacher) / whole scope (admin, HOD)


class MyScopeArgs(_Args):
    """No arguments: the caller's own scope comes from the server-side session."""


class SectionComparisonArgs(_DateRange):
    metric: Literal["attendance", "marks"] = "attendance"


ARG_MODELS = {
    "resolve_entity": ResolveEntityArgs,
    "attendance_stats": AttendanceStatsArgs,
    "low_attendance_list": LowAttendanceArgs,
    "marks_stats": MarksStatsArgs,
    "attendance_trend": AttendanceTrendArgs,
    "tests_created_by_teacher": TestsCreatedArgs,
    "my_scope": MyScopeArgs,
    "section_comparison": SectionComparisonArgs,
}

EMPTY_SCOPE_MESSAGE = ("No batch or subject is assigned to you yet. The institute admin or your HOD can assign "
                       "batches (Teaching assignments / timetable).")

NOT_FOUND = {"status": "not_found"}
FORBIDDEN = {"status": "forbidden"}


# ---- helpers --------------------------------------------------------------
def _params(ctx: SessionCtx, **extra: Any) -> dict[str, Any]:
    is_admin = ctx.role == "admin"
    tid = None if is_admin else (ctx.teacher_id or ctx.user_id)
    return {"inst": ctx.institute_id, "is_admin": is_admin, "tid": tid, **extra}


# batches visible to the caller
_SCOPE_BATCHES = """
scope_batches as (
  select b.id, b.name
  from agent_demo.batches b
  where b.institute_id = %(inst)s
    and (%(is_admin)s or exists (
      select 1 from agent_demo.teaching_assignments ta
      where ta.batch_id = b.id and ta.institute_id = %(inst)s
        and ta.teacher_id = %(tid)s::uuid and ta.is_active))
)"""

# sessions visible to the caller (teachers: only their own)
_SESSION_FILTER = "cs.institute_id = %(inst)s and cs.status = 'held' and (%(is_admin)s or cs.teacher_id = %(tid)s::uuid)"


def _num(v: Any, nd: int = 2) -> Optional[float]:
    if v is None:
        return None
    return round(float(v), nd)


def _pct(good: int, total: int) -> Optional[float]:
    if not total:
        return None
    return round(good * 100.0 / total, 1)


def _like_tokens(name: str) -> list[str]:
    toks = [t for t in re.split(r"\s+", name.strip()) if t]
    return ["%" + t.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%" for t in toks]


def _result(matches: list[dict[str, str]], total: int) -> dict[str, Any]:
    if total == 0:
        return {"status": "not_found", "matches": []}
    if total > 1:
        return {"status": "ambiguous", "matches": matches[:MAX_MATCHES]}
    return {"status": "ok", "matches": matches[:1]}


# ---- tools ----------------------------------------------------------------
def resolve_entity(ctx: SessionCtx, args: ResolveEntityArgs) -> dict[str, Any]:
    with connect(readonly=True) as conn:
        if args.kind == "section":
            norm = normalize_section(args.name)
            rows = conn.execute(
                f"with {_SCOPE_BATCHES} select sb.id, sb.name, (select count(*) from agent_demo.students s "
                "where s.batch_id = sb.id and s.institute_id = %(inst)s and s.is_active) as n from scope_batches sb "
                "where lower(name) = lower(%(n)s) or (%(n)s ~ '^[0-9]+$' and name ~ ('^' || %(n)s || '[A-Za-z]$')) "
                "order by name, id",
                _params(ctx, n=norm),
            ).fetchall()
            return _hinted(_result([{"id": str(r["id"]), "label": r["name"]} for r in rows], len(rows)),
                           {str(r["id"]): student_hint("Batch", r["n"]) for r in rows})

        cleaned = strip_honorifics(args.name)
        toks = _like_tokens(cleaned)
        if not toks:
            return {"status": "not_found", "matches": []}

        if args.kind == "teacher":
            # teachers of the caller's own institute only; returns the auth user id
            rows = conn.execute(
                "select user_id as id, user_name as label from agent_demo.institute_members "
                "where institute_id = %(inst)s and status = 'active' and role = 'teacher' "
                "and user_name ilike all(%(toks)s::text[]) order by user_name, user_id limit 6",
                _params(ctx, toks=toks),
            ).fetchall()
            return _hinted(_result([{"id": str(r["id"]), "label": r["label"]} for r in rows], len(rows)),
                           {str(r["id"]): "Teacher" for r in rows})

        rows = conn.execute(
            f"with {_SCOPE_BATCHES} "
            "select s.id, s.name || ' (' || sb.name || ')' as label, sb.name as batch, s.roll_no "
            "from agent_demo.students s "
            "join scope_batches sb on sb.id = s.batch_id "
            "where s.institute_id = %(inst)s and s.is_active and s.name ilike all(%(toks)s::text[]) "
            "order by s.name, sb.name, s.id limit 6",
            _params(ctx, toks=toks),
        ).fetchall()
        return _hinted(_result([{"id": str(r["id"]), "label": r["label"]} for r in rows], len(rows)),
                       {str(r["id"]): f"{r['batch']} · roll {r['roll_no']}" for r in rows})


def student_hint(kind: str, n: int, dept: Optional[str] = None) -> str:
    """'Electronics · 22 students' / 'Batch · 1 student' / 'Batch · 0 students (empty)'."""
    n = int(n or 0)
    count = f"{n} student" + ("" if n == 1 else "s") + (" (empty)" if n == 0 else "")
    return f"{dept or kind} · {count}"


def _hinted(res: dict[str, Any], hints: dict[str, str]) -> dict[str, Any]:
    """Hints only on ambiguous results (chips need them; single matches keep the 1.1 shape)."""
    if res.get("status") == "ambiguous":
        res["matches"] = [{**m, "hint": hints.get(m["id"])} for m in res["matches"]]
    return res


def rank_rows(rows: list[dict]) -> dict[str, Any]:
    """Rank by pct desc (ties share a rank: 1, 2, 3, 3, 5); rows without data go last with rank None.
    -> {rows, highest: [names], lowest: [names], no_data: [names]}"""
    have = sorted((r for r in rows if r["pct"] is not None), key=lambda r: (-r["pct"], r["name"], r["id"]))
    none = sorted((r for r in rows if r["pct"] is None), key=lambda r: (r["name"], r["id"]))
    prev, rank = None, 0
    for i, r in enumerate(have, 1):
        if r["pct"] != prev:
            rank, prev = i, r["pct"]
        r["rank"] = rank
    for r in none:
        r["rank"] = None
    hi = [r["name"] for r in have if r["pct"] == have[0]["pct"]] if have else []
    lo = [r["name"] for r in have if r["pct"] == have[-1]["pct"]] if have else []
    return {"rows": have + none, "highest": hi, "lowest": lo, "no_data": [r["name"] for r in none]}


def attendance_stats(ctx: SessionCtx, args: AttendanceStatsArgs) -> dict[str, Any]:
    p = _params(ctx, eid=args.entity_id, df=args.date_from, dt=args.date_to)
    with connect(readonly=True) as conn:
        if args.scope == "teacher":
            row = conn.execute(
                "select user_id, user_name from agent_demo.institute_members "
                "where institute_id = %(inst)s and user_id = %(eid)s and status = 'active' and role = 'teacher'",
                p,
            ).fetchone()
            if not row:
                return NOT_FOUND
            if not p["is_admin"] and row["user_id"] != p["tid"]:
                return FORBIDDEN
            label = row["user_name"]
            sess_where = "cs.teacher_id = %(eid)s"
            rec_where = sess_where
        elif args.scope == "section":
            row = conn.execute(f"with {_SCOPE_BATCHES} select id, name from scope_batches where id = %(eid)s", p).fetchone()
            if not row:
                return NOT_FOUND
            label = row["name"]
            sess_where = "cs.batch_id = %(eid)s"
            rec_where = sess_where
        else:
            row = conn.execute(
                f"with {_SCOPE_BATCHES} select s.id, s.batch_id, s.name || ' (' || sb.name || ')' as label "
                "from agent_demo.students s join scope_batches sb on sb.id = s.batch_id "
                "where s.institute_id = %(inst)s and s.id = %(eid)s",
                p,
            ).fetchone()
            if not row:
                return NOT_FOUND
            label = row["label"]
            p["bid"] = row["batch_id"]
            sess_where = "cs.batch_id = %(bid)s"
            rec_where = "ar.student_id = %(eid)s"

        sessions = conn.execute(
            f"select count(*) as n from agent_demo.class_sessions cs where {_SESSION_FILTER} and {sess_where} "
            "and cs.session_date between %(df)s and %(dt)s",
            p,
        ).fetchone()["n"]
        agg = conn.execute(
            "select count(*) as marked, "
            "count(*) filter (where ar.status = 'present') as present, "
            "count(*) filter (where ar.status = 'late') as late, "
            "count(*) filter (where ar.status = 'absent') as absent, "
            "count(*) filter (where ar.status = 'leave') as leave "
            "from agent_demo.attendance_records ar join agent_demo.class_sessions cs on cs.id = ar.session_id "
            f"where ar.institute_id = %(inst)s and {_SESSION_FILTER} and {sess_where} and {rec_where} "
            "and cs.session_date between %(df)s and %(dt)s",
            p,
        ).fetchone()
    return {
        "status": "ok",
        "entity": label,
        "date_from": args.date_from.isoformat(),
        "date_to": args.date_to.isoformat(),
        "sessions": int(sessions),
        "marked": int(agg["marked"]),
        "present": int(agg["present"]),
        "late": int(agg["late"]),
        "absent": int(agg["absent"]),
        "leave": int(agg["leave"]),
        "pct": _pct(agg["present"] + agg["late"], agg["marked"]),
    }


def low_attendance_list(ctx: SessionCtx, args: LowAttendanceArgs) -> dict[str, Any]:
    p = _params(ctx, df=args.date_from, dt=args.date_to, sid=args.section_id, thr=args.threshold)
    with connect(readonly=True) as conn:
        if args.section_id is not None:
            ok = conn.execute(f"with {_SCOPE_BATCHES} select 1 from scope_batches where id = %(sid)s", p).fetchone()
            if not ok:
                return NOT_FOUND
        rows = conn.execute(
            f"with {_SCOPE_BATCHES}, per as ("
            " select s.id, s.name, sb.name as section, count(*) as marked,"
            "        count(*) filter (where ar.status in ('present','late')) as good"
            " from agent_demo.attendance_records ar"
            " join agent_demo.class_sessions cs on cs.id = ar.session_id"
            " join agent_demo.students s on s.id = ar.student_id"
            " join scope_batches sb on sb.id = s.batch_id"
            f" where ar.institute_id = %(inst)s and s.institute_id = %(inst)s and s.is_active and {_SESSION_FILTER}"
            "   and cs.batch_id = s.batch_id and cs.session_date between %(df)s and %(dt)s"
            "   and (%(sid)s::uuid is null or s.batch_id = %(sid)s::uuid)"
            " group by s.id, s.name, sb.name)"
            " select id, name, section, marked, round(good * 100.0 / marked, 1) as pct from per"
            " where good * 100.0 / marked < %(thr)s order by pct, name, id",
            p,
        ).fetchall()
    return {
        "status": "ok",
        "threshold": args.threshold,
        "rows": [
            {"student_id": str(r["id"]), "name": r["name"], "section": r["section"],
             "pct": _num(r["pct"], 1), "marked": int(r["marked"])}
            for r in rows
        ],
    }


def marks_stats(ctx: SessionCtx, args: MarksStatsArgs) -> dict[str, Any]:
    p = _params(ctx, eid=args.entity_id, subj=args.subject_id)
    test_filter = (
        "t.institute_id = %(inst)s and (%(is_admin)s or t.teacher_id = %(tid)s::uuid) "
        "and (%(subj)s::uuid is null or t.subject_id = %(subj)s::uuid)"
    )
    with connect(readonly=True) as conn:
        if args.scope == "section":
            row = conn.execute(f"with {_SCOPE_BATCHES} select id, name from scope_batches where id = %(eid)s", p).fetchone()
            if not row:
                return NOT_FOUND
            rows = conn.execute(
                "select t.name, t.test_date, t.max_marks, avg(m.marks_obtained) as avg,"
                " max(m.marks_obtained) as highest, min(m.marks_obtained) as lowest,"
                " count(*) filter (where m.marks_obtained is null) as absent"
                " from agent_demo.tests t join agent_demo.test_marks m on m.test_id = t.id"
                f" where t.batch_id = %(eid)s and {test_filter}"
                " group by t.id, t.name, t.test_date, t.max_marks order by t.test_date, t.name",
                p,
            ).fetchall()
            overall = conn.execute(
                "select avg(m.marks_obtained * 100.0 / t.max_marks) as v"
                " from agent_demo.tests t join agent_demo.test_marks m on m.test_id = t.id"
                f" where t.batch_id = %(eid)s and {test_filter}",
                p,
            ).fetchone()["v"]
            tests = [
                {"test": r["name"], "date": r["test_date"].isoformat(), "max_marks": _num(r["max_marks"]),
                 "avg": _num(r["avg"]),
                 "avg_pct": _num(r["avg"] * 100 / r["max_marks"], 1) if r["avg"] is not None else None,
                 "highest": _num(r["highest"]), "lowest": _num(r["lowest"]), "absent": int(r["absent"])}
                for r in rows
            ]
            return {"status": "ok", "entity": row["name"], "tests": tests, "overall_avg_pct": _num(overall, 1)}

        row = conn.execute(
            f"with {_SCOPE_BATCHES} select s.id, s.name || ' (' || sb.name || ')' as label "
            "from agent_demo.students s join scope_batches sb on sb.id = s.batch_id "
            "where s.institute_id = %(inst)s and s.id = %(eid)s",
            p,
        ).fetchone()
        if not row:
            return NOT_FOUND
        rows = conn.execute(
            "select t.name, t.test_date, t.max_marks, m.marks_obtained"
            " from agent_demo.tests t join agent_demo.test_marks m on m.test_id = t.id"
            f" where m.student_id = %(eid)s and {test_filter} order by t.test_date, t.name",
            p,
        ).fetchall()
    tests = []
    pcts = []
    for r in rows:
        mk = r["marks_obtained"]
        pct = _num(mk * 100 / r["max_marks"], 1) if mk is not None else None
        if mk is not None:
            pcts.append(Decimal(mk) * 100 / r["max_marks"])
        tests.append({"test": r["name"], "date": r["test_date"].isoformat(), "max_marks": _num(r["max_marks"]),
                      "marks": _num(mk), "pct": pct, "absent": mk is None})
    overall = _num(sum(pcts) / len(pcts), 1) if pcts else None
    return {"status": "ok", "entity": row["label"], "tests": tests, "overall_avg_pct": overall}


# ---- seed versions of the two newer tools (same output schema as the real adapter) -------------
def attendance_trend(ctx: SessionCtx, args: AttendanceTrendArgs) -> dict[str, Any]:
    base = attendance_stats(ctx, AttendanceStatsArgs(scope=args.scope, entity_id=args.entity_id,
                                                     date_from=args.date_from, date_to=args.date_to))
    if base.get("status") != "ok":
        return base
    p = _params(ctx, eid=args.entity_id, df=args.date_from, dt=args.date_to, g=args.granularity)
    if args.scope == "teacher":
        where = "cs.teacher_id = %(eid)s"
    elif args.scope == "section":
        where = "cs.batch_id = %(eid)s"
    else:
        where = "ar.student_id = %(eid)s"
    with connect(readonly=True) as conn:
        rows = conn.execute(
            "select date_trunc(%(g)s, cs.session_date)::date as period, count(*) as marked, "
            "count(*) filter (where ar.status in ('present', 'late')) as good "
            "from agent_demo.attendance_records ar join agent_demo.class_sessions cs on cs.id = ar.session_id "
            f"where ar.institute_id = %(inst)s and {_SESSION_FILTER} and {where} "
            "and cs.session_date between %(df)s and %(dt)s group by 1 order by 1 limit 120", p).fetchall()
    return {"status": "ok", "entity": base["entity"], "granularity": args.granularity,
            "date_from": args.date_from.isoformat(), "date_to": args.date_to.isoformat(),
            "points": [{"period": r["period"].isoformat(), "pct": _pct(r["good"], r["marked"]),
                        "marked": int(r["marked"])} for r in rows]}


def tests_created_by_teacher(ctx: SessionCtx, args: TestsCreatedArgs) -> dict[str, Any]:
    p = _params(ctx, df=args.date_from, dt=args.date_to)
    is_admin = ctx.role == "admin"
    target = args.teacher_id or (None if is_admin else (ctx.teacher_id or ctx.user_id))
    with connect(readonly=True) as conn:
        label = "Institute"
        if target is not None:
            row = conn.execute("select user_id, user_name from agent_demo.institute_members where institute_id = %(inst)s "
                               "and user_id = %(t)s and status = 'active'", {**p, "t": target}).fetchone()
            if not row:
                return NOT_FOUND
            if not is_admin and row["user_id"] != p["tid"]:
                return FORBIDDEN
            label = row["user_name"]
        rows = conn.execute(
            "select t.name, t.test_date, t.max_marks, m.user_name from agent_demo.tests t "
            "join agent_demo.institute_members m on m.user_id = t.teacher_id and m.institute_id = t.institute_id "
            "where t.institute_id = %(inst)s and t.test_date between %(df)s and %(dt)s "
            "and (%(t)s::uuid is null or t.teacher_id = %(t)s::uuid) order by t.test_date desc, t.name",
            {**p, "t": target}).fetchall()
    by: dict[str, int] = {}
    for r in rows:
        by[r["user_name"]] = by.get(r["user_name"], 0) + 1
    return {"status": "ok", "teacher": label, "date_from": args.date_from.isoformat(), "date_to": args.date_to.isoformat(),
            "count": len(rows), "by_teacher": [{"teacher": k, "count": v} for k, v in sorted(by.items())],
            "recent": [{"title": r["name"], "class_grade": None, "subject": None, "total_marks": _num(r["max_marks"]),
                        "status": None, "created_on": r["test_date"].isoformat()} for r in rows[:10]]}


# ---- scope + comparison (seed; same output schema as agent/real/schema_adapter.py) ----------------------------
def _scope_rows(conn, ctx: SessionCtx) -> list[dict]:
    rows = conn.execute(
        f"with {_SCOPE_BATCHES} select sb.id, sb.name, "
        "(select count(*) from agent_demo.students s where s.batch_id = sb.id and s.institute_id = %(inst)s "
        " and s.is_active) as n, "
        "array(select distinct sj.name from agent_demo.teaching_assignments ta join agent_demo.subjects sj "
        " on sj.id = ta.subject_id where ta.batch_id = sb.id and ta.institute_id = %(inst)s and ta.is_active "
        " and (%(is_admin)s or ta.teacher_id = %(tid)s::uuid) order by 1) as subjects, "
        "array(select distinct ta.subject_id::text from agent_demo.teaching_assignments ta where ta.batch_id = sb.id "
        " and ta.institute_id = %(inst)s and ta.is_active and (%(is_admin)s or ta.teacher_id = %(tid)s::uuid)) "
        " as subject_ids "
        "from scope_batches sb", _params(ctx)).fetchall()
    return sorted(rows, key=lambda r: (r["name"].lower(), str(r["id"])))  # same order in seed and real mode


def my_scope(ctx: SessionCtx, args: MyScopeArgs) -> dict[str, Any]:
    role = "admin" if ctx.role == "admin" else "teacher"
    with connect(readonly=True) as conn:
        rows = _scope_rows(conn, ctx)
        subj = {r["id"]: r["name"] for r in conn.execute(
            "select id::text as id, name from agent_demo.subjects where institute_id = %(inst)s",
            _params(ctx)).fetchall()}
    batches = [{"id": str(r["id"]), "name": r["name"], "student_count": int(r["n"]), "role": role, "section": None,
                "department": None, "subjects": list(r["subjects"])} for r in rows]
    count: dict[str, int] = {}
    for r in rows:
        for sid in r["subject_ids"]:
            count[sid] = count.get(sid, 0) + 1
    out = {"status": "ok", "caller_role": ctx.role, "batches": batches,
           "subjects": sorted(({"id": k, "name": subj.get(k, "—"), "batch_count": v} for k, v in count.items()),
                              key=lambda x: (x["name"], x["id"])),
           "students_in_scope": sum(b["student_count"] for b in batches), "empty": not batches}
    if not batches:
        out["message"] = EMPTY_SCOPE_MESSAGE
    return out


def section_comparison(ctx: SessionCtx, args: SectionComparisonArgs) -> dict[str, Any]:
    p = _params(ctx, df=args.date_from, dt=args.date_to)
    with connect(readonly=True) as conn:
        scope = _scope_rows(conn, ctx)
        if args.metric == "attendance":
            agg = conn.execute(
                "select cs.batch_id, count(distinct cs.id) as sessions, count(ar.id) as marked, "
                "count(ar.id) filter (where ar.status in ('present','late')) as good "
                "from agent_demo.class_sessions cs left join agent_demo.attendance_records ar "
                "  on ar.session_id = cs.id and ar.institute_id = %(inst)s "
                f"where {_SESSION_FILTER} and cs.session_date between %(df)s and %(dt)s group by cs.batch_id",
                p).fetchall()
            by = {r["batch_id"]: (int(r["sessions"]), int(r["marked"]), _pct(r["good"], r["marked"])) for r in agg}
        else:
            agg = conn.execute(
                "select t.batch_id, count(distinct t.id) as tests, count(m.marks_obtained) as graded, "
                "avg(m.marks_obtained * 100.0 / t.max_marks) as pct "
                "from agent_demo.tests t join agent_demo.test_marks m on m.test_id = t.id "
                "where t.institute_id = %(inst)s and (%(is_admin)s or t.teacher_id = %(tid)s::uuid) "
                "and t.test_date between %(df)s and %(dt)s group by t.batch_id", p).fetchall()
            by = {r["batch_id"]: (int(r["tests"]), int(r["graded"]), _num(r["pct"], 1)) for r in agg}
    none = (0, 0, None)
    rows = [{"id": str(b["id"]), "name": b["name"], "student_count": int(b["n"]),
             "sessions": by.get(b["id"], none)[0], "marked": by.get(b["id"], none)[1],
             "pct": by.get(b["id"], none)[2]} for b in scope]
    out = {"status": "ok", "metric": args.metric, "date_from": args.date_from.isoformat(),
           "date_to": args.date_to.isoformat(), **rank_rows(rows)}
    if not scope:
        out["message"] = EMPTY_SCOPE_MESSAGE
    return out


def scope_entities(ctx: SessionCtx) -> list[dict[str, Any]]:
    """For /agent/suggestions: batch / teacher / subject names in the caller's scope (seed)."""
    sc = my_scope(ctx, MyScopeArgs())
    out = [{"kind": "batch", "id": b["id"], "name": b["name"], "student_count": b["student_count"]}
           for b in sc["batches"]]
    out += [{"kind": "subject", "id": x["id"], "name": x["name"]} for x in sc["subjects"]]
    with connect(readonly=True) as conn:
        rows = conn.execute("select user_id, user_name from agent_demo.institute_members where institute_id = %(inst)s "
                            "and status = 'active' and role = 'teacher' and (%(is_admin)s or user_id = %(tid)s::uuid) "
                            "order by user_name", _params(ctx)).fetchall()
    out += [{"kind": "teacher", "id": str(r["user_id"]), "name": r["user_name"]} for r in rows if r["user_name"]]
    return out
