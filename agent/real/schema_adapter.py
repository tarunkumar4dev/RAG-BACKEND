"""SchemaAdapter: the insight tools on REAL a4ai tables (see docs/AGENT_PLAN.md "Real schema map").

Same arg models and output schemas as agent/tools/insights.py (seed), so the UI never changes.
Rules: parameterized SQL only; every query filtered by ctx.institute_id; scope mirrors
a4ai/supabase/migrations/01_access_helpers.sql:
  admin   -> every batch of the institute
  HOD     -> batches of own department(s)            (+ own classes)
  proctor -> batches of proctor sections             (+ own classes)
  teacher -> (batch, subject) pairs from teaching_assignments ∪ timetable_slots, and own class_sessions
Data minimisation: only names, roll numbers, batch names and numbers leave this module. Never phone, email,
parent contact, address, gender, user ids of students.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional

from ..db_real import connect_real
from ..schemas import SessionCtx
from ..text import normalize_section, strip_honorifics
from ..tools.insights import (FORBIDDEN, MAX_MATCHES, NOT_FOUND, AttendanceStatsArgs, AttendanceTrendArgs,
                              LowAttendanceArgs, MarksStatsArgs, ResolveEntityArgs, TestsCreatedArgs, _like_tokens)

IST_DATE = "((%s at time zone 'UTC') + interval '5 hours 30 minutes')::date"  # no tz database needed


def _r1(v: Any) -> Optional[float]:
    if v is None:
        return None
    return float(Decimal(str(v)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _r2(v: Any) -> Optional[float]:
    if v is None:
        return None
    return float(Decimal(str(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _pct(good: int, total: int) -> Optional[float]:
    return _r1(Decimal(good) * 100 / Decimal(total)) if total else None


@dataclass(frozen=True)
class Scope:
    inst: str
    uid: str
    is_admin: bool
    full: list            # batches fully visible (admin / HOD dept / proctor section)
    tb: list              # teaching pairs: batch ids ...
    ts: list              # ... and subject ids (same length)
    vis: list             # full ∪ teaching batches

    def p(self, **kw) -> dict:
        return {"inst": self.inst, "uid": self.uid, "full": self.full, "tb": self.tb, "ts": self.ts,
                "vis": self.vis, **kw}


def load_scope(c, ctx: SessionCtx) -> Scope:
    inst, uid = str(ctx.institute_id), str(ctx.user_id)
    is_admin = ctx.role == "admin"
    if is_admin:
        full = [r["id"] for r in c.execute("select id from batches where institute_id = %s", (inst,)).fetchall()]
    else:
        full = [r["id"] for r in c.execute(
            "with hod as (select m.department_id from institute_members m where m.institute_id = %(inst)s "
            "  and m.user_id = %(uid)s and m.status = 'active' and m.role = 'hod' and m.department_id is not null), "
            "proc as (select pa.section_id from proctor_assignments pa join institute_members m on m.id = pa.teacher_id "
            "  where m.user_id = %(uid)s and m.institute_id = %(inst)s and m.status = 'active' "
            "  and pa.institute_id = %(inst)s and coalesce(pa.is_active, true)) "
            "select b.id from batches b left join sections s on s.id = b.section_id "
            "where b.institute_id = %(inst)s and (coalesce(b.department_id, s.department_id) in (select * from hod) "
            "  or b.section_id in (select * from proc))", {"inst": inst, "uid": uid}).fetchall()]
    pairs = c.execute(
        "select ta.batch_id, ta.subject_id from teaching_assignments ta join batches b on b.id = ta.batch_id "
        "where ta.teacher_id = %(uid)s and ta.institute_id = %(inst)s and b.institute_id = %(inst)s "
        "and coalesce(ta.is_active, true) "
        "union select ts.batch_id, ts.subject_id from timetable_slots ts join batches b on b.id = ts.batch_id "
        "where ts.teacher_id = %(uid)s and ts.institute_id = %(inst)s and b.institute_id = %(inst)s "
        "and coalesce(ts.is_active, true)", {"inst": inst, "uid": uid}).fetchall()
    tb = [r["batch_id"] for r in pairs]
    ts = [r["subject_id"] for r in pairs]
    vis = list(dict.fromkeys(full + tb))
    return Scope(inst, uid, is_admin, full, tb, ts, vis)


# sessions the caller may see (always this institute, only held classes)
SESS = ("cs.institute_id = %(inst)s and cs.status = 'conducted' and (cs.batch_id = any(%(full)s::uuid[]) "
        "or (cs.teacher_id = %(uid)s and cs.batch_id = any(%(vis)s::uuid[])) "
        "or exists (select 1 from unnest(%(tb)s::uuid[], %(ts)s::uuid[]) as tp(b, s) "
        "where tp.b = cs.batch_id and tp.s = cs.subject_id))")
STUDENT_VISIBLE = ("st.institute_id = %(inst)s and coalesce(st.is_active, true) "
                   "and (st.batch_id = any(%(vis)s::uuid[]) or st.lab_batch_id = any(%(vis)s::uuid[]))")


def _visible_teachers(c, sc: Scope) -> dict:
    """user_id -> name of staff the caller may ask about."""
    if sc.is_admin:
        rows = c.execute("select user_id, user_name from institute_members where institute_id = %(inst)s "
                         "and status = 'active' and role in ('admin','institute','hod','proctor','teacher')",
                         sc.p()).fetchall()
    else:
        rows = c.execute(
            "select m.user_id, m.user_name from institute_members m where m.institute_id = %(inst)s "
            "and m.status = 'active' and (m.user_id = %(uid)s "
            " or m.user_id in (select ta.teacher_id from teaching_assignments ta where ta.institute_id = %(inst)s "
            "   and ta.batch_id = any(%(full)s::uuid[]) and coalesce(ta.is_active, true)) "
            " or m.user_id in (select ts.teacher_id from timetable_slots ts where ts.institute_id = %(inst)s "
            "   and ts.batch_id = any(%(full)s::uuid[]) and coalesce(ts.is_active, true)) "
            " or m.department_id in (select h.department_id from institute_members h where h.institute_id = %(inst)s "
            "   and h.user_id = %(uid)s and h.status = 'active' and h.role = 'hod' and h.department_id is not null))",
            sc.p()).fetchall()
    return {r["user_id"]: (r["user_name"] or "—") for r in rows}


def _result(matches: list[dict]) -> dict:
    if not matches:
        return {"status": "not_found", "matches": []}
    if len(matches) == 1:
        return {"status": "ok", "matches": matches}
    return {"status": "ambiguous", "matches": matches[:MAX_MATCHES], "total_matches": len(matches)}


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ---- tools ----------------------------------------------------------------
def resolve_entity(ctx: SessionCtx, args: ResolveEntityArgs) -> dict[str, Any]:
    with connect_real() as c:
        sc = load_scope(c, ctx)
        if args.kind == "section":
            keys = list({_norm_key(args.name), _norm_key(normalize_section(args.name))} - {""})
            rows = c.execute("select id, name from batches where id = any(%(vis)s::uuid[]) and "
                             "regexp_replace(lower(name), '[^a-z0-9]', '', 'g') = any(%(k)s) order by name, id",
                             sc.p(k=keys)).fetchall()
            return _result([{"id": str(r["id"]), "label": r["name"]} for r in rows])
        toks = _like_tokens(strip_honorifics(args.name))
        if not toks:
            return {"status": "not_found", "matches": []}
        if args.kind == "teacher":
            vt = _visible_teachers(c, sc)
            pats = [t.lower() for t in toks]
            hits = [(uid, n) for uid, n in vt.items() if all(p.strip("%") in (n or "").lower() for p in pats)]
            hits.sort(key=lambda x: (x[1], str(x[0])))
            return _result([{"id": str(u), "label": n} for u, n in hits])
        where = " and ".join(f"st.name ilike %(t{i})s" for i in range(len(toks)))
        rows = c.execute(
            "select st.id, st.name, st.roll_no, b.name as batch from students st left join batches b on b.id = st.batch_id "
            f"where {STUDENT_VISIBLE} and {where} order by st.name, b.name, st.id limit 20",
            sc.p(**{f"t{i}": t for i, t in enumerate(toks)})).fetchall()
        return _result([{"id": str(r["id"]), "label": f"{r['name']} ({r['batch'] or '—'}"
                         + (f", roll {r['roll_no']})" if r["roll_no"] else ")")} for r in rows])


def _entity(c, sc: Scope, scope: str, eid) -> tuple[Optional[str], str, dict] | dict:
    """-> (label, extra where on sessions/records, params) or an error dict."""
    if scope == "teacher":
        vt = _visible_teachers(c, sc)
        if eid not in vt:
            known = c.execute("select 1 from institute_members where institute_id = %(inst)s and user_id = %(e)s "
                              "and status = 'active'", sc.p(e=eid)).fetchone()
            return FORBIDDEN if known else NOT_FOUND
        return vt[eid], "cs.teacher_id = %(e)s", {}
    if scope == "section":
        row = c.execute("select name from batches where id = %(e)s and id = any(%(vis)s::uuid[])", sc.p(e=eid)).fetchone()
        if not row:
            return NOT_FOUND
        return row["name"], "cs.batch_id = %(e)s", {}
    row = c.execute("select st.name, b.name as batch from students st left join batches b on b.id = st.batch_id "
                    f"where st.id = %(e)s and {STUDENT_VISIBLE}", sc.p(e=eid)).fetchone()
    if not row:
        return NOT_FOUND
    return f"{row['name']} ({row['batch'] or '—'})", "ar.student_id = %(e)s", {}


def _agg(c, sc: Scope, where: str, df: date, dt: date, eid) -> dict:
    return c.execute(
        "select count(distinct cs.id) as sessions, count(ar.id) as marked, "
        "count(ar.id) filter (where ar.status = 'present') as present, "
        "count(ar.id) filter (where ar.status = 'late') as late, "
        "count(ar.id) filter (where ar.status = 'absent') as absent, "
        "count(ar.id) filter (where ar.status = 'leave') as leave "
        "from class_sessions cs join attendance_records ar on ar.session_id = cs.id and ar.institute_id = %(inst)s "
        f"where {SESS} and {where} and cs.session_date between %(df)s and %(dt)s",
        sc.p(e=eid, df=df, dt=dt)).fetchone()


def attendance_stats(ctx: SessionCtx, args: AttendanceStatsArgs) -> dict[str, Any]:
    with connect_real() as c:
        sc = load_scope(c, ctx)
        ent = _entity(c, sc, args.scope, args.entity_id)
        if isinstance(ent, dict):
            return ent
        label, where, _ = ent
        cur = _agg(c, sc, where, args.date_from, args.date_to, args.entity_id)
        days = (args.date_to - args.date_from).days + 1
        prev = _agg(c, sc, where, args.date_from - timedelta(days=days), args.date_from - timedelta(days=1),
                    args.entity_id)
    return {"status": "ok", "entity": label, "date_from": args.date_from.isoformat(),
            "date_to": args.date_to.isoformat(), "sessions": int(cur["sessions"]), "marked": int(cur["marked"]),
            "present": int(cur["present"]), "late": int(cur["late"]), "absent": int(cur["absent"]),
            "leave": int(cur["leave"]), "pct": _pct(cur["present"] + cur["late"], cur["marked"]),
            "prev_pct": _pct(prev["present"] + prev["late"], prev["marked"])}


def low_attendance_list(ctx: SessionCtx, args: LowAttendanceArgs) -> dict[str, Any]:
    with connect_real() as c:
        sc = load_scope(c, ctx)
        if args.section_id is not None and args.section_id not in sc.vis:
            return NOT_FOUND
        rows = c.execute(
            "with per as (select st.id, st.name, b.name as section, count(ar.id) as marked, "
            "  count(ar.id) filter (where ar.status in ('present','late')) as good "
            "  from students st left join batches b on b.id = st.batch_id "
            "  join attendance_records ar on ar.student_id = st.id and ar.institute_id = %(inst)s "
            "  join class_sessions cs on cs.id = ar.session_id "
            f"  where {STUDENT_VISIBLE} and {SESS} and cs.session_date between %(df)s and %(dt)s "
            "  and (%(sid)s::uuid is null or st.batch_id = %(sid)s::uuid or st.lab_batch_id = %(sid)s::uuid "
            "       or cs.batch_id = %(sid)s::uuid) "
            "  group by st.id, st.name, b.name having count(ar.id) > 0) "
            "select id, name, section, marked, good from per where good * 100.0 / marked < %(thr)s "
            "order by good * 1.0 / marked, name, id limit 200",
            sc.p(df=args.date_from, dt=args.date_to, sid=args.section_id, thr=args.threshold)).fetchall()
    return {"status": "ok", "threshold": args.threshold,
            "rows": [{"student_id": str(r["id"]), "name": r["name"], "section": r["section"],
                      "pct": _pct(r["good"], r["marked"]), "marked": int(r["marked"])} for r in rows]}


def attendance_trend(ctx: SessionCtx, args: AttendanceTrendArgs) -> dict[str, Any]:
    with connect_real() as c:
        sc = load_scope(c, ctx)
        ent = _entity(c, sc, args.scope, args.entity_id)
        if isinstance(ent, dict):
            return ent
        label, where, _ = ent
        rows = c.execute(
            "select date_trunc(%(g)s, cs.session_date)::date as period, count(ar.id) as marked, "
            "count(ar.id) filter (where ar.status in ('present','late')) as good "
            "from class_sessions cs join attendance_records ar on ar.session_id = cs.id and ar.institute_id = %(inst)s "
            f"where {SESS} and {where} and cs.session_date between %(df)s and %(dt)s group by 1 order by 1 limit 120",
            sc.p(e=args.entity_id, df=args.date_from, dt=args.date_to, g=args.granularity)).fetchall()
    return {"status": "ok", "entity": label, "granularity": args.granularity,
            "date_from": args.date_from.isoformat(), "date_to": args.date_to.isoformat(),
            "points": [{"period": r["period"].isoformat(), "pct": _pct(r["good"], r["marked"]),
                        "marked": int(r["marked"])} for r in rows]}


# marks: graded assignment submissions ∪ checked copies (check_sheets). There is no single marks table.
ITEMS = """
items as (
  select 'assignment' as kind, a.id, a.title as test, %(ist_a)s as on_date, a.max_marks::numeric as max_marks, a.batch_id
  from assignments a
  where a.institute_id = %%(inst)s and (a.batch_id = any(%%(full)s::uuid[])
        or (a.batch_id = any(%%(vis)s::uuid[]) and a.created_by = %%(uid)s))
  union all
  select 'check_copies', k.id, k.title, %(ist_k)s, k.total_marks, k.batch_id
  from check_sessions k join batches b on b.id = k.batch_id and b.institute_id = %%(inst)s
  where k.batch_id = any(%%(full)s::uuid[]) or (k.batch_id = any(%%(vis)s::uuid[]) and k.teacher_id = %%(uid)s)
),
marks as (
  select 'assignment' as kind, s.assignment_id as item_id, s.student_id, s.grade::numeric as mark
  from submissions s where s.grade is not null
  union all
  select 'check_copies', h.session_id, h.student_id, h.final_marks
  from check_sheets h where h.final_marks is not null and h.status in ('checked', 'approved')
)""" % {"ist_a": IST_DATE % "coalesce(a.deadline, a.created_at)", "ist_k": IST_DATE % "k.created_at"}


def marks_stats(ctx: SessionCtx, args: MarksStatsArgs) -> dict[str, Any]:
    with connect_real() as c:
        sc = load_scope(c, ctx)
        subj = None
        if args.subject_id is not None:
            r = c.execute("select name from subjects where id = %(s)s and institute_id = %(inst)s",
                          sc.p(s=args.subject_id)).fetchone()
            if not r:
                return NOT_FOUND
            subj = r["name"]
        subj_sql = ("and (i.kind = 'check_copies' and exists (select 1 from check_sessions k2 where k2.id = i.id "
                    "and k2.subject ilike %(subj)s))") if subj else ""
        if args.scope == "section":
            b = c.execute("select name from batches where id = %(e)s and id = any(%(vis)s::uuid[])",
                          sc.p(e=args.entity_id)).fetchone()
            if not b:
                return NOT_FOUND
            rows = c.execute(
                f"with {ITEMS}, roster as (select count(*) n from students st where st.institute_id = %(inst)s "
                "  and coalesce(st.is_active, true) and (st.batch_id = %(e)s or st.lab_batch_id = %(e)s)) "
                "select i.kind, i.test, i.on_date, i.max_marks, count(m.mark) as graded, avg(m.mark) as avg, "
                "max(m.mark) as hi, min(m.mark) as lo, (select n from roster) as roster, "
                "array_remove(array_agg(m.mark * 100.0 / nullif(i.max_marks, 0)), null) as pcts "
                "from items i left join marks m on m.kind = i.kind and m.item_id = i.id "
                "  and m.student_id in (select st.id from students st where st.institute_id = %(inst)s "
                "  and (st.batch_id = %(e)s or st.lab_batch_id = %(e)s)) "
                f"where i.batch_id = %(e)s {subj_sql} group by i.kind, i.id, i.test, i.on_date, i.max_marks "
                "order by i.on_date, i.test", sc.p(e=args.entity_id, subj=subj)).fetchall()
            pooled = [p for r in rows for p in (r["pcts"] or [])]
            tests = [{"test": r["test"], "kind": r["kind"], "date": r["on_date"].isoformat() if r["on_date"] else None,
                      "max_marks": _r2(r["max_marks"]), "avg": _r2(r["avg"]),
                      "avg_pct": _r1(r["avg"] * 100 / r["max_marks"]) if r["avg"] is not None and r["max_marks"] else None,
                      "highest": _r2(r["hi"]), "lowest": _r2(r["lo"]),
                      "absent": max(int(r["roster"]) - int(r["graded"]), 0)} for r in rows]
            overall = _r1(sum(pooled) / len(pooled)) if pooled else None
            return {"status": "ok", "entity": b["name"], "tests": tests, "overall_avg_pct": overall}
        st = c.execute("select st.name, b.name as batch, st.batch_id, st.lab_batch_id from students st "
                       f"left join batches b on b.id = st.batch_id where st.id = %(e)s and {STUDENT_VISIBLE}",
                       sc.p(e=args.entity_id)).fetchone()
        if not st:
            return NOT_FOUND
        rows = c.execute(
            f"with {ITEMS} select i.kind, i.test, i.on_date, i.max_marks, m.mark from items i "
            "left join marks m on m.kind = i.kind and m.item_id = i.id and m.student_id = %(e)s "
            f"where i.batch_id = any(array[%(b1)s, %(b2)s]::uuid[]) {subj_sql} order by i.on_date, i.test",
            sc.p(e=args.entity_id, b1=st["batch_id"], b2=st["lab_batch_id"], subj=subj)).fetchall()
    tests, pcts = [], []
    for r in rows:
        pct = _r1(r["mark"] * 100 / r["max_marks"]) if r["mark"] is not None and r["max_marks"] else None
        if r["mark"] is not None and r["max_marks"]:
            pcts.append(Decimal(r["mark"]) * 100 / Decimal(r["max_marks"]))
        tests.append({"test": r["test"], "kind": r["kind"], "date": r["on_date"].isoformat() if r["on_date"] else None,
                      "max_marks": _r2(r["max_marks"]), "marks": _r2(r["mark"]), "pct": pct, "absent": r["mark"] is None})
    return {"status": "ok", "entity": f"{st['name']} ({st['batch'] or '—'})", "tests": tests,
            "overall_avg_pct": _r1(sum(pcts) / len(pcts)) if pcts else None}


def tests_created_by_teacher(ctx: SessionCtx, args: TestsCreatedArgs) -> dict[str, Any]:
    with connect_real() as c:
        sc = load_scope(c, ctx)
        vt = _visible_teachers(c, sc)
        target = args.teacher_id or (None if sc.is_admin or ctx.role == "hod" else ctx.user_id)
        if target is not None and target not in vt:
            known = c.execute("select 1 from institute_members where institute_id = %(inst)s and user_id = %(t)s "
                              "and status = 'active'", sc.p(t=target)).fetchone()
            return FORBIDDEN if known else NOT_FOUND
        ids = [target] if target is not None else list(vt.keys())
        created = IST_DATE % "t.created_at"
        rows = c.execute(
            f"select t.teacher_id, t.exam_title, t.class_grade, t.subject, t.total_marks, t.status, {created} as on_date "
            f"from tests t where t.teacher_id = any(%(ids)s::uuid[]) and {created} between %(df)s and %(dt)s "
            "order by t.created_at desc, t.id limit 500", sc.p(ids=ids, df=args.date_from, dt=args.date_to)).fetchall()
    by: dict[str, int] = {}
    for r in rows:
        by[vt[r["teacher_id"]]] = by.get(vt[r["teacher_id"]], 0) + 1
    return {"status": "ok", "teacher": vt[target] if target is not None else "Institute" if sc.is_admin else "Department",
            "date_from": args.date_from.isoformat(), "date_to": args.date_to.isoformat(), "count": len(rows),
            "by_teacher": [{"teacher": k, "count": v} for k, v in sorted(by.items())],
            "recent": [{"title": r["exam_title"], "class_grade": r["class_grade"], "subject": r["subject"],
                        "total_marks": _r2(r["total_marks"]), "status": r["status"],
                        "created_on": r["on_date"].isoformat()} for r in rows[:10]]}
