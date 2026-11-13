"""Read-only trial export tools: defaulters_export, proctor_month_report.

Each builds an .xlsx (openpyxl), stores it in the agent file store (agent/files.py), and returns a dict that the
agent turns into a `file` event. The file is owner-only, expires in 1 hour, never touches Supabase.
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from ..db import connect
from ..schemas import SessionCtx, now_ist


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DefaultersExportArgs(_Args):
    """Export a list of students below an attendance threshold over a period."""
    date_from: date
    date_to: date
    threshold: float = Field(default=75, ge=0, le=100)
    section_id: Optional[UUID] = None  # None = all sections in scope


class ProctorMonthReportArgs(_Args):
    """Subject-wise attendance matrix for a section in a given month."""
    section_id: UUID
    month: str = Field(pattern=r"^\d{4}-\d{2}$")  # "2026-10"


def _check_openpyxl():
    try:
        import openpyxl
        return openpyxl
    except ImportError:
        return None


def defaulters_export(ctx: SessionCtx, args: DefaultersExportArgs) -> dict[str, Any]:
    """Build an .xlsx of students below the attendance threshold. Returns a file-ready dict."""
    openpyxl = _check_openpyxl()
    if openpyxl is None:
        return {"status": "error", "error": "openpyxl not installed"}

    from openpyxl import Workbook
    from ..files import store_file

    with connect(readonly=True) as conn:
        # Get batches in scope
        section_filter = ""
        params: dict = {"inst": str(ctx.institute_id), "df": str(args.date_from), "dt": str(args.date_to)}
        if args.section_id:
            section_filter = "AND b.section_id = %(sid)s"
            params["sid"] = str(args.section_id)

        rows = conn.execute(f"""
            WITH scope_batches AS (
                SELECT b.id AS batch_id, b.name AS batch_name
                FROM batches b
                JOIN sections sec ON sec.id = b.section_id
                WHERE b.institute_id = %(inst)s AND coalesce(b.is_lab, false) = false
                {section_filter}
            ),
            student_att AS (
                SELECT s.id, s.name, s.roll_no, sb.batch_name,
                       count(DISTINCT cs.id) AS sessions_held,
                       count(DISTINCT CASE WHEN ar.status IN ('present', 'late') THEN ar.session_id END) AS attended
                FROM students s
                JOIN scope_batches sb ON sb.batch_id = s.batch_id
                JOIN class_sessions cs ON cs.batch_id = s.batch_id
                     AND cs.session_date BETWEEN %(df)s AND %(dt)s
                     AND cs.status = 'conducted'
                LEFT JOIN attendance_records ar ON ar.session_id = cs.id AND ar.student_id = s.id
                WHERE s.institute_id = %(inst)s AND coalesce(s.is_active, true)
                GROUP BY s.id, s.name, s.roll_no, sb.batch_name
                HAVING count(DISTINCT cs.id) > 0
            )
            SELECT name, roll_no, batch_name, sessions_held, attended,
                   round(100.0 * attended / sessions_held, 1) AS pct
            FROM student_att
            WHERE round(100.0 * attended / sessions_held, 1) < %(thr)s
            ORDER BY pct, batch_name, roll_no
        """, {**params, "thr": args.threshold}).fetchall()

    # Build workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Defaulters"
    headers = ["Roll No", "Name", "Batch", "Sessions Held", "Attended", "Attendance %"]
    ws.append(headers)
    for r in rows:
        ws.append([r["roll_no"], r["name"], r["batch_name"],
                   r["sessions_held"], r["attended"], float(r["pct"])])

    # Style header
    from openpyxl.styles import Font
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = max(12, len(str(col[0].value or "")) + 4)

    name = f"defaulters_{args.date_from}_{args.date_to}.xlsx"
    fid = store_file(ctx, wb, name, kind="defaulters_export")
    return {"status": "ok", "file_id": fid, "name": name, "rows": len(rows),
            "period": f"{args.date_from} to {args.date_to}", "threshold": args.threshold}


def proctor_month_report(ctx: SessionCtx, args: ProctorMonthReportArgs) -> dict[str, Any]:
    """Build a subject-wise attendance matrix for a section-month."""
    openpyxl = _check_openpyxl()
    if openpyxl is None:
        return {"status": "error", "error": "openpyxl not installed"}

    from openpyxl import Workbook
    from ..files import store_file

    year, month = args.month.split("-")
    # Last day of month
    import calendar
    last_day = calendar.monthrange(int(year), int(month))[1]
    date_from = f"{args.month}-01"
    date_to = f"{args.month}-{last_day:02d}"

    with connect(readonly=True) as conn:
        # Get students and their attendance per subject
        rows = conn.execute("""
            WITH section_batches AS (
                SELECT b.id AS batch_id FROM batches b
                WHERE b.section_id = %(sid)s AND coalesce(b.is_lab, false) = false
                  AND b.institute_id = %(inst)s
            ),
            student_subject AS (
                SELECT s.id AS student_id, s.name, s.roll_no,
                       sub.name AS subject_name, sub.id AS subject_id,
                       count(DISTINCT cs.id) AS held,
                       count(DISTINCT CASE WHEN ar.status IN ('present', 'late') THEN ar.session_id END) AS attended
                FROM students s
                JOIN section_batches sb ON sb.batch_id = s.batch_id
                JOIN class_sessions cs ON cs.batch_id = s.batch_id
                     AND cs.session_date BETWEEN %(df)s AND %(dt)s AND cs.status = 'conducted'
                JOIN subjects sub ON sub.id = cs.subject_id AND sub.institute_id = %(inst)s
                LEFT JOIN attendance_records ar ON ar.session_id = cs.id AND ar.student_id = s.id
                WHERE s.institute_id = %(inst)s AND coalesce(s.is_active, true)
                GROUP BY s.id, s.name, s.roll_no, sub.name, sub.id
            )
            SELECT student_id, name, roll_no, subject_name, held, attended,
                   CASE WHEN held > 0 THEN round(100.0 * attended / held, 1) ELSE 0 END AS pct
            FROM student_subject
            ORDER BY roll_no, subject_name
        """, {"sid": str(args.section_id), "inst": str(ctx.institute_id),
              "df": date_from, "dt": date_to}).fetchall()

    # Pivot: students × subjects
    subjects: list[str] = sorted(set(r["subject_name"] for r in rows))
    students: dict[str, dict] = {}
    for r in rows:
        key = r["student_id"]
        if key not in students:
            students[key] = {"roll_no": r["roll_no"], "name": r["name"], "subjects": {}}
        students[key]["subjects"][r["subject_name"]] = {
            "held": r["held"], "attended": r["attended"], "pct": float(r["pct"])}

    wb = Workbook()
    ws = wb.active
    ws.title = f"Report {args.month}"
    headers = ["Roll No", "Name"] + [f"{s} %" for s in subjects] + ["Overall %"]
    ws.append(headers)

    for sid in sorted(students, key=lambda k: students[k]["roll_no"] or ""):
        st = students[sid]
        row = [st["roll_no"], st["name"]]
        total_held, total_att = 0, 0
        for s in subjects:
            d = st["subjects"].get(s, {"pct": 0, "held": 0, "attended": 0})
            row.append(d["pct"])
            total_held += d["held"]
            total_att += d["attended"]
        overall = round(100 * total_att / total_held, 1) if total_held else 0
        row.append(overall)
        ws.append(row)

    from openpyxl.styles import Font
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = max(12, len(str(col[0].value or "")) + 4)

    name = f"proctor_report_{args.month}.xlsx"
    fid = store_file(ctx, wb, name, kind="proctor_month_report")
    return {"status": "ok", "file_id": fid, "name": name, "rows": len(students),
            "month": args.month, "subjects": len(subjects)}
