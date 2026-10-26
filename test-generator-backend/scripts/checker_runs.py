"""
Shared by shadow_regrade.py and export_ai_marks.py: env, a read-only-by-default DB connection, and the rows of one
assignment's checked copies. A "run" is either the production marks (label "prod" = check_sheets.ai_result) or a
shadow re-grade stored in check_shadow_results under its label (26_checker_p0.sql).

Exports contain student names: write them outside the repo (or to a git-ignored path), never commit them.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv  # noqa: E402

# A worktree has no .env of its own; the main checkout's is the local one.
for env in (BACKEND / ".env", Path("C:/Users/login/Downloads/RAG-BACKEND/test-generator-backend/.env")):
    if env.exists():
        load_dotenv(env, override=False)
        break

PROD = "prod"
CHECKED = ("checked", "needs_review", "published")


def connect():
    """Direct Postgres (same credentials as the worker), autocommit, read-only unless writable=True."""
    from app.core.db_pool import get_db_connection
    conn = get_db_connection()
    if conn is None:
        raise SystemExit("Direct DB not configured (DATABASE_URL / DB_PASSWORD in .env)")
    return conn


def rows(conn, sql: str, args: tuple = ()) -> list:
    raw = getattr(conn, "_conn", conn)
    cur = raw.cursor()
    try:
        cur.execute(sql, args)
        return [r[0] for r in cur.fetchall()]
    finally:
        cur.close()


def copies(conn, assignment_id: str) -> list:
    """One dict per checked attempt of the assignment: submission + sheet + student + the copy job's usage."""
    return rows(conn, """
        select json_build_object(
                 'submission_id', s.id, 'status', s.status, 'answer_key_version', s.answer_key_version,
                 'files', s.files, 'student_id', s.student_id, 'submitter_key', s.submitter_key,
                 'roll', coalesce(st.roll_no, s.guest_roll), 'name', coalesce(st.name, s.guest_name),
                 'sheet', row_to_json(sh), 'usage', j.usage)
          from public.submissions s
          join public.check_sheets sh on sh.submission_id = s.id
          left join public.students st on st.id = s.student_id
          left join lateral (select usage from public.check_jobs
                              where submission_id = s.id and kind = 'check' order by updated_at desc limit 1) j on true
         where s.assignment_id = %s and s.status = any(%s)
         order by coalesce(st.roll_no, s.guest_roll), s.attempt_no""", (assignment_id, list(CHECKED)))


def run_results(conn, assignment_id: str, label: str) -> list:
    """[{submission_id, roll, name, model, ai_result}] for one run."""
    if label == PROD:
        return [{"submission_id": c["submission_id"], "roll": c["roll"], "name": c["name"],
                 "model": (c["sheet"] or {}).get("model_version"), "ai_result": (c["sheet"] or {}).get("ai_result") or {}}
                for c in copies(conn, assignment_id)]
    return rows(conn, """
        select json_build_object('submission_id', r.submission_id, 'roll', coalesce(st.roll_no, s.guest_roll),
                                 'name', coalesce(st.name, s.guest_name), 'model', r.model, 'ai_result', r.ai_result)
          from public.check_shadow_results r
          join public.submissions s on s.id = r.submission_id
          left join public.students st on st.id = s.student_id
         where r.assignment_id = %s and r.run_label = %s
         order by coalesce(st.roll_no, s.guest_roll)""", (assignment_id, label))


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
