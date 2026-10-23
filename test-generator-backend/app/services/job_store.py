"""
Database side of the checker job worker (contract: docs/upload-flow-api.md §4, functions from 18_upload_flow.sql).

Direct Postgres (app.core.db_pool, same credentials as Modules / Worksheet) instead of PostgREST, because the
worker needs real transactions: a sheet's marks, its audit row and the job checkpoint are written together, so a
retry after a crash never grades the same copy twice. Claim / finish always go through a4_claim_check_job /
a4_finish_check_job; the state → submission status sync stays in the database triggers.

Every call takes a fresh connection from the factory and gives it back (the pool hands out autocommit
connections; `tx=True` turns autocommit off for one unit of work).
"""

import contextlib
from typing import Callable, Optional

from psycopg2.extras import Json

from app.routers.smart_assignment_router import NOT_A_CHECK_TYPES


def _raw(conn):
    return getattr(conn, "_conn", conn)      # db_pool wraps connections in a proxy


class JobStore:
    def __init__(self, connect: Callable):
        self._connect = connect

    @contextlib.contextmanager
    def _cur(self, tx: bool = False):
        conn = self._connect()
        if conn is None:
            raise RuntimeError("database unavailable")
        raw = _raw(conn)
        try:
            if tx:
                raw.autocommit = False
            cur = raw.cursor()
            try:
                yield cur
                if tx:
                    raw.commit()
            except Exception:
                if tx:
                    raw.rollback()
                raise
            finally:
                cur.close()
        finally:
            if tx:
                try:
                    raw.autocommit = True
                except Exception:
                    pass
            conn.close()

    def _one(self, sql: str, args: tuple = ()):
        with self._cur() as cur:
            cur.execute(sql, args)
            row = cur.fetchone()
            return row[0] if row else None

    def _all(self, sql: str, args: tuple = ()) -> list:
        with self._cur() as cur:
            cur.execute(sql, args)
            return [r[0] for r in cur.fetchall()]

    # ── queue ─────────────────────────────────────────────────────────
    def claim(self, max_running: int) -> Optional[dict]:
        return self._one("select row_to_json(j) from public.a4_claim_check_job(%s) j", (max_running,))

    def finish(self, job_id: str, ok: bool, code: str = None, error: str = None, usage: dict = None) -> dict:
        return self._one("select row_to_json(f) from public.a4_finish_check_job(%s, %s, %s, %s, %s) f",
                         (job_id, ok, code, error, Json(usage) if usage is not None else None))

    def job(self, job_id: str) -> Optional[dict]:
        return self._one("select row_to_json(j) from public.check_jobs j where id = %s", (job_id,))

    def not_before(self, job_id: str, seconds: float) -> None:
        """Gemini asked us to wait longer than the default backoff (Retry-After)."""
        with self._cur() as cur:
            cur.execute("update public.check_jobs set next_run_at = greatest(next_run_at, now() + make_interval(secs => %s))"
                        " where id = %s and state = 'queued'", (seconds, job_id))

    def requeue_unchanged(self, job_id: str, delay_s: int, checkpoint: dict = None, usage: dict = None) -> None:
        """Back to the queue without using a retry (rubric still building elsewhere, chunks still running)."""
        with self._cur() as cur:
            cur.execute("""update public.check_jobs
                              set state = 'queued', locked_at = null, retries = greatest(retries - 1, 0),
                                  next_run_at = now() + make_interval(secs => %s),
                                  checkpoint = coalesce(%s, checkpoint), usage = coalesce(%s, usage), updated_at = now()
                            where id = %s and state = 'running'""",
                        (delay_s, Json(checkpoint) if checkpoint is not None else None,
                         Json(usage) if usage is not None else None, job_id))

    def set_checkpoint(self, job_id: str, checkpoint: dict) -> None:
        with self._cur() as cur:
            cur.execute("update public.check_jobs set checkpoint = %s, updated_at = now() where id = %s",
                        (Json(checkpoint), job_id))

    def use_up_retries(self, job_id: str) -> None:
        with self._cur() as cur:
            cur.execute("update public.check_jobs set retries = max_retries where id = %s", (job_id,))

    # ── chunks of a long copy (kind 'check', submission_id null, checkpoint.parent = the copy's job) ──
    def add_chunks(self, parent: dict, chunks: list) -> None:
        with self._cur(tx=True) as cur:
            for ch in chunks:
                cur.execute("""insert into public.check_jobs (kind, teacher_id, institute_id, session_id, sheet_id,
                                                              state, finished_at, idempotency_key, checkpoint)
                               values ('check', %s, %s, %s, %s, %s, case when %s = 'done' then now() end, %s, %s)
                               on conflict (idempotency_key) do nothing""",
                            (parent["teacher_id"], parent["institute_id"], parent["session_id"], parent["sheet_id"],
                             ch["state"], ch["state"], f"chunk:{parent['id']}:{ch['checkpoint']['chunk']}",
                             Json(ch["checkpoint"])))

    def restart_chunks(self, job_ids: list) -> None:
        with self._cur() as cur:
            cur.execute("""update public.check_jobs
                              set state = 'queued', retries = 0, next_run_at = now(), locked_at = null, finished_at = null,
                                  last_error_code = null, error = null, updated_at = now()
                            where id = any(%s::uuid[]) and state in ('failed', 'cancelled')""", (list(job_ids),))

    def chunks_of(self, parent_id: str) -> list:
        return self._all("""select row_to_json(j) from public.check_jobs j
                             where kind = 'check' and submission_id is null and checkpoint ->> 'parent' = %s
                             order by (checkpoint ->> 'chunk')::int""", (str(parent_id),))

    def wake(self, job_id: str) -> None:
        with self._cur() as cur:
            cur.execute("update public.check_jobs set next_run_at = now(), updated_at = now()"
                        " where id = %s and state = 'queued'", (job_id,))

    def show_checking(self, submission_id: str) -> None:
        """A parked long copy is still being checked (its chunks are): students see 'checking', not 'queued'."""
        with self._cur() as cur:
            cur.execute("update public.submissions set status = 'checking' where id = %s and status = 'queued'",
                        (submission_id,))

    # ── rows ──────────────────────────────────────────────────────────
    def row(self, table: str, row_id: str, key: str = "id") -> Optional[dict]:
        assert table in ("check_sheets", "check_sessions", "submissions", "assignments", "assignment_ai_keys")
        assert key in ("id", "assignment_id", "submission_id")
        return self._one(f"select row_to_json(t) from public.{table} t where {key} = %s", (row_id,))

    def other_students_with(self, assignment_id: str, student_id: str, shas: list) -> list:
        """sha256 values among `shas` that another student of this assignment also uploaded."""
        if not shas:
            return []
        return self._all("""select distinct f ->> 'sha256' from public.submissions s, jsonb_array_elements(s.files) f
                             where s.assignment_id = %s and s.student_id <> %s and f ->> 'sha256' = any(%s)""",
                         (assignment_id, student_id, list(shas)))

    def set_reject_reasons(self, submission_id: str, problems: list) -> None:
        """The state trigger appended one {code: bad_file, message} entry; replace it with one entry per file."""
        with self._cur() as cur:
            cur.execute("""update public.submissions
                              set reject_reasons = (reject_reasons - (jsonb_array_length(reject_reasons) - 1)) || %s
                            where id = %s and status = 'rejected' and jsonb_array_length(reject_reasons) > 0""",
                        (Json(problems), submission_id))

    # ── rubric: one job per (session = assignment, answer-key version), result cached in check_sessions.rubric ──
    def rubric_job(self, session_id: str, key_version: int) -> Optional[dict]:
        return self._one("select row_to_json(j) from public.check_jobs j where idempotency_key = %s",
                         (f"rubric:{session_id}:k{key_version}",))

    def ensure_rubric_job(self, copy_job: dict, assignment_id: str, key_version: int) -> dict:
        """The rubric job of this assignment + key version (created now if missing; a 'done' one is re-opened when
        the cache is gone). kind 'check' + submission_id null + checkpoint.stage 'rubric': the queue's kind check
        needs a sheet, so it carries the sheet of the copy that asked first (the state triggers ignore it)."""
        key = f"rubric:{copy_job['session_id']}:k{key_version}"
        with self._cur() as cur:
            cur.execute("""insert into public.check_jobs (kind, teacher_id, institute_id, session_id, sheet_id,
                                                          idempotency_key, checkpoint)
                           values ('check', %s, %s, %s, %s, %s, %s)
                           on conflict (idempotency_key) do update
                              set state = 'queued', retries = 0, next_run_at = now(), locked_at = null,
                                  finished_at = null, last_error_code = null, error = null, updated_at = now()
                            where check_jobs.state = 'done'""",
                        (copy_job["teacher_id"], copy_job["institute_id"], copy_job["session_id"], copy_job["sheet_id"],
                         key, Json({"stage": "rubric", "assignment_id": str(assignment_id), "key_version": key_version})))
        return self.rubric_job(copy_job["session_id"], key_version)

    def rubric_building(self, session_id: str) -> None:
        with self._cur() as cur:
            cur.execute("update public.check_sessions set rubric_status = 'building', rubric_error = null, updated_at = now()"
                        " where id = %s", (session_id,))

    def wake_copies(self, session_id: str) -> int:
        """Copies parked while their rubric was being built → due now (picked up by this run or the next sweep)."""
        with self._cur() as cur:
            cur.execute("""update public.check_jobs set next_run_at = now(), updated_at = now()
                            where session_id = %s and kind = 'check' and submission_id is not null and state = 'queued'
                              and next_run_at > now()""", (session_id,))
            return cur.rowcount

    def requeue_waiting_key(self, assignment_id: str) -> int:
        """Attempts that went to waiting_key because the rubric failed: queue them again now that it exists."""
        ids = self._all("select id::text from public.submissions where assignment_id = %s and status = 'waiting_key'",
                        (assignment_id,))
        for sid in ids:
            self.enqueue_submission(sid)
        return len(ids)

    def save_rubric(self, session_id: str, rubric: dict) -> None:
        with self._cur() as cur:
            cur.execute("""update public.check_sessions set rubric = %s, rubric_status = 'ready', rubric_error = null,
                                  total_marks = %s, strictness = coalesce(%s, strictness), updated_at = now()
                            where id = %s""",
                        (Json(rubric), rubric.get("total_marks"), rubric.get("strictness"), session_id))

    def rubric_failed(self, session_id: str, message: str) -> None:
        with self._cur() as cur:
            cur.execute("update public.check_sessions set rubric_status = 'failed', rubric_error = %s, updated_at = now()"
                        " where id = %s", (message[:300], session_id))

    # ── results ───────────────────────────────────────────────────────
    def save_result(self, job_id: str, sheet_id: str, fields: dict, audit: Optional[dict], checkpoint: dict) -> None:
        """The marks, the audit row (= one check in the teacher's daily count) and the job's 'graded' checkpoint
        in one transaction: a retry after this never calls Gemini again."""
        with self._cur(tx=True) as cur:
            cur.execute("""update public.check_sheets
                              set status = 'checked', ai_result = %s, ai_marks = %s, max_marks = %s, error = null,
                                  checked_at = now(), updated_at = now(), needs_review_reasons = %s::public.a4_review_reason[],
                                  confidence = %s, model_version = %s
                            where id = %s""",
                        (Json(fields["ai_result"]), fields["ai_marks"], fields["max_marks"],
                         list(fields["needs_review_reasons"]), fields["confidence"], fields["model_version"], sheet_id))
            if audit:
                self._insert_audit(cur, audit)
            cur.execute("update public.check_jobs set checkpoint = %s, updated_at = now() where id = %s",
                        (Json(checkpoint), job_id))

    def mark_sheet_failed(self, sheet_id: str, message: str) -> None:
        with self._cur() as cur:
            cur.execute("update public.check_sheets set status = 'failed', error = %s, updated_at = now() where id = %s",
                        (message[:300], sheet_id))

    def mark_sheet_checking(self, sheet_id: str) -> None:
        with self._cur() as cur:
            cur.execute("update public.check_sheets set status = 'checking', error = null, updated_at = now()"
                        " where id = %s and status <> 'approved'", (sheet_id,))

    @staticmethod
    def _insert_audit(cur, a: dict) -> None:
        cur.execute("""insert into public.smart_assignment_checks
                         (id, teacher_id, student_name, student_class, student_roll, subject, class_grade, qp_filename,
                          ss_filename, is_valid, total_obtained, max_marks, percentage, needs_review_count, strictness,
                          overall_remarks, result_json, assignment_id, submission_id, check_sheet_id)
                       values (gen_random_uuid(), %(teacher_id)s, %(student_name)s, %(student_class)s, %(student_roll)s,
                               %(subject)s, %(class_grade)s, %(qp_filename)s, %(ss_filename)s, %(is_valid)s,
                               %(total_obtained)s, %(max_marks)s, %(percentage)s, %(needs_review_count)s, %(strictness)s,
                               %(overall_remarks)s, %(result_json)s, %(assignment_id)s, %(submission_id)s,
                               %(check_sheet_id)s)""",
                    {**a, "result_json": Json(a["result_json"])})

    def insert_audit(self, audit: dict) -> None:
        with self._cur() as cur:
            self._insert_audit(cur, audit)

    # ── limits ────────────────────────────────────────────────────────
    def checks_today(self, teacher_id: str, since, exclude_job_id: str) -> int:
        """AI checks the teacher used today (audit rows; publish / unpublish / override / regrade rows are not
        checks) + copies of theirs being checked right now by other runs (not saved yet)."""
        return int(self._one("""select (select count(*) from public.smart_assignment_checks
                                         where teacher_id = %s and created_at >= %s
                                           and (result_json ->> 'type' is null or result_json ->> 'type' <> all(%s)))
                                     + (select count(*) from public.check_jobs
                                         where teacher_id = %s and kind = 'check' and submission_id is not null
                                           and state = 'running' and id <> %s
                                           and coalesce(checkpoint ->> 'graded', 'false') <> 'true')""",
                             (teacher_id, since, list(NOT_A_CHECK_TYPES), teacher_id, exclude_job_id)) or 0)

    def spend_since(self, since) -> float:
        """Estimated Gemini spend (₹) of every job touched since `since` (usage.est_cost_inr, all runs)."""
        return float(self._one("""select coalesce(sum((usage ->> 'est_cost_inr')::numeric), 0) from public.check_jobs
                                   where updated_at >= %s and usage ? 'est_cost_inr'""", (since,)) or 0)

    # ── retention / regrade ──────────────────────────────────────────
    def orphan_files(self, limit: int) -> list:
        return self._all("select f from public.a4_orphan_submission_files(%s) f", (limit,))

    def enqueue_submission(self, submission_id: str) -> Optional[str]:
        v = self._one("select public.a4_enqueue_submission(%s)", (submission_id,))
        return str(v) if v else None
