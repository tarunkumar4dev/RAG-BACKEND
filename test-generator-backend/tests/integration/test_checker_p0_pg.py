"""
fix/checker-p0 on the real queue SQL (25_review_reason_fallback + 26_checker_p0, see conftest.py):
failed_retryable rounds, the sweep's alert, flagged copies stay hidden until approved / edited, a backup model is
flagged, checker_job_health, check_shadow_results is service-only.
"""

import pytest

from app.services import job_worker, smart_assignment_service as svc
from app.services.answer_checker import copy_check
from app.services.job_store import JobStore
from tests.integration.conftest import A1, A9, FIXTURES, S2, T1, make_attempt

SHEET = (FIXTURES / "answer_sheet_messy.pdf").read_bytes()
PDF = [(SHEET, "pdf", "application/pdf")]


@pytest.fixture(autouse=True)
def needs_26(db):
    if not db.one("select to_regclass('public.checker_job_health') is not null"):
        pytest.skip("26_checker_p0.sql not in this checkout (A4AI_PRODUCT_DIR)")


def worker(db, storage, **kw) -> job_worker.JobWorker:
    kw.setdefault("budget_s", 10_000)
    return job_worker.JobWorker(JobStore(db.connect), storage, **kw)


def as_teacher(db, sql: str, args: tuple = ()):
    conn = db.connect()
    try:
        cur = conn.cursor()
        cur.execute("begin; select tst.as_user(%s)", (T1,))
        cur.execute(sql, args)
        row = cur.fetchone()
        cur.execute("commit")
        return row[0] if row else None
    finally:
        conn.close()


def sweep(db) -> dict:
    return db.one("select public.a4_checker_sweep()")


def fail_with(monkeypatch, exc):
    def boom(*a, **kw):
        raise exc
    monkeypatch.setattr(copy_check, "grade_copy", boom)


def run_out_of_retries(db, storage, job):
    w = worker(db, storage)
    for _ in range(3):
        db.due_now(job)
        w.run()


# ── 1. AI failures wait for another round instead of failing ─────────

def test_round_after_round_then_failed_after_the_last(db, storage, monkeypatch):
    sub, job = make_attempt(db, storage, PDF)
    fail_with(monkeypatch, svc.GeminiBusy(503))
    run_out_of_retries(db, storage, job)
    j = db.job(job)
    assert (j["state"], j["retry_rounds"], j["last_error_code"]) == ("failed_retryable", 0, "GEMINI_5XX")
    assert db.sub(sub)["status"] == "queued"

    assert sweep(db)["retried"] == 0                     # not due yet (30 min)
    db.due_now(job)
    assert sweep(db)["retried"] == 1
    j = db.job(job)
    assert (j["state"], j["retries"], j["retry_rounds"]) == ("queued", 0, 1)

    db.q("update check_jobs set retry_rounds = public.a4_retry_rounds_max() where id = %s", (job,))
    run_out_of_retries(db, storage, job)
    assert db.job(job)["state"] == "failed"
    assert db.sub(sub)["status"] == "failed"             # teacher's Check again works as before


def test_retryable_copy_is_checked_in_a_later_round(db, storage, monkeypatch):
    sub, job = make_attempt(db, storage, PDF)
    real = copy_check.grade_copy
    fail_with(monkeypatch, TimeoutError("read timed out"))
    run_out_of_retries(db, storage, job)
    assert db.job(job)["state"] == "failed_retryable" and db.job(job)["last_error_code"] == "TIMEOUT"

    monkeypatch.setattr(copy_check, "grade_copy", real)
    db.due_now(job)
    sweep(db)
    worker(db, storage).run()
    assert db.job(job)["state"] == "done"
    assert db.sub(sub)["status"] in ("checked", "needs_review")


def test_bad_key_is_gemini_config_and_waits(db, storage, monkeypatch):
    sub, job = make_attempt(db, storage, PDF)
    fail_with(monkeypatch, svc.GeminiConfigError(403, "gemini-3.8-flash"))
    run_out_of_retries(db, storage, job)
    j = db.job(job)
    assert (j["state"], j["last_error_code"]) == ("failed_retryable", "GEMINI_CONFIG")
    assert db.sub(sub)["status"] == "queued"


def test_other_errors_still_fail_for_a_human_retry(db, storage, monkeypatch):
    sub, job = make_attempt(db, storage, PDF)
    fail_with(monkeypatch, RuntimeError("bug"))
    run_out_of_retries(db, storage, job)
    assert db.job(job)["state"] == "failed" and db.sub(sub)["status"] == "failed"


def test_stuck_run_out_of_retries_becomes_retryable(db, storage):
    _, job = make_attempt(db, storage, PDF)
    db.q("update check_jobs set state = 'running', retries = max_retries, locked_at = now() - interval '7 minutes' "
         "where id = %s", (job,))
    sweep(db)
    assert db.job(job)["state"] == "failed_retryable"


# ── 2. flagged copies stay hidden until the teacher approves / edits ──

def _checked(db, storage, asg=A1, student=None):
    kw = {"student": student} if student else {}
    sub, job = make_attempt(db, storage, PDF, asg=asg, **kw)
    worker(db, storage).run()
    assert db.sub(sub)["status"] == "needs_review", db.sub(sub)       # the messy fixture is always flagged
    return sub


def test_publish_all_skips_flagged_copies(db, storage):
    sub = _checked(db, storage)
    r = as_teacher(db, "select public.publish_results(%s::uuid, null)", (A1,))
    assert r["published"] == [] and r["blocked"] == [{"id": sub, "status": "needs_review", "reason": "needs_review"}]
    assert db.sub(sub)["status"] == "needs_review" and db.sub(sub)["grade"] is None
    r = as_teacher(db, "select public.publish_results(null, %s::uuid[])", ([sub],))     # the row's Publish too
    assert r["published"] == []


def test_approve_review_publishes_and_is_audited(db, storage):
    sub = _checked(db, storage)
    r = as_teacher(db, "select public.approve_review(%s)", (sub,))
    assert r["state"] == "ok" and r["total"] is not None
    s, sh = db.sub(sub), db.sheet(sub)
    assert s["status"] == "published" and sh["review_approved_at"] and sh["review_approved_by"] == T1
    assert db.one("select count(*) from smart_assignment_checks where submission_id = %s "
                  "and result_json->>'type' = 'publish'", (sub,)) == 1
    assert as_teacher(db, "select public.approve_review(%s)", (sub,))["state"] == "already_published"


def test_an_edit_counts_as_the_review(db, storage):
    sub = _checked(db, storage)
    q = db.sheet(sub)["ai_result"]["graded_questions"][0]
    r = as_teacher(db, "select public.override_marks(%s, %s, %s, 'checked by hand')", (sub, q["q_no"], 0))
    assert r["state"] == "ok" and db.sheet(sub)["review_approved_at"]
    r = as_teacher(db, "select public.publish_results(%s::uuid, null)", (A1,))
    assert r["published"] == [sub]


def test_approve_review_is_refused_for_other_teachers_and_anon(db, storage):
    sub = _checked(db, storage)
    conn = db.connect()
    try:
        cur = conn.cursor()
        cur.execute("begin; select tst.as_user('00000000-0000-0000-0000-0000000000ff')")
        cur.execute("select public.approve_review(%s)", (sub,))
        assert cur.fetchone()[0]["state"] == "forbidden"
        cur.execute("rollback")
    finally:
        conn.close()
    assert db.one("select has_function_privilege('anon', 'public.approve_review(uuid)', 'execute')") is False


def test_already_published_copies_are_not_touched_by_the_migration(db, storage):
    """26 adds columns + rules for future publishing only: a copy published before stays published."""
    sub = _checked(db, storage)
    db.q("select public.a4_publish_submission(%s, null, false)", (sub,))
    db.q("update check_sheets set review_approved_at = null where submission_id = %s", (sub,))
    sweep(db)
    assert db.sub(sub)["status"] == "published"


# ── 3. a backup model is never silent ────────────────────────────────

def test_backup_model_copy_is_flagged_and_not_auto_published(db, storage, monkeypatch):
    from app.services.answer_checker import portal_result
    real = copy_check.grade_copy

    def by_backup(paths, rubric, strictness="medium"):
        result, ordered, usage = real(paths, rubric, strictness)
        return result, ordered, {**usage, "model": "gemini-3-flash-preview", "fallback_used": True}

    monkeypatch.setattr(copy_check, "grade_copy", by_backup)
    monkeypatch.setattr(portal_result, "review", lambda result, rubric, _r=portal_result.review: (
        [x for x in _r(result, rubric)[0] if x == "fallback_model"], 0.95, {}))
    sub, _ = make_attempt(db, storage, PDF, asg=A9, student=S2)          # A9 = auto_publish
    worker(db, storage).run()
    sh = db.sheet(sub)
    assert sh["needs_review_reasons"] == ["fallback_model"] and sh["model_version"] == "gemini-3-flash-preview"
    assert db.sub(sub)["status"] == "needs_review"                       # never auto-published


# ── 4. health view + alert ────────────────────────────────────────────

def test_health_view_and_sweep_alert_on_old_jobs(db, storage):
    _, job = make_attempt(db, storage, PDF)
    db.q("update check_jobs set created_at = now() - interval '40 minutes', next_run_at = now() + interval '1 hour' "
         "where id = %s", (job,))
    h = db.one("select row_to_json(h) from checker_job_health h where job_id = %s", (job,))
    assert h["job_type"] == "copy" and h["assignment_id"] == A1 and h["age_min"] >= 40 and h["state"] == "queued"
    assert sweep(db)["stale"] >= 1
    assert [r["job_id"] for r in JobStore(db.connect).stale_jobs(15)] == [job]


# ── 5. shadow re-grades: service only ────────────────────────────────

def test_shadow_results_are_invisible_to_teachers(db, storage):
    sub = _checked(db, storage)
    db.q("""insert into check_shadow_results (run_label, submission_id, assignment_id, ai_result, ai_marks)
            values ('t', %s, %s, '{}'::jsonb, 1)""", (sub, A1))
    conn = db.connect()
    try:
        cur = conn.cursor()
        cur.execute("begin; select tst.as_user(%s)", (T1,))
        with pytest.raises(Exception, match="permission denied"):
            cur.execute("select count(*) from check_shadow_results")
        cur.execute("rollback")
    finally:
        conn.close()
    assert db.sub(sub)["status"] == "needs_review"                      # nothing visible changed
