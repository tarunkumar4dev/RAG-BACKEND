"""
Job worker end to end on the real 18_upload_flow.sql queue (see conftest.py): claim, finish, backoff, state →
submission status triggers, quota, budget, chunks, retention, regrade.
"""

import io
import uuid
import logging
import threading
import time
from datetime import datetime, timedelta, timezone

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.services import job_worker, smart_assignment_service as svc
from app.services.answer_checker import copy_check
from app.services.job_store import JobStore
from app.services.regrade import regrade
from tests.integration.conftest import A1, A9, FIXTURES, S1, S2, T1, make_attempt

SHEET = (FIXTURES / "answer_sheet_messy.pdf").read_bytes()      # 7 written pages
IST = timezone(timedelta(hours=5, minutes=30))


def worker(db, storage, **kw) -> job_worker.JobWorker:
    kw.setdefault("budget_s", 10_000)          # tests never run out of time
    return job_worker.JobWorker(JobStore(db.connect), storage, **kw)


def pdf_pages(n: int) -> bytes:
    """n written pages (the fixture's pages repeated)."""
    src = pymupdf.open(stream=SHEET, filetype="pdf")
    out = pymupdf.open()
    while out.page_count < n:
        out.insert_pdf(src, from_page=0, to_page=min(src.page_count, n - out.page_count) - 1)
    return out.tobytes()


def page_jpeg(i: int = 0) -> bytes:
    doc = pymupdf.open(stream=SHEET, filetype="pdf")
    return doc[i].get_pixmap(dpi=110).tobytes("jpeg")


@pytest.fixture
def grades(monkeypatch):
    """Counts real (mock-Gemini) grade_copy calls."""
    calls = []
    real = copy_check.grade_copy

    def counting(paths, rubric, strictness="medium"):
        out = real(paths, rubric, strictness)
        calls.append(out[2]["pages"])             # pages graded in this call
        return out

    monkeypatch.setattr(copy_check, "grade_copy", counting)
    return calls


# ── happy path + no recompute ────────────────────────────────────────

def test_copy_is_checked_and_marks_stored(db, storage, grades):
    sub, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    assert db.sub(sub)["status"] == "queued"
    stats = worker(db, storage).run()
    assert stats["jobs"] == 3 and grades == [7]          # copy waits → rubric job → copy
    j, s, sh = db.job(job), db.sub(sub), db.sheet(sub)
    assert j["state"] == "done" and j["checkpoint"]["graded"] is True
    assert sh["status"] == "checked" and sh["ai_marks"] is not None and sh["model_version"] == "mock"
    assert 0 < float(sh["confidence"]) <= 1
    assert sh["ai_result"]["meta"]["answer_key_version"] == 1
    assert len(sh["ai_result"]["norm_pages"]) == 7                       # red pen draws on these
    assert sh["per_question"] and {"q_no", "marks", "feedback"} <= set(sh["per_question"][0])
    assert s["status"] == ("needs_review" if sh["needs_review_reasons"] else "checked")
    assert "empty_answer" in sh["needs_review_reasons"]                 # Q6 of the recording is not attempted
    assert db.one("select count(*) from smart_assignment_checks where submission_id = %s", (sub,)) == 1


def test_retry_after_saved_marks_does_not_call_gemini_again(db, storage, grades, monkeypatch):
    sub, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    real_finish = JobStore.finish
    crashed = []

    class Died(BaseException):                      # the function was killed: nothing in the worker runs after
        pass

    def crash_once(self, job_id, ok, *a, **kw):
        if ok and str(job_id) == job and not crashed:      # the copy's finish (the rubric job finishes normally)
            crashed.append(1)
            raise Died()
        return real_finish(self, job_id, ok, *a, **kw)

    monkeypatch.setattr(JobStore, "finish", crash_once)
    with pytest.raises(Died):
        worker(db, storage).run()                   # saved, then died before finish (job still running)
    assert db.job(job)["state"] == "running" and grades == [7]
    db.q("update check_jobs set state = 'queued', locked_at = null where id = %s", (job,))   # what the sweep does
    worker(db, storage).run()
    assert db.job(job)["state"] == "done" and grades == [7]          # no second grade
    assert db.one("select count(*) from smart_assignment_checks where submission_id = %s", (sub,)) == 1


def test_auto_publish_only_without_review_reasons(db, storage, monkeypatch):
    sub_review, _ = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")], asg=A9)
    worker(db, storage).run()
    assert db.sub(sub_review)["status"] == "needs_review"               # never auto-published

    from app.services.answer_checker import portal_result
    monkeypatch.setattr(portal_result, "review", lambda result, rubric: ([], 0.95, {}))
    sub_clean, _ = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")], asg=A9, student=S2)
    worker(db, storage).run()
    s = db.sub(sub_clean)
    assert s["status"] == "published" and s["grade"] is not None
    assert db.one("select result_json->>'type' from smart_assignment_checks where submission_id = %s "
                  "and result_json->>'type' is not null", (sub_clean,)) == "auto_publish"


# ── concurrency ──────────────────────────────────────────────────────

def test_two_parallel_run_calls_check_each_job_once(db, storage, grades, monkeypatch):
    from app.main import app
    from app.routers import internal_jobs_router

    monkeypatch.setenv("JOB_SECRET", "s3cret")
    make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()        # builds the session rubric first (a second job meanwhile would wait 20 s)
    grades.clear()
    for st in (S1, S2, S1, S2):
        make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")], student=st)
    monkeypatch.setattr(internal_jobs_router, "_worker", lambda: worker(db, storage))
    barrier = threading.Barrier(2)
    codes = []

    def kick():
        with TestClient(app) as client:
            barrier.wait()
            codes.append(client.post("/internal/jobs/run", headers={"X-Job-Secret": "s3cret"}).status_code)

    threads = [threading.Thread(target=kick) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    assert codes == [202, 202]
    assert len(grades) == 4                                              # 4 jobs, each graded exactly once
    assert db.one("select count(*) from check_jobs where state = 'done' and submission_id is not null") == 5
    assert db.one("select count(*) from check_jobs where idempotency_key like 'rubric:%%'") == 1
    assert db.one("select max(retries) from check_jobs") == 1            # nobody claimed a job twice
    assert db.one("select count(*) from smart_assignment_checks") == 5


# ── retries / backoff ────────────────────────────────────────────────

def _busy(monkeypatch, retry_after=None):
    def boom(*a, **kw):
        raise svc.GeminiBusy(429, retry_after=retry_after)
    monkeypatch.setattr(copy_check, "grade_copy", boom)


def _seconds_until(db, job_id):
    return db.one("select extract(epoch from next_run_at - now()) from check_jobs where id = %s", (job_id,))


def test_gemini_429_backs_off_then_waits_for_a_later_round_after_3_runs(db, storage, monkeypatch):
    sub, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    _busy(monkeypatch)
    w = worker(db, storage)

    w.run()
    j = db.job(job)
    assert j["state"] == "queued" and j["retries"] == 1 and j["last_error_code"] == "GEMINI_429"
    assert 25 < _seconds_until(db, job) <= 30                             # backoff 30 s
    assert "busy" in j["error"] and db.sub(sub)["status"] == "queued"     # student sees 'checking soon'

    assert w.run()["jobs"] == 0                                           # not due yet
    db.due_now(job)
    w.run()
    assert db.job(job)["retries"] == 2 and 55 < _seconds_until(db, job) <= 60     # 60 s

    db.due_now(job)
    w.run()
    j = db.job(job)
    # 26_checker_p0: an AI failure is never final by itself — the sweep tries again in 30 min
    assert j["state"] == "failed_retryable" and j["retries"] == 3 and j["last_error_code"] == "GEMINI_429"
    assert j["error"] == "AI not available after 3 tries (AI busy). It will be checked again automatically."
    assert 1790 < _seconds_until(db, job) <= 1800
    assert db.sub(sub)["status"] == "queued"                              # nothing published, student waits
    assert db.sheet(sub)["status"] != "failed"


def test_retry_after_from_gemini_is_respected(db, storage, monkeypatch):
    _, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    _busy(monkeypatch, retry_after=95)
    worker(db, storage).run()
    assert 90 < _seconds_until(db, job) <= 95


# ── validation → rejected (does not use up an attempt) ───────────────

def _rejected(db, sub):
    s = db.sub(sub)
    assert s["status"] == "rejected"
    return s["reject_reasons"]


def test_31_page_pdf_is_rejected(db, storage, grades):
    sub, job = make_attempt(db, storage, [(pdf_pages(31), "pdf", "application/pdf")])
    worker(db, storage).run()
    reasons = _rejected(db, sub)
    assert reasons[0] == {"code": "too_many_pages", "page_order": 0,
                          "message": "The copy has 31 pages; max 30 allowed. Remove extra pages and upload again."}
    assert db.job(job)["last_error_code"] == "BAD_FILE" and grades == []


def test_page_cap_follows_the_assignment(db, storage, grades):
    db.q("update assignments set max_pages = 5 where id = %s", (A1,))
    sub, _ = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()
    assert "max 5 allowed" in _rejected(db, sub)[0]["message"]


def test_fake_jpg_with_pdf_inside_is_rejected(db, storage, grades):
    sub, _ = make_attempt(db, storage, [(page_jpeg(0), "jpg", "image/jpeg"), (SHEET, "jpg", "image/jpeg")])
    worker(db, storage).run()
    reasons = _rejected(db, sub)
    assert [r["page_order"] for r in reasons] == [2]                     # page 1 was fine
    assert reasons[0]["message"] == "Page 2: named .jpg but is a PDF. Upload the original file without renaming it."
    assert grades == []


def test_password_pdf_is_rejected(db, storage, grades):
    doc = pymupdf.open(stream=SHEET, filetype="pdf")
    locked = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="1234", owner_pw="owner")
    sub, _ = make_attempt(db, storage, [(locked, "pdf", "application/pdf")])
    worker(db, storage).run()
    assert _rejected(db, sub)[0]["message"] == "Page 1: the PDF is password-protected. Upload it without a password."
    assert grades == []


def test_heic_copy_is_checked(db, storage, grades):
    pillow_heif = pytest.importorskip("pillow_heif")
    from PIL import Image
    pillow_heif.register_heif_opener()
    db.q("update assignments set allowed_mimes = allowed_mimes || array['image/heic'] where id = %s", (A1,))
    heics = []
    for i in range(2):
        im = Image.open(io.BytesIO(page_jpeg(i)))
        buf = io.BytesIO()
        im.save(buf, format="HEIF", quality=85)
        heics.append((buf.getvalue(), "heic", "image/heic"))
    sub, job = make_attempt(db, storage, heics)
    worker(db, storage).run()
    assert db.job(job)["state"] == "done", db.job(job)
    assert db.sub(sub)["status"] in ("checked", "needs_review") and grades == [2]


def test_exif_gps_is_removed_from_the_stored_photo(db, storage, grades):
    from PIL import Image
    im = Image.open(io.BytesIO(page_jpeg(0)))
    exif = Image.Exif()
    exif[0x0110] = "Pixel 9"                                   # camera model
    exif[0x8825] = {1: "N", 2: (28.0, 36.0, 0.0), 3: "E", 4: (77.0, 12.0, 0.0)}   # GPS
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    sub, job = make_attempt(db, storage, [(buf.getvalue(), "jpg", "image/jpeg")])
    worker(db, storage).run()
    assert db.job(job)["state"] == "done"
    path = db.sub(sub)["files"][0]["storage_path"]
    with Image.open(io.BytesIO(storage.download("submissions", path))) as stored:
        assert not stored.getexif() and b"Pixel 9" not in storage.download("submissions", path)


def test_same_file_as_another_student_is_flagged_for_the_teacher(db, storage):
    make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")], student=S2)
    sub, _ = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()
    flags = db.sheet(sub)["ai_result"]["sheet_flags"]
    assert any(f["code"] == "DUPLICATE_OTHER_STUDENT" for f in flags)
    assert "DUPLICATE" not in str(db.sheet(sub)["per_question"])         # never in what the student sees


# ── long copies ──────────────────────────────────────────────────────

def test_20_page_copy_is_checked_in_2_chunks_and_merged(db, storage, grades):
    sub, job = make_attempt(db, storage, [(pdf_pages(20), "pdf", "application/pdf")])
    worker(db, storage).run()
    chunks = db.q("select state, checkpoint->>'chunk', jsonb_array_length(checkpoint->'pages') from check_jobs "
                  "where checkpoint->>'parent' = %s order by 2", (job,))
    assert chunks == [("done", "0", 10), ("done", "1", 10)]
    assert sorted(grades) == [10, 10]                                    # one grade call per chunk
    j, sh = db.job(job), db.sheet(sub)
    assert j["state"] == "done" and j["retries"] == 1                    # parking didn't use up retries
    assert sh["ai_result"]["meta"]["chunks"] == 2
    assert [p["index"] for p in sh["ai_result"]["norm_pages"]] == list(range(20))
    gq = sh["ai_result"]["graded_questions"]
    assert len({g["q_no"] for g in gq}) == len(gq)                        # merged by question, no duplicates
    assert all(g["page_index"] is None or 0 <= g["page_index"] < 20 for g in gq)
    assert float(sh["ai_marks"]) <= float(sh["max_marks"])
    assert db.sub(sub)["status"] in ("checked", "needs_review")
    assert db.one("select count(*) from smart_assignment_checks where submission_id = %s", (sub,)) == 1


def test_parked_long_copy_shows_checking(db, storage, grades, monkeypatch):
    sub, job = make_attempt(db, storage, [(pdf_pages(20), "pdf", "application/pdf")])
    w = worker(db, storage)
    for _ in range(3):                                                   # copy waits → rubric job → copy splits
        w.handle(w.store.claim(5))
    assert db.job(job)["state"] == "queued" and db.sub(sub)["status"] == "checking"


# ── limits ───────────────────────────────────────────────────────────

def _audit_rows(db, n, kind=None):
    db.q("""insert into smart_assignment_checks (id, teacher_id, result_json)
            select gen_random_uuid(), %s, case when %s::text is null then '{}'::jsonb
                                               else jsonb_build_object('type', %s::text) end
              from generate_series(1, %s)""", (T1, kind, kind, n))


def test_quota_exceeded_waits_for_next_ist_day(db, storage, grades):
    _audit_rows(db, 100)
    sub, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage, daily_limit=100).run()
    j = db.job(job)
    assert j["state"] == "blocked_quota" and j["retries"] == 0 and grades == []
    assert db.sub(sub)["status"] == "waiting_quota"
    tomorrow = (datetime.now(IST) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    assert abs(datetime.fromisoformat(j["next_run_at"]).timestamp() - tomorrow.timestamp()) < 2


def test_publish_and_audit_rows_do_not_use_up_the_quota(db, storage, grades):
    _audit_rows(db, 99)
    for kind in ("publish", "auto_publish", "unpublish", "teacher_override", "regrade"):
        _audit_rows(db, 20, kind)
    sub, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage, daily_limit=100).run()
    assert db.job(job)["state"] == "done" and grades == [7]


def test_checks_today_route_filter_excludes_audit_types(monkeypatch):
    """The PostgREST count used by the teacher routes excludes the same types."""
    from app.routers import smart_assignment_router as r
    seen = {}

    class Q:
        def __getattr__(self, name):
            def f(*a, **kw):
                seen.setdefault(name, []).append(a)
                return self
            return f

        def execute(self):
            return type("R", (), {"count": 3})()

    monkeypatch.setattr(r, "get_supabase_admin", lambda: type("C", (), {"table": lambda self, t: Q()})())
    assert r._checks_today(T1) == 3
    flt = seen["or_"][0][0]
    for t in ("teacher_override", "publish", "auto_publish", "unpublish", "regrade"):
        assert t in flt
    assert "+05:30" in seen["gte"][0][1]                                 # IST day


def test_daily_cost_cap_pauses_new_jobs_and_alerts(db, storage, grades, caplog):
    db.q("""insert into check_jobs (kind, state, usage, idempotency_key, finished_at)
            values ('retention', 'done', '{"est_cost_inr": 151.5}', 'spent-today', now())""")
    sub, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    with caplog.at_level(logging.ERROR, logger="app.services.job_worker"):
        stats = worker(db, storage, daily_budget_inr=150).run()
    assert stats == {"jobs": 0, "paused_budget": True} and grades == []
    assert db.job(job)["state"] == "queued" and db.sub(sub)["status"] == "queued"
    assert any("ALERT" in r.message and "budget" in r.message for r in caplog.records)
    worker(db, storage, daily_budget_inr=1000).run()                     # cap raised → runs again
    assert db.job(job)["state"] == "done"


def test_job_usage_carries_an_estimated_cost(db, storage):
    _, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()
    u = db.job(job)["usage"]
    assert u["gemini_calls"] == 1 and "est_cost_inr" in u              # the grade call only
    ru = db.one("select usage from check_jobs where idempotency_key like 'rubric:%%'")
    assert ru["gemini_calls"] == 1 and "est_cost_inr" in ru            # the rubric call, on its own job


# ── retention ────────────────────────────────────────────────────────

def test_retention_deletes_unattached_uploads_through_storage(db, storage):
    import uuid
    from tests.integration.conftest import INST
    old = [f"{INST}/{A1}/{S1}/{uuid.uuid4()}.jpg" for _ in range(3)]
    for p in old:
        storage.put("submissions", p, b"x")
        db.q("insert into storage.objects (bucket_id, name, created_at) values ('submissions', %s, now() - interval '2 days')",
             (p,))
    sub, _ = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])          # attached: kept
    kept = db.sub(sub)["files"][0]["storage_path"]
    db.q("update storage.objects set created_at = now() - interval '2 days' where name = %s", (kept,))
    db.q("update check_jobs set state = 'cancelled'")                   # only the retention job below
    rid = db.one("select public.a4_enqueue_orphan_cleanup()")
    assert rid
    worker(db, storage).run()
    assert sorted(p for _, p in storage.removed) == sorted(old)
    assert db.job(rid)["state"] == "done" and db.job(rid)["usage"]["deleted"] == 3
    assert storage.download("submissions", kept) is not None


# ── regrade ──────────────────────────────────────────────────────────

def _unpublish_as_teacher(db):
    def unpublish(ids, reason):
        conn = db.connect()
        try:
            cur = conn.cursor()
            cur.execute("begin; select tst.as_user(%s)", (T1,))
            cur.execute("select public.unpublish_results(%s::uuid[], %s)", (ids, reason))
            r = cur.fetchone()[0]
            cur.execute("commit")
            return r
        finally:
            conn.close()
    return unpublish


def test_regrade_after_key_change_keeps_old_marks_in_audit(db, storage, grades):
    sub, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()
    db.q("select public.a4_publish_submission(%s, %s, false)", (sub, T1))
    old = db.sub(sub)
    assert old["status"] == "published"
    storage.put("answer-keys", "00000000-0000-0000-0000-0000000000a2/key-v2.pdf",
                (FIXTURES / "answer_key.pdf").read_bytes())
    db.q("update assignment_ai_keys set answer_key_path = '00000000-0000-0000-0000-0000000000a2/key-v2.pdf' "
         "where assignment_id = %s", (A1,))

    r = regrade(JobStore(db.connect), A1, [sub], T1, _unpublish_as_teacher(db))
    assert r["state"] == "ok" and [q["id"] for q in r["queued"]] == [sub] and r["answer_key_version"] == 2
    audit = db.one("select result_json from smart_assignment_checks where submission_id = %s "
                   "and result_json->>'type' = 'regrade'", (sub,))
    assert float(audit["previous_marks"]) == float(old["grade"]) and audit["previous_answer_key_version"] == 1
    assert audit["previous_per_question"]
    assert db.one("select count(*) from smart_assignment_checks where submission_id = %s "
                  "and result_json->>'type' = 'unpublish'", (sub,)) == 1
    assert db.sub(sub)["status"] == "queued" and db.sub(sub)["grade"] is None

    worker(db, storage).run()
    assert len(grades) == 2 and db.sheet(sub)["ai_result"]["meta"]["answer_key_version"] == 2
    assert db.sub(sub)["status"] in ("checked", "needs_review")
    assert db.one("select count(*) from check_jobs where submission_id = %s and state = 'done'", (sub,)) == 2

    again = regrade(JobStore(db.connect), A1, [sub], T1, _unpublish_as_teacher(db))
    assert again["queued"] == [] and again["skipped"] == [{"id": sub, "reason": "key_unchanged"}]


def test_rubric_timeout_is_retried_not_key_missing(db, storage, monkeypatch):
    import httpx
    def hang(*a, **kw):
        raise httpx.ReadTimeout("model did not answer")
    monkeypatch.setattr(svc, "build_rubric", hang)
    sub, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()
    rj = db.one("select row_to_json(j) from check_jobs j where idempotency_key like 'rubric:%%'")
    assert rj["state"] == "queued" and rj["last_error_code"] == "TIMEOUT" and rj["retries"] == 1
    j = db.job(job)                                       # the copy waits; none of its retries used
    assert j["state"] == "queued" and j["retries"] == 0 and j["last_error_code"] is None
    assert db.sub(sub)["status"] == "queued"


def test_teacher_retry_restarts_a_long_copy_whose_chunk_failed(db, storage, grades, monkeypatch):
    """19_upload_link_portal.retry_checks re-opens the copy's job but keeps its checkpoint (stage 'chunks')."""
    if not db.one("select to_regprocedure('public.retry_checks(uuid[])') is not null"):
        pytest.skip("migration 19 not in this checkout")
    sub, job = make_attempt(db, storage, [(pdf_pages(20), "pdf", "application/pdf")])
    real = copy_check.grade_copy
    calls = {"n": 0}

    def second_chunk_breaks(paths, rubric, strictness="medium"):
        calls["n"] += 1
        if calls["n"] == 2:
            raise svc.CallCapExceeded("cap")                 # final for that chunk
        return real(paths, rubric, strictness)

    monkeypatch.setattr(copy_check, "grade_copy", second_chunk_breaks)
    worker(db, storage).run()
    assert db.sub(sub)["status"] == "failed" and db.job(job)["state"] == "failed"

    conn = db.connect()
    try:
        cur = conn.cursor()
        cur.execute("begin; select tst.as_user(%s)", (T1,))
        cur.execute("select public.retry_checks(%s::uuid[])", ([sub],))
        r = cur.fetchone()[0]
        cur.execute("commit")
    finally:
        conn.close()
    assert r["queued"] == [sub]
    monkeypatch.setattr(copy_check, "grade_copy", real)
    worker(db, storage).run()
    assert db.job(job)["state"] == "done", db.job(job)
    assert db.sub(sub)["status"] in ("checked", "needs_review")
    assert db.sheet(sub)["ai_result"]["meta"]["chunks"] == 2



# ── rubric = its own job, once per assignment + key version ──────────

def test_rubric_is_built_once_for_many_copies(db, storage, grades, monkeypatch):
    builds = []
    real = svc.build_rubric

    def counting(*a, **kw):
        builds.append(svc._call_timeout.get())
        return real(*a, **kw)

    monkeypatch.setattr(svc, "build_rubric", counting)
    subs = [make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")], student=st)[0] for st in (S1, S2, S1)]
    worker(db, storage).run()
    assert builds == [job_worker.RUBRIC_CALL_TIMEOUT_S]                 # once, with the rubric stage timeout
    assert all(db.sub(s)["status"] in ("checked", "needs_review") for s in subs) and len(grades) == 3
    assert db.one("select max(retries) from check_jobs where submission_id is not null") == 1
    rj = db.one("select row_to_json(j) from check_jobs j where idempotency_key like 'rubric:%%'")
    version = db.one("select version from assignment_ai_keys where assignment_id = %s", (A1,))
    assert rj["state"] == "done" and rj["checkpoint"]["key_version"] == version and rj["submission_id"] is None


def test_grade_call_uses_the_grade_stage_timeout(db, storage, monkeypatch):
    seen = []
    real = copy_check.grade_copy

    def spy(*a, **kw):
        seen.append(svc._call_timeout.get())
        return real(*a, **kw)

    monkeypatch.setattr(copy_check, "grade_copy", spy)
    make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()
    assert seen == [job_worker.GRADE_CALL_TIMEOUT_S]
    assert job_worker.WORK_BUDGET_S - job_worker.MIN_LEFT_TO_CLAIM_S + job_worker.RUBRIC_CALL_TIMEOUT_S < 300


def test_failed_rubric_sends_copies_to_waiting_key_then_new_key_checks_them(db, storage, grades):
    storage.buckets["answer-keys"].clear()                       # key file unreadable → rubric fails for good
    v1 = db.one("select version from assignment_ai_keys where assignment_id = %s", (A1,))
    subs = [make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")], student=st) for st in (S1, S2)]
    worker(db, storage).run()
    rj = db.one("select row_to_json(j) from check_jobs j where idempotency_key like 'rubric:%%'")
    assert rj["state"] == "failed" and rj["last_error_code"] == "KEY_MISSING"
    for sub, job in subs:
        assert db.sub(sub)["status"] == "waiting_key"
        j = db.job(job)
        assert j["error"] == "Marking scheme could not be built (key unreadable). Attach the answer key again to retry."
        assert db.sheet(sub)["error"] == j["error"]               # the teacher sees why
    assert grades == []

    storage.put("answer-keys", "00000000-0000-0000-0000-0000000000a2/key-new.pdf",
                (FIXTURES / "answer_key.pdf").read_bytes())
    db.q("update assignment_ai_keys set answer_key_path = '00000000-0000-0000-0000-0000000000a2/key-new.pdf' "
         "where assignment_id = %s", (A1,))                     # version 2 → 18's trigger re-queues waiting_key
    worker(db, storage).run()
    assert all(db.sub(sub)["status"] in ("checked", "needs_review") for sub, _ in subs) and len(grades) == 2
    assert db.one("select count(*) from check_jobs where idempotency_key like %s and state = 'done'",
                  (f"rubric:%:k{v1 + 1}",)) == 1


def test_stage_timings_are_logged_with_the_job_id(db, storage, caplog):
    _, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    with caplog.at_level(logging.INFO):
        worker(db, storage).run()
    tag = f"job={job[:8]}"
    rj = db.one("select id::text from check_jobs where idempotency_key like 'rubric:%%'")
    lines = [r.getMessage() for r in caplog.records]
    assert any(f"job={rj[:8]} stage=rubric_gemini" in m for m in lines)
    assert any(f"job={rj[:8]} rubric tokens" in m for m in lines)       # the Gemini call line carries the job id
    for name in ("download_validate", "grade page_prep=", "save"):
        assert any(f"{tag} stage={name}" in m for m in lines), name
    assert any(f"{tag} grade tokens" in m for m in lines)
    assert {"download_validate", "page_prep", "gemini_grade", "grade_total"} <= set(db.job(job)["usage"]["stages"])



# ── guest attempts on an open link (23_open_link: student_id null, owner = submitter_key) ──

def test_guest_submission_rubric_check_publish(db, storage, grades):
    from tests.integration.conftest import make_guest_attempt
    sub, job, key = make_guest_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    s = db.sub(sub)
    assert s["student_id"] is None and s["status"] == "queued"
    worker(db, storage).run()
    rj = db.one("select row_to_json(j) from check_jobs j where idempotency_key like 'rubric:%%'")
    assert rj["state"] == "done"                                          # rubric job first …
    j, sh = db.job(job), db.sheet(sub)
    assert j["state"] == "done" and grades == [7]                         # … then the check
    assert sh["status"] == "checked" and sh["student_name"] == "Ravi Guest" and sh["student_id"] is None
    assert db.sub(sub)["status"] in ("checked", "needs_review")
    db.q("select public.a4_publish_submission(%s, %s, false)", (sub, T1))
    s = db.sub(sub)
    assert s["status"] == "published" and s["grade"] is not None
    assert db.one("select count(*) from smart_assignment_checks where submission_id = %s", (sub,)) == 2  # check + publish


def test_guest_files_outside_their_own_folder_are_never_read(db, storage, grades):
    from tests.integration.conftest import INST, make_guest_attempt
    sub, job, key = make_guest_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    other = f"{INST}/{A1}/guest/{uuid.uuid4()}/{uuid.uuid4()}.pdf"      # another guest's folder
    storage.put("submissions", other, SHEET)
    db.q("""update submissions set files = jsonb_set(files, '{0,storage_path}', to_jsonb(%s::text)) where id = %s""",
         (other, sub))
    db.q("update check_sheets set pages = array[%s] where submission_id = %s", (other, sub))
    worker(db, storage).run()
    assert db.sub(sub)["status"] == "rejected" and grades == []
    assert db.sub(sub)["reject_reasons"][0]["code"] == "not_uploaded"


def test_duplicate_flag_compares_owners_including_guests(db, storage):
    from tests.integration.conftest import make_guest_attempt
    make_guest_attempt(db, storage, [(SHEET, "pdf", "application/pdf")], name="First Guest")
    sub2, _, _ = make_guest_attempt(db, storage, [(SHEET, "pdf", "application/pdf")], name="Second Guest")
    worker(db, storage).run()
    assert any(f["code"] == "DUPLICATE_OTHER_STUDENT" for f in db.sheet(sub2)["ai_result"]["sheet_flags"])
    # the same guest's own earlier upload is not "another student"
    store = JobStore(db.connect)
    key2 = db.sub(sub2)["submitter_key"]
    sha = db.sub(sub2)["files"][0]["sha256"]
    assert store.other_students_with(A1, None, key2, [sha]) == [sha]          # first guest has it
    db.q("delete from submissions where guest_name = 'First Guest'")
    assert store.other_students_with(A1, None, key2, [sha]) == []


# ── what the checker spent is written down (answer_checker/usage_cost.py) ─────────────────────────

def _bill(inp: int, out: int, tag: str = "RUBRIC") -> None:
    """One billed reply on the active usage meter, the way _generate counts it."""
    from types import SimpleNamespace
    svc._count_call(tag)
    svc._log_usage(tag, "gemini-3.8-flash", SimpleNamespace(candidates=[], usage_metadata=SimpleNamespace(
        prompt_token_count=inp, candidates_token_count=out, total_token_count=inp + out, thoughts_token_count=0)), 1.0)


def test_rubric_carries_what_it_cost(db, storage, monkeypatch):
    real = svc.build_rubric

    def billing(*a, **kw):
        rubric = real(*a, **kw)
        _bill(10_000, 2_000)
        return rubric

    monkeypatch.setattr(svc, "build_rubric", billing)
    make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()
    ju = db.one("select usage from check_jobs where idempotency_key like 'rubric:%%'")
    b = db.one("""select rubric -> 'build' from check_sessions
                   where id = (select session_id from check_jobs where idempotency_key like 'rubric:%%')""")
    assert b["source"] == "job" and b["input_tokens"] >= 10_000 and "stages" not in b
    for k in ("gemini_calls", "billed_calls", "input_tokens", "output_tokens", "thinking_tokens", "est_cost_inr", "model"):
        assert b[k] == ju[k], k                                 # the session says what the rubric job's row says


def test_failed_rubric_run_keeps_its_billed_usage(db, storage, monkeypatch):
    import httpx

    def reply_then_timeout(*a, **kw):
        _bill(8_000, 1_000)
        raise httpx.ReadTimeout("the next call did not answer")

    monkeypatch.setattr(svc, "build_rubric", reply_then_timeout)
    make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()
    rj = db.one("select row_to_json(j) from check_jobs j where idempotency_key like 'rubric:%%'")
    assert rj["state"] == "queued" and rj["last_error_code"] == "TIMEOUT"
    u = rj["usage"]
    assert (u["input_tokens"], u["output_tokens"], u["billed_calls"]) == (8_000, 1_000, 1) and u["est_cost_inr"] > 0


def test_failed_grade_run_keeps_its_billed_usage(db, storage, monkeypatch):
    def over_the_cap(*a, **kw):
        e = svc.CallCapExceeded("cap")
        e.usage = {"gemini_calls": 2, "billed_calls": 2, "input_tokens": 9_000, "output_tokens": 600,
                   "thinking_tokens": 0, "model": "gemini-3.8-flash"}
        raise e

    monkeypatch.setattr(copy_check, "grade_copy", over_the_cap)
    _, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()
    j = db.job(job)
    assert j["state"] == "failed" and j["last_error_code"] == "CALL_CAP"
    u = j["usage"]
    assert (u["billed_calls"], u["input_tokens"], u["output_tokens"]) == (2, 9_000, 600)
    assert u["est_cost_inr"] == job_worker.est_cost_inr(u) > 0 and u["model"] == "gemini-3.8-flash"


def test_unreadable_replies_keep_their_usage(db, storage, monkeypatch):
    """Two billed replies, neither JSON: what they cost must be on the row. (Whether the job then ends or is
    retried depends on migration 35 making AI_UNREADABLE final; until then the queue SQL re-queues it, and the
    billed-call cap of 3 per copy bounds that. Not asserted here.)"""
    def unreadable(*a, **kw):
        e = svc.UnreadableReply("not json")
        e.usage = {"gemini_calls": 2, "billed_calls": 2, "input_tokens": 9_000, "output_tokens": 600,
                   "thinking_tokens": 0, "model": "gemini-3.8-flash"}
        raise e

    monkeypatch.setattr(copy_check, "grade_copy", unreadable)
    _, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()
    j = db.job(job)
    assert (j["usage"]["billed_calls"], j["usage"]["input_tokens"]) == (2, 9_000)
    assert j["usage"]["est_cost_inr"] > 0


def test_usage_of_a_failed_run_and_the_run_that_succeeds_add_up(db, storage, monkeypatch):
    real = copy_check.grade_copy
    seen = {"n": 0}

    def flaky(paths, rubric, strictness="medium"):
        seen["n"] += 1
        if seen["n"] == 1:
            e = TimeoutError("timed out after a billed reply")
            e.usage = {"gemini_calls": 1, "billed_calls": 1, "input_tokens": 7_000, "output_tokens": 500,
                       "thinking_tokens": 0, "model": "gemini-3.8-flash"}
            raise e
        out = real(paths, rubric, strictness)
        seen["second"] = dict(out[2])
        return out

    monkeypatch.setattr(copy_check, "grade_copy", flaky)
    _, job = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    w = worker(db, storage)
    w.run()
    j = db.job(job)
    assert j["state"] == "queued" and j["usage"]["input_tokens"] == 7_000 and j["usage"]["billed_calls"] == 1
    db.due_now(job)
    w.run()
    j = db.job(job)
    assert j["state"] == "done"
    second = seen["second"]
    assert j["usage"]["input_tokens"] == 7_000 + second["input_tokens"]
    assert j["usage"]["gemini_calls"] == 1 + second["gemini_calls"]
    assert j["usage"]["est_cost_inr"] == job_worker.est_cost_inr(j["usage"])


# ── a failed rebuild writes only rubric.build (JobStore.record_rubric_build, jsonb_set under a row lock) ──

def _built_session(db, storage) -> str:
    """A session whose rubric was built by the rubric job on the mock Gemini."""
    make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()
    sid = db.one("select session_id from check_jobs where idempotency_key like 'rubric:%%'")
    assert db.one("select jsonb_array_length(rubric -> 'questions') from check_sessions where id = %s", (sid,)) > 0
    return sid


def _rubric_of(db, sid) -> dict:
    return db.one("select rubric from check_sessions where id = %s", (sid,))


def test_failed_rebuild_writes_only_rubric_build_and_a_concurrent_edit_survives(db, storage, monkeypatch):
    from psycopg2.extras import Json
    from app.routers import copy_checker_router as ccr
    sid = _built_session(db, storage)
    before = _rubric_of(db, sid)

    def read_session(c, session_id):
        row = db.one("select row_to_json(s) from check_sessions s where id = %s", (session_id,))
        return {**row, "source": "upload", "qp_paths": ["x/q.pdf"]}

    def write_session(c, session_id, fields):          # applies every field it is given, rubric included
        for k, v in fields.items():
            db.q(f"update check_sessions set {k} = %s where id = %s",
                 (Json(v) if isinstance(v, (dict, list)) else v, session_id))
        return fields

    def reply_then_fail(qp, key):
        _bill(8_000, 1_000)
        # a teacher edits a question's marks while the rebuild is still running
        db.q("update check_sessions set rubric = jsonb_set(rubric, '{questions,0,marks}', '99'::jsonb) where id = %s",
             (sid,))
        raise ValueError("the reply was not a rubric")

    monkeypatch.setattr(ccr, "_get_session", read_session)
    monkeypatch.setattr(ccr, "_update_session", write_session)
    monkeypatch.setattr(ccr, "_download_own", lambda c, p, prefix: "/tmp/x.pdf")
    monkeypatch.setattr(ccr, "_cleanup", lambda *p: None)
    monkeypatch.setattr(ccr, "build_rubric", reply_then_fail)
    monkeypatch.setattr(ccr, "_rubric_store", lambda: JobStore(db.connect))
    with pytest.raises(ValueError):
        ccr._build_session_rubric(None, sid, True)

    after = _rubric_of(db, sid)
    assert after["questions"][0]["marks"] == 99                                   # the edit survived
    assert after["build"]["input_tokens"] == before["build"]["input_tokens"] + 8_000
    assert after["build"]["gemini_calls"] == before["build"]["gemini_calls"] + 1
    expected = {k: v for k, v in before.items() if k != "build"}
    expected["questions"][0]["marks"] = 99
    assert {k: v for k, v in after.items() if k != "build"} == expected          # nothing else in the rubric moved
    assert db.one("select rubric_status from check_sessions where id = %s", (sid,)) == "failed"


def test_record_rubric_build_holds_the_row_so_a_concurrent_edit_waits_and_both_survive(db, storage):
    sid = _built_session(db, storage)
    store = JobStore(db.connect)
    locked = threading.Event()

    def slow_merge(build):
        locked.set()
        time.sleep(0.6)                                  # the row is locked while an edit arrives
        return {**(build or {}), "input_tokens": 12_345}

    writer = threading.Thread(target=lambda: store.record_rubric_build(sid, slow_merge))
    writer.start()
    assert locked.wait(10)
    t0 = time.time()
    db.q("update check_sessions set rubric = jsonb_set(rubric, '{questions,0,marks}', '77'::jsonb) where id = %s",
         (sid,))
    waited = time.time() - t0
    writer.join()
    after = _rubric_of(db, sid)
    assert waited > 0.3                                  # the edit queued behind the lock, so the writes were ordered
    assert after["questions"][0]["marks"] == 77 and after["build"]["input_tokens"] == 12_345


def test_record_rubric_build_does_nothing_without_a_rubric(db, storage):
    sid = _built_session(db, storage)
    db.q("update check_sessions set rubric = null where id = %s", (sid,))
    assert JobStore(db.connect).record_rubric_build(sid, lambda b: {"x": 1}) is False
    assert db.one("select rubric is null from check_sessions where id = %s", (sid,)) is True


# ═══════════════════════════════════════════════════════════════════════
# Usage events (migration 36 — the table is created on the fly in the
# test database, since the scratch DB only carries migrations up to 26)
# ═══════════════════════════════════════════════════════════════════════

from app.services.answer_checker.usage_cost import emit_usage_event

_EVENTS_DDL = """\
create table if not exists public.checker_usage_events (
  id              uuid primary key default gen_random_uuid(),
  created_at      timestamptz not null default now(),
  kind            text not null check (kind in ('rubric', 'copy', 'chunk', 'shadow')),
  route           text not null check (route in ('job', 'sync')),
  session_id      uuid,
  sheet_id        uuid,
  job_id          uuid,
  model           text,
  input_tokens    bigint not null default 0,
  output_tokens   bigint not null default 0,
  thinking_tokens bigint not null default 0,
  gemini_calls    int not null default 0,
  billed_calls    int not null default 0,
  est_cost_inr    numeric(10,4) not null default 0,
  ok              boolean not null,
  error_code      text
);
"""


@pytest.fixture
def events_table(db):
    db.q(_EVENTS_DDL)
    db.q("delete from checker_usage_events")
    yield db
    db.q("drop table if exists checker_usage_events")


def test_emit_usage_event_inserts_a_row(events_table):
    db = events_table
    ok = emit_usage_event(
        db.connect, kind="rubric", route="sync", ok=True,
        usage={"input_tokens": 5000, "output_tokens": 1000, "thinking_tokens": 200,
               "gemini_calls": 2, "billed_calls": 2, "model": "gemini-3.8-flash"},
        session_id=str(S1),
    )
    assert ok is True
    row = db.q("select kind, route, session_id, model, input_tokens, output_tokens, ok, error_code "
               "from checker_usage_events limit 1")
    assert row and len(row) == 1
    r = row[0]
    assert r[0] == "rubric" and r[1] == "sync" and str(r[2]) == str(S1)
    assert r[3] == "gemini-3.8-flash" and r[4] == 5000 and r[5] == 1000 and r[6] is True and r[7] is None


def test_emit_usage_event_records_a_failed_copy(events_table):
    db = events_table
    ok = emit_usage_event(
        db.connect, kind="copy", route="job", ok=False,
        usage={"input_tokens": 9000, "output_tokens": 3000, "thinking_tokens": 500,
               "gemini_calls": 1, "billed_calls": 1, "model": "gemini-3.8-flash"},
        session_id=str(S1), sheet_id=str(uuid.uuid4()), job_id=str(uuid.uuid4()),
        error_code="GEMINI_429",
    )
    assert ok is True
    r = db.q("select ok, error_code, est_cost_inr from checker_usage_events limit 1")[0]
    assert r[0] is False and r[1] == "GEMINI_429" and r[2] > 0


def test_emit_returns_false_without_the_table(db):
    db.q("drop table if exists checker_usage_events")
    ok = emit_usage_event(
        db.connect, kind="rubric", route="sync", ok=True,
        usage={"input_tokens": 100},
    )
    assert ok is False


def test_job_worker_run_emits_usage_event(events_table, storage):
    db = events_table
    sub, asg_id, sid, sheet_id = make_attempt(db, storage, SHEET)
    w = worker(db, storage)
    w.run()
    rows = db.q("select kind, route, ok, session_id, sheet_id, job_id from checker_usage_events "
                "where kind in ('rubric', 'copy') order by created_at")
    kinds = [r[0] for r in rows]
    assert "rubric" in kinds and "copy" in kinds
    for r in rows:
        assert r[2] is True             # ok
        assert r[3] is not None          # session_id
        assert r[5] is not None          # job_id

