"""
Job worker end to end on the real 18_upload_flow.sql queue (see conftest.py): claim, finish, backoff, state →
submission status triggers, quota, budget, chunks, retention, regrade.
"""

import io
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
    assert stats["jobs"] == 1 and grades == [7]
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
        if ok and not crashed:
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
    assert db.one("select count(*) from check_jobs where state = 'done'") == 5
    assert db.one("select max(retries) from check_jobs") == 1            # nobody claimed a job twice
    assert db.one("select count(*) from smart_assignment_checks") == 5


# ── retries / backoff ────────────────────────────────────────────────

def _busy(monkeypatch, retry_after=None):
    def boom(*a, **kw):
        raise svc.GeminiBusy(429, retry_after=retry_after)
    monkeypatch.setattr(copy_check, "grade_copy", boom)


def _seconds_until(db, job_id):
    return db.one("select extract(epoch from next_run_at - now()) from check_jobs where id = %s", (job_id,))


def test_gemini_429_backs_off_then_fails_after_3_runs(db, storage, monkeypatch):
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
    assert j["state"] == "failed" and j["retries"] == 3 and j["last_error_code"] == "GEMINI_429"
    assert "3 baar" in j["error"]
    assert db.sub(sub)["status"] == "failed"
    sh = db.sheet(sub)
    assert sh["status"] == "failed" and "3 baar" in sh["error"]          # the teacher sees why


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
    assert reasons[0]["code"] == "too_many_pages" and "31 pages" in reasons[0]["message"]
    assert "max 30" in reasons[0]["message"]
    assert db.job(job)["last_error_code"] == "BAD_FILE" and grades == []


def test_page_cap_follows_the_assignment(db, storage, grades):
    db.q("update assignments set max_pages = 5 where id = %s", (A1,))
    sub, _ = make_attempt(db, storage, [(SHEET, "pdf", "application/pdf")])
    worker(db, storage).run()
    assert "max 5 pages" in _rejected(db, sub)[0]["message"]


def test_fake_jpg_with_pdf_inside_is_rejected(db, storage, grades):
    sub, _ = make_attempt(db, storage, [(page_jpeg(0), "jpg", "image/jpeg"), (SHEET, "jpg", "image/jpeg")])
    worker(db, storage).run()
    reasons = _rejected(db, sub)
    assert [r["page_order"] for r in reasons] == [2]                     # page 1 was fine
    assert ".jpg" in reasons[0]["message"] and "PDF" in reasons[0]["message"]
    assert grades == []


def test_password_pdf_is_rejected(db, storage, grades):
    doc = pymupdf.open(stream=SHEET, filetype="pdf")
    locked = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="1234", owner_pw="owner")
    sub, _ = make_attempt(db, storage, [(locked, "pdf", "application/pdf")])
    worker(db, storage).run()
    assert "password" in _rejected(db, sub)[0]["message"]
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
    w.handle(w.store.claim(5))                                           # only the split
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
    assert u["gemini_calls"] >= 2 and "est_cost_inr" in u              # rubric + grade


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
