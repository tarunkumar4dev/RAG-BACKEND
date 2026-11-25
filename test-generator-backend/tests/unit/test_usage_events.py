"""Usage events (migration 36) and the emit_usage_event helper.

emit_usage_event is best-effort: it never raises, tolerates a missing table, and returns bool.
The sync route and the job worker call it on every rubric / copy / chunk success and failure.
"""
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.routers import copy_checker_router as ccr
from app.services import job_worker as jw
from app.services import smart_assignment_service as svc
from app.services.answer_checker import usage_cost as uc


# ── emit_usage_event unit tests ───────────────────────────────────────────────

class FakeCursor:
    def __init__(self, pgcode=None):
        self.rows = []
        self.pgcode = pgcode

    def execute(self, sql, args=None):
        if self.pgcode:
            exc = Exception("table gone")
            exc.pgcode = self.pgcode
            raise exc
        if args is not None:
            self.rows.append(args)

    def close(self):
        pass


class FakeConn:
    def __init__(self, cursor):
        self._cur = cursor
        self.closed = False

    def cursor(self):
        return self._cur

    def close(self):
        self.closed = True


def test_emit_writes_a_row_and_returns_true():
    cur = FakeCursor()
    conn = FakeConn(cur)
    ok = uc.emit_usage_event(
        lambda: conn, kind="rubric", route="sync", ok=True,
        usage={"input_tokens": 1000, "output_tokens": 200, "thinking_tokens": 50,
               "gemini_calls": 1, "billed_calls": 1, "model": "gemini-3.8-flash"},
        session_id="sess-1",
    )
    assert ok is True
    assert len(cur.rows) == 1
    args = cur.rows[0]
    assert args[0] == "rubric" and args[1] == "sync"
    assert args[2] == "sess-1"           # session_id
    assert args[3] is None               # sheet_id
    assert args[4] is None               # job_id
    assert args[5] == "gemini-3.8-flash" # model
    assert args[6] == 1000               # input_tokens
    assert args[7] == 200                # output_tokens
    assert args[8] == 50                 # thinking_tokens
    assert args[12] is True              # ok
    assert args[13] is None              # error_code
    assert conn.closed


def test_emit_silently_returns_false_when_table_missing():
    cur = FakeCursor(pgcode="42P01")
    conn = FakeConn(cur)
    ok = uc.emit_usage_event(
        lambda: conn, kind="copy", route="job", ok=False,
        usage={"input_tokens": 500, "output_tokens": 100},
        error_code="TIMEOUT",
    )
    assert ok is False
    assert conn.closed


def test_emit_silently_returns_false_when_connect_fails():
    ok = uc.emit_usage_event(
        lambda: None, kind="rubric", route="sync", ok=True,
        usage={"input_tokens": 100},
    )
    assert ok is False


def test_emit_silently_returns_false_on_unexpected_error(caplog):
    def boom():
        raise RuntimeError("pool exhausted")

    with caplog.at_level(logging.WARNING):
        ok = uc.emit_usage_event(
            boom, kind="copy", route="sync", ok=False,
            usage={"input_tokens": 100}, error_code="RuntimeError",
        )
    assert ok is False
    assert "pool exhausted" in caplog.text


def test_emit_skips_failed_events_with_no_tokens():
    cur = FakeCursor()
    conn = FakeConn(cur)
    ok = uc.emit_usage_event(
        lambda: conn, kind="copy", route="sync", ok=False,
        usage={"gemini_calls": 2, "billed_calls": 0},
        error_code="TIMEOUT",
    )
    assert ok is False
    assert len(cur.rows) == 0
    assert not conn.closed


def test_emit_records_successful_events_even_with_no_tokens():
    cur = FakeCursor()
    conn = FakeConn(cur)
    ok = uc.emit_usage_event(
        lambda: conn, kind="rubric", route="sync", ok=True,
        usage={},
    )
    assert ok is True
    assert len(cur.rows) == 1


# ── sync route emits events ──────────────────────────────────────────────────

PRIMARY = ccr.CHECKER_MODEL


def reply(inp, out, think=0):
    return SimpleNamespace(text="{}", candidates=[], usage_metadata=SimpleNamespace(
        prompt_token_count=inp, candidates_token_count=out, total_token_count=inp + out + think,
        thoughts_token_count=think))


def bill(inp=1000, out=200, think=300, model="gemini-3.8-flash", tag="RUBRIC"):
    svc._count_call(tag)
    svc._log_usage(tag, model, reply(inp, out, think), 1.0)


UPLOAD = {"source": "upload", "qp_paths": ["u/s/paper/1.pdf"]}
PRIOR_BUILD = uc.rubric_build_record({"gemini_calls": 1, "billed_calls": 1, "input_tokens": 1000, "output_tokens": 200,
                                      "thinking_tokens": 0, "model": "gemini-3.8-flash"}, None, "sync")
PRIOR = {"questions": [{"q_no": "1"}], "total_marks": 5, "build": PRIOR_BUILD}


class FakeStoreSimple:
    def __init__(self, build=None, has_rubric=True):
        self.build, self.has_rubric, self.calls = build, has_rubric, []

    def record_rubric_build(self, session_id, merge):
        self.calls.append(session_id)
        if not self.has_rubric:
            return False
        self.build = merge(self.build)
        return True


def ok_build():
    bill(1000, 200, 300)
    return {"questions": [], "total_marks": 40, "_rubric_time_s": 3.2}


def bad_after_billing():
    bill(8000, 1000, 0)
    raise ValueError("the reply was not a rubric")


def test_sync_rubric_success_emits_usage_event(monkeypatch):
    events = []
    monkeypatch.setattr(ccr, "emit_usage_event", lambda connect, **kw: events.append(kw) or True)
    store = FakeStoreSimple(dict(PRIOR_BUILD))
    monkeypatch.setattr(ccr, "_rubric_store", lambda: store)
    monkeypatch.setattr(ccr, "_get_session", lambda c, sid: dict(UPLOAD))
    monkeypatch.setattr(ccr, "_update_session", lambda c, sid, f: f)
    monkeypatch.setattr(ccr, "_download_own", lambda c, p, prefix: "/tmp/x.pdf")
    monkeypatch.setattr(ccr, "_cleanup", lambda *p: None)
    monkeypatch.setattr(ccr, "build_rubric", lambda qp, key: ok_build())
    ccr._build_session_rubric(None, "sid", False)
    assert len(events) == 1
    ev = events[0]
    assert ev["kind"] == "rubric" and ev["route"] == "sync" and ev["ok"] is True
    assert ev["session_id"] == "sid"
    assert ev["usage"]["input_tokens"] == 1000


def test_sync_rubric_failure_emits_usage_event(monkeypatch):
    events = []
    monkeypatch.setattr(ccr, "emit_usage_event", lambda connect, **kw: events.append(kw) or True)
    store = FakeStoreSimple(dict(PRIOR_BUILD))
    monkeypatch.setattr(ccr, "_rubric_store", lambda: store)
    monkeypatch.setattr(ccr, "_get_session", lambda c, sid: {**UPLOAD, "rubric": PRIOR, "rubric_status": "ready"})
    monkeypatch.setattr(ccr, "_update_session", lambda c, sid, f: f)
    monkeypatch.setattr(ccr, "_download_own", lambda c, p, prefix: "/tmp/x.pdf")
    monkeypatch.setattr(ccr, "_cleanup", lambda *p: None)
    monkeypatch.setattr(ccr, "build_rubric", lambda qp, key: bad_after_billing())
    with pytest.raises(ValueError):
        ccr._build_session_rubric(None, "sid", True)
    assert len(events) == 1
    ev = events[0]
    assert ev["kind"] == "rubric" and ev["route"] == "sync" and ev["ok"] is False
    assert ev["error_code"] == "ValueError"
    assert ev["usage"]["input_tokens"] == 8000


def test_sync_rubric_failure_without_tokens_still_emits(monkeypatch):
    events = []
    monkeypatch.setattr(ccr, "emit_usage_event", lambda connect, **kw: events.append(kw) or True)
    monkeypatch.setattr(ccr, "_rubric_store", lambda: FakeStoreSimple())
    monkeypatch.setattr(ccr, "_get_session", lambda c, sid: dict(UPLOAD))
    monkeypatch.setattr(ccr, "_update_session", lambda c, sid, f: f)
    monkeypatch.setattr(ccr, "_download_own", lambda c, p, prefix: "/tmp/x.pdf")
    monkeypatch.setattr(ccr, "_cleanup", lambda *p: None)

    class ReadTimeout(Exception):
        pass

    monkeypatch.setattr(ccr, "build_rubric", lambda qp, key: (_ for _ in ()).throw(ReadTimeout("timed out")))
    with pytest.raises(ReadTimeout):
        ccr._build_session_rubric(None, "sid", False)
    assert len(events) == 1
    ev = events[0]
    assert ev["ok"] is False and ev["error_code"] == "ReadTimeout"
    assert not uc.has_tokens(ev["usage"])     # meter ran but no Gemini call was made


# ── sync route: copy check emits events ──────────────────────────────────────

class FakeDB:
    def __init__(self):
        self.updates = []
        self.sheet = {"id": "s1", "session_id": "sess", "pages": ["u/p1.jpg"], "student_name": "", "roll_no": ""}

    def table(self, name):
        return FakeQ(self)


class FakeQ:
    def __init__(self, db):
        self.db, self.fields = db, None

    def select(self, *a, **k):
        return self

    def eq(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def update(self, fields):
        self.fields = fields
        return self

    def execute(self):
        if self.fields is not None:
            self.db.updates.append(dict(self.fields))
            return SimpleNamespace(data=[{"id": "s1", **self.fields}])
        return SimpleNamespace(data=[self.db.sheet])


def usage_of(model=PRIMARY, **extra):
    return {"model": model, "models_by_tag": {"GRADE": model}, "fallback_used": model != PRIMARY, "pages": 1,
            "gemini_calls": 1, "billed_calls": 1, "input_tokens": 9000, "output_tokens": 3000, "thinking_tokens": 500,
            **extra}


def test_sync_copy_check_success_emits_event(monkeypatch):
    events = []
    monkeypatch.setattr(ccr, "emit_usage_event", lambda connect, **kw: events.append(kw) or True)
    db = FakeDB()
    monkeypatch.setattr(ccr, "_get_session", lambda c, sid: {"rubric_status": "ready", "rubric": {"questions": []},
                                                            "strictness": "medium", "class_grade": "10",
                                                            "subject": "x", "title": "t"})
    monkeypatch.setattr(ccr, "_download_own", lambda c, p, prefix: "/tmp/x.jpg")
    monkeypatch.setattr(ccr, "_cleanup", lambda *p: None)
    monkeypatch.setattr(ccr, "_save_norm_pages", lambda *a, **k: [])
    monkeypatch.setattr(ccr, "_save_to_db", lambda *a, **k: None)
    u = usage_of()
    monkeypatch.setattr(ccr, "grade_copy",
                        lambda files, rubric, strictness: ({"graded_questions": [], "total_marks_awarded": 3,
                                                            "max_marks": 5}, [], dict(u)))
    c = SimpleNamespace(db=db, user=SimpleNamespace(id="u"))
    ccr._check_sheet(c, "s1")
    copy_events = [e for e in events if e["kind"] == "copy"]
    assert len(copy_events) == 1
    ev = copy_events[0]
    assert ev["ok"] is True and ev["route"] == "sync" and ev["sheet_id"] == "s1"
    assert ev["session_id"] == "sess"


def test_sync_copy_check_failure_emits_event(monkeypatch):
    events = []
    monkeypatch.setattr(ccr, "emit_usage_event", lambda connect, **kw: events.append(kw) or True)
    db = FakeDB()
    monkeypatch.setattr(ccr, "_get_session", lambda c, sid: {"rubric_status": "ready", "rubric": {"questions": []},
                                                            "strictness": "medium"})
    monkeypatch.setattr(ccr, "_download_own", lambda c, p, prefix: "/tmp/x.jpg")
    monkeypatch.setattr(ccr, "_cleanup", lambda *p: None)
    monkeypatch.setattr(ccr, "_save_norm_pages", lambda *a, **k: [])

    u = usage_of()
    def fail_grade(files, rubric, strictness):
        e = svc.UnreadableReply("not json")
        e.usage = dict(u)
        raise e

    monkeypatch.setattr(ccr, "grade_copy", fail_grade)
    c = SimpleNamespace(db=db, user=SimpleNamespace(id="u"))
    with pytest.raises(svc.UnreadableReply):
        ccr._check_sheet(c, "s1")
    copy_events = [e for e in events if e["kind"] == "copy"]
    assert len(copy_events) == 1
    ev = copy_events[0]
    assert ev["ok"] is False and ev["error_code"] == "UnreadableReply"
    assert ev["usage"]["input_tokens"] == 9000


# ── job worker emits events ──────────────────────────────────────────────────

def test_job_worker_emit_on_failure(monkeypatch):
    events = []
    monkeypatch.setattr(uc, "emit_usage_event", lambda connect, **kw: events.append(kw) or True)
    monkeypatch.setattr(jw, "emit_usage_event", lambda connect, **kw: events.append(kw) or True)

    class FakeJobStore:
        _connect = lambda: None

        def finish(self, job_id, ok, code=None, message=None, usage=None):
            return {"state": "failed"}

        def mark_sheet_failed(self, sheet_id, msg):
            pass

    worker = jw.JobWorker.__new__(jw.JobWorker)
    worker.store = FakeJobStore()
    worker._run_usage = {"input_tokens": 500, "output_tokens": 100, "gemini_calls": 1, "billed_calls": 1}
    worker._stages = {"grade": 2.5}
    worker._job_tag = "job=test1234"
    worker._job = {"id": "j1", "session_id": "s1", "sheet_id": "sh1", "submission_id": "sub1",
                   "checkpoint": {}, "retries": 0, "max_retries": 3, "usage": {}}
    worker._fail(worker._job, svc.GeminiBusy(429))
    assert len(events) == 1
    ev = events[0]
    assert ev["kind"] == "copy" and ev["route"] == "job" and ev["ok"] is False
    assert ev["error_code"] == "GEMINI_429"
    assert ev["job_id"] == "j1" and ev["sheet_id"] == "sh1"


def test_job_kind_classification():
    assert jw.JobWorker._job_kind({"checkpoint": {"stage": "rubric"}, "kind": "check"}) == "rubric"
    assert jw.JobWorker._job_kind({"checkpoint": {"parent": "p1"}, "kind": "check"}) == "chunk"
    assert jw.JobWorker._job_kind({"checkpoint": {}, "kind": "check", "submission_id": "s"}) == "copy"
    assert jw.JobWorker._job_kind({"checkpoint": {}, "kind": "retention"}) == "retention"


def test_job_worker_does_not_emit_for_retention():
    events = []

    def capture(connect, **kw):
        events.append(kw)
        return True

    worker = jw.JobWorker.__new__(jw.JobWorker)
    worker.store = MagicMock()
    worker.store._connect = lambda: None
    with patch.object(uc, "emit_usage_event", capture), patch.object(jw, "emit_usage_event", capture):
        worker._emit({"id": "j1", "checkpoint": {}, "kind": "retention"}, ok=True)
    assert len(events) == 0
