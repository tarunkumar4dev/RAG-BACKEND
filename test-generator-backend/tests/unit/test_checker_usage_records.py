"""What the checker spent is written down (21 Nov 2026): the rubric's cost, model_version on the direct route, and the
usage of failed attempts whose replies carried token counts. Marking is not touched by any of this.

The real router functions run; only the database, storage and the Gemini network are replaced.
"""
import logging
from types import SimpleNamespace

import pytest

from app.routers import copy_checker_router as ccr
from app.services import job_worker as jw
from app.services import smart_assignment_service as svc
from app.services.answer_checker import usage_cost as uc

PRIMARY = ccr.CHECKER_MODEL


class ReadTimeout(Exception):
    """Stands in for the HTTP library's read timeout."""


def reply(inp, out, think=0):
    return SimpleNamespace(text="{}", candidates=[], usage_metadata=SimpleNamespace(
        prompt_token_count=inp, candidates_token_count=out, total_token_count=inp + out + think,
        thoughts_token_count=think))


def bill(inp=1000, out=200, think=300, model="gemini-3.8-flash", tag="RUBRIC"):
    """One billed Gemini reply, counted on the active usage meter exactly like _generate does."""
    svc._count_call(tag)
    svc._log_usage(tag, model, reply(inp, out, think), 1.0)


# ── the helpers ─────────────────────────────────────────────────────────────────────────────────────────────

def test_pricing_moved_but_job_worker_still_exports_it():
    for name in ("add_usage", "est_cost_inr", "price_in_per_m_inr", "price_out_per_m_inr", "_COUNTERS",
                 "rubric_build_record"):
        assert getattr(jw, name) is getattr(uc, name)


def test_has_tokens_only_when_a_reply_came_back():
    assert uc.has_tokens({"input_tokens": 5}) and uc.has_tokens({"thinking_tokens": 1})
    assert not uc.has_tokens({"gemini_calls": 3, "billed_calls": 0}) and not uc.has_tokens(None)


def test_build_record_adds_attempts_and_prices_the_total():
    one = {"gemini_calls": 1, "billed_calls": 1, "input_tokens": 1000, "output_tokens": 200, "thinking_tokens": 300,
           "model": "gemini-3.8-flash", "stages": {"x": 1}}
    two = {**one, "input_tokens": 2000, "model": "gemini-3.6-flash", "fallback_used": True}
    first = uc.rubric_build_record(one, None, "sync")
    both = uc.rubric_build_record(two, first, "sync")
    assert "stages" not in first and first["source"] == "sync"
    assert (both["gemini_calls"], both["billed_calls"], both["input_tokens"]) == (2, 2, 3000)
    assert both["models"] == ["gemini-3.6-flash", "gemini-3.8-flash"] and both["fallback_used"]
    assert both["est_cost_inr"] == uc.est_cost_inr({"input_tokens": 3000, "output_tokens": 400,
                                                    "thinking_tokens": 600}) > first["est_cost_inr"] > 0


def test_answering_model_prefers_a_backup_so_it_is_never_hidden():
    assert uc.answering_model({"model": "m1"}, "m1") == "m1"
    assert uc.answering_model({"model": "m1", "models_by_tag": {"GRADE": "backup"}}, "m1") == "backup"
    assert uc.answering_model({"model": "backup", "models": ["m1", "backup"]}, "m1") == "backup"
    assert uc.answering_model({"model": "mock"}, "m1") == "mock"          # GEMINI_MOCK runs say so


def test_answering_model_is_none_when_the_usage_names_no_model():
    """An unknown model is stored as unknown, never as the primary."""
    assert uc.answering_model({}, "m1") is None and uc.answering_model(None, "m1") is None
    assert uc.answering_model({"gemini_calls": 1, "input_tokens": 5}, "m1") is None
    assert uc.answering_model({"models": ["mock"]}, "m1") is None


def test_without_build_hides_the_cost_and_leaves_the_rest_alone():
    rubric = {"questions": [{"q_no": "1"}], "model": "m1", "build": {"est_cost_inr": 3.2, "input_tokens": 9}}
    out = uc.without_build(rubric)
    assert out == {"questions": [{"q_no": "1"}], "model": "m1"} and "build" in rubric      # a copy, not a mutation
    assert uc.without_build(None) is None and uc.without_build({"questions": []}) == {"questions": []}


# ── the direct route's rubric build ─────────────────────────────────────────────────────────────────────────

class FakeStore:
    """JobStore.record_rubric_build without a database: applies `merge` to the build it holds."""

    def __init__(self, build=None, has_rubric=True, boom=False):
        self.build, self.has_rubric, self.boom, self.calls = build, has_rubric, boom, []

    def record_rubric_build(self, session_id, merge):
        self.calls.append(session_id)
        if self.boom:
            raise RuntimeError("database unavailable")
        if not self.has_rubric:
            return False
        self.build = merge(self.build)
        return True


class Fx:
    def __init__(self, monkeypatch, session, build, store=None):
        self.updates = []
        self.store = store or FakeStore(dict(PRIOR_BUILD))
        monkeypatch.setattr(ccr, "_rubric_store", lambda: self.store)
        monkeypatch.setattr(ccr, "_get_session", lambda c, sid: session)
        monkeypatch.setattr(ccr, "_update_session", lambda c, sid, f: self.updates.append(dict(f)) or f)
        monkeypatch.setattr(ccr, "_download_own", lambda c, p, prefix: "/tmp/x.pdf")
        monkeypatch.setattr(ccr, "_cleanup", lambda *p: None)
        monkeypatch.setattr(ccr, "build_rubric", lambda qp, key: build())

    @property
    def last(self):
        return self.updates[-1]


UPLOAD = {"source": "upload", "qp_paths": ["u/s/paper/1.pdf"]}
PRIOR_BUILD = uc.rubric_build_record({"gemini_calls": 1, "billed_calls": 1, "input_tokens": 1000, "output_tokens": 200,
                                      "thinking_tokens": 0, "model": "gemini-3.8-flash"}, None, "sync")
PRIOR = {"questions": [{"q_no": "1"}], "total_marks": 5, "build": PRIOR_BUILD}


def ok_build():
    bill(1000, 200, 300)
    return {"questions": [], "total_marks": 40, "_rubric_time_s": 3.2}


def test_rubric_build_usage_is_saved_with_the_rubric(monkeypatch):
    fx = Fx(monkeypatch, dict(UPLOAD), ok_build)
    ccr._build_session_rubric(None, "sid", False)
    rubric = fx.last["rubric"]
    b = rubric["build"]
    assert (b["input_tokens"], b["output_tokens"], b["thinking_tokens"]) == (1000, 200, 300)
    assert (b["gemini_calls"], b["billed_calls"], b["model"], b["source"]) == (1, 1, "gemini-3.8-flash", "sync")
    assert b["est_cost_inr"] == uc.est_cost_inr(b) > 0
    assert not [k for k in rubric if k.startswith("_")]                  # internal keys are still stripped


def test_a_rebuild_adds_to_the_earlier_builds(monkeypatch):
    fx = Fx(monkeypatch, {**UPLOAD, "rubric": PRIOR, "rubric_status": "ready"}, ok_build)
    ccr._build_session_rubric(None, "sid", True)
    b = fx.last["rubric"]["build"]
    assert (b["gemini_calls"], b["input_tokens"]) == (2, 2000)


def test_rebuild_from_a_saved_test_keeps_the_history(monkeypatch):
    class Q:
        def __init__(self, data):
            self.data = data

        def __getattr__(self, name):
            return lambda *a, **k: self

        def execute(self):
            return SimpleNamespace(data=self.data)

    monkeypatch.setattr(ccr, "get_supabase_admin", lambda: SimpleNamespace(table=lambda n: Q([{"id": "t1"}])))
    monkeypatch.setattr(ccr, "rubric_from_test_fast", lambda t, qs: {"questions": [], "total_marks": 5})
    fx = Fx(monkeypatch, {"source": "test", "test_id": "t1", "rubric": PRIOR, "rubric_status": "ready"}, ok_build)
    c = SimpleNamespace(user=SimpleNamespace(id="u"))
    ccr._build_session_rubric(c, "sid", True)
    assert fx.last["rubric"]["build"] == PRIOR_BUILD                     # no Gemini call, nothing added or lost


def bad_after_billing():
    bill(8000, 1000, 0)
    raise ValueError("the reply was not a rubric")


def test_failed_rebuild_adds_the_attempt_to_rubric_build_and_writes_no_rubric(monkeypatch):
    fx = Fx(monkeypatch, {**UPLOAD, "rubric": PRIOR, "rubric_status": "ready"}, bad_after_billing)
    with pytest.raises(ValueError):
        ccr._build_session_rubric(None, "sid", True)
    # the session update carries the status only: the rubric itself is never written back
    assert fx.last == {"rubric_status": "failed", "rubric_error": ccr.PAPER_UNREADABLE_MSG}
    assert all("rubric" not in u for u in fx.updates)
    assert fx.store.calls == ["sid"]
    assert (fx.store.build["gemini_calls"], fx.store.build["input_tokens"]) == (2, 9000)   # 1000 earlier + 8000 now


def test_a_timeout_without_a_reply_does_not_touch_the_rubric(monkeypatch):
    def hang():
        raise ReadTimeout("read timed out")

    fx = Fx(monkeypatch, {**UPLOAD, "rubric": PRIOR, "rubric_status": "ready"}, hang)
    with pytest.raises(ReadTimeout):
        ccr._build_session_rubric(None, "sid", True)
    assert fx.last == {"rubric_status": "failed", "rubric_error": ccr.BUSY_MSG} and fx.store.calls == []


def test_failed_first_build_after_billed_replies_is_logged_with_its_cost(monkeypatch, caplog):
    fx = Fx(monkeypatch, dict(UPLOAD), bad_after_billing, store=FakeStore(has_rubric=False))
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError):
        ccr._build_session_rubric(None, "sid", False)
    assert fx.last == {"rubric_status": "failed", "rubric_error": ccr.PAPER_UNREADABLE_MSG}
    assert "no rubric to attach it to" in caplog.text and "'input_tokens': 8000" in caplog.text


def test_recording_a_failed_build_never_hides_the_build_error(monkeypatch, caplog):
    fx = Fx(monkeypatch, {**UPLOAD, "rubric": PRIOR, "rubric_status": "ready"}, bad_after_billing,
            store=FakeStore(boom=True))
    with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="not a rubric"):
        ccr._build_session_rubric(None, "sid", True)
    assert fx.last["rubric_status"] == "failed"
    assert "recording it failed (RuntimeError)" in caplog.text and "'input_tokens': 8000" in caplog.text


def test_the_rubric_api_response_has_no_cost_data(monkeypatch):
    import asyncio
    session = {"rubric_status": "ready", "rubric": {**PRIOR, "model": "m1"}}
    monkeypatch.setattr(ccr, "_build_session_rubric", lambda c, sid, force, deadline=None: session)
    out = asyncio.run(ccr.session_rubric("00000000-0000-0000-0000-000000000001", ccr.RubricRequest(),
                                         SimpleNamespace(user=SimpleNamespace(id="u"))))
    assert out["full_rubric"]["questions"] == PRIOR["questions"] and out["full_rubric"]["model"] == "m1"
    assert "build" not in out["full_rubric"] and "build" in session["rubric"]          # stored, not served
    assert "est_cost_inr" not in str(out)


# ── the direct route's copy check ───────────────────────────────────────────────────────────────────────────

class Q:
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


class DB:
    def __init__(self):
        self.updates = []
        self.sheet = {"id": "s1", "session_id": "sess", "pages": ["u/p1.jpg"], "student_name": "", "roll_no": ""}

    def table(self, name):
        return Q(self)


def usage_of(model=PRIMARY, **extra):
    return {"model": model, "models_by_tag": {"GRADE": model}, "fallback_used": model != PRIMARY, "pages": 1,
            "gemini_calls": 1, "billed_calls": 1, "input_tokens": 9000, "output_tokens": 3000, "thinking_tokens": 500,
            **extra}


def sheet_fx(monkeypatch, grade):
    db = DB()
    monkeypatch.setattr(ccr, "_get_session", lambda c, sid: {"rubric_status": "ready", "rubric": {"questions": []},
                                                            "strictness": "medium", "class_grade": "10",
                                                            "subject": "x", "title": "t"})
    monkeypatch.setattr(ccr, "_download_own", lambda c, p, prefix: "/tmp/x.jpg")
    monkeypatch.setattr(ccr, "_cleanup", lambda *p: None)
    monkeypatch.setattr(ccr, "_save_norm_pages", lambda *a, **k: [])
    monkeypatch.setattr(ccr, "_save_to_db", lambda *a, **k: None)
    monkeypatch.setattr(ccr, "grade_copy", grade)
    return db, SimpleNamespace(db=db, user=SimpleNamespace(id="u"))


def graded(usage):
    return lambda files, rubric, strictness: ({"graded_questions": [], "total_marks_awarded": 3, "max_marks": 5},
                                              [], dict(usage))


def test_direct_check_writes_model_version(monkeypatch):
    db, c = sheet_fx(monkeypatch, graded(usage_of()))
    ccr._check_sheet(c, "s1")
    done = db.updates[-1]
    assert done["status"] == "checked" and done["model_version"] == PRIMARY
    assert done["ai_result"]["usage"]["input_tokens"] == 9000            # the result still carries its usage


def test_direct_check_stores_no_model_when_the_usage_names_none(monkeypatch):
    usage = {k: v for k, v in usage_of().items() if k not in ("model", "models_by_tag")}
    db, c = sheet_fx(monkeypatch, graded(usage))
    ccr._check_sheet(c, "s1")
    assert db.updates[-1]["status"] == "checked" and db.updates[-1]["model_version"] is None


def test_direct_check_model_version_names_a_backup_model(monkeypatch):
    db, c = sheet_fx(monkeypatch, graded(usage_of("backup-model")))
    ccr._check_sheet(c, "s1")
    done = db.updates[-1]
    assert done["model_version"] == "backup-model"
    assert [f["code"] for f in done["ai_result"]["sheet_flags"]] == ["FALLBACK_MODEL"]   # flagged as before


def test_failed_check_after_a_billed_grade_logs_what_it_cost(monkeypatch, caplog):
    db, c = sheet_fx(monkeypatch, graded(usage_of()))
    monkeypatch.setattr(ccr, "_save_norm_pages", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("storage")))
    with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
        ccr._check_sheet(c, "s1")
    failed = db.updates[-1]
    assert failed["status"] == "failed" and "ai_result" not in failed and "model_version" not in failed
    assert "still cost" in caplog.text and "'input_tokens': 9000" in caplog.text and "est_cost_inr" in caplog.text


def test_failed_check_logs_the_usage_grade_copy_attached(monkeypatch, caplog):
    def unreadable(files, rubric, strictness):
        e = svc.UnreadableReply("not json")
        e.usage = usage_of(billed_calls=2, input_tokens=18000)           # two billed replies, none usable
        raise e

    db, c = sheet_fx(monkeypatch, unreadable)
    with caplog.at_level(logging.WARNING), pytest.raises(svc.UnreadableReply):
        ccr._check_sheet(c, "s1")
    assert db.updates[-1]["status"] == "failed"
    assert "still cost" in caplog.text and "'input_tokens': 18000" in caplog.text and "'billed_calls': 2" in caplog.text


def test_failed_check_without_a_reply_logs_no_cost(monkeypatch, caplog):
    def busy(files, rubric, strictness):
        raise svc.GeminiBusy(429)

    db, c = sheet_fx(monkeypatch, busy)
    with caplog.at_level(logging.WARNING), pytest.raises(svc.GeminiBusy):
        ccr._check_sheet(c, "s1")
    assert db.updates[-1]["status"] == "failed" and "still cost" not in caplog.text
