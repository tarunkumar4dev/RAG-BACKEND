"""fix/checker-p0 (docs/audit/inventory.md): a backup model is never silent, key / model errors alert and wait, and
unfinished jobs are reported. Pure worker logic with a fake store; the SQL side is in
tests/integration/test_job_worker_pg.py."""

import logging

import pytest

from app.services import job_worker, smart_assignment_service as svc
from app.services.answer_checker import portal_result


class FakeStore:
    def __init__(self, stale=None, fail=False):
        self.saved = []
        self.finished = []
        self._stale = stale or []
        self._fail = fail

    def save_result(self, job_id, sheet_id, fields, audit, checkpoint):
        self.saved.append(fields)

    def finish(self, job_id, ok, code=None, error=None, usage=None):
        self.finished.append((ok, code, error))
        return {"state": "done" if ok else "queued"}

    def stale_jobs(self, minutes):
        if self._fail:
            raise RuntimeError("relation checker_job_health does not exist")
        return [j for j in self._stale if j["age_min"] >= minutes]


def _result(model, fallback=False):
    q = {"q_no": "1", "marks_awarded": 2, "max_marks": 2, "confidence": "high", "counted": True,
         "not_attempted": False, "legibility_issue": False, "review_reasons": [], "subpart_grades": []}
    return {"graded_questions": [q], "total_marks_awarded": 2, "max_marks": 2, "sheet_flags": [],
            "usage": {"model": model, "fallback_used": fallback}}


def _save(worker, result, rubric=None):
    job = {"id": "j1", "teacher_id": "t1", "checkpoint": {}, "usage": {}}
    worker._save(job, {"id": "sh1"}, {"id": "s1", "assignment_id": "a1"}, {"title": "Econ"}, {"version": 1},
                 rubric or {"model": svc.CHECKER_MODEL}, result, [], {"pages": 2})
    return worker.store.saved[-1]


def test_fallback_flag_becomes_a_review_reason():
    result = {"graded_questions": [], "sheet_flags": [{"code": "FALLBACK_MODEL"}, {"code": "BLANK_PAGE"}]}
    reasons, _, _ = portal_result.review(result, {})
    assert reasons == ["blank_pages", "fallback_model"]


def test_copy_marked_by_a_backup_model_is_flagged_with_the_model():
    w = job_worker.JobWorker(FakeStore(), storage=None)
    fields = _save(w, _result("gemini-3-flash-preview", fallback=True))
    assert "fallback_model" in fields["needs_review_reasons"]
    assert fields["model_version"] == "gemini-3-flash-preview"
    flag = [f for f in fields["ai_result"]["sheet_flags"] if f["code"] == "FALLBACK_MODEL"][0]
    assert "gemini-3-flash-preview" in flag["message"] and svc.CHECKER_MODEL in flag["message"]
    assert fields["ai_result"]["meta"]["models"]["grade"] == ["gemini-3-flash-preview"]


def test_copy_marked_by_the_primary_is_not_flagged():
    w = job_worker.JobWorker(FakeStore(), storage=None)
    fields = _save(w, _result(svc.CHECKER_MODEL))
    assert fields["needs_review_reasons"] == []
    assert fields["model_version"] == svc.CHECKER_MODEL
    assert fields["ai_result"]["meta"]["primary_model"] == svc.CHECKER_MODEL


def test_rubric_built_by_a_backup_model_flags_every_copy():
    w = job_worker.JobWorker(FakeStore(), storage=None)
    fields = _save(w, _result(svc.CHECKER_MODEL), rubric={"model": "gemini-2.5-flash", "fallback_used": True})
    assert "fallback_model" in fields["needs_review_reasons"]
    assert "marking scheme" in fields["ai_result"]["sheet_flags"][-1]["message"]


def test_chunk_usage_keeps_every_model():
    u = job_worker.add_usage({}, {"model": svc.CHECKER_MODEL, "gemini_calls": 1})
    u = job_worker.add_usage(u, {"model": "gemini-3-flash-preview", "fallback_used": True, "gemini_calls": 1})
    u = job_worker.add_usage(u, {"model": svc.CHECKER_MODEL, "gemini_calls": 1})
    assert u["models"] == sorted({svc.CHECKER_MODEL, "gemini-3-flash-preview"})
    assert u["fallback_used"] is True and u["gemini_calls"] == 3
    w = job_worker.JobWorker(FakeStore(), storage=None)
    result = _result(svc.CHECKER_MODEL)
    result["usage"] = u
    assert "fallback_model" in _save(w, result)["needs_review_reasons"]


def test_key_or_model_error_is_gemini_config_with_an_alert(caplog):
    w = job_worker.JobWorker(FakeStore(), storage=None)
    with caplog.at_level(logging.ERROR):
        code, message, _ = w._classify(svc.GeminiConfigError(403, "gemini-3.8-flash"))
    assert code == "GEMINI_CONFIG" and "automatically" in message
    assert any("ALERT" in r.message and "403" in r.message for r in caplog.records if r.levelno == logging.ERROR)


def test_provider_failure_out_of_retries_says_it_will_be_tried_again():
    store = FakeStore()
    w = job_worker.JobWorker(store, storage=None)
    w._fail({"id": "j1", "retries": 3, "max_retries": 3, "checkpoint": {}}, svc.GeminiBusy(503))
    ok, code, error = store.finished[-1]
    assert (ok, code) == (False, "GEMINI_5XX") and "checked again automatically" in error


def test_other_failure_out_of_retries_still_asks_for_retry():
    store = FakeStore()
    w = job_worker.JobWorker(store, storage=None)
    w._fail({"id": "j1", "retries": 3, "max_retries": 3, "checkpoint": {}}, RuntimeError("boom"))
    assert "Use Retry" in store.finished[-1][2]


def test_stale_jobs_are_logged_at_error(caplog):
    stale = [{"job_id": "j-old", "assignment_id": "a-1", "age_min": 42, "state": "queued", "retries": 1,
              "retry_rounds": 0, "last_error_code": "TIMEOUT"},
             {"job_id": "j-new", "assignment_id": "a-1", "age_min": 3, "state": "running", "retries": 1,
              "retry_rounds": 0, "last_error_code": None}]
    w = job_worker.JobWorker(FakeStore(stale=stale), storage=None)
    with caplog.at_level(logging.ERROR):
        w.alert_stale()
    lines = [r.message for r in caplog.records if r.levelno == logging.ERROR]
    assert len(lines) == 1 and "j-old" in lines[0] and "a-1" in lines[0] and "42 min" in lines[0]


def test_missing_health_view_never_blocks_checking(caplog):
    w = job_worker.JobWorker(FakeStore(fail=True), storage=None)
    w.alert_stale()                     # no exception
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.parametrize("model,expected", [(svc.CHECKER_MODEL, False), ("mock", False), ("gemini-2.5-flash", True)])
def test_meter_marks_fallback_only_for_other_models(model, expected):
    from types import SimpleNamespace
    with svc.usage_meter() as m:
        svc._log_usage("GRADE", model, SimpleNamespace(usage_metadata=None), 0.1)
    assert m.fallback_used is expected and m.models_by_tag == {"GRADE": model}
