"""Checker bundle #2: one copy-check job run (all retries + fallbacks) fits inside the function's max duration.
Fake clock and fake Gemini client; no network."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from app.services import job_worker as jw
from app.services import smart_assignment_service as svc

from audit_support import no_network  # noqa: F401


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def env(monkeypatch, no_network):  # noqa: F811
    clock = Clock()
    monkeypatch.setattr(svc.time, "monotonic", clock)
    monkeypatch.setattr(svc.time, "sleep", lambda s: setattr(clock, "t", clock.t + s))
    monkeypatch.setattr(svc, "GEMINI_MOCK", False)
    monkeypatch.setattr(svc, "CHECKER_MODEL", "primary")
    monkeypatch.setattr(svc, "CHECKER_FALLBACK_MODELS", ["fb1"])
    monkeypatch.setattr(svc, "CHECKER_API_KEY", "test-key")
    svc._clients.clear()
    timeouts, calls = [], []

    def make(seconds_per_call: float, code: int = 503):
        class Client:
            def __init__(self, api_key, http_options):
                t = http_options["timeout"] / 1000
                self.models = SimpleNamespace(generate_content=lambda model, contents, config: _call(model, t))

        def _call(model, t):
            calls.append(model)
            timeouts.append(t)
            clock.t += min(seconds_per_call, t)          # a call never runs longer than its own timeout
            raise genai_errors.APIError(code, {"error": {"code": code, "message": "x", "status": "X"}},
                                        SimpleNamespace(headers={}))
        monkeypatch.setattr(svc.genai, "Client", Client)
        return clock, timeouts, calls
    yield make
    svc._clients.clear()


def test_three_attempts_never_pass_the_run_deadline(env):
    clock, timeouts, calls = env(seconds_per_call=140)
    start = clock.t
    budget = jw.JOB_MAX_DURATION_S - jw.JOB_RUN_MARGIN_S          # 300 - 30 = 270 s by default
    with svc.usage_meter(), svc.call_settings(timeout_s=jw.GRADE_CALL_TIMEOUT_S, deadline=start + budget):
        with pytest.raises((svc.GeminiBusy, svc.RunDeadline)):
            svc._generate_json(svc._get_client(), ["x"], "GRADE")
    assert clock.t - start <= budget
    assert all(t <= jw.GRADE_CALL_TIMEOUT_S for t in timeouts)
    assert len(calls) <= svc.MAX_ATTEMPTS_PER_RUN


def test_call_timeout_is_cut_to_the_time_left(env):
    clock, timeouts, _ = env(seconds_per_call=1)
    with svc.usage_meter(), svc.call_settings(timeout_s=150, deadline=clock.t + 90):
        with pytest.raises(svc.GeminiBusy):
            svc._generate(svc._get_client(), ["x"], "GRADE")
    assert timeouts[0] == 90


def test_no_call_starts_with_less_than_min_left(env):
    clock, timeouts, calls = env(seconds_per_call=1)
    with svc.usage_meter(), svc.call_settings(timeout_s=150, deadline=clock.t + svc.MIN_CALL_S - 1):
        with pytest.raises(svc.RunDeadline):
            svc._generate_json(None, ["x"], "GRADE")
    assert calls == []                                           # nothing paid


def test_run_deadline_is_a_retryable_timeout_for_the_worker():
    code, _msg, _ = jw.JobWorker._classify(jw.JobWorker, svc.RunDeadline("x"))
    assert code == "TIMEOUT" and code in jw._RETRY_CODES


def test_defaults_match_vercel_limit():
    """Vercel function limit verified 13 Nov 2026: 300 s (defaultResourceConfig.functionDefaultTimeout)."""
    assert jw.GRADE_CALL_TIMEOUT_S == 150 and svc.CHECKER_TIMEOUT_S == 60
    assert jw.JOB_MAX_DURATION_S == 300 and jw.JOB_RUN_MARGIN_S == 30
    assert jw.GRADE_CALL_TIMEOUT_S <= jw.JOB_MAX_DURATION_S - jw.JOB_RUN_MARGIN_S - svc.MIN_CALL_S
