"""fix/checker-p0 merged into main (21 Nov 2026): the rules where the two met.

- one shared deadline per direct request (rubric + grade inside Vercel's 300 s), checked before every Gemini call;
- both budgets before every call: the run's attempts (3 primary + 1 per fallback model) and the copy's billed calls;
- only replies are billed (429 / 5xx / timeouts are not);
- the dead test_checker router / service are gone.
No network, no database: fake Gemini client, fake clock.
"""
import json
import time as _time
from pathlib import Path
from types import SimpleNamespace

import pytest
from google import genai
from google.genai import errors as genai_errors

from app.services import smart_assignment_service as svc

BACKEND = Path(__file__).resolve().parents[2]


def api_error(code: int) -> genai_errors.APIError:
    return genai_errors.APIError(code, {"error": {"code": code, "message": "x", "status": "X"}},
                                 SimpleNamespace(headers={}))


class ReadTimeout(Exception):
    """Stands in for the HTTP library's read timeout (what the SDK raises when a call runs out of time)."""


class Clock:
    """Fake time module for svc: monotonic/time only move when a call 'takes' time or something sleeps."""

    def __init__(self, now: float = 10_000.0):
        self.now = now

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def sleep(self, s):
        self.now += s

    def __getattr__(self, name):
        return getattr(_time, name)


def reply(text):
    return SimpleNamespace(text=text, candidates=[], usage_metadata=SimpleNamespace(
        prompt_token_count=10, candidates_token_count=5, total_token_count=15, thoughts_token_count=0))


class ScriptedModels:
    """Each script item: an int (API error code), "timeout" (the call uses its whole timeout, then fails), or
    reply text. `took` = seconds each call moves the clock."""

    def __init__(self, clock, script, took=1.0, timeout=None):
        self.clock, self.script, self.took, self.timeout = clock, list(script), took, timeout
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        item = self.script.pop(0) if self.script else "timeout"
        if item == "timeout":
            self.clock.now += self.timeout or 60
            raise ReadTimeout("read timed out")
        self.clock.now += self.took
        if isinstance(item, int):
            raise api_error(item)
        return reply(item)


@pytest.fixture
def env(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(svc, "time", clock)
    monkeypatch.setattr(svc, "GEMINI_MOCK", False)
    monkeypatch.setattr(svc, "CHECKER_MODEL", "primary")
    monkeypatch.setattr(svc, "CHECKER_FALLBACK_MODELS", ["fb1"])
    monkeypatch.setattr(svc, "CHECKER_ALLOW_FALLBACK", False)
    monkeypatch.setattr(svc, "CHECKER_PRIMARY_ATTEMPTS", 3)

    def client(script, **kw):
        models = ScriptedModels(clock, script, **kw)
        return SimpleNamespace(models=models), models
    return SimpleNamespace(clock=clock, client=client)


# ── shared deadline ──────────────────────────────────────────────────────────────────────────────────────────

def test_shared_deadline_no_call_with_too_little_time_left(env):
    """A request that has run 275 s of its 280 s: no Gemini call is started (DeadlineReached, a timeout)."""
    client, models = env.client(['{"ok": true}'])
    deadline = svc.request_deadline()
    env.clock.now += 275
    with svc.call_settings(timeout_s=150, deadline=deadline), pytest.raises(svc.DeadlineReached) as e:
        svc._generate_json(client, ["x"], "GRADE")
    assert models.calls == []
    assert svc.is_timeout(e.value)                    # → TIMEOUT in the worker, AI busy (503) on the routes


def test_request_deadline_is_280s_from_now(env):
    assert svc.request_deadline() - env.clock.monotonic() == svc.SYNC_REQUEST_BUDGET_S == 280


def test_no_retry_that_would_run_past_the_deadline(env):
    """Grade call times out after 150 s with 130 s left: a second 150 s call can't fit, so it isn't started."""
    client, models = env.client(["timeout", '{"ok": true}'], timeout=150)
    start = env.clock.now
    with svc.call_settings(timeout_s=150, deadline=start + 280), pytest.raises(ReadTimeout):
        svc._generate_json(client, ["x"], "GRADE")
    assert models.calls == ["primary"]
    assert env.clock.now - start <= 280


class FakeClient(genai.Client):
    """A genai.Client for isinstance(), without the network; built per timeout like svc._get_client does."""
    models = None

    def __init__(self, models, timeout):
        self.models, self.timeout = models, timeout


def test_check_sheet_rubric_plus_grade_stay_inside_one_deadline(env, monkeypatch):
    """The real _check_sheet → _build_session_rubric → grade path (DB, storage and the Gemini network stubbed).
    The inline rubric takes 200 s; the grade call then gets only what is left (75 s, not its 150 s), times out,
    and is not retried: the request ends before 280 s and the copy says AI busy instead of Vercel killing it."""
    from app.routers import copy_checker_router as ccr
    clock = env.clock
    start = clock.now
    seen = {}

    class Chain:                                    # c.db.table(...).select/update/eq/limit(...).execute()
        def __init__(self):
            self.updates = []

        def __getattr__(self, name):
            def step(*a, **k):
                if name == "update":
                    self.updates.append(a[0])
                return self
            return step

        def execute(self):
            return SimpleNamespace(data=[{"id": "s1", "session_id": "sess", "pages": ["u/p1.jpg"]}])

    db = Chain()
    c = SimpleNamespace(db=db, user=SimpleNamespace(id="u"))
    monkeypatch.setattr(ccr, "_get_session", lambda c, sid: {"source": "upload", "qp_paths": ["u/qp.pdf"],
                                                            "rubric_status": "none"})
    monkeypatch.setattr(ccr, "_update_session", lambda c, sid, fields: dict(fields))
    monkeypatch.setattr(ccr, "_download_own", lambda c, p, prefix: "/tmp/x")
    monkeypatch.setattr(ccr, "_cleanup", lambda *p: None)

    def build_rubric(qp, key):
        seen["rubric_deadline"] = svc._deadline.get()
        clock.now += 200                            # a slow 40-mark paper
        return {"questions": [], "total_marks": 40}
    monkeypatch.setattr(ccr, "build_rubric", build_rubric)

    grade_models = ScriptedModels(clock, ["timeout", "timeout", "timeout"])
    timeouts = []

    def get_client(timeout_s=None):
        t = timeout_s or svc._call_timeout.get()
        timeouts.append(t)
        grade_models.timeout = t
        return FakeClient(grade_models, t)
    monkeypatch.setattr(svc, "_get_client", get_client)

    def grade_copy(files, rubric, strictness):
        seen["grade_deadline"] = svc._deadline.get()
        svc._generate_json(svc._get_client(), ["x"], "GRADE")
    monkeypatch.setattr(ccr, "grade_copy", grade_copy)

    with pytest.raises(ReadTimeout):
        ccr._check_sheet(c, "s1")
    assert seen["rubric_deadline"] == seen["grade_deadline"] == start + 280     # one shared deadline
    assert grade_models.calls == ["primary"]                                    # no retry past it
    assert timeouts == [150, 75]                    # stage client, then the capped one: 80 s left − 2 → 75
    assert clock.now - start <= 280
    assert db.updates[-1]["status"] == "failed" and db.updates[-1]["error"] == ccr.BUSY_MSG


# ── both budgets ─────────────────────────────────────────────────────────────────────────────────────────────

def test_run_budget_stops_the_run_while_billed_budget_remains(env):
    """503 every time: 3 primary attempts use the run's budget, the copy still has 3 billed calls left."""
    client, models = env.client([503] * 10)
    with svc.usage_meter() as meter, svc.billed_budget(svc.MAX_BILLED_PER_COPY):
        with pytest.raises(svc.GeminiBusy):
            svc._generate_json(client, ["x"], "GRADE")
        left = svc._billed_left.get()[0]
    assert models.calls == ["primary"] * 3 and meter.calls == 3
    assert left == 3 and meter.billed_calls == 0


def test_with_fallback_on_the_budget_is_3_primary_plus_1_per_fallback(env, monkeypatch):
    monkeypatch.setattr(svc, "CHECKER_ALLOW_FALLBACK", True)
    client, models = env.client([503] * 10)
    assert svc.run_attempt_budget() == 4
    with svc.usage_meter() as meter, pytest.raises(svc.GeminiBusy):
        svc._generate_json(client, ["x"], "GRADE")
    assert models.calls == ["primary"] * 3 + ["fb1"] and meter.calls == 4


def test_fallback_answer_is_flagged_and_billed(env, monkeypatch):
    monkeypatch.setattr(svc, "CHECKER_ALLOW_FALLBACK", True)
    client, models = env.client([503, 503, 503, '{"ok": true}'])
    with svc.usage_meter() as meter, svc.billed_budget(3):
        assert svc._generate_json(client, ["x"], "GRADE") == {"ok": True}
        left = svc._billed_left.get()[0]
    assert meter.fallback_used and meter.as_dict()["models_by_tag"] == {"GRADE": "fb1"}
    assert meter.billed_calls == 1 and left == 2


def test_billed_budget_stops_the_run_while_run_budget_remains(env):
    """Earlier runs already billed 2 of 3: one unreadable reply is billed, the re-ask is refused (call cap)."""
    client, models = env.client(["not json", "not json"])
    with svc.usage_meter() as meter, svc.billed_budget(svc.MAX_BILLED_PER_COPY - 2):
        with pytest.raises(svc.CallCapExceeded):
            svc._generate_json(client, ["x"], "GRADE")
    assert models.calls == ["primary"] and meter.billed_calls == 1


def test_billed_cap_holds_across_retry_rounds(env):
    """Three job runs (retry rounds) of one copy, each told what earlier runs billed, as job_worker does:
    never more than MAX_BILLED_PER_COPY billed replies in total."""
    billed_total = 0
    for _ in range(3):
        client, models = env.client(["not json"] * 5)
        with svc.usage_meter() as meter, svc.billed_budget(svc.MAX_BILLED_PER_COPY - billed_total):
            with pytest.raises((svc.UnreadableReply, svc.CallCapExceeded)):
                svc._generate_json(client, ["x"], "GRADE")
        billed_total += meter.billed_calls
    assert billed_total == svc.MAX_BILLED_PER_COPY == 3


def test_reask_after_an_unreadable_reply_uses_the_same_run_budget(env):
    """503, 503, unreadable reply (3 of 3 attempts): no budget is left for the re-ask → UnreadableReply."""
    client, models = env.client([503, 503, "not json", '{"ok": true}'])
    with pytest.raises(svc.UnreadableReply):
        svc._generate_json(client, ["x"], "GRADE")
    assert len(models.calls) == 3


# ── billing ──────────────────────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("errors", [[429], [503], [500, 504], ["timeout"]])
def test_retries_are_not_billed(env, errors):
    client, models = env.client([*errors, json.dumps({"ok": True})], timeout=5)
    with svc.usage_meter() as meter, svc.billed_budget(3):
        assert svc._generate_json(client, ["x"], "GRADE") == {"ok": True}
        left = svc._billed_left.get()[0]
    assert len(models.calls) == len(errors) + 1
    assert meter.billed_calls == 1 and left == 2          # only the reply


# ── dead code ────────────────────────────────────────────────────────────────────────────────────────────────

def test_dead_test_checker_code_is_gone():
    assert not (BACKEND / "app/routers/test_checker_router.py").exists()
    assert not (BACKEND / "app/services/test_checker_service.py").exists()
    refs = [p for base in ("app", "tests", "scripts") for p in (BACKEND / base).rglob("*.py")
            if p.name != Path(__file__).name
            and ("test_checker_router" in p.read_text(encoding="utf-8", errors="ignore")
                 or "test_checker_service" in p.read_text(encoding="utf-8", errors="ignore"))]
    assert refs == []
