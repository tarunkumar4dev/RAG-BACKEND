"""Gate-1 audit §2: retry + fallback amplification (docs/COST_MAP.md §2). Mocked providers only.

Scenario from the brief: the provider answers 429, then 500, then keeps failing (worst case) — or answers 200
with an unusable body (the only failure Google/DeepSeek bill for). These tests pin TODAY's worst case so it cannot
grow unnoticed; the `xfail(strict=True)` tests state the target (≤ 3 paid calls per user action) and will start
failing — i.e. must be flipped to normal tests — once a fix lands.
"""
from __future__ import annotations

import json

import pytest

from app.services import smart_assignment_service as svc
from app.services import test_generator_service as g
from app.models.test_generator import ChapterSection, TestGenerationRequest

from audit_support import FakeGenModels, fake_parse_batch_factory, force_gemini, no_network  # noqa: F401

TARGET_MAX_PAID_PER_ACTION = 3


def _req(*chapters):
    return TestGenerationRequest(exam_title="Audit", subject="Science", class_grade="10", teacher_id="u",
                                 chapters=list(chapters))


LIGHT = ChapterSection(chapter="Light", quantity=5, difficulty="medium", format="mcq", marks_per_question=1)
ELEC = ChapterSection(chapter="Electricity", quantity=5, difficulty="medium", format="short_answer",
                      marks_per_question=3)
CHUNKS = [{"chapter": "Light", "content": "light " * 300}, {"chapter": "Electricity", "content": "current " * 300}]


@pytest.fixture
def gen(monkeypatch, no_network, force_gemini):  # noqa: F811
    """Generator with two models (primary + fallback, as in production config) and no real sleeping."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "GEMINI_GEN_MODEL", "primary")
    monkeypatch.setattr(settings, "GEMINI_FALLBACK_MODEL", "fallback")
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    monkeypatch.setattr(g, "_parse_batch", fake_parse_batch_factory())

    def make(script):
        models = FakeGenModels(script)
        client = type("C", (), {"models": models})()
        monkeypatch.setattr(g, "_get_gemini_client", lambda: client)
        return models
    return make


def _fail_429_then_500_forever(n):
    return Exception("429 RESOURCE_EXHAUSTED") if n == 1 else Exception("500 INTERNAL")


ONE_BATCH = ChapterSection(chapter="Light", quantity=4, difficulty="medium", format="mcq", marks_per_question=1)
# quantity 4 -> ask 5 -> a single batch, so the call order is deterministic


# ── test generator (Gate 2 #4: first attempt per batch + PAID_EXTRA_BUDGET=3 extras per paper, no top-ups) ──

def test_generator_429_then_500_then_fallback_ok(gen):
    """429 -> retry (extra 1) -> 500 -> primary exhausted -> fallback model (extra 2) -> usable."""
    models = gen(lambda n: Exception("429 RESOURCE_EXHAUSTED") if n == 1 else
                 (Exception("500 INTERNAL") if n == 2 else json.dumps({"questions": []})))
    qs = g.generate_questions(_req(ONE_BATCH), CHUNKS)
    assert len(qs) == 4
    assert [m for m, _ in models.calls] == ["primary", "primary", "fallback"]


def test_generator_429_then_500_forever_stops_after_budget(gen):
    """Two batches (5 + 1 questions): 2 first attempts + 3 extras = 5 provider requests, then a fast 503."""
    models = gen(_fail_429_then_500_forever)
    with pytest.raises(g.GenerationError) as e:
        g.generate_questions(_req(LIGHT), CHUNKS)
    assert e.value.status_code == 503 and str(e.value) == g.MSG_PAPER_FAILED
    assert len(models.calls) == 2 + g.PAID_EXTRA_BUDGET == 5
    assert g.MAX_RETRIES == 2 and g.PAID_EXTRA_BUDGET == 3


def test_generator_paid_calls_capped_one_chapter(gen):
    """Every reply billed but unusable: 2 first attempts + 3 extras = 5 paid calls (was 10)."""
    models = gen(lambda n: "UNUSABLE")
    with pytest.raises(g.GenerationError) as e:
        g.generate_questions(_req(LIGHT), CHUNKS)
    assert e.value.status_code == 503
    # exact count depends on which parallel batch spends the last unit; never more than first attempts + budget
    assert 3 <= len(models.calls) <= 2 + g.PAID_EXTRA_BUDGET


def test_generator_paid_calls_capped_cbse_paper(gen):
    """2-chapter CBSE paper, every reply unusable: 14 first-wave batches (cost_guard_baseline) + 3 extras at most
    (was 100)."""
    models = gen(lambda n: "UNUSABLE")
    with pytest.raises(g.GenerationError) as e:
        g.generate_questions(_req(LIGHT, ELEC), CHUNKS, cbse_pattern=True)
    assert e.value.status_code == 503
    assert len(models.calls) <= 14 + TARGET_MAX_PAID_PER_ACTION


def test_generator_no_top_up_rounds(gen):
    """A short batch is returned short: no second paid round for the shortfall."""
    short = fake_parse_batch_factory()
    calls = []

    def parse(raw, chapter, request, section_key=None):
        calls.append(1)
        return short(raw, chapter, request, section_key)[:2]
    import app.services.test_generator_service as mod
    mod._parse_batch = parse
    models = gen(lambda n: json.dumps({"questions": []}))
    qs = g.generate_questions(_req(ONE_BATCH), CHUNKS)
    assert len(qs) == 2 and len(models.calls) == 1


def test_generator_deepseek_fallback_counts_against_budget(gen, monkeypatch):
    """LLM_PROVIDER=deepseek: DeepSeek retry (1) + Gemini fallback (2) + Gemini retry (3) -> then 503."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "LLM_PROVIDER", "deepseek")
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "audit-not-a-key")
    ds_calls = []

    class Resp:
        status_code = 500
        text = "boom"

    def fake_post(url, **kw):
        ds_calls.append(url)
        return Resp()
    monkeypatch.setattr(g.httpx, "post", fake_post)
    models = gen(_fail_429_then_500_forever)
    with pytest.raises(g.GenerationError) as e:
        g.generate_questions(_req(ONE_BATCH), CHUNKS)
    assert e.value.status_code == 503
    assert len(ds_calls) + len(models.calls) == 1 + TARGET_MAX_PAID_PER_ACTION


# ── copy checker (smart_assignment_service) ─────────────────────────────────────────────────────────────────

@pytest.fixture
def checker(monkeypatch, no_network):  # noqa: F811
    monkeypatch.setattr(svc, "GEMINI_MOCK", False)
    monkeypatch.setattr(svc, "CHECKER_MODEL", "primary")
    monkeypatch.setattr(svc, "CHECKER_FALLBACK_MODELS", ["fb1", "fb2"])
    monkeypatch.setattr(svc.time, "sleep", lambda s: None)
    from google.genai import errors as genai_errors
    from types import SimpleNamespace

    def err(code):
        return genai_errors.APIError(code, {"error": {"code": code, "message": "x", "status": "X"}},
                                     SimpleNamespace(headers={}))

    def make(script):
        calls = []

        class Models:
            def generate_content(self, model, contents, config):
                calls.append(model)
                item = script[len(calls) - 1] if len(calls) <= len(script) else script[-1]
                if isinstance(item, int):
                    raise err(item)
                return SimpleNamespace(text=item, usage_metadata=SimpleNamespace(
                    prompt_token_count=1, candidates_token_count=1, total_token_count=2, thoughts_token_count=0),
                    candidates=[])
        return SimpleNamespace(models=Models()), calls
    return make


def test_checker_429_then_500_is_two_unbilled_calls_per_run(checker):
    client, calls = checker([429, 500])
    with svc.usage_meter() as meter:
        with pytest.raises(svc.GeminiBusy):
            svc._generate_json(client, ["x"], "GRADE")
    assert calls == ["primary", "primary"] and meter.calls == 2


def test_checker_attempts_per_run_capped_at_3(checker):
    """503 on every model: 3 attempts (not 6), then GeminiBusy for the worker to re-queue."""
    client, calls = checker([503] * 6)
    with svc.usage_meter() as meter:
        with pytest.raises(svc.GeminiBusy):
            svc._generate_json(client, ["x"], "GRADE")
    assert len(calls) == svc.MAX_ATTEMPTS_PER_RUN == 3 and meter.calls == 3


def test_checker_paid_calls_per_copy_capped():
    """Billed worst case (200 + unreadable JSON twice): UnreadableReply is FINAL, so the worker does not re-run
    the copy -> 2 paid grading calls per copy (was 2 x 3 runs = 6)."""
    from app.services import job_worker as jw
    from types import SimpleNamespace
    client, calls = None, None

    class Models:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append(model)
            return SimpleNamespace(text="not json", usage_metadata=SimpleNamespace(
                prompt_token_count=1, candidates_token_count=1, total_token_count=2, thoughts_token_count=0),
                candidates=[])
    import pytest as _pt
    m = Models()
    with _pt.MonkeyPatch.context() as mp:
        mp.setattr(svc, "GEMINI_MOCK", False)
        mp.setattr(svc, "CHECKER_MODEL", "primary")
        mp.setattr(svc, "CHECKER_FALLBACK_MODELS", ["fb1", "fb2"])
        mp.setattr(svc.time, "sleep", lambda s: None)
        with svc.usage_meter():
            with pytest.raises(svc.UnreadableReply) as e:
                svc._generate_json(client or SimpleNamespace(models=m), ["x"], "GRADE")
    paid_per_run = len(m.calls)
    code, msg, _ = jw.JobWorker._classify(jw.JobWorker, e.value)
    assert paid_per_run == 2 and getattr(e.value, "final", False) is True
    assert code == "AI_UNREADABLE" and "manually" in msg
    assert paid_per_run * 1 <= TARGET_MAX_PAID_PER_ACTION      # final -> exactly one run
