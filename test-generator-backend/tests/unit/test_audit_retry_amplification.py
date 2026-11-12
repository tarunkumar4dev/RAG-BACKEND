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


# ── test generator ───────────────────────────────────────────────────────────────────────────────────────────

def test_generator_429_then_500_then_ok_costs_one_paid_call(gen):
    models = gen(lambda n: Exception("429 RESOURCE_EXHAUSTED") if n == 1 else
                 (Exception("500 INTERNAL") if n == 2 else json.dumps({"questions": []})))
    qs = g.generate_questions(_req(LIGHT), CHUNKS)
    assert len(qs) == 5
    # 2 failed attempts (not billed) + the successful batches
    assert [m for m, _ in models.calls][:3] == ["primary", "primary", "primary"]


def test_generator_worst_case_attempts_per_chapter_today(gen):
    """1 chapter x 5 questions: ask 6 -> batches [5, 1] + up to 3 top-up rounds = 5 run_batch;
    each run_batch tries 2 models x MAX_RETRIES(4) = 8 attempts -> 40 provider requests."""
    models = gen(_fail_429_then_500_forever)
    with pytest.raises(g.GenerationError):
        g.generate_questions(_req(LIGHT), CHUNKS)
    assert g.MAX_RETRIES == 4
    assert len(models.calls) == 40


def test_generator_worst_case_paid_calls_per_chapter_today(gen):
    """Billed worst case: every reply is HTTP 200 but unusable -> no retry inside the model, next model, then
    top-up rounds: 5 run_batch x 2 models = 10 billed calls for ONE chapter."""
    models = gen(lambda n: "UNUSABLE")
    with pytest.raises(g.GenerationError):
        g.generate_questions(_req(LIGHT), CHUNKS)
    assert len(models.calls) == 10


def test_generator_worst_case_paid_calls_per_paper_today(gen):
    """A 2-chapter CBSE-pattern paper (the common teacher action) with every reply unusable."""
    models = gen(lambda n: "UNUSABLE")
    with pytest.raises(g.GenerationError):
        g.generate_questions(_req(LIGHT, ELEC), CHUNKS, cbse_pattern=True)
    assert len(models.calls) == 100     # 10 jobs (sections x chapters x formats) x 5 run_batch x 2 models


@pytest.mark.xfail(strict=True, reason="AUDIT: no per-paper call cap in the generator (target <= 3 paid retries)")
def test_generator_paid_calls_capped(gen):
    models = gen(lambda n: "UNUSABLE")
    with pytest.raises(g.GenerationError):
        g.generate_questions(_req(LIGHT), CHUNKS)
    assert len(models.calls) <= TARGET_MAX_PAID_PER_ACTION


def test_generator_deepseek_provider_doubles_attempts(gen, monkeypatch):
    """LLM_PROVIDER=deepseek (what the local .env sets): every model in the loop first retries DeepSeek 4x."""
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
    with pytest.raises(g.GenerationError):
        g.generate_questions(_req(LIGHT), CHUNKS)
    assert len(ds_calls) == 40          # 5 run_batch x 2 models x 4 DeepSeek attempts
    assert len(models.calls) == 40      # + the same 40 Gemini attempts


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


def test_checker_worst_case_per_copy_today(checker):
    """Per job run: <= 2 attempts x 3 models (503 path) = 6 requests, under the 12-call sheet cap. The job is re-run
    up to max_retries = 3 (check_jobs default, migration 18) -> 18 requests per copy. Billed worst case
    (200 + unreadable JSON): 2 per run x 3 runs = 6 full multimodal grading calls (~Rs 3.5 each, measured p50)."""
    client, calls = checker([503] * 6)
    with svc.usage_meter() as meter:
        with pytest.raises(svc.GeminiBusy):
            svc._generate_json(client, ["x"], "GRADE")
    assert len(calls) == 6 and meter.calls == 6
    assert svc.MAX_GEMINI_CALLS_PER_SHEET == 12

    client, calls = checker(["not json", "not json"])
    with svc.usage_meter():
        with pytest.raises(ValueError):
            svc._generate_json(client, ["x"], "GRADE")
    paid_per_run = len(calls)
    assert paid_per_run == 2
    assert paid_per_run * 3 == 6


@pytest.mark.xfail(strict=True, reason="AUDIT: a copy can be billed up to 6 grading calls (2 per run x 3 runs)")
def test_checker_paid_calls_per_copy_capped():
    assert 2 * 3 <= TARGET_MAX_PAID_PER_ACTION
