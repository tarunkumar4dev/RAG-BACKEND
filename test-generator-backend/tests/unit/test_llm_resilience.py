"""Upstream overload (17 Nov 22:53: gemini-3.5-flash-lite 503 "model is overloaded"): switch to the fallback model
instead of retrying the primary, keep a partial paper when the paid-call budget runs out, and answer a paper with
zero questions caused by the provider with the friendly 503 + Retry-After, not a 500. Mocked models only."""
import asyncio
import concurrent.futures
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.v1.endpoints import test_generator as ep
from app.core.auth import AuthUser
from app.models.test_generator import ChapterSection, TestGenerationRequest
from app.services import test_generator_service as g

from audit_support import FakeGenModels, fake_parse_batch_factory, force_gemini, no_network  # noqa: F401
from test_p0_job_status import FakeDB

OVERLOADED = "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'The model is overloaded. Please try again later.'}}"
OK = json.dumps({"questions": []})
_REAL_POOL = concurrent.futures.ThreadPoolExecutor


class InlineExecutor(concurrent.futures.Executor):
    """Runs each submitted batch at once, in order, so which batch meets the spent budget is deterministic."""

    def __init__(self, *a, **k):
        pass

    def submit(self, fn, *args, **kwargs):
        f = concurrent.futures.Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except BaseException as e:  # noqa: BLE001
            f.set_exception(e)
        return f


@pytest.fixture
def gen(monkeypatch, no_network, force_gemini):  # noqa: F811
    from app.core.config import settings
    monkeypatch.setattr(settings, "GEMINI_GEN_MODEL", "primary")
    monkeypatch.setattr(settings, "GEMINI_FALLBACK_MODEL", "fallback")
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    monkeypatch.setattr(g.concurrent.futures, "ThreadPoolExecutor", InlineExecutor)
    parse = fake_parse_batch_factory()
    monkeypatch.setattr(g, "_parse_batch", parse)

    def make(script, per_reply=None):
        if per_reply is not None:
            monkeypatch.setattr(g, "_parse_batch", lambda *a, **k: parse(*a, **k)[:per_reply])
        models = FakeGenModels(script)
        client = SimpleNamespace(models=models)
        monkeypatch.setattr(g, "_get_gemini_client", lambda: client)
        return models
    return make


def _req(*chapters):
    return TestGenerationRequest(exam_title="T", subject="Science", class_grade="10", teacher_id="u",
                                 chapters=list(chapters))


def _ch(name="Light", n=4):
    return ChapterSection(chapter=name, quantity=n, difficulty="medium", format="mcq", marks_per_question=1)


CHUNKS = [{"chapter": "Light", "content": "light " * 300}, {"chapter": "Electricity", "content": "current " * 300}]


def _calls(models):
    return [m for m, _ in models.calls]


# ── 1. overload -> fallback at once; 429 still retries the same model ─────────────────────────────────────────

def test_primary_503_goes_straight_to_fallback_without_a_primary_retry(gen):
    models = gen(lambda n: Exception(OVERLOADED) if n == 1 else OK)
    with g.paper_usage() as stats:
        qs = g.generate_questions(_req(_ch(n=4)), CHUNKS)       # 4 -> ask 5 -> one batch
    assert len(qs) == 4
    assert _calls(models) == ["primary", "fallback"]
    assert stats.extra == 1                                     # the fallback only; no primary retry was paid for


def test_429_is_still_retried_on_the_same_model(gen):
    models = gen(lambda n: Exception("429 RESOURCE_EXHAUSTED") if n == 1 else OK)
    with g.paper_usage() as stats:
        g.generate_questions(_req(_ch(n=4)), CHUNKS)
    assert _calls(models) == ["primary", "primary"] and stats.extra == 1


def test_later_batches_start_on_the_fallback_once_the_primary_is_overloaded(gen):
    models = gen(lambda n: Exception(OVERLOADED) if n == 1 else OK)
    with g.paper_usage() as stats:
        qs = g.generate_questions(_req(_ch(n=5)), CHUNKS)       # 5 -> ask 6 -> batches of 5 and 1
    assert len(qs) == 5
    assert _calls(models) == ["primary", "fallback", "fallback"]
    assert stats.extra == 1                                     # batch 2's first attempt is on the fallback, free


def test_the_answer_checker_keeps_its_retry_on_overload(gen):
    """The checker has no fallback model; it still gets its one retry on the same model."""
    models = gen(lambda n: Exception(OVERLOADED) if n == 1 else OK)
    client = g._get_gemini_client()
    assert g._call_gemini_model(client, "p", "checker", temperature=0.1) == OK
    assert _calls(models) == ["checker", "checker"]


# ── 2 + 3. through the endpoint: partial paper -> 200; nothing because of the provider -> 503 ────────────────

USER = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def endpoint(monkeypatch, gen):
    db = FakeDB()
    monkeypatch.setattr(ep, "get_supabase_admin", lambda: db)
    monkeypatch.setattr(ep, "reserve", lambda *a, **k: SimpleNamespace(usage={}))
    monkeypatch.setattr(ep, "consume", lambda rsv: {"used": 1, "limit": 10, "remaining": 9})
    monkeypatch.setattr(ep, "release", lambda rsv: None)
    monkeypatch.setattr(ep, "log_usage", lambda *a, **k: None)
    monkeypatch.setattr(ep, "retrieve_context", lambda *a, **k: CHUNKS)

    def call(rows):
        req = ep.FrontendGenerateRequest(subject="Science", cbsePattern=False,
                                         simpleData=[ep.FrontendChapterRow(topic=t, quantity=n) for t, n in rows])
        user = AuthUser(id=USER, email=None, is_anonymous=False, role="authenticated")
        return asyncio.run(ep.generate_from_frontend(req, user, "k-1"))
    call.db = db
    return call


def test_budget_runs_out_after_6_of_10_questions_returns_the_6(endpoint, gen):
    # 10 -> ask 11 -> batches of 5, 5, 1. Batches 1 and 2 give 3 questions each; batch 3 meets a rate-limited
    # provider: primary 429, retry (extra 1), 429, fallback (extra 2), 429, retry refused -> budget spent.
    models = gen(lambda n: OK if n <= 2 else Exception("429 RESOURCE_EXHAUSTED"), per_reply=3)
    resp = endpoint([("Light", 10)])
    assert resp.ok is True and resp.totalQuestions == 6
    assert (resp.meta["requestedQuestions"], resp.meta["deliveredQuestions"]) == (10, 6)
    assert len(models.calls) == 3 + g.PAID_EXTRA_BUDGET        # first attempts + the 2 extras, nothing more
    assert [s for _, s in endpoint.db.history] == ["generating", "draft"]


@pytest.mark.parametrize("rows", [[("Light", 10)], [("Light", 5), ("Electricity", 5)]],
                         ids=["one-chapter", "two-chapters"])
def test_every_call_overloaded_is_a_503_with_retry_after(endpoint, gen, rows):
    gen(lambda n: Exception(OVERLOADED))
    with pytest.raises(HTTPException) as e:
        endpoint(rows)
    assert e.value.status_code == 503
    assert e.value.headers == {"Retry-After": "120"}
    assert e.value.detail == g.MSG_PAPER_FAILED
    assert [s for _, s in endpoint.db.history] == ["generating", "failed"]


def test_overload_with_real_parallel_batches(endpoint, gen, monkeypatch):
    """Same as above with the production thread pools (batches in parallel): still a 503, never a 500, and the
    paper never makes more than its first attempts + PAID_EXTRA_BUDGET calls."""
    monkeypatch.setattr(g.concurrent.futures, "ThreadPoolExecutor", _REAL_POOL)
    models = gen(lambda n: Exception(OVERLOADED))
    with pytest.raises(HTTPException) as e:
        endpoint([("Light", 5), ("Electricity", 5)])          # 2 chapters x batches of 5 + 1 = 4 first attempts
    assert e.value.status_code == 503 and e.value.headers == {"Retry-After": "120"}
    assert len(models.calls) <= 4 + g.PAID_EXTRA_BUDGET


def test_a_genuine_error_with_no_questions_is_still_a_500(endpoint, gen):
    gen(lambda n: Exception("400 INVALID_ARGUMENT: response_mime_type is not supported"))
    with pytest.raises(HTTPException) as e:
        endpoint([("Light", 10)])
    assert e.value.status_code == 500


def test_budget_values_unchanged():
    assert g.MAX_RETRIES == 2 and g.PAID_EXTRA_BUDGET == 2
