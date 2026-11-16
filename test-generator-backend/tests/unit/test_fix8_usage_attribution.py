"""Gate 2 #8: papers and worksheets record input/output tokens with feature + institute_id (ids only, no names),
on the existing tests.request_payload column and as one AI_USAGE log line. No new tables, no network."""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from app.core import usage_log
from app.services import test_generator_service as g

from audit_support import no_network  # noqa: F401
from test_audit_quota import USER, _safe_call, env  # noqa: F401  (fixture)

INST = "aaaaaaaa-0000-0000-0000-000000000001"


@pytest.fixture(autouse=True)
def inst(monkeypatch):
    usage_log.clear_cache()
    monkeypatch.setattr(usage_log, "lookup_institute", lambda uid: INST if uid == USER else None)
    yield
    usage_log.clear_cache()


def _lines(caplog):
    return [json.loads(r.getMessage().split("AI_USAGE ", 1)[1]) for r in caplog.records
            if r.getMessage().startswith("AI_USAGE ")]


def _gen_with_calls(n_calls, fail=False):
    def gen(*a, **k):
        for _ in range(n_calls):
            g._record_call("gemini-3.5-flash-lite", 2700, 1500)
        if fail:
            raise g.GenerationError(g.MSG_PAPER_FAILED, 503)
        return SimpleNamespace()
    return gen


def test_paper_usage_logged_and_stored_on_tests_row(env, caplog):  # noqa: F811
    db = env(limit=10, generate=_gen_with_calls(4))
    with caplog.at_level(logging.INFO, logger="a4ai.usage"):
        assert _safe_call("p1") == "ok"
    (rec,) = _lines(caplog)
    assert rec["feature"] == "paper" and rec["institute_id"] == INST and rec["user_id"] == USER
    assert (rec["calls"], rec["input_tokens"], rec["output_tokens"]) == (4, 10800, 6000)
    assert rec["est_cost_usd"] == round((10800 * 0.30 + 6000 * 2.50) / 1e6, 6)
    assert db.tests_rows[0]["request_payload"]["ai_usage"] == rec


def test_failed_paper_still_logs_what_it_spent(env, caplog):  # noqa: F811
    env(limit=10, generate=_gen_with_calls(3, fail=True))
    with caplog.at_level(logging.INFO, logger="a4ai.usage"):
        assert _safe_call("p2") == 503
    (rec,) = _lines(caplog)
    assert rec["outcome"] == "failed" and rec["calls"] == 3 and rec["institute_id"] == INST


def test_no_names_or_text_in_the_record(env, caplog):  # noqa: F811
    env(limit=10, generate=_gen_with_calls(1))
    with caplog.at_level(logging.INFO, logger="a4ai.usage"):
        _safe_call("p3")
    (rec,) = _lines(caplog)
    assert set(rec) == {"feature", "institute_id", "user_id", "outcome", "calls", "input_tokens", "output_tokens",
                        "est_cost_usd", "models"}


def test_teacher_without_institute_is_none(monkeypatch, caplog):
    with caplog.at_level(logging.INFO, logger="a4ai.usage"):
        rec = usage_log.log_usage("worksheet_direct", "someone-else", {"calls": 1, "input_tokens": 5})
    assert rec["institute_id"] is None


def test_institute_lookup_failure_never_breaks(monkeypatch):
    def boom(uid):
        raise RuntimeError("db down")
    monkeypatch.setattr(usage_log, "lookup_institute", boom)
    usage_log.clear_cache()
    assert usage_log.log_usage("paper", USER, {"calls": 1})["institute_id"] is None


def test_worksheet_direct_logs_usage(monkeypatch, no_network, caplog):  # noqa: F811
    import asyncio
    import google.genai as genai_mod
    from app.api.v1.endpoints import modules as ep
    from app.core.auth import AuthUser
    from app.services import module_service as ms

    class Client:
        def __init__(self, **_):
            self.models = SimpleNamespace(generate_content=lambda model, contents, config: SimpleNamespace(
                text=json.dumps({"questions": []}),
                usage_metadata=SimpleNamespace(prompt_token_count=15000, candidates_token_count=2500,
                                               thoughts_token_count=500)))
    monkeypatch.setattr(genai_mod, "Client", Client)
    monkeypatch.setattr(ms, "get_supabase", lambda: SimpleNamespace(storage=SimpleNamespace(
        from_=lambda b: SimpleNamespace(download=lambda p: b"%PDF-fake"))))
    monkeypatch.setattr(ms.ModuleService, "_extract_pdf", staticmethod(lambda path: ("Text. " * 100, 1, False)))
    monkeypatch.setattr(ep, "reserve", lambda u, a, k: SimpleNamespace(open=True))
    monkeypatch.setattr(ep, "consume", lambda r: {})
    monkeypatch.setattr(ep, "release", lambda r: None)
    ep._worksheet_limiter._hits.clear()
    user = AuthUser(id=USER, email=None, is_anonymous=False, role="authenticated")
    with caplog.at_level(logging.INFO, logger="a4ai.usage"):
        asyncio.run(ep.generate_worksheet_direct({"storage_path": f"{USER}/worksheets/w.pdf"}, user))
    (rec,) = _lines(caplog)
    assert rec["feature"] == "worksheet_direct" and rec["institute_id"] == INST
    assert (rec["input_tokens"], rec["output_tokens"]) == (15000, 3000)
