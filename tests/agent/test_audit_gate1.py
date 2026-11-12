"""Gate-1 audit, agent + voice side (docs/COST_MAP.md §2, §3, §4, §7). Mocked providers, no network.

*_today tests pin current behaviour; xfail(strict=True) tests state the correct behaviour and flip when fixed.
"""
from __future__ import annotations

import json
import threading
import uuid

import pytest

from _agent_ids import ADMIN, DECOY, DECOY_ADMIN, MAIN, SHARMA
from agent import config
from agent.core import Agent
from agent.llm import LLMResponse
from agent.schemas import SessionCtx
from agent.voice import limits
from agent.voice import config as vc
from agent.voice.providers import Chain, FakeSTT, ProviderDown, ProviderError, set_providers
from test_voice_phase1 import H, api  # noqa: F401  (fixture)
from voice_fixtures import webm

TARGET_MAX_PAID_PER_ACTION = 3
CTX = SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN)


class Http429(Exception):
    status_code = 429


class Http500(Exception):
    status_code = 500


class Failing:
    """429, then 500 forever (the brief's scenario)."""
    model = "audit-model"

    def __init__(self):
        self.calls = 0

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False, cancel=None):
        self.calls += 1
        raise Http429("rate") if self.calls == 1 else Http500("boom")


class FailingSTT(FakeSTT):
    def __init__(self, name, code):
        super().__init__(text="x", name=name)
        self.code, self.calls = code, 0

    def transcribe(self, *a, **k):
        self.calls += 1
        raise ProviderError(self.name, f"http_{self.code}", retryable=True)


# ── §2 retry + fallback amplification ─────────────────────────────────────────────────────────────────────────

def test_agent_turn_429_then_500_is_three_requests(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKOFF_BASE_S", 0.0)
    prov = Failing()
    evs = list(Agent(prov, CTX, str(uuid.uuid4())).run("attendance batao"))
    assert prov.calls == 1 + config.LLM_RETRIES == 3
    assert evs[-1]["type"] == "final"
    assert any(e["type"] == "error" and e.get("retryable") for e in evs)


class FlakyThenTool:
    """Fails twice, answers with a tool call, then fails for good: 3 + 2 requests, 3 retries, then the budget."""
    model = "audit-model"

    def __init__(self):
        self.calls = 0

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False, cancel=None):
        from agent.llm import ToolCall
        self.calls += 1
        if self.calls in (1, 2, 4, 5, 6):
            raise Http500("boom")
        return LLMResponse(text="", model=self.model, tool_calls=[ToolCall("t1", "my_scope", {})])


def test_agent_turn_retries_capped_at_3_across_steps(monkeypatch):
    """Gate 2 #4: TURN_RETRY_BUDGET=3 across all steps of one turn (LLM_RETRIES=2 is per step)."""
    monkeypatch.setattr(config, "LLM_BACKOFF_BASE_S", 0.0)
    prov = FlakyThenTool()
    evs = list(Agent(prov, CTX, str(uuid.uuid4())).run("meri classes batao"))
    assert prov.calls == 5                                   # step 1: fail, fail, ok; step 2: fail, fail -> stop
    assert evs[-1]["usage"]["retries"] == config.TURN_RETRY_BUDGET == 3
    assert any(e["type"] == "error" and e.get("retryable") for e in evs)


def test_voice_stt_fallback_is_two_requests():
    sarvam, gemini = FailingSTT("sarvam", 429), FailingSTT("gemini", 500)
    with pytest.raises(ProviderDown):
        Chain([sarvam, gemini]).transcribe(b"x", "audio/webm", "hi-IN")
    assert (sarvam.calls, gemini.calls) == (1, 1)


# ── §3 quota correctness (voice) ──────────────────────────────────────────────────────────────────────────────

def _daily_used(uid):
    d = limits._daily.get(str(uid))
    return d[1] if d else 0


def test_voice_failure_after_quota_check_is_counted_today(api):  # noqa: F811
    c, *_ = api
    set_providers([FailingSTT("sarvam", 500)], None)
    r = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(2))
    assert r.status_code == 503
    assert _daily_used(ADMIN) == 1          # charged although nothing was transcribed


def test_voice_rejected_upload_is_counted_today(api):  # noqa: F811
    c, *_ = api
    r = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}),
               content=b"\x00" * (vc.max_bytes() + 1))
    assert r.status_code == 413
    assert _daily_used(ADMIN) == 1          # counted before the size / format checks


@pytest.mark.xfail(strict=True, reason="AUDIT: check_user counts before validation and before STT")
def test_voice_failed_request_not_counted(api):  # noqa: F811
    c, *_ = api
    set_providers([FailingSTT("sarvam", 500)], None)
    c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(2))
    assert _daily_used(ADMIN) == 0


def test_voice_concurrent_last_unit_single_process_ok(monkeypatch):
    limits.reset()
    monkeypatch.setenv("VOICE_USER_DAILY", "1")
    monkeypatch.setattr(vc, "user_daily", lambda env=None: 1)
    results = []
    ts = [threading.Thread(target=lambda: results.append(limits.check_user("u-race"))) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sum(r is None for r in results) == 1     # correct inside one process (lock)
    limits.reset()


def test_voice_limits_are_per_process_today():
    """Counters live in module dicts: a second worker / instance starts at zero and a restart forgets them."""
    assert isinstance(limits._daily, dict) and isinstance(limits._recent, dict)


def test_voice_retry_same_key_after_success_not_counted_twice(api):  # noqa: F811
    c, *_ = api
    h = H(ADMIN, **{"Content-Type": "audio/webm", "Idempotency-Key": "k1"})
    assert c.post("/agent/voice/transcribe", headers=h, content=webm(2)).status_code == 200
    assert c.post("/agent/voice/transcribe", headers=h, content=webm(2)).status_code == 200
    assert _daily_used(ADMIN) == 1          # correct


def test_voice_retry_same_key_after_failure_counted_again_today(api):  # noqa: F811
    c, *_ = api
    h = H(ADMIN, **{"Content-Type": "audio/webm", "Idempotency-Key": "k2"})
    set_providers([FailingSTT("sarvam", 500)], None)
    c.post("/agent/voice/transcribe", headers=h, content=webm(2))
    c.post("/agent/voice/transcribe", headers=h, content=webm(2))
    assert _daily_used(ADMIN) == 2


def test_voice_speak_has_no_per_user_quota_today(api):  # noqa: F811
    c, *_ = api
    for _ in range(5):
        c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "Aaj attendance 92% hai."})
    assert _daily_used(ADMIN) == 0          # TTS (paid per character) is never counted


def test_voice_default_daily_cap_is_not_plan_aware():
    assert vc.USER_DAILY_DEFAULT == 100     # same for every plan, incl. Free


# ── §4 cache isolation (agent) ────────────────────────────────────────────────────────────────────────────────

def test_entity_cache_never_crosses_institutes():
    from agent.suggest import EntityCache
    cache = EntityCache(ttl_s=300)
    a = SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN)
    b = SessionCtx(user_id=DECOY_ADMIN, role="admin", institute_id=DECOY)
    same_user_other_inst = SessionCtx(user_id=ADMIN, role="admin", institute_id=DECOY)
    assert cache.get(a, lambda c: [{"inst": str(c.institute_id)}]) == [{"inst": str(MAIN)}]
    assert cache.get(b, lambda c: [{"inst": str(c.institute_id)}]) == [{"inst": str(DECOY)}]
    assert cache.get(same_user_other_inst, lambda c: [{"inst": str(c.institute_id)}]) == [{"inst": str(DECOY)}]
    teacher = SessionCtx(user_id=ADMIN, role="teacher", institute_id=MAIN, teacher_id=ADMIN)
    assert cache.get(teacher, lambda c: ["teacher-scope"]) == ["teacher-scope"]   # role is in the key


def test_identity_cache_key_is_per_user():
    from agent.auth import _IDENTITY, cached_identity, invalidate_identity
    invalidate_identity()
    a = cached_identity(("live", str(ADMIN), "f"), lambda: SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN))
    b = cached_identity(("live", str(DECOY_ADMIN), "f"),
                        lambda: SessionCtx(user_id=DECOY_ADMIN, role="admin", institute_id=DECOY))
    assert a.institute_id == MAIN and b.institute_id == DECOY and len(_IDENTITY) == 2
    invalidate_identity()


def test_idempotency_cache_is_per_user():
    limits.reset()
    limits.idem_put(str(ADMIN), "same-key", {"text": "admin's words"})
    assert limits.idem_get(str(DECOY_ADMIN), "same-key") is None
    limits.reset()


def test_stale_window_after_direct_db_writes_today():
    """Frontend writes institute_members / students straight to Supabase (no backend call), so nothing can
    invalidate these: a removed teacher keeps agent access <= 60 s; a new student is unknown to suggestions /
    voice name-repair <= 300 s."""
    from agent.suggest import ENTITY_TTL_S
    assert ENTITY_TTL_S == 300 and vc.ENTITY_TTL_S == 300
    import os
    assert float(os.getenv("AGENT_IDENTITY_CACHE_S", "60")) == 60


# ── §7 attribution ────────────────────────────────────────────────────────────────────────────────────────────

def _trace_rows(monkeypatch, tmp_path):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setattr(config, "TRACE_PATH", str(path))
    prov = type("P", (), {"model": "audit-model",
                          "complete": lambda self, *a, **k: LLMResponse(text="Theek hai.", model="audit-model",
                                                                        tokens_in=10, tokens_out=5)})()
    sharma = SessionCtx(user_id=SHARMA, role="teacher", institute_id=MAIN, teacher_id=SHARMA)
    list(Agent(prov, sharma, str(uuid.uuid4())).run("Riya Sharma ki attendance"))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_agent_trace_has_no_names_and_no_institute_today(monkeypatch, tmp_path):
    rows = _trace_rows(monkeypatch, tmp_path)
    blob = json.dumps(rows)
    assert rows and "Riya" not in blob                        # message text / names never traced (correct)
    assert any(r.get("kind") == "llm" and "cost_usd" in r for r in rows)
    assert str(MAIN) not in blob and str(SHARMA) not in blob  # cost rows carry request_id only


@pytest.mark.xfail(strict=True, reason="AUDIT: agent/voice trace rows have no institute_id / feature field")
def test_agent_trace_carries_institute_and_feature(monkeypatch, tmp_path):
    rows = _trace_rows(monkeypatch, tmp_path)
    llm = [r for r in rows if r.get("kind") == "llm"]
    assert llm and all(r.get("institute_id") == str(MAIN) and r.get("feature") for r in llm)
