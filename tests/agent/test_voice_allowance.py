"""Gate 2 #5: monthly per-plan voice allowance (agent/voice/allowance.py). No DB, no network."""
from __future__ import annotations

import pytest

from _agent_ids import ADMIN, MAIN
from agent.schemas import SessionCtx
from agent.voice import allowance
from test_voice_phase1 import H, api  # noqa: F401
from voice_fixtures import webm

DEMO = SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN, data_source="demo")
LIVE = SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN, data_source="live")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    allowance.reset()
    monkeypatch.setattr(allowance, "live_consume", lambda uid: False)   # never touch a DB here
    yield
    allowance.reset()


def test_demo_default_and_counting(monkeypatch):
    monkeypatch.delenv("VOICE_DEMO_MONTHLY_LIMIT", raising=False)
    a = allowance.get(DEMO)
    assert (a.limit, a.used, a.source) == (allowance.DEMO_MONTHLY_DEFAULT, 0, "demo")
    allowance.consume(DEMO)
    assert allowance.get(DEMO).used == 1 and allowance.check(DEMO) is None


def test_live_unset_is_zero_and_blocked(monkeypatch):
    monkeypatch.setattr(allowance, "live_fetch", lambda uid: None)      # migration 32 not applied
    assert allowance.check(LIVE) == allowance.MSG_NOT_IN_PLAN
    monkeypatch.setattr(allowance, "live_fetch", lambda uid: {"limit": 0, "used": 0})
    allowance.reset()
    assert allowance.check(LIVE) == allowance.MSG_NOT_IN_PLAN


def test_live_limit_and_month_done(monkeypatch):
    monkeypatch.setattr(allowance, "live_fetch", lambda uid: {"limit": 2, "used": 1})
    assert allowance.check(LIVE) is None
    allowance.consume(LIVE)                      # write refused -> local count 1, max(db 1, local 1) = 1
    allowance.consume(LIVE)                      # local 2 -> blocked
    assert "2/2" in (allowance.check(LIVE) or "")
    monkeypatch.setattr(allowance, "live_fetch", lambda uid: {"limit": -1, "used": 999})
    allowance.reset()
    assert allowance.check(LIVE) is None         # -1 = unlimited


def test_live_blocked_user_gets_403_before_stt(api, monkeypatch):  # noqa: F811
    """Through the API: a live user whose plan has no voice is refused before any provider call."""
    c, _prov, _st, _trace, stt, _tts = api
    import agent.voice.api as vapi
    monkeypatch.setattr(allowance, "live_fetch", lambda uid: {"limit": 0, "used": 0})
    live_ctx = LIVE

    def gate_live():
        return live_ctx
    c.app.dependency_overrides[vapi.voice_gate] = gate_live
    try:
        r = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(2))
        assert r.status_code == 403 and "plan" in r.json()["detail"]
        r = c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "hi"})
        assert r.status_code == 403
    finally:
        c.app.dependency_overrides.pop(vapi.voice_gate, None)
    assert stt.calls == []          # refused before any STT call
