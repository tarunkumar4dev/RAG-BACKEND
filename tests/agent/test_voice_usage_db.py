"""Voice upgrade phase 3: usage counts live in the agent store (shared across processes, survive restarts);
institute_id + feature on every cost record; the admin usage endpoint. No network."""
from __future__ import annotations

import json
import os
import uuid

import pytest

from _agent_ids import ADMIN, DECOY, DECOY_ADMIN, MAIN, SHARMA
from agent import config
from agent.core import Agent
from agent.llm import LLMResponse
from agent.schemas import SessionCtx
from agent.store import Owner, SQLiteStore, set_store
from agent.voice import allowance
from agent.voice import config as vc
from test_voice_phase1 import H, api, events  # noqa: F401
from voice_fixtures import webm

CTX = SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN, data_source="demo")


def test_two_processes_share_the_count(tmp_path, monkeypatch):
    """Two SQLiteStore instances on one file = two server processes. Exhaust the daily limit through one, the other
    sees it exhausted; a 'restart' (new instance) still sees the count."""
    path = tmp_path / "store.db"
    a, b = SQLiteStore(path), SQLiteStore(path)
    monkeypatch.setattr(vc, "user_daily", lambda env=None: 3)
    allowance.reset()
    set_store(a)
    for _ in range(3):
        assert allowance.check(CTX) is None
        allowance.consume(CTX)
    assert "Aaj ki voice limit" in (allowance.check(CTX) or "")
    set_store(b)                                     # the other process
    assert allowance.get(CTX).used_day == 3 and "Aaj ki" in (allowance.check(CTX) or "")
    set_store(SQLiteStore(path))                     # after a restart
    assert allowance.get(CTX).used == 3
    set_store(None)


def test_counts_are_per_institute_and_user(tmp_path):
    st = SQLiteStore(tmp_path / "s.db")
    set_store(st)
    other_inst = SessionCtx(user_id=ADMIN, role="admin", institute_id=DECOY, data_source="demo")
    other_user = SessionCtx(user_id=SHARMA, role="teacher", institute_id=MAIN, teacher_id=SHARMA, data_source="demo")
    allowance.consume(CTX); allowance.consume(CTX)
    allowance.consume(other_inst)
    assert allowance.get(CTX).used == 2 and allowance.get(other_inst).used == 1 and allowance.get(other_user).used == 0
    rows = st.voice_usage_by_institute(allowance.month_period())
    assert {r["institute_id"]: r["queries"] for r in rows} == {str(MAIN): 2, str(DECOY): 1}
    assert all(set(r) == {"institute_id", "users", "queries"} for r in rows)   # no user ids
    set_store(None)


def test_transcribe_counts_only_on_success_in_the_store(api):  # noqa: F811
    c, *_ = api
    allowance.reset()
    assert allowance.get(CTX).used_day == 0
    assert c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(2)).status_code == 200
    assert allowance.get(CTX).used_day == 1 and allowance.get(CTX).used == 1
    r = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=b"\x00" * 40)
    assert r.status_code == 422 and allowance.get(CTX).used_day == 1
    cfg = c.get("/agent/voice/config", headers=H(ADMIN)).json()["limits"]
    assert cfg["day_used"] == 1 and cfg["month_used"] == 1


def test_daily_cap_blocks_before_stt(api, monkeypatch):  # noqa: F811
    c, _prov, _st, _trace, stt, _tts = api
    monkeypatch.setattr(vc, "user_daily", lambda env=None: 1)
    allowance.consume(CTX)
    r = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(2))
    assert r.status_code == 429 and "Aaj ki" in r.json()["detail"] and int(r.headers["retry-after"]) > 0
    assert stt.calls == []


def test_admin_endpoint_own_institute_and_ops_token(api, monkeypatch):  # noqa: F811
    c, *_ = api
    allowance.consume(CTX)
    allowance.consume(SessionCtx(user_id=DECOY_ADMIN, role="admin", institute_id=DECOY, data_source="demo"))
    r = c.get("/agent/admin/voice-usage", headers=H(ADMIN))
    assert r.status_code == 200
    body = r.json()
    assert body["month"] == allowance.month_period()
    assert [x["institute_id"] for x in body["institutes"]] == [str(MAIN)]
    assert c.get("/agent/admin/voice-usage", headers=H(SHARMA)).status_code == 403     # teacher
    monkeypatch.setenv("AGENT_METRICS_TOKEN", "ops-secret")
    r = c.get("/agent/admin/voice-usage", headers={"Authorization": "Bearer ops-secret"})
    assert {x["institute_id"] for x in r.json()["institutes"]} == {str(MAIN), str(DECOY)}
    assert c.get("/agent/admin/voice-usage?month=2026-13", headers={"Authorization": "Bearer ops-secret"}).json()["month"] == allowance.month_period()


def test_agent_and_voice_trace_rows_carry_institute_and_feature(api, monkeypatch, tmp_path):  # noqa: F811
    c, _prov, _st, trace_path, *_ = api
    c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(2))
    c.post("/agent/voice/chat", headers=H(ADMIN), json={"message": "attendance batao"})
    rows = [json.loads(l) for l in trace_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    kinds = {r["kind"] for r in rows}
    assert "voice" in kinds and "llm" in kinds
    for r in rows:
        if r["kind"] in ("voice", "llm", "route", "tool"):
            assert r.get("institute_id") == str(MAIN), r
            assert r.get("feature") == "voice", r
    blob = json.dumps(rows)
    assert str(ADMIN) not in blob                      # never a user id


def test_text_agent_trace_feature_is_agent(monkeypatch, tmp_path):
    path = tmp_path / "t.jsonl"
    monkeypatch.setattr(config, "TRACE_PATH", str(path))
    prov = type("P", (), {"model": "m", "complete": lambda self, *a, **k: LLMResponse(text="ok", model="m", tokens_in=5, tokens_out=2)})()
    list(Agent(prov, CTX, str(uuid.uuid4())).run("hi"))
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
    assert all(r["institute_id"] == str(MAIN) and r["feature"] == "agent" for r in rows if r["kind"] == "llm")
