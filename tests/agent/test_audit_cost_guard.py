"""Gate-1 audit §6, agent side: cost guard for "voice query" and "agent question" (docs/COST_MAP.md §6).

Mocked STT / LLM / TTS. Happy path = transcribe (1 STT call) -> agent turn with one read tool (2 LLM calls) ->
speak (1 TTS call). Static prompt size = system prompt + tool schemas sent on every LLM call (chars/4; the
conversation itself is not counted). Same baseline file and 20 % rule as
test-generator-backend/tests/unit/test_audit_cost_guard.py; refresh with COST_GUARD_WRITE=1.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from _agent_ids import ADMIN
from agent.llm import LLMResponse, ToolCall
from test_voice_phase1 import H, api  # noqa: F401  (fixture)
from voice_fixtures import webm

BASELINE = Path(__file__).resolve().parents[1] / "cost_guard_baseline.json"
TOLERANCE = 1.20
MEASURED: dict[str, dict] = {}


def _tokens(s: str) -> int:
    return (len(s) + 3) // 4


class Recording:
    model = "claude-haiku-5-5"

    def __init__(self):
        self.calls = []

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False, cancel=None):
        self.calls.append(_tokens(system) + _tokens(json.dumps(tools)))
        if len(self.calls) == 1:
            return LLMResponse(text="", model=self.model, tool_calls=[ToolCall("t1", "my_scope", {})])
        return LLMResponse(text="Aapke 3 sections hain.\nSPEECH: Aapke teen sections hain.", model=self.model)


@pytest.fixture
def rec(api, monkeypatch):  # noqa: F811
    import agent.api as api_mod
    prov = Recording()
    monkeypatch.setattr(api_mod, "get_provider", lambda: prov)
    monkeypatch.setattr(api_mod, "fast_provider_for", lambda p: None)
    return api, prov


def test_measure_voice_query(rec):
    (c, _prov, _st, _trace, stt, tts), prov = rec
    stt_calls = tts_calls = 0
    r = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(2))
    assert r.status_code == 200
    stt_calls += 1
    r = c.post("/agent/voice/chat", headers=H(ADMIN), json={"message": "meri classes batao"})
    assert r.status_code == 200
    r = c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "Aapke teen sections hain."})
    assert r.status_code in (200, 204)
    tts_calls += 1
    MEASURED["voice_query"] = {"calls": stt_calls + len(prov.calls) + tts_calls, "prompt_tokens": sum(prov.calls)}


def test_measure_agent_question(rec):
    (c, *_), prov = rec
    r = c.post("/agent/chat", headers=H(ADMIN), json={"message": "meri classes batao"})
    assert r.status_code == 200
    MEASURED["agent_question"] = {"calls": len(prov.calls), "prompt_tokens": sum(prov.calls)}


def test_zz_compare_with_baseline():
    assert MEASURED
    base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    if os.getenv("COST_GUARD_WRITE") == "1":
        base.update(MEASURED)
        BASELINE.write_text(json.dumps(dict(sorted(base.items())), indent=2) + "\n", encoding="utf-8")
        pytest.skip(f"baseline written: {sorted(MEASURED)}")
    problems = []
    for op, now in MEASURED.items():
        if op not in base:
            problems.append(f"{op}: not in baseline (measured {now}); run with COST_GUARD_WRITE=1")
            continue
        for k in ("calls", "prompt_tokens"):
            if now[k] > base[op][k] * TOLERANCE:
                problems.append(f"{op}.{k}: {now[k]} > {base[op][k]} x {TOLERANCE}")
    assert not problems, "; ".join(problems)
