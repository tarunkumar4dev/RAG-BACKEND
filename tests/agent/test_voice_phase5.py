"""Voice phase 5: the 40-utterance set (shape + offline layers), fallback paths with fake providers (primary STT
down, both down, TTS down), local stage latency, the audio eval refuses to invent results, and WER/SNR helpers."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from _agent_ids import ADMIN
from agent.voice.providers import FakeSTT, FakeTTS, set_providers
from test_voice_phase1 import H, api, events  # noqa: F401
from voice_fixtures import webm

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "evals" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_regression_set_shape():
    ev = _load("run_voice_evals")
    cases = ev.load_cases()
    assert len(cases) == 40 and len({c["id"] for c in cases}) == 40
    kinds = [c["kind"] for c in cases]
    assert kinds.count("out_of_scope") == 6 and kinds.count("injection") == 4 and kinds.count("approve_by_voice") == 4
    assert {c["lang"] for c in cases} == {"hi", "en", "hi-en"} and {c["role"] for c in cases} == {"admin", "teacher"}
    assert any("ma'am" in c["heard"] for c in cases) and any(" sir " in f" {c['heard']} " for c in cases)


def test_regression_set_offline_layers_all_pass():
    ev = _load("run_voice_evals")
    from agent.voice import repair
    repair.CACHE.clear()
    rows = [{**c, **ev.offline(c)} for c in ev.load_cases()]
    bad = [(r["id"], r["repaired_text"]) for r in rows if not r["offline_ok"]]
    assert not bad, bad
    # unsafe-action gate: every approve-by-voice utterance is blocked before the model
    assert all(r["guard"] for r in rows if r["kind"] == "approve_by_voice")
    assert not any(r["guard"] for r in rows if r["kind"] != "approve_by_voice" and not r.get("expect_guard"))


def test_fallback_primary_stt_down(api):
    c, prov, st, trace, stt, tts = api
    set_providers([FakeSTT(fail="ConnectTimeout", name="sarvam"), FakeSTT(text="meri classes dikhao", name="gemini")], tts)
    j = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(3)).json()
    assert j["provider"] == "gemini" and j["next"] == "auto"


def test_fallback_both_stt_down_tells_the_user_to_type(api):
    c, prov, st, trace, stt, tts = api
    set_providers([FakeSTT(fail="ConnectTimeout", name="sarvam"), FakeSTT(fail="ServerError", name="gemini")], tts)
    r = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(3))
    assert r.status_code == 503 and "Likh kar poochhiye" in r.json()["detail"]
    # the breaker opens after 3 failures in a row: the 4th request does not even try the dead providers
    for _ in range(2):
        c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(3))
    rec = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines() if '"kind":"voice"' in l]
    c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(3))
    rec = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines() if '"kind":"voice"' in l]
    assert rec[-1]["outcome"] == "failed" and rec[-1]["tried"] == ["sarvam", "gemini"]
    assert rec[-1]["stages_ms"]["stt"] < 50  # both skipped: answered at once, not after two timeouts


def test_fallback_tts_down_chat_still_answers(api):
    c, prov, st, trace, stt, tts = api
    set_providers([stt], FakeTTS(fail="ServerError"))
    prov.responses = ["Theek hai.\nSPEECH: ठीक है।"]
    evs = events(c.post("/agent/voice/chat", headers=H(ADMIN), json={"message": "meri classes"}))
    assert [e["type"] for e in evs][-1] == "final" and any(e["type"] == "speech" for e in evs)
    assert c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "ठीक है।"}).status_code == 503  # UI: caption + speechSynthesis


def test_local_stage_latency_is_small(api):
    """Decode + repair on the seed DB (vendor stages excluded: they need a key and a 4G phone)."""
    c, prov, st, trace, stt, tts = api
    stt.text = "Priya Sharmaa ki is hafte ki attendance"
    for _ in range(10):
        c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(9))
    rec = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines() if '"kind":"voice"' in l]
    dec = sorted(r["stages_ms"]["decode"] for r in rec)
    rep = sorted(r["stages_ms"]["repair"] for r in rec)
    assert dec[len(dec) // 2] < 20 and rep[-1] < 1000  # first repair loads the entity list; the rest hit the cache
    assert rep[len(rep) // 2] < 150


def test_audio_eval_refuses_without_key_or_recordings(monkeypatch, capsys):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    mod = _load("run_voice_audio_eval")
    assert mod.main(["--provider", "sarvam"]) == 2
    out = capsys.readouterr().out
    assert "NOT run" in out and "No synthetic results" in out


def test_wer_and_mixing_helpers():
    mod = _load("run_voice_audio_eval")
    assert mod.wer("sharma ma'am ki attendance", "sharma ma'am ki attendance") == 0.0
    assert mod.wer("sharma ma'am ki attendance", "sharma mam ki attendance") == 0.25
    assert mod.wer("a b", "") == 1.0
    import array, io, wave
    sp = array.array("h", [1000 if i % 2 else -1000 for i in range(16000)])
    nz = mod.babble(16000)
    data = mod.mix(sp, nz, 10)
    with wave.open(io.BytesIO(data)) as w:
        assert w.getframerate() == 16000 and w.getnframes() == 16000
