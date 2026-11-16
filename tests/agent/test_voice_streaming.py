"""Streaming TTS + barge-in (voice upgrade, phases 1-2). Mocked LLM / TTS, no network, no key."""
from __future__ import annotations

import base64
import threading
import time

import pytest

from _agent_ids import ADMIN
from agent.llm import LLMResponse
from agent.store import Owner
from agent.voice import stream_tts
from agent.voice.providers import FakeTTS, set_providers
from test_voice_phase1 import H, api, events  # noqa: F401  (fixture)

SENTENCES = ["10A की attendance 80% रही।", "10B की 76% रही।", "10C सबसे ऊपर है।", "11A में गिरावट है।",
             "बाकी सब ठीक है।"]


class SlowStreamer:
    """Streams 'SPEECH: s1 s2 s3 s4 s5' sentence by sentence, 200 ms apart, then the written answer."""
    model = "claude-haiku-5-5"
    supports_cancel = True
    supports_text_stream = True

    def __init__(self, gap_s: float = 0.2):
        self.gap_s = gap_s
        self.started = threading.Event()
        self.t_start = None
        self.t_end = None

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False, cancel=None, on_text=None):
        self.t_start = time.monotonic()
        self.started.set()
        text = "SPEECH: "
        if on_text:
            on_text("SPEECH: ")
        for s in SENTENCES:
            if cancel is not None and cancel.wait(self.gap_s):
                from agent.llm import LLMAborted
                raise LLMAborted(self.model, 10, max_tokens)
            piece = s + " "
            text += piece
            if on_text:
                on_text(piece)
        tail = "\nAttendance ka poora hisaab upar card mein hai."
        text += tail
        if on_text:
            on_text(tail)
        self.t_end = time.monotonic()
        return LLMResponse(text=text, model=self.model, tokens_in=10, tokens_out=40)


@pytest.fixture
def streaming(api, monkeypatch):  # noqa: F811
    import agent.api as api_mod
    c, _prov, _st, _trace, stt, _ = api
    tts = FakeTTS()
    set_providers([stt], tts)
    prov = SlowStreamer()
    monkeypatch.setattr(api_mod, "get_provider", lambda: prov)
    monkeypatch.setattr(api_mod, "fast_provider_for", lambda p: None)
    return c, prov, tts


def test_five_sentences_five_tts_calls_in_order_fast_first_audio(streaming):
    c, prov, tts = streaming
    r = c.post("/agent/voice/chat", headers=H(ADMIN), json={"message": "attendance batao", "tts": "stream"})
    assert r.status_code == 200
    evs = events(r)
    audio = [e for e in evs if e["type"] == "audio"]
    assert len(tts.calls) == 5                                     # one TTS call per sentence
    assert [a["index"] for a in audio] == [0, 1, 2, 3, 4]          # in order
    assert all(base64.b64decode(a["data"]).startswith(b"ID3") for a in audio)
    assert audio[0]["ttfa_ms"] is not None and audio[0]["ttfa_ms"] < 2000
    llm_ms = (prov.t_end - prov.t_start) * 1000
    assert audio[0]["ttfa_ms"] < llm_ms                            # first audio before the model finished writing
    types = [e["type"] for e in evs]
    assert types.index("audio") < types.index("final") and types[-1] == "final"
    sp = [e for e in evs if e["type"] == "speech"][0]["text"]
    assert sp.startswith("10A") and "SPEECH" not in "".join(e.get("text", "") for e in evs if e["type"] == "text_delta")


def test_audio_is_never_stored_in_history(streaming, api):  # noqa: F811
    c, *_ = streaming
    st = api[2]
    r = c.post("/agent/voice/chat", headers=H(ADMIN), json={"message": "attendance batao", "tts": "stream"})
    sid = [e for e in events(r) if e["type"] == "session"][0]["session_id"]
    stored = c.get(f"/agent/sessions/{sid}", headers=H(ADMIN)).json()
    assert "audio" not in str(stored) or '"type": "audio"' not in str(stored)
    for m in stored.get("messages", []):
        assert all(e.get("type") != "audio" for e in (m.get("events") or []))


def test_default_is_the_old_behaviour_no_audio(streaming):
    c, _prov, tts = streaming
    r = c.post("/agent/voice/chat", headers=H(ADMIN), json={"message": "attendance batao"})
    assert not [e for e in events(r) if e["type"] == "audio"] and tts.calls == []


def test_barge_in_cancels_the_running_turn(streaming):
    """A second voice turn with cancel_current=True stops the first: its LLM stream is closed, nothing more is
    spoken, and its `final` says cancelled."""
    c, prov, tts = streaming
    prov.gap_s = 0.5                                               # first answer is slow
    first: dict = {}

    def run_first():
        first["r"] = c.post("/agent/voice/chat", headers=H(ADMIN), json={"message": "attendance batao", "tts": "stream"})
    t = threading.Thread(target=run_first)
    t.start()
    assert prov.started.wait(5)
    time.sleep(0.6)                                                # mid-answer: ~1 sentence written
    prov.gap_s = 0.0
    prov.started.clear()
    r2 = c.post("/agent/voice/chat", headers=H(ADMIN),
                json={"message": "nahi, 10B batao", "tts": "stream", "cancel_current": True})
    t.join(10)
    assert r2.status_code == 200
    ev1 = events(first["r"])
    assert ev1[-1]["type"] == "final" and ev1[-1]["cancelled"] is True
    assert len([e for e in ev1 if e["type"] == "audio"]) < 5       # the rest of the first answer is never spoken


def test_run_registry_cancel_owner():
    from agent.runs import RunRegistry
    reg = RunRegistry()
    a = Owner(user_id="u1", institute_id="i1")
    b = Owner(user_id="u2", institute_id="i1")
    ea, eb = reg.register("r1", a), reg.register("r2", b)
    assert reg.cancel_owner(a) == 1 and ea.is_set() and not eb.is_set()


# ---- unit: chunker + pipeline --------------------------------------------------------------------------------------

def test_split_ready_sentences_and_long_runs():
    chunks, rest = stream_tts.split_ready("Pehla vakya. Doosra vakya? Teesra", final=False)
    assert chunks == ["Pehla vakya.", "Doosra vakya?"] and rest.strip() == "Teesra"
    long = "shabd " * 30
    chunks, rest = stream_tts.split_ready(long, final=False)
    assert chunks and all(len(c) <= stream_tts.MAX_CHUNK_CHARS for c in chunks)
    chunks, rest = stream_tts.split_ready("aakhri tukda", final=True)
    assert chunks == ["aakhri tukda"] and rest == ""


def test_pipeline_ignores_text_before_speech_and_after_its_line():
    tts = FakeTTS()
    s = stream_tts.SpeechStream(tts, "x", 1.0)
    s.begin_step()
    for d in ["Kuch prose pehle.\n", "SPEECH: Ek. ", "Do.\nYeh likha hua jawab hai. Teen."]:
        s.feed(d)
    s.end_step()
    deadline = time.time() + 3
    out = []
    while s.pending() and time.time() < deadline:
        out += s.ready()
        time.sleep(0.01)
    out += s.ready()
    assert [p["chars"] for p in out] == [3, 3] and len(tts.calls) == 2


def test_pipeline_discards_a_tool_call_step_and_cancel_drops_pending():
    tts = FakeTTS()
    s = stream_tts.SpeechStream(tts, "x", 1.0)
    s.begin_step()
    s.discard_step()
    s.feed("SPEECH: ")          # nothing captured: discard reset the step but no begin -> line state cleared
    s.cancel()
    s.feed("Ek. Do.")
    assert s.ready() == [] and not s.pending()


def test_failed_chunk_is_skipped_not_blocking():
    class Flaky(FakeTTS):
        def synthesize(self, text, language_code, speaker, pace):
            self.calls.append(text)
            if text.startswith("Do"):
                raise RuntimeError("tts down")
            yield b"ID3ok"
    s = stream_tts.SpeechStream(Flaky(), "x", 1.0)
    s.ensure("Ek. Do. Teen.")
    deadline = time.time() + 3
    out = []
    while s.pending() and time.time() < deadline:
        out += s.ready()
        time.sleep(0.01)
    out += s.ready()
    assert [p["index"] for p in out] == [0, 2] and s.errors == 1
