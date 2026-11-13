"""Voice layer, phase 1: container decoding (30 s / 1 MB on the bytes), providers + fallback chain + breaker, limits
+ idempotency, entity repair, speech split, the three endpoints (auth, kill switch, allowlist, scope), no audio or
transcript text in the trace, and the `speech` event through the real agent loop with a scripted model."""
from __future__ import annotations

import json
import uuid

import pytest

from _agent_ids import ADMIN, DECOY, DECOY_ADMIN, MAIN, SHARMA, VERMA
from agent.llm import LLMResponse, ToolCall
from agent.schemas import AgentEventAdapter, SessionCtx
from agent.store import MemoryStore, set_store
from agent.voice import audio, limits, repair, speech
from agent.voice import config as vc
from agent.voice.providers import (Breaker, Chain, FakeSTT, FakeTTS, NoTTS, ProviderDown, ProviderError,
                                   SarvamSTT, SarvamTTS, set_providers)
from voice_fixtures import mp4, mp4_fragmented, ogg, wav, webm

ADMIN_CTX = SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN)
SHARMA_CTX = SessionCtx(user_id=SHARMA, role="teacher", institute_id=MAIN, teacher_id=SHARMA)
DECOY_CTX = SessionCtx(user_id=DECOY_ADMIN, role="admin", institute_id=DECOY)
SECRET = "ZQX-private-7731"  # a transcript marker that must never reach the trace


# ---------------- audio containers ----------------
@pytest.mark.parametrize("build,container", [(webm, "webm"), (mp4, "mp4"), (mp4_fragmented, "mp4"), (ogg, "ogg"), (wav, "wav")])
def test_duration_is_read_from_the_container(build, container):
    info = audio.inspect(build(8.0))
    assert info.container == container
    assert 7900 <= info.duration_ms <= 8100, info


def test_webm_with_info_duration_and_without():
    assert 4950 <= audio.inspect(webm(5.0, with_duration=True)).duration_ms <= 5050
    assert 4950 <= audio.inspect(webm(5.0)).duration_ms <= 5050  # MediaRecorder writes no Duration


def test_truncated_webm_still_measures_what_is_there():
    b = webm(6.0)
    info = audio.inspect(b[: len(b) - 7])  # stop() mid-block
    assert 5500 <= info.duration_ms <= 6050


@pytest.mark.parametrize("bad", [b"", b"\x00" * 40, b"ID3\x03\x00" + b"\x00" * 100, b"RIFF\x00\x00\x00\x00WAVEjunk"])
def test_unknown_or_corrupt_containers_are_rejected(bad):
    with pytest.raises(audio.AudioError):
        audio.inspect(bad)


# ---------------- providers / chain / breaker ----------------
def test_chain_falls_back_to_gemini_then_fails():
    a, b = FakeSTT(fail="Timeout", name="sarvam"), FakeSTT(text="meri classes dikhao", name="gemini")
    r = Chain([a, b], Breaker()).transcribe(b"x", "audio/webm")
    assert r.transcript.provider == "gemini" and r.tried == ["sarvam", "gemini"]
    with pytest.raises(ProviderDown) as e:
        Chain([FakeSTT(fail="x", name="sarvam"), FakeSTT(fail="y", name="gemini")], Breaker()).transcribe(b"x", "audio/webm")
    assert e.value.tried == ["sarvam", "gemini"]


def test_breaker_opens_after_consecutive_failures_and_recovers():
    t = [0.0]
    br = Breaker(failures=2, open_s=60, clock=lambda: t[0])
    bad, good = FakeSTT(fail="Timeout", name="sarvam"), FakeSTT(text="ok", name="gemini")
    ch = Chain([bad, good], br)
    ch.transcribe(b"x", "audio/webm")
    ch.transcribe(b"x", "audio/webm")
    assert not br.allows("sarvam")
    r = ch.transcribe(b"x", "audio/webm")
    assert r.skipped == ["sarvam"] and len(bad.calls) == 2  # skipped while open
    t[0] = 61
    assert br.allows("sarvam")  # half-open: one try again


def test_sarvam_stt_request_shape_v3_translit_and_v4_keyterms(monkeypatch):
    class Resp:
        status_code = 200

        def json(self):
            return {"request_id": "r", "transcript": "Sharma maam ki attendance", "language_code": "hi-IN", "language_probability": 0.91}

    class Client:
        def __init__(self):
            self.calls = []

        def post(self, url, headers=None, files=None, data=None):
            self.calls.append({"url": url, "headers": headers, "files": files, "data": data})
            return Resp()
    c = Client()
    p = SarvamSTT(api_key="k", model="saaras:v3", mode="translit", client=c)
    t = p.transcribe(b"abc", "audio/webm", keyterms=["Priya Sharma"])
    call = c.calls[0]
    assert call["url"].endswith("/speech-to-text") and call["headers"] == {"api-subscription-key": "k"}
    assert call["data"] == {"model": "saaras:v3", "language_code": "unknown", "mode": "translit"}  # v3: no keyterms
    assert call["files"]["file"][2] == "audio/webm" and t.language_probability == 0.91
    SarvamSTT(api_key="k", model="saaras:v4", mode=None, client=c).transcribe(b"abc", "audio/mp4", keyterms=["Priya Sharma"])
    assert c.calls[1]["data"] == {"model": "saaras:v4", "language_code": "unknown", "keyterms": '["Priya Sharma"]'}
    assert "mode" not in c.calls[1]["data"]


def test_sarvam_stt_errors_are_type_names_only():
    class Client:
        def post(self, *a, **k):
            raise TimeoutError(SECRET)
    with pytest.raises(ProviderError) as e:
        SarvamSTT(api_key="k", client=Client()).transcribe(b"x", "audio/webm")
    assert e.value.kind == "TimeoutError" and SECRET not in str(e.value)
    assert SarvamSTT(api_key=None, client=Client()).api_key is None
    with pytest.raises(ProviderError) as e2:
        SarvamSTT(api_key=None, client=Client()).transcribe(b"x", "audio/webm")
    assert e2.value.kind == "not_configured" and not e2.value.retryable


def test_sarvam_tts_streams_mp3_with_allowlisted_fields():
    class Stream:
        status_code = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_bytes(self):
            yield b"ID3"
            yield b"rest"

    class Client:
        def __init__(self):
            self.calls = []

        def stream(self, method, url, headers=None, json=None):
            self.calls.append((method, url, headers, json))
            return Stream()
    c = Client()
    out = b"".join(SarvamTTS(api_key="k", model="bulbul:v4-flash", client=c).synthesize("नमस्ते", "hi-IN", "ishita_enhi_companion", 1.1))
    assert out == b"ID3rest"
    m, url, h, body = c.calls[0]
    assert url.endswith("/text-to-speech/stream") and h["api-subscription-key"] == "k"
    assert body == {"text": "नमस्ते", "language_code": "hi-IN", "speaker": "ishita_enhi_companion", "model": "bulbul:v4-flash",
                    "pace": 1.1, "output_audio_codec": "mp3", "speech_sample_rate": 22050}


# ---------------- limits / idempotency ----------------
def test_voice_limits_per_10min_and_daily(monkeypatch):
    limits.reset()
    monkeypatch.setenv("VOICE_USER_PER_10MIN", "2")
    monkeypatch.setenv("VOICE_USER_DAILY", "3")
    now = 1_800_000_000.0
    assert limits.check_user("u", now) is None and limits.check_user("u", now) is None
    hit = limits.check_user("u", now)
    assert hit and hit.scope == "user_10min" and "likh" in hit.message
    assert limits.check_user("u", now + 601) is None  # window moved
    hit = limits.check_user("u", now + 1300)
    assert hit and hit.scope == "user_daily" and "100" not in hit.message and "3" in hit.message
    assert limits.check_user("v", now) is None  # another user
    limits.reset()


def test_idempotency_cache_per_user():
    limits.reset()
    limits.idem_put("u", "k1", {"transcript": "x"}, now=100.0)
    assert limits.idem_get("u", "k1", now=200.0) == {"transcript": "x"}
    assert limits.idem_get("v", "k1", now=200.0) is None
    assert limits.idem_get("u", "k1", now=100.0 + vc.IDEMPOTENCY_TTL_S + 1) is None
    assert limits.idem_get("u", None) is None
    limits.reset()


# ---------------- speech helpers ----------------
@pytest.mark.parametrize("text,lang", [
    ("Sharma ma'am ki is mahine ki attendance kya hai", "hi"),
    ("What is the attendance of 10A this month", "en"),
    ("10A ki attendance show karo this week", "hi-en"),
    ("शर्मा मैम की attendance", "hi-en"),
    ("", "en"),
])
def test_classify_language(text, lang):
    assert speech.classify(text) == lang


def test_confidence_from_language_probability_and_length():
    assert speech.confidence("", 0.9) == 0.0
    assert speech.confidence("haan", 0.9) == 0.63
    assert speech.confidence("meri classes dikhao", 0.9) == 0.9
    assert speech.confidence("meri classes dikhao", None) == 0.8


def test_split_speech_line_and_tts_language():
    prose, sp = speech.split_speech("10A ki attendance is mahine 80.7% rahi.\n\nSPEECH: 10A की attendance 80.7% रही।")
    assert prose == "10A ki attendance is mahine 80.7% rahi." and sp == "10A की attendance 80.7% रही।"
    assert speech.split_speech("no line here") == ("no line here", None)
    assert speech.tts_language_code(sp) == "hi-IN" and speech.tts_language_code("All good.", "en") == "en-IN"


# ---------------- entity repair ----------------
def _loader(ctx):
    return [{"kind": "teacher", "id": str(SHARMA), "name": "Priya Sharma"},
            {"kind": "teacher", "id": str(VERMA), "name": "Rakesh Verma"},
            {"kind": "batch", "id": "b1", "name": "10A", "student_count": 20},
            {"kind": "batch", "id": "b2", "name": "10B", "student_count": 20},
            {"kind": "student", "id": "s1", "name": "Rahul Kumar"},
            {"kind": "student", "id": "s2", "name": "Rahul Kumar"}]


def test_repair_fixes_a_misheard_teacher_name_after_resolve_confirms():
    repair.CACHE.clear()
    calls = []

    def resolve(ctx, kind, name):
        calls.append((kind, name))
        return {"status": "ok", "matches": [{"id": str(SHARMA), "label": "Priya Sharma"}]}
    r = repair.repair(ADMIN_CTX, "Priya Sharmaa ki is mahine ki attendance", loader=_loader, resolve=resolve)
    assert r.changed and r.text == "Priya Sharma ki is mahine ki attendance"
    assert r.entity["kind"] == "teacher" and calls == [("teacher", "Priya Sharma")] and r.db_calls == 1


def test_repair_leaves_exact_names_alone_without_a_db_call():
    repair.CACHE.clear()
    r = repair.repair(ADMIN_CTX, "Rakesh Verma ki attendance", loader=_loader, resolve=lambda *a: pytest.fail("no DB"))
    assert not r.changed and r.entity["name"] == "Rakesh Verma" and r.db_calls == 0


def test_repair_ambiguous_name_gives_chips_never_auto_picks():
    repair.CACHE.clear()

    def resolve(ctx, kind, name):
        return {"status": "ambiguous", "matches": [{"id": "s1", "label": "Rahul Kumar (10A)", "hint": "10A · roll 1"},
                                                    {"id": "s2", "label": "Rahul Kumar (10B)", "hint": "10B · roll 1"}]}
    r = repair.repair(ADMIN_CTX, "Rahul Kumaar ke marks batao", loader=_loader, resolve=resolve)
    assert r.clarify and [c["label"] for c in r.clarify["chips"]] == ["Rahul Kumar (10A)", "Rahul Kumar (10B)"]
    assert r.clarify["chips"][0]["value"] == "Rahul Kumar (10A) ke marks batao" and not r.changed


def test_repair_only_sees_the_callers_scope():
    repair.CACHE.clear()
    r = repair.repair(DECOY_CTX, "Priya Sharmaa ki attendance", loader=lambda ctx: [], resolve=lambda *a: pytest.fail("no DB"))
    assert not r.changed and r.entity is None and r.clarify is None


def test_repair_ignores_function_words_and_short_tokens():
    repair.CACHE.clear()
    r = repair.repair(ADMIN_CTX, "is mahine ki attendance batao", loader=_loader, resolve=lambda *a: pytest.fail("no DB"))
    assert not r.changed and r.entity is None


# ---------------- endpoints ----------------
class Scripted:
    model = "claude-sonnet-5-5"

    def __init__(self, responses=(), default="Theek hai.\nSPEECH: ठीक है।"):
        self.responses, self.calls, self.default = list(responses), [], default

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        r = self.responses.pop(0) if self.responses else self.default
        return LLMResponse(text=r, model=self.model) if isinstance(r, str) else r


@pytest.fixture()
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_DEV_AUTH", "1")
    monkeypatch.setenv("VOICE_ENABLED", "1")
    monkeypatch.delenv("VOICE_INSTITUTE_ALLOWLIST", raising=False)
    monkeypatch.delenv("VERCEL", raising=False)
    from fastapi.testclient import TestClient
    import agent.api as api_mod
    from agent import config
    from agent.metrics import METRICS
    from agent.rate_limit import reset as reset_rate_limits
    from agent.voice.api import clear_phrase_cache
    monkeypatch.setattr(config, "TRACE_PATH", str(tmp_path / "trace.jsonl"))
    st = MemoryStore()
    set_store(st)
    reset_rate_limits()
    limits.reset()
    repair.CACHE.clear()
    clear_phrase_cache()
    METRICS.reset()
    prov = Scripted()
    monkeypatch.setattr(api_mod, "get_provider", lambda: prov)
    stt = FakeSTT(text="Sharma ma'am ki is mahine ki attendance", language_code="hi-IN", probability=0.95, name="sarvam")
    tts = FakeTTS()
    set_providers([stt], tts)
    yield TestClient(api_mod.create_app({})), prov, st, tmp_path / "trace.jsonl", stt, tts
    set_providers(None, None)
    set_store(None)
    reset_rate_limits()
    limits.reset()


def H(u, **extra):
    return {"x-agent-dev-user": str(u), **extra}


def events(resp):
    return [json.loads(line[6:]) for line in resp.text.splitlines() if line.startswith("data: ")]


def trace_text(path):
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_config_hides_voice_when_off_or_not_allowlisted(api, monkeypatch):
    c, *_ = api
    j = c.get("/agent/voice/config", headers=H(ADMIN)).json()
    assert j["enabled"] and j["tts"] == "sarvam" and j["default_speaker"] in {s["id"] for s in j["speakers"]}
    assert {s["gender"] for s in j["speakers"]} == {"female", "male"} and j["paces"] == [0.9, 1.0, 1.1]
    assert j["max_seconds"] == 30 and j["max_bytes"] == 1_048_576 and "audio/webm" in j["formats"] and "audio/mp4" in j["formats"]
    monkeypatch.setenv("VOICE_ENABLED", "0")
    j = c.get("/agent/voice/config", headers=H(ADMIN)).json()
    assert j == {**j, "enabled": False, "reason": "kill_switch"}
    assert c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(2)).status_code == 503
    monkeypatch.setenv("VOICE_ENABLED", "1")
    monkeypatch.setenv("VOICE_INSTITUTE_ALLOWLIST", str(DECOY))
    assert c.get("/agent/voice/config", headers=H(ADMIN)).json()["reason"] == "not_allowlisted"
    assert c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(2)).status_code == 403
    assert c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "hi"}).status_code == 403
    assert c.post("/agent/voice/chat", headers=H(ADMIN), json={"message": "hi"}).status_code == 403
    assert c.post("/agent/voice/transcribe", headers=H(DECOY_ADMIN, **{"Content-Type": "audio/webm"}), content=webm(2)).status_code == 200


def test_voice_endpoints_need_auth(api):
    c, *_ = api
    assert c.get("/agent/voice/config").status_code == 401
    assert c.post("/agent/voice/transcribe", content=webm(2), headers={"Content-Type": "audio/webm"}).status_code == 401
    assert c.post("/agent/voice/speak", json={"text": "hi"}).status_code == 401
    assert c.post("/agent/voice/chat", json={"message": "hi"}).status_code == 401


def test_transcribe_ok_language_confidence_and_repair(api):
    c, prov, st, trace, stt, _ = api
    r = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm;codecs=opus", "Idempotency-Key": "abc"}), content=webm(7.5))
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["transcript"] == "Sharma ma'am ki is mahine ki attendance" and j["language"] == "hi"
    assert j["confidence"] == 0.95 and 7400 <= j["duration_ms"] <= 7600 and j["provider"] == "sarvam"
    assert j["clarify"] is None and j["replayed"] is False
    assert stt.calls[0]["mime"] == "audio/webm" and stt.calls[0]["keyterms"] == []  # v3: no keyterms sent
    # same key -> cached, provider not called again
    r2 = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm", "Idempotency-Key": "abc"}), content=webm(7.5))
    assert r2.json()["replayed"] is True and len(stt.calls) == 1
    # another user, same key -> fresh
    c.post("/agent/voice/transcribe", headers=H(SHARMA, **{"Content-Type": "audio/webm", "Idempotency-Key": "abc"}), content=webm(3))
    assert len(stt.calls) == 2
    rec = [json.loads(l) for l in trace_text(trace).splitlines() if '"kind":"voice"' in l]
    assert rec and rec[0]["endpoint"] == "transcribe" and rec[0]["outcome"] == "ok" and rec[0]["provider"] == "sarvam"
    assert set(rec[0]["stages_ms"]) == {"decode", "stt", "repair"} and rec[0]["audio_bytes"] > 0 and rec[0]["cost_inr"] > 0


def test_transcribe_repairs_names_from_the_callers_scope_only(api):
    c, prov, st, trace, stt, _ = api
    stt.text = "Priya Sharmaa ki is hafte ki attendance"
    j = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(4)).json()
    assert j["transcript"] == "Priya Sharma ki is hafte ki attendance" and j["repaired"] is True
    # the decoy institute's admin never sees MAIN's teacher
    j2 = c.post("/agent/voice/transcribe", headers=H(DECOY_ADMIN, **{"Content-Type": "audio/webm"}), content=webm(4)).json()
    assert j2["transcript"] == "Priya Sharmaa ki is hafte ki attendance" and j2["repaired"] is False


def test_transcribe_ambiguous_student_returns_chips(api):
    c, prov, st, trace, stt, _ = api
    stt.text = "Rahul Kumaar ke marks batao"
    j = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(4)).json()
    assert j["clarify"] and len(j["clarify"]["chips"]) == 2 and all("Rahul Kumar" in ch["label"] for ch in j["clarify"]["chips"])
    assert j["transcript"] == "Rahul Kumaar ke marks batao"  # untouched: the user picks
    rec = [json.loads(l) for l in trace_text(trace).splitlines() if '"kind":"voice"' in l]
    assert rec[-1]["outcome"] == "clarified"


def test_transcribe_limits_are_enforced_on_the_bytes(api, monkeypatch):
    c, *_ = api
    hdr = H(ADMIN, **{"Content-Type": "audio/webm"})
    r = c.post("/agent/voice/transcribe", headers=hdr, content=webm(31))
    assert r.status_code == 422 and "30 second" in r.json()["detail"]
    r = c.post("/agent/voice/transcribe", headers=hdr, content=webm(5) + b"\x00" * 1_048_600)
    assert r.status_code == 413
    r = c.post("/agent/voice/transcribe", headers=hdr, content=b"\x00" * 200)
    assert r.status_code == 422 and "format" in r.json()["detail"]
    assert c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "text/plain"}), content=b"hello").status_code == 415
    assert c.post("/agent/voice/transcribe", headers=hdr, content=b"").status_code == 422
    # client lies about the duration in a header: ignored, the container decides
    r = c.post("/agent/voice/transcribe", headers={**hdr, "X-Duration-Ms": "1000"}, content=webm(31))
    assert r.status_code == 422
    # iOS Safari: audio/mp4 (fragmented) is accepted
    assert c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/mp4"}), content=mp4_fragmented(6)).status_code == 200


def test_transcribe_rate_limit_friendly_message(api, monkeypatch):
    c, *_ = api
    monkeypatch.setenv("VOICE_USER_DAILY", "2")
    hdr = H(ADMIN, **{"Content-Type": "audio/webm"})
    assert c.post("/agent/voice/transcribe", headers=hdr, content=webm(2)).status_code == 200
    assert c.post("/agent/voice/transcribe", headers=hdr, content=webm(2)).status_code == 200
    r = c.post("/agent/voice/transcribe", headers=hdr, content=webm(2))
    assert r.status_code == 429 and "likh" in r.json()["detail"] and int(r.headers["retry-after"]) > 0
    assert c.post("/agent/voice/transcribe", headers=H(SHARMA, **{"Content-Type": "audio/webm"}), content=webm(2)).status_code == 200


def test_transcribe_falls_back_to_gemini_then_to_typing(api):
    c, prov, st, trace, stt, tts = api
    set_providers([FakeSTT(fail="Timeout", name="sarvam"), FakeSTT(text="meri classes dikhao", name="gemini")], tts)
    j = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(3)).json()
    assert j["provider"] == "gemini" and j["transcript"] == "meri classes dikhao"
    set_providers([FakeSTT(fail="Timeout", name="sarvam"), FakeSTT(fail="ServerError", name="gemini")], tts)
    r = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(3))
    assert r.status_code == 503 and "Likh kar" in r.json()["detail"]
    rec = [json.loads(l) for l in trace_text(trace).splitlines() if '"kind":"voice"' in l]
    assert rec[-1]["outcome"] == "failed" and rec[-1]["tried"] == ["sarvam", "gemini"]
    assert rec[-2]["provider"] == "gemini" and rec[-2]["tried"] == ["sarvam", "gemini"]


def test_transcribe_empty_speech(api):
    c, prov, st, trace, stt, _ = api
    stt.text = "   "
    j = c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(3)).json()
    assert j["transcript"] == "" and j["confidence"] == 0.0
    rec = [json.loads(l) for l in trace_text(trace).splitlines() if '"kind":"voice"' in l]
    assert rec[-1]["outcome"] == "empty"


def test_speak_allowlist_streaming_and_static_cache(api):
    c, prov, st, trace, stt, tts = api
    assert c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "hi", "speaker": "evil_voice"}).status_code == 422
    assert c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "hi", "pace": 1.5}).status_code == 422
    assert c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "x" * 301}).status_code == 422
    assert c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "hi", "extra": 1}).status_code == 422
    r = c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "10A की attendance 80% रही।", "speaker": "shubh_enhi_companion", "pace": 0.9})
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/mpeg") and r.content.startswith(b"ID3")
    assert tts.calls[-1] == {"chars": 26, "language_code": "hi-IN", "speaker": "shubh_enhi_companion", "pace": 0.9}
    # data answers are never cached: same text -> provider called again
    c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "10A की attendance 80% रही।", "speaker": "shubh_enhi_companion", "pace": 0.9})
    assert len(tts.calls) == 2
    # static phrase: first call synthesises, second is served from cache
    r = c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "-", "phrase": "repeat", "language": "hi"})
    assert r.status_code == 200 and len(tts.calls) == 3 and tts.calls[-1]["chars"] == len("Dobara boliye.")
    r = c.post("/agent/voice/speak", headers=H(SHARMA), json={"text": "-", "phrase": "repeat", "language": "hi"})
    assert r.status_code == 200 and len(tts.calls) == 3
    rec = [json.loads(l) for l in trace_text(trace).splitlines() if '"kind":"voice"' in l and '"speak"' in l]
    assert [x["outcome"] for x in rec] == ["ok", "ok", "ok", "cached"] and rec[0]["cost_inr"] > 0 and rec[-1]["cost_inr"] == 0


def test_speak_tts_down_and_none(api):
    c, prov, st, trace, stt, tts = api
    set_providers([stt], FakeTTS(fail="ServerError"))
    r = c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "hello"})
    assert r.status_code == 503
    set_providers([stt], NoTTS())
    assert c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": "hello"}).status_code == 204


def test_voice_chat_emits_speech_and_stores_a_normal_turn(api):
    c, prov, st, trace, stt, tts = api
    prov.responses = ["Sharma ma'am ki attendance is mahine theek rahi.\nSPEECH: Sharma ma'am की attendance ठीक रही।"]
    r = c.post("/agent/voice/chat", headers=H(ADMIN), json={"message": "Sharma ma'am ki is mahine ki attendance"})
    assert r.status_code == 200
    evs = events(r)
    types = [e["type"] for e in evs]
    assert types[0] == "session" and types[-1] == "final" and "speech" in types
    for e in evs:
        AgentEventAdapter.validate_python(e)
    td = next(e for e in evs if e["type"] == "text_delta")
    assert "SPEECH" not in td["text"] and td["text"] == "Sharma ma'am ki attendance is mahine theek rahi."
    sp = next(e for e in evs if e["type"] == "speech")
    assert sp["text"] == "Sharma ma'am की attendance ठीक रही।" and sp["language"] == "hi-en"
    assert "VOICE TURN" in prov.calls[0]["system"] and "SPEECH:" in prov.calls[0]["system"]
    sid = evs[0]["session_id"]
    msgs = c.get(f"/agent/sessions/{sid}/messages", headers=H(ADMIN)).json()["items"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[0]["text"] == "Sharma ma'am ki is mahine ki attendance"
    assert "SPEECH" not in (msgs[1]["text"] or "")
    # a text turn in the same session has no voice addendum and no speech event
    r2 = c.post("/agent/chat", headers=H(ADMIN), json={"message": "aur pichhle hafte?", "session_id": sid})
    assert "speech" not in [e["type"] for e in events(r2)] and "VOICE TURN" not in prov.calls[1]["system"]
    rec = [json.loads(l) for l in trace_text(trace).splitlines() if '"kind":"voice"' in l]
    assert rec[-1]["endpoint"] == "chat" and rec[-1]["outcome"] == "answered" and rec[-1]["speech"] is True
    assert "turn" in rec[-1]["stages_ms"]


def test_voice_chat_approve_by_voice_never_runs_the_model(api):
    c, prov, st, trace, stt, tts = api
    r = c.post("/agent/voice/chat", headers=H(SHARMA), json={"message": "haan kar do, approve kar do"})
    evs = events(r)
    types = [e["type"] for e in evs]
    assert prov.calls == [] and "pending_approval" not in types
    assert types == ["session", "text_delta", "ui_action", "speech", "final"]
    assert evs[1]["text"] == "Voice se approve nahi hota. Screen par Confirm dabaiye."
    assert evs[2]["action"] == "highlight_confirm" and evs[4]["usage"]["route"] == "guard"
    sid = evs[0]["session_id"]
    msgs = c.get(f"/agent/sessions/{sid}/messages", headers=H(SHARMA)).json()["items"]
    assert len(msgs) == 2 and msgs[1]["events"][2]["type"] == "ui_action"
    rec = [json.loads(l) for l in trace_text(trace).splitlines() if '"kind":"voice"' in l]
    assert rec[-1]["outcome"] == "refused" and rec[-1]["guarded"] is True


def test_no_transcript_or_speech_text_in_trace_or_logs(api, caplog):
    c, prov, st, trace, stt, tts = api
    import logging
    caplog.set_level(logging.DEBUG)
    stt.text = f"{SECRET} ki attendance"
    c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(3))
    prov.responses = [f"{SECRET} theek hai.\nSPEECH: {SECRET} ठीक है।"]
    c.post("/agent/voice/chat", headers=H(ADMIN), json={"message": f"{SECRET} ki attendance"})
    c.post("/agent/voice/speak", headers=H(ADMIN), json={"text": f"{SECRET} ठीक है।"})
    assert SECRET not in trace_text(trace)
    assert SECRET not in caplog.text
    assert "audio_bytes" in trace_text(trace) and "\\u001aE" not in trace_text(trace)


def test_voice_metrics_have_fixed_label_vocabulary(api):
    c, *_ = api
    c.post("/agent/voice/transcribe", headers=H(ADMIN, **{"Content-Type": "audio/webm"}), content=webm(3))
    m = c.get("/agent/metrics").text
    assert 'agent_voice_requests_total{endpoint="transcribe",outcome="ok",provider="sarvam"} 1' in m
    assert 'agent_voice_stage_seconds_count{endpoint="transcribe",stage="stt"} 1' in m
    assert str(ADMIN) not in m and str(MAIN) not in m


def test_old_chat_endpoint_is_unchanged(api):
    c, prov, *_ = api
    prov.responses = ["Theek hai."]
    r = c.post("/agent/chat", headers=H(ADMIN), json={"message": "meri classes"})
    types = [e["type"] for e in events(r)]
    assert types == ["session", "text_delta", "final"]
    assert "VOICE TURN" not in prov.calls[0]["system"]
