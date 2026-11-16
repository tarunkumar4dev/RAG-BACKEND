"""Phase 5: one shared keep-alive HTTP client for Sarvam; startup warmup catches a bad key. No network."""
from __future__ import annotations

import logging
from types import SimpleNamespace

from agent.voice import providers as pv
from agent.voice.providers import FakeSTT, FakeTTS, SarvamSTT, SarvamTTS, set_providers


def test_http_client_is_shared_per_timeout_and_keepalive():
    pv.close_http_clients()
    a, b, c = pv._http_client(8.0), pv._http_client(8.0), pv._http_client(10.0)
    assert a is b and a is not c
    import httpx
    assert isinstance(a, httpx.Client)
    pv.close_http_clients()
    assert pv._http_client(8.0) is not a


def test_warmup_skips_fakes_and_when_off(monkeypatch):
    stt, tts = FakeSTT(), FakeTTS()
    set_providers([stt], tts)
    assert pv.warmup({"VOICE_ENABLED": "1"}) == {} and stt.calls == [] and tts.calls == []
    assert pv.warmup({"VOICE_ENABLED": "1", "VOICE_WARMUP": "0"}) == {}
    assert pv.warmup({}) == {}
    set_providers(None, None)


class _Resp:
    def __init__(self, code, body=None, stream=False):
        self.status_code, self._body = code, body or {}

    def json(self):
        return self._body

    def iter_bytes(self):
        yield b"ID3"

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_warmup_reports_a_bad_key_and_opens_the_connection(caplog):
    """A fake Sarvam HTTP client: STT answers 401 (bad key) -> error logged, named; TTS answers 200 -> ok."""
    calls = []

    class Client:
        def post(self, url, **kw):
            calls.append(url)
            return _Resp(401)

        def stream(self, method, url, **kw):
            calls.append(url)
            return _Resp(200)
    stt = SarvamSTT(api_key="k", client=Client())
    tts = SarvamTTS(api_key="k", client=Client())
    set_providers([stt], tts)
    with caplog.at_level(logging.WARNING):
        out = pv.warmup({"VOICE_ENABLED": "1"})
    set_providers(None, None)
    assert out == {"sarvam_stt": "http_401", "sarvam_tts": "ok"}
    assert any("SARVAM_API_KEY" in r.getMessage() and r.levelno == logging.ERROR for r in caplog.records)
    assert any(u.endswith("/speech-to-text") for u in calls) and any("/text-to-speech" in u for u in calls)


def test_silence_wav_is_a_valid_short_wav():
    w = pv._silence_wav(0.4)
    assert w[:4] == b"RIFF" and w[8:12] == b"WAVE" and len(w) == 44 + 2 * 6400
