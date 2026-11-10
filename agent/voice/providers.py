"""STT / TTS provider interfaces + Sarvam, Gemini (STT fallback) and Fake (tests) implementations.

Interfaces are deliberately small so a streaming STT (Sarvam realtime WebSocket) can be added later without touching
the endpoints: `transcribe(audio, mime, hints) -> Transcript`, `synthesize(text, language, speaker, pace) -> bytes
chunks`. No provider logs audio or text; errors carry a type name only.

Fallback order and the in-process failure counter live in `Chain` (STT) — Sarvam -> Gemini -> ProviderDown. A
provider that failed BREAKER_FAILURES times in a row is skipped for BREAKER_OPEN_S seconds.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional, Protocol

from . import config as vc


@dataclass
class Transcript:
    text: str
    language_code: Optional[str]      # provider BCP-47 (hi-IN / en-IN / …) or None
    language_probability: Optional[float]
    provider: str
    model: str
    latency_ms: float = 0.0
    raw_keys: tuple = ()               # for the trace: which response keys came back (never values)


class ProviderError(Exception):
    """One provider failed for this request (timeout, 5xx, bad response). The chain moves on."""

    def __init__(self, provider: str, kind: str, retryable: bool = True):
        super().__init__(f"{provider}:{kind}")
        self.provider, self.kind, self.retryable = provider, kind, retryable


class ProviderDown(Exception):
    """Every provider in the chain failed (or none is configured)."""

    def __init__(self, tried: list[str]):
        super().__init__("all providers failed")
        self.tried = tried


class STTProvider(Protocol):
    name: str

    def transcribe(self, audio: bytes, mime: str, keyterms: Optional[list[str]] = None) -> Transcript: ...


class TTSProvider(Protocol):
    name: str
    supports_stream: bool

    def synthesize(self, text: str, language_code: str, speaker: str, pace: float) -> Iterator[bytes]: ...


# ---- breaker ----------------------------------------------------------------------------------------------------
class Breaker:
    """Consecutive-failure counter per provider. Open = skip it for a while. Process-local, no persistence."""

    def __init__(self, failures: int = vc.BREAKER_FAILURES, open_s: float = vc.BREAKER_OPEN_S,
                 clock: Callable[[], float] = time.monotonic):
        self.failures, self.open_s, self.clock = failures, open_s, clock
        self._lock = threading.Lock()
        self._count: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def allows(self, name: str) -> bool:
        with self._lock:
            until = self._open_until.get(name, 0.0)
            if until and self.clock() < until:
                return False
            if until:
                self._open_until.pop(name, None)  # half-open: one try
            return True

    def ok(self, name: str) -> None:
        with self._lock:
            self._count[name] = 0
            self._open_until.pop(name, None)

    def fail(self, name: str) -> None:
        with self._lock:
            n = self._count.get(name, 0) + 1
            self._count[name] = n
            if n >= self.failures:
                self._open_until[name] = self.clock() + self.open_s
                self._count[name] = 0

    def state(self) -> dict[str, str]:
        with self._lock:
            now = self.clock()
            return {k: ("open" if v > now else "closed") for k, v in self._open_until.items()}

    def reset(self) -> None:
        with self._lock:
            self._count.clear()
            self._open_until.clear()


# ---- Sarvam ------------------------------------------------------------------------------------------------------
def _http_client(timeout: float):
    import httpx
    return httpx.Client(timeout=httpx.Timeout(timeout, connect=3.0))


class SarvamSTT:
    """POST https://api.sarvam.ai/speech-to-text (multipart). saaras:v3 + mode=translit -> Roman script; saaras:v4
    takes `keyterms` (names from the caller's scope) but transcribes in native script."""
    name = "sarvam"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, mode: Optional[str] = None,
                 client: Any = None, timeout_s: float = vc.STT_TIMEOUT_S):
        self.api_key = api_key or vc.sarvam_key()
        self.model = model or vc.stt_model()
        self.mode = mode if mode is not None else vc.stt_mode()
        self.client = client
        self.timeout_s = timeout_s

    def transcribe(self, audio: bytes, mime: str, keyterms: Optional[list[str]] = None) -> Transcript:
        if not self.api_key:
            raise ProviderError(self.name, "not_configured", retryable=False)
        ext = {"audio/webm": "webm", "audio/mp4": "m4a", "audio/ogg": "ogg", "audio/wav": "wav"}.get(mime, "bin")
        data: dict[str, str] = {"model": self.model, "language_code": "unknown"}
        if self.mode and not self.model.startswith("saaras:v4"):
            data["mode"] = self.mode
        if keyterms and self.model.startswith("saaras:v4"):
            data["keyterms"] = json.dumps(keyterms[:50])
        t0 = time.time()
        try:
            client = self.client or _http_client(self.timeout_s)
            r = client.post(f"{vc.SARVAM_BASE}/speech-to-text", headers={"api-subscription-key": self.api_key},
                            files={"file": (f"audio.{ext}", audio, mime)}, data=data)
        except Exception as e:  # noqa: BLE001  timeouts / connection errors: type name only
            raise ProviderError(self.name, type(e).__name__) from None
        if r.status_code in (401, 403):
            raise ProviderError(self.name, f"http_{r.status_code}", retryable=False)
        if r.status_code >= 400:
            raise ProviderError(self.name, f"http_{r.status_code}", retryable=r.status_code in (408, 429) or r.status_code >= 500)
        try:
            j = r.json()
        except ValueError:
            raise ProviderError(self.name, "bad_json") from None
        text = j.get("transcript")
        if not isinstance(text, str):
            raise ProviderError(self.name, "no_transcript")
        lp = j.get("language_probability")
        return Transcript(text=text.strip(), language_code=j.get("language_code"),
                          language_probability=float(lp) if isinstance(lp, (int, float)) else None,
                          provider=self.name, model=self.model, latency_ms=(time.time() - t0) * 1000,
                          raw_keys=tuple(sorted(k for k in j.keys() if isinstance(k, str))))


class SarvamTTS:
    """POST /text-to-speech/stream (binary mp3 body, chunked) — falls back to /text-to-speech (base64 wav) if the
    stream endpoint answers 404/405 on this account."""
    name = "sarvam"
    supports_stream = True

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, client: Any = None,
                 timeout_s: float = vc.TTS_TIMEOUT_S):
        self.api_key = api_key or vc.sarvam_key()
        self.model = model or vc.tts_model()
        self.client = client
        self.timeout_s = timeout_s
        self.mime = "audio/mpeg"

    def _body(self, text: str, language_code: str, speaker: str, pace: float) -> dict:
        return {"text": text, "language_code": language_code, "speaker": speaker, "model": self.model,
                "pace": pace, "output_audio_codec": "mp3", "speech_sample_rate": 22050}

    def synthesize(self, text: str, language_code: str, speaker: str, pace: float) -> Iterator[bytes]:
        if not self.api_key:
            raise ProviderError(self.name, "not_configured", retryable=False)
        headers = {"api-subscription-key": self.api_key, "Content-Type": "application/json"}
        body = self._body(text, language_code, speaker, pace)
        client = self.client or _http_client(self.timeout_s)
        try:
            with client.stream("POST", f"{vc.SARVAM_BASE}/text-to-speech/stream", headers=headers, json=body) as r:
                if r.status_code in (404, 405):
                    yield from self._rest(client, headers, body)
                    return
                if r.status_code >= 400:
                    raise ProviderError(self.name, f"http_{r.status_code}", retryable=r.status_code >= 500 or r.status_code == 429)
                for chunk in r.iter_bytes():
                    if chunk:
                        yield chunk
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            raise ProviderError(self.name, type(e).__name__) from None

    def _rest(self, client, headers: dict, body: dict) -> Iterator[bytes]:
        import base64
        r = client.post(f"{vc.SARVAM_BASE}/text-to-speech", headers=headers, json=body)
        if r.status_code >= 400:
            raise ProviderError(self.name, f"http_{r.status_code}")
        audios = (r.json() or {}).get("audios") or []
        if not audios:
            raise ProviderError(self.name, "no_audio")
        self.mime = "audio/mpeg" if body.get("output_audio_codec") == "mp3" else "audio/wav"
        yield base64.b64decode(audios[0])


# ---- Gemini (fallback STT) ---------------------------------------------------------------------------------------
_GEMINI_PROMPT = ("Transcribe this short audio from an Indian school office. Reply ONLY with JSON "
                  '{"transcript": "...", "language": "hi"|"en"|"hi-en"}. Write Hindi words in Roman (Latin) script, '
                  "the way Hinglish is typed (e.g. 'Sharma ma'am ki is mahine ki attendance kya hai'). "
                  "If there is no speech, transcript is an empty string. Never add anything else.")


class GeminiSTT:
    name = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, client: Any = None,
                 timeout_s: float = vc.GEMINI_STT_TIMEOUT_S):
        from .. import config as agent_config
        self.api_key = api_key or vc.gemini_key()
        self.model = model or agent_config.GEMINI_MODEL
        self.client = client
        self.timeout_s = timeout_s

    def transcribe(self, audio: bytes, mime: str, keyterms: Optional[list[str]] = None) -> Transcript:
        if not self.api_key and self.client is None:
            raise ProviderError(self.name, "not_configured", retryable=False)
        t0 = time.time()
        try:
            from google.genai import types
            client = self.client
            if client is None:
                from google import genai
                client = genai.Client(api_key=self.api_key, http_options={"timeout": int(self.timeout_s * 1000)})
            prompt = _GEMINI_PROMPT
            if keyterms:
                prompt += " Names that may occur: " + ", ".join(keyterms[:50]) + "."
            r = client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=audio, mime_type=mime), types.Part(text=prompt)],
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0,
                                                   max_output_tokens=400))
            text = (getattr(r, "text", None) or "").strip()
        except Exception as e:  # noqa: BLE001
            raise ProviderError(self.name, type(e).__name__) from None
        try:
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0]
            j = json.loads(text[text.find("{"):text.rfind("}") + 1])
        except (ValueError, IndexError):
            raise ProviderError(self.name, "bad_json") from None
        tr = j.get("transcript")
        if not isinstance(tr, str):
            raise ProviderError(self.name, "no_transcript")
        lang = {"hi": "hi-IN", "en": "en-IN", "hi-en": "hi-IN"}.get(str(j.get("language") or "").lower())
        return Transcript(text=tr.strip(), language_code=lang, language_probability=None, provider=self.name,
                          model=self.model, latency_ms=(time.time() - t0) * 1000, raw_keys=tuple(sorted(j.keys())))


# ---- fakes (tests) -----------------------------------------------------------------------------------------------
class FakeSTT:
    """Scripted: `script` maps nothing — it returns `text` (or raises) regardless of audio. `fail` = raise every time."""
    name = "fake"

    def __init__(self, text: str = "Sharma ma'am ki is mahine ki attendance", language_code: Optional[str] = "hi-IN",
                 probability: Optional[float] = 0.95, fail: Optional[str] = None, name: str = "fake",
                 latency_ms: float = 0.0):
        self.text, self.language_code, self.probability, self.fail = text, language_code, probability, fail
        self.name, self.latency_ms = name, latency_ms
        self.calls: list[dict] = []

    def transcribe(self, audio: bytes, mime: str, keyterms: Optional[list[str]] = None) -> Transcript:
        self.calls.append({"bytes": len(audio), "mime": mime, "keyterms": list(keyterms or [])})
        if self.fail:
            raise ProviderError(self.name, self.fail)
        return Transcript(text=self.text, language_code=self.language_code, language_probability=self.probability,
                          provider=self.name, model="fake", latency_ms=self.latency_ms)


class FakeTTS:
    name = "fake"
    supports_stream = True
    mime = "audio/mpeg"

    def __init__(self, fail: Optional[str] = None):
        self.fail = fail
        self.calls: list[dict] = []

    def synthesize(self, text: str, language_code: str, speaker: str, pace: float) -> Iterator[bytes]:
        self.calls.append({"chars": len(text), "language_code": language_code, "speaker": speaker, "pace": pace})
        if self.fail:
            raise ProviderError(self.name, self.fail)
        yield b"ID3fake-audio:" + str(len(text)).encode()
        yield b"|end"


class NoTTS:
    """`VOICE_TTS=none`: the client uses browser speechSynthesis; /speak answers 204."""
    name = "none"
    supports_stream = False
    mime = "audio/mpeg"

    def synthesize(self, text: str, language_code: str, speaker: str, pace: float) -> Iterator[bytes]:
        raise ProviderError(self.name, "disabled", retryable=False)


# ---- chain -------------------------------------------------------------------------------------------------------
@dataclass
class ChainResult:
    transcript: Transcript
    tried: list[str] = field(default_factory=list)   # provider names that were attempted (in order)
    skipped: list[str] = field(default_factory=list)  # providers skipped because their breaker was open


class Chain:
    """Fallback order with a breaker per provider. Sarvam -> Gemini -> ProviderDown."""

    def __init__(self, providers: list[STTProvider], breaker: Optional[Breaker] = None):
        self.providers = providers
        self.breaker = breaker or BREAKER

    def transcribe(self, audio: bytes, mime: str, keyterms: Optional[list[str]] = None) -> ChainResult:
        tried: list[str] = []
        skipped: list[str] = []
        for p in self.providers:
            if not self.breaker.allows(p.name):
                skipped.append(p.name)
                continue
            tried.append(p.name)
            try:
                t = p.transcribe(audio, mime, keyterms)
            except ProviderError as e:
                if e.retryable:
                    self.breaker.fail(p.name)
                continue
            self.breaker.ok(p.name)
            return ChainResult(transcript=t, tried=tried, skipped=skipped)
        raise ProviderDown(tried + skipped)


BREAKER = Breaker()
_lock = threading.Lock()
_stt_chain: Optional[Chain] = None
_tts: Optional[TTSProvider] = None


def stt_chain(env=os.environ) -> Chain:
    """Default chain from env: Sarvam (if key) -> Gemini (if key). Tests replace it with set_providers()."""
    global _stt_chain
    with _lock:
        if _stt_chain is None:
            ps: list[STTProvider] = []
            if vc.sarvam_key(env):
                ps.append(SarvamSTT())
            if vc.gemini_key(env):
                ps.append(GeminiSTT())
            _stt_chain = Chain(ps)
        return _stt_chain


def tts_provider(env=os.environ) -> TTSProvider:
    global _tts
    with _lock:
        if _tts is None:
            if (env.get("VOICE_TTS") or "").strip().lower() == "none" or not vc.sarvam_key(env):
                _tts = NoTTS()
            else:
                _tts = SarvamTTS()
        return _tts


def set_providers(stt: Optional[list[STTProvider]] = None, tts: Optional[TTSProvider] = None,
                  breaker: Optional[Breaker] = None) -> None:
    """Tests / dev: install fakes. None resets to env-built providers."""
    global _stt_chain, _tts
    with _lock:
        _stt_chain = Chain(stt, breaker) if stt is not None else None
        _tts = tts
    (breaker or BREAKER).reset()
