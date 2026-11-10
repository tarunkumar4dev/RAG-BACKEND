"""Voice endpoints (mounted by agent/api.py):

  GET  /agent/voice/config      what the UI needs: enabled?, speakers, paces, limits, consent text version
  POST /agent/voice/transcribe  raw audio body (what MediaRecorder produced) -> {transcript, language, confidence, …}
  POST /agent/voice/speak       {text, language?, speaker?, pace?} -> streamed audio/mpeg (204 when TTS is "none")
  POST /agent/voice/chat        the same ChatRequest / SSE as POST /agent/chat, plus a `speech` event

Same auth (get_session), same store, same history. VOICE_ENABLED=1 is the kill switch; VOICE_INSTITUTE_ALLOWLIST
the pilot gate. Audio is read into memory, inspected, sent to the STT provider and dropped. Transcript text is never
logged; the trace keeps ids, sizes, durations, providers and outcomes.
"""
from __future__ import annotations

import logging
import re
import threading
import time
import uuid
from typing import Iterator, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from ..auth import get_session
from ..schemas import ChatRequest, SessionCtx
from ..trace import Tracer
from . import config as vc
from . import limits
from .audio import ACCEPTED_MIMES, AudioError, inspect
from .providers import ProviderDown, ProviderError, stt_chain, tts_provider
from .repair import keyterms as scope_keyterms
from .repair import repair as repair_entities
from .speech import classify, confidence as confidence_of, tts_language_code

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent/voice", tags=["agent-voice"])

CONSENT_VERSION = "2026-11-10"
VOICE_OFF_MESSAGE = "Voice abhi band hai. Likh kar poochhiye."
VOICE_NOT_ENABLED_MESSAGE = "Voice abhi aapke institute ke liye on nahi hai. Likh kar poochhiye."
MSG_TYPED = "Awaaz abhi samajh nahi aayi (voice service down). Likh kar poochhiye."
MSG_TOO_LONG = "Itna lamba audio nahi le sakte: ek baar mein 30 second tak boliye."
MSG_TOO_BIG = "Audio file bahut badi hai (1 MB tak)."
MSG_FORMAT = "Ye audio format samajh nahi aaya. Chrome ya Safari ka mic use karein, ya likh kar poochhiye."
MSG_EMPTY = "Kuch sunai nahi diya. Dobara boliye."
REFUSAL_RE = re.compile(r"^\s*(ye main nahi kar sakt[ai]|i can'?t do that|main ye nahi kar sakt[ai])", re.IGNORECASE)


# ---- models ------------------------------------------------------------------------------------------------------
class Speaker(BaseModel):
    id: str
    label: str
    gender: Literal["female", "male"]


class VoiceConfig(BaseModel):
    enabled: bool
    reason: Optional[Literal["kill_switch", "not_allowlisted", "not_configured"]] = None
    tts: Literal["sarvam", "none"]
    stt_provider: Optional[str] = None
    speakers: list[Speaker]
    default_speaker: str
    paces: list[float]
    max_seconds: int
    max_bytes: int
    formats: list[str]
    consent_version: str
    limits: dict[str, int]


class ClarifyChipOut(BaseModel):
    label: str
    value: str
    hint: Optional[str] = None


class ClarifyOut(BaseModel):
    question: str
    chips: list[ClarifyChipOut] = Field(max_length=5)


class TranscribeResponse(BaseModel):
    """`clarify` set = the name was ambiguous inside the caller's scope: show the chips, do NOT auto-send."""
    request_id: str
    transcript: str
    language: Literal["hi", "en", "hi-en"]
    confidence: float = Field(ge=0, le=1)
    duration_ms: int
    provider: str
    repaired: bool = False
    clarify: Optional[ClarifyOut] = None
    replayed: bool = False   # served from the Idempotency-Key cache


class SpeakRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=vc.SPEAK_MAX_CHARS)
    language: Optional[Literal["hi", "en", "hi-en"]] = None
    speaker: Optional[str] = None
    pace: Optional[float] = None
    phrase: Optional[str] = None   # a STATIC_PHRASES key: served from the audio cache (text is ignored)


# ---- helpers -----------------------------------------------------------------------------------------------------
def voice_gate(ctx: SessionCtx = Depends(get_session)) -> SessionCtx:
    if not vc.enabled():
        raise HTTPException(503, VOICE_OFF_MESSAGE)
    if not vc.institute_allowed(ctx.institute_id):
        raise HTTPException(403, VOICE_NOT_ENABLED_MESSAGE)
    return ctx


def _config_for(ctx: SessionCtx) -> VoiceConfig:
    chain = stt_chain()
    tts = tts_provider()
    enabled, reason = True, None
    if not vc.enabled():
        enabled, reason = False, "kill_switch"
    elif not vc.institute_allowed(ctx.institute_id):
        enabled, reason = False, "not_allowlisted"
    elif not chain.providers:
        enabled, reason = False, "not_configured"
    return VoiceConfig(
        enabled=enabled, reason=reason, tts="none" if tts.name == "none" else "sarvam",
        stt_provider=chain.providers[0].name if chain.providers else None,
        speakers=[Speaker(id=k, label=v["label"], gender=v["gender"]) for k, v in vc.SPEAKERS.items()],
        default_speaker=vc.DEFAULT_SPEAKER, paces=list(vc.PACES), max_seconds=vc.max_seconds(), max_bytes=vc.max_bytes(),
        formats=list(ACCEPTED_MIMES), consent_version=CONSENT_VERSION,
        limits={"per_10min": vc.user_per_10min(), "per_day": vc.user_daily()})


@router.get("/config", response_model=VoiceConfig)
def voice_config(ctx: SessionCtx = Depends(get_session)):
    """Never 403: a disabled institute gets enabled=false so the mic button is hidden, not broken."""
    return _config_for(ctx)


# ---- transcribe --------------------------------------------------------------------------------------------------
async def _read_limited(request: Request, limit: int) -> bytes:
    buf = bytearray()
    async for chunk in request.stream():
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(413, MSG_TOO_BIG)
    return bytes(buf)


def _transcribe_sync(ctx: SessionCtx, audio: bytes, request_id: str) -> tuple[dict, dict]:
    """Runs in a worker thread: decode -> STT chain -> entity repair. Returns (response dict, trace dict)."""
    tracer = Tracer(request_id=request_id)
    stages: dict[str, float] = {}
    t = time.time()
    try:
        info = inspect(audio)
    except AudioError as e:
        tracer.voice("transcribe", "rejected", stages, audio_bytes=len(audio), reason=e.code)
        raise HTTPException(422, MSG_FORMAT) from None
    stages["decode"] = (time.time() - t) * 1000
    if info.duration_ms > vc.max_seconds() * 1000:
        tracer.voice("transcribe", "rejected", stages, audio_bytes=len(audio), duration_ms=info.duration_ms, reason="too_long")
        raise HTTPException(422, MSG_TOO_LONG)
    kt = scope_keyterms(ctx) if vc.stt_model().startswith("saaras:v4") else None
    t = time.time()
    try:
        res = stt_chain().transcribe(audio, info.mime, kt)
    except ProviderDown as d:
        stages["stt"] = (time.time() - t) * 1000
        tracer.voice("transcribe", "failed", stages, audio_bytes=len(audio), duration_ms=info.duration_ms,
                     tried=d.tried, reason="providers_down")
        raise HTTPException(503, MSG_TYPED) from None
    stages["stt"] = (time.time() - t) * 1000
    tr = res.transcript
    seconds = info.duration_ms / 1000.0
    cost = vc.stt_cost_inr(seconds, tr.provider)
    text = tr.text.strip()
    if not text:
        tracer.voice("transcribe", "empty", stages, provider=tr.provider, model=tr.model, audio_bytes=len(audio),
                     duration_ms=info.duration_ms, cost_inr=cost, tried=res.tried, skipped=res.skipped)
        return ({"request_id": request_id, "transcript": "", "language": "hi", "confidence": 0.0,
                 "duration_ms": info.duration_ms, "provider": tr.provider}, {})
    t = time.time()
    rep = repair_entities(ctx, text)
    stages["repair"] = (time.time() - t) * 1000
    out = {"request_id": request_id, "transcript": rep.text, "language": classify(rep.text, tr.language_code),
           "confidence": confidence_of(rep.text, tr.language_probability), "duration_ms": info.duration_ms,
           "provider": tr.provider, "repaired": rep.changed, "clarify": rep.clarify}
    tracer.voice("transcribe", "clarified" if rep.clarify else "ok", stages, provider=tr.provider, model=tr.model,
                 audio_bytes=len(audio), duration_ms=info.duration_ms, cost_inr=cost, tried=res.tried,
                 skipped=res.skipped, repaired=rep.changed, repair_db_calls=rep.db_calls, words=len(rep.text.split()),
                 confidence=out["confidence"], language=out["language"])
    return out, {}


@router.post("/transcribe", response_model=TranscribeResponse,
             responses={413: {"description": "audio over VOICE_MAX_BYTES"}, 422: {"description": "format / over 30 s"},
                        429: {"description": "per-user voice limit"}, 503: {"description": "voice off or STT down -> type"}})
async def transcribe(request: Request, ctx: SessionCtx = Depends(voice_gate),
                     idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key", max_length=80),
                     content_type: Optional[str] = Header(None)):
    """Body = the recorded audio bytes (Content-Type audio/webm | audio/mp4 | audio/ogg | audio/wav). The container
    is decoded for its real duration; the client's own headers are not trusted."""
    uid = str(ctx.user_id)
    cached = limits.idem_get(uid, idempotency_key)
    if cached is not None:
        return {**cached, "replayed": True}
    hit = limits.check_user(uid)
    if hit is not None:
        raise HTTPException(429, hit.message, headers={"Retry-After": str(hit.retry_after_s)})
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct and ct not in ACCEPTED_MIMES and ct != "application/octet-stream":
        raise HTTPException(415, MSG_FORMAT)
    audio = await _read_limited(request, vc.max_bytes())
    if not audio:
        raise HTTPException(422, MSG_EMPTY)
    request_id = uuid.uuid4().hex[:16]
    out, _ = await run_in_threadpool(_transcribe_sync, ctx, audio, request_id)
    del audio
    limits.idem_put(uid, idempotency_key, out)
    return out


# ---- speak -------------------------------------------------------------------------------------------------------
_PHRASE_CACHE: dict[tuple, bytes] = {}
_PHRASE_LOCK = threading.Lock()


def _phrase_text(key: str, lang: str) -> Optional[str]:
    p = vc.STATIC_PHRASES.get(key)
    if not p:
        return None
    return p["en"] if lang == "en" else p["hi"]


@router.post("/speak", response_class=StreamingResponse,
             responses={200: {"content": {"audio/mpeg": {}}, "description": "audio stream"},
                        204: {"description": "TTS is off (VOICE_TTS=none): use browser speechSynthesis"},
                        422: {"description": "speaker / pace not in the allowlist"}, 503: {"description": "TTS down"}})
def speak(body: SpeakRequest, ctx: SessionCtx = Depends(voice_gate)):
    """Speaker and pace only from the allowlist (`GET /agent/voice/config`). Static phrases (`phrase`) come from an
    in-process cache; data answers are never cached (they carry names and numbers)."""
    speaker = body.speaker or vc.DEFAULT_SPEAKER
    if speaker not in vc.SPEAKERS:
        raise HTTPException(422, "unknown speaker")
    pace = 1.0 if body.pace is None else float(body.pace)
    if pace not in vc.PACES:
        raise HTTPException(422, "pace must be one of 0.9, 1.0, 1.1")
    tts = tts_provider()
    if tts.name == "none":
        return Response(status_code=204)
    lang = body.language or classify(body.text)
    text = body.text
    if body.phrase:
        text = _phrase_text(body.phrase, lang) or ""
        if not text:
            raise HTTPException(422, "unknown phrase")
    lc = tts_language_code(text, lang)
    request_id = uuid.uuid4().hex[:16]
    tracer = Tracer(request_id=request_id)
    model = getattr(tts, "model", "")
    key = (body.phrase, lang, speaker, pace, model)
    if body.phrase:
        with _PHRASE_LOCK:
            hit = _PHRASE_CACHE.get(key)
        if hit is not None:
            tracer.voice("speak", "cached", {}, provider=tts.name, model=model, chars=len(text), cost_inr=0.0, phrase=body.phrase)
            return Response(content=hit, media_type=getattr(tts, "mime", "audio/mpeg"), headers={"X-Agent-Request-Id": request_id})

    def stream() -> Iterator[bytes]:
        t0 = time.time()
        got = bytearray()
        try:
            for chunk in tts.synthesize(text, lc, speaker, pace):
                if body.phrase:
                    got.extend(chunk)
                yield chunk
        except ProviderError as e:
            tracer.voice("speak", "failed", {"tts": (time.time() - t0) * 1000}, provider=tts.name, model=model,
                         chars=len(text), reason=e.kind)
            return
        if body.phrase and got:
            with _PHRASE_LOCK:
                _PHRASE_CACHE[key] = bytes(got)
        tracer.voice("speak", "ok", {"tts": (time.time() - t0) * 1000}, provider=tts.name, model=model, chars=len(text),
                     cost_inr=vc.tts_cost_inr(len(text), model), phrase=body.phrase, language=lang)

    # first chunk eagerly, so a provider failure becomes a clean 503 instead of an empty 200
    it = stream()
    try:
        first = next(it)
    except StopIteration:
        raise HTTPException(503, "TTS abhi available nahi hai. Screen par jawab padh sakte hain.")

    def body_iter() -> Iterator[bytes]:
        yield first
        yield from it

    return StreamingResponse(body_iter(), media_type=getattr(tts, "mime", "audio/mpeg"),
                             headers={"Cache-Control": "no-store", "X-Agent-Request-Id": request_id})


def clear_phrase_cache() -> None:
    with _PHRASE_LOCK:
        _PHRASE_CACHE.clear()


# ---- voice chat --------------------------------------------------------------------------------------------------
def outcome_of(events: list[dict]) -> str:
    """answered | clarified | refused | navigated | failed — from the event stream (ids/types only)."""
    types = [e.get("type") for e in events]
    if "error" in types:
        return "failed"
    if any(e.get("type") == "ui_action" and e.get("action") == "navigate" for e in events):
        return "navigated"
    if "clarify" in types:
        return "clarified"
    text = " ".join(e.get("text", "") for e in events if e.get("type") == "text_delta")
    if REFUSAL_RE.match(text):
        return "refused"
    if any(e.get("type") == "ui_action" and e.get("action") == "highlight_confirm" for e in events):
        return "refused"
    return "answered"


@router.post("/chat", responses={200: {"description": "SSE, as POST /agent/chat, plus a `speech` event"}})
def voice_chat(body: ChatRequest, ctx: SessionCtx = Depends(voice_gate)):
    """The transcript becomes a normal chat message (same session, history, retention, clear-all). The only
    differences: the model is told the message was spoken (untrusted, keep it short, one-line refusals) and ends its
    answer with a `speech` event. Writes stay behind the on-screen approval gate — this endpoint never approves."""
    from ..api import run_chat
    from .guard import guard_turn
    t0 = time.time()
    guarded = guard_turn(body.message or "")

    def on_done(events: list[dict], agent) -> None:
        Tracer(request_id=agent.request_id).voice(
            "chat", outcome_of(events), {"turn": (time.time() - t0) * 1000}, provider="agent",
            model=getattr(agent, "route", ""), chars=sum(len(e.get("text", "")) for e in events if e.get("type") == "speech"),
            cost_inr=round(agent.cost * vc.INR_PER_USD, 5), speech=any(e.get("type") == "speech" for e in events),
            guarded=guarded is not None)

    return run_chat(body, ctx, voice=True, on_done=on_done, canned=guarded)
