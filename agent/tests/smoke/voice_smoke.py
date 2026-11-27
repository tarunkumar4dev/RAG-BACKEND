#!/usr/bin/env python3
"""Nightly voice-pipeline smoke test for Sarvam STT + TTS.

Standalone — no imports from the agent app. Only `requests` is required at runtime. Reads SARVAM_API_KEY
from the environment and exits 0 cleanly when it is not set, so the CI job stays green on forks and PR
branches that don't carry the secret.

Three tiers (seven calls total, well under the $0.05 budget cap):
  STT (3): round-trip TTS-generated reference audio → STT twice, then a silence WAV to prove STT does not
           crash on empty input. We round-trip real audio because a sine wave would legitimately produce an
           empty transcript on Sarvam saaras:v3 and the task asks us to assert "non-empty string".
  TTS (3): two short Hinglish phrases + one 600-char long text (latency bounded at 10s).
  E2E (1): STT (feed the reference audio) → mocked LLM response → TTS. No Gemini call — mocking the LLM
           keeps a nightly job cheap and isolated from LLM-side outages.

Budget: $0.05 hard cap. Priced from Sarvam's rates (STT ₹0.49/min, TTS ₹1/1000 chars). If the running
total passes the cap, the script aborts before the next call and reports which tests did not run.

Exit: 0 when every test passed OR the API key is missing (skip). 1 on any failure, including budget
exceeded. The script never raises uncaught — a crash in one test is captured as a failed TestResult.

Prints a JSON summary to stdout:
  {"passed": N, "failed": N, "skipped": bool, "total_cost_usd": 0.XX, "results": [...], "cap_usd": 0.05}
Each result:
  {"test": "stt_roundtrip_1", "status": "pass"|"fail"|"skipped", "latency_ms": N,
   "cost_usd": 0.XX, "error": null|"msg"}
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import struct
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import requests  # the ONE allowed runtime dep

# --- Sarvam config (literals, never imported from agent) -------------------------------------------------

SARVAM_BASE = os.environ.get("SARVAM_BASE", "https://api.sarvam.ai").rstrip("/")
STT_MODEL = "saaras:v3"
STT_MODE = "translit"
TTS_MODEL = "bulbul:v4-flash"
TTS_SPEAKER = "ishita_enhi_companion"
TTS_LANGUAGE = "hi-IN"

# --- Pricing (Sarvam, from docs/VOICE_PLAN.md §3, 10 Nov 2026) -------------------------------------------

INR_PER_USD = 83.0
STT_INR_PER_SECOND = 0.49 / 60.0    # ₹0.49/min
TTS_INR_PER_CHAR = 1.0 / 1000.0     # ₹1/1000 chars

# --- Bounds from the task spec ---------------------------------------------------------------------------

DEFAULT_BUDGET_USD = 0.05
STT_LATENCY_LIMIT_MS = 3000
TTS_LATENCY_LIMIT_MS = 3000
LONG_TTS_TIMEOUT_MS = 10000
E2E_LATENCY_LIMIT_MS = 8000

# --- Fixtures --------------------------------------------------------------------------------------------

PHRASE_1 = "Namaste, yeh ek smoke test hai."
PHRASE_2 = "Class ki attendance check kar rahe hain."
LONG_SENTENCE = ("Aaj classroom me bachche ikatthe hue the aur teacher ne roll call ki, "
                 "phir attendance register me sab ka naam likha. ")
LONG_TEXT = (LONG_SENTENCE * 6)[:600]  # 600 chars, comfortably over the 500 threshold
MOCK_LLM_REPLY = "Haan, samajh gaya."   # fixed response in the e2e tier — no LLM is called


# --- Audio helpers ---------------------------------------------------------------------------------------


def silence_wav(seconds: float = 0.5, rate: int = 16000) -> bytes:
    """16-bit mono WAV of silence — for the "STT does not crash on empty audio" test."""
    n = int(seconds * rate)
    data = b"\x00\x00" * n
    hdr = (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
           + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16))
    return hdr + b"data" + struct.pack("<I", len(data)) + data


def _wav_duration_s(wav: bytes, default: float = 1.0) -> float:
    """Byte-level WAV duration parse. Returns a conservative default when the header is unreadable."""
    try:
        if wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
            return default
        # Walk chunks for 'fmt ' and 'data'
        i = 12
        sample_rate = channels = bits = 0
        data_len = 0
        while i + 8 <= len(wav):
            cid = wav[i:i + 4]
            clen = struct.unpack("<I", wav[i + 4:i + 8])[0]
            if cid == b"fmt " and clen >= 16:
                channels = struct.unpack("<H", wav[i + 10:i + 12])[0]
                sample_rate = struct.unpack("<I", wav[i + 12:i + 16])[0]
                bits = struct.unpack("<H", wav[i + 22:i + 24])[0]
            elif cid == b"data":
                data_len = clen
                break
            i += 8 + clen + (clen & 1)
        if sample_rate and channels and bits and data_len:
            return data_len / float(sample_rate * channels * (bits // 8))
    except Exception:
        pass
    return default


# --- Budget ----------------------------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    """Hard cap hit — the script aborts before any further paid call."""


@dataclass
class Budget:
    cap_usd: float
    total_inr: float = 0.0

    @property
    def total_usd(self) -> float:
        return self.total_inr / INR_PER_USD

    def _check(self) -> None:
        if self.total_usd > self.cap_usd:
            raise BudgetExceeded(f"budget {self.total_usd:.4f} USD > cap {self.cap_usd:.4f}")

    def add_stt(self, duration_s: float) -> float:
        cost_inr = max(duration_s, 0.0) * STT_INR_PER_SECOND
        self.total_inr += cost_inr
        self._check()
        return cost_inr / INR_PER_USD

    def add_tts(self, chars: int) -> float:
        cost_inr = max(chars, 0) * TTS_INR_PER_CHAR
        self.total_inr += cost_inr
        self._check()
        return cost_inr / INR_PER_USD


# --- Results ---------------------------------------------------------------------------------------------


@dataclass
class TestResult:
    test: str
    status: str                     # "pass" | "fail" | "skipped"
    latency_ms: int
    cost_usd: float = 0.0
    error: Optional[str] = None


# --- Sarvam calls ----------------------------------------------------------------------------------------


def call_stt(api_key: str, audio: bytes, mime: str, timeout: float = 10.0) -> tuple[dict, float]:
    """POST /speech-to-text (multipart). Returns (response_json, latency_ms)."""
    t0 = time.time()
    ext = {"audio/wav": "wav", "audio/webm": "webm", "audio/mp4": "m4a", "audio/ogg": "ogg"}.get(mime, "wav")
    r = requests.post(
        f"{SARVAM_BASE}/speech-to-text",
        headers={"api-subscription-key": api_key},
        files={"file": (f"audio.{ext}", audio, mime)},
        data={"model": STT_MODEL, "mode": STT_MODE, "language_code": "unknown"},
        timeout=timeout,
    )
    latency_ms = (time.time() - t0) * 1000
    if r.status_code >= 400:
        raise RuntimeError(f"stt http {r.status_code}: {r.text[:200]}")
    return r.json(), latency_ms


def call_tts(api_key: str, text: str, codec: str = "wav", timeout: float = 10.0) -> tuple[bytes, float]:
    """POST /text-to-speech (non-stream, base64 body). Returns (audio_bytes, latency_ms).

    Uses the REST (non-stream) endpoint so audio arrives as a single base64 blob — simpler for a smoke
    runner and the WAV codec plays nicely with /speech-to-text on the round trip.
    """
    t0 = time.time()
    r = requests.post(
        f"{SARVAM_BASE}/text-to-speech",
        headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
        json={
            "text": text,
            "language_code": TTS_LANGUAGE,
            "speaker": TTS_SPEAKER,
            "model": TTS_MODEL,
            "pace": 1.0,
            "output_audio_codec": codec,
            "speech_sample_rate": 22050,
        },
        timeout=timeout,
    )
    latency_ms = (time.time() - t0) * 1000
    if r.status_code >= 400:
        raise RuntimeError(f"tts http {r.status_code}: {r.text[:200]}")
    audios = (r.json() or {}).get("audios") or []
    if not audios:
        raise RuntimeError("tts: no 'audios' in response")
    return base64.b64decode(audios[0]), latency_ms


# --- Tests -----------------------------------------------------------------------------------------------


def _record(results: list[TestResult], test: str, status: str, latency_ms: float,
            cost_usd: float = 0.0, error: Optional[str] = None) -> None:
    results.append(TestResult(test=test, status=status, latency_ms=int(latency_ms),
                              cost_usd=round(cost_usd, 6), error=error))


def run_tts_short(api_key: str, text: str, label: str, budget: Budget, results: list[TestResult],
                  codec: str = "wav") -> Optional[bytes]:
    """One short TTS call: pass if 200 + non-empty audio + latency < 3s. Returns audio bytes or None."""
    try:
        cost_usd = budget.add_tts(len(text))
        audio, latency_ms = call_tts(api_key, text, codec=codec, timeout=10.0)
    except BudgetExceeded as e:
        _record(results, label, "fail", 0, 0.0, f"budget_exceeded: {e}")
        raise
    except Exception as e:
        _record(results, label, "fail", 0, 0.0, f"{type(e).__name__}: {e}")
        return None
    ok = latency_ms < TTS_LATENCY_LIMIT_MS and len(audio) > 0
    err = None if ok else f"latency {latency_ms:.0f}ms (limit {TTS_LATENCY_LIMIT_MS}) or empty audio ({len(audio)}B)"
    _record(results, label, "pass" if ok else "fail", latency_ms, cost_usd, err)
    return audio if ok else None


def run_tts_long(api_key: str, budget: Budget, results: list[TestResult]) -> None:
    """Long-text TTS: must not time out within 10s. Audio length > 0."""
    label = "tts_long_600_chars"
    try:
        cost_usd = budget.add_tts(len(LONG_TEXT))
        audio, latency_ms = call_tts(api_key, LONG_TEXT, codec="mp3", timeout=10.0)
    except BudgetExceeded as e:
        _record(results, label, "fail", 0, 0.0, f"budget_exceeded: {e}")
        raise
    except Exception as e:
        _record(results, label, "fail", 0, 0.0, f"{type(e).__name__}: {e}")
        return
    ok = latency_ms < LONG_TTS_TIMEOUT_MS and len(audio) > 0
    err = None if ok else f"latency {latency_ms:.0f}ms (limit {LONG_TTS_TIMEOUT_MS}) or empty audio ({len(audio)}B)"
    _record(results, label, "pass" if ok else "fail", latency_ms, cost_usd, err)


def run_stt_real(api_key: str, audio: bytes, label: str, budget: Budget, results: list[TestResult]) -> None:
    """STT on a real (TTS-generated) WAV — must yield a non-empty transcript within 3s."""
    duration_s = _wav_duration_s(audio, default=1.0)
    try:
        cost_usd = budget.add_stt(duration_s)
        resp, latency_ms = call_stt(api_key, audio, "audio/wav", timeout=10.0)
    except BudgetExceeded as e:
        _record(results, label, "fail", 0, 0.0, f"budget_exceeded: {e}")
        raise
    except Exception as e:
        _record(results, label, "fail", 0, 0.0, f"{type(e).__name__}: {e}")
        return
    transcript = resp.get("transcript")
    ok = (isinstance(transcript, str) and transcript.strip() != ""
          and latency_ms < STT_LATENCY_LIMIT_MS)
    if not isinstance(transcript, str):
        err = f"no transcript key (got {list(resp.keys())})"
    elif not transcript.strip():
        err = "empty transcript"
    elif latency_ms >= STT_LATENCY_LIMIT_MS:
        err = f"latency {latency_ms:.0f}ms >= {STT_LATENCY_LIMIT_MS}"
    else:
        err = None
    _record(results, label, "pass" if ok else "fail", latency_ms, cost_usd, err)


def run_stt_silence(api_key: str, budget: Budget, results: list[TestResult]) -> None:
    """STT on a silence WAV — must not crash. Empty transcript is OK; absence of the key is not."""
    label = "stt_silence_graceful"
    audio = silence_wav(0.5)
    duration_s = _wav_duration_s(audio, default=0.5)
    try:
        cost_usd = budget.add_stt(duration_s)
        resp, latency_ms = call_stt(api_key, audio, "audio/wav", timeout=10.0)
    except BudgetExceeded as e:
        _record(results, label, "fail", 0, 0.0, f"budget_exceeded: {e}")
        raise
    except Exception as e:
        _record(results, label, "fail", 0, 0.0, f"{type(e).__name__}: {e}")
        return
    # Graceful = 200 with a transcript key present (empty string is fine for silence).
    if "transcript" not in resp:
        _record(results, label, "fail", latency_ms, cost_usd,
                f"no transcript key in response (got {list(resp.keys())})")
        return
    _record(results, label, "pass", latency_ms, cost_usd, None)


def run_e2e(api_key: str, reference_audio: Optional[bytes], budget: Budget,
            results: list[TestResult]) -> None:
    """STT → mocked LLM → TTS. Total latency < 8s."""
    label = "e2e_stt_mock_llm_tts"
    if reference_audio is None:
        _record(results, label, "skipped", 0, 0.0,
                "no reference audio (earlier TTS failed) — cannot run e2e")
        return
    t0 = time.time()
    duration_s = _wav_duration_s(reference_audio, default=1.0)
    try:
        stt_cost = budget.add_stt(duration_s)
        stt_resp, stt_latency_ms = call_stt(api_key, reference_audio, "audio/wav", timeout=10.0)
        transcript = stt_resp.get("transcript", "")
        if not isinstance(transcript, str) or not transcript.strip():
            total_ms = (time.time() - t0) * 1000
            _record(results, label, "fail", total_ms, stt_cost, "stt returned empty transcript in e2e")
            return
        # Mocked LLM — no Gemini call, no network. Fixed reply decoupled from input content so a smoke
        # failure here is TTS-attributable, not LLM-attributable.
        reply = MOCK_LLM_REPLY
        tts_cost = budget.add_tts(len(reply))
        audio, _ = call_tts(api_key, reply, codec="mp3", timeout=10.0)
    except BudgetExceeded as e:
        total_ms = (time.time() - t0) * 1000
        _record(results, label, "fail", total_ms, 0.0, f"budget_exceeded: {e}")
        raise
    except Exception as e:
        total_ms = (time.time() - t0) * 1000
        _record(results, label, "fail", total_ms, 0.0, f"{type(e).__name__}: {e}")
        return
    total_ms = (time.time() - t0) * 1000
    total_cost = stt_cost + tts_cost
    ok = total_ms < E2E_LATENCY_LIMIT_MS and len(audio) > 0
    err = None if ok else f"total {total_ms:.0f}ms (limit {E2E_LATENCY_LIMIT_MS}) or empty audio ({len(audio)}B)"
    _record(results, label, "pass" if ok else "fail", total_ms, total_cost, err)


# --- Main ------------------------------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Sarvam voice-pipeline smoke test.")
    p.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD,
                   help=f"Hard cap on this run's Sarvam spend (default ${DEFAULT_BUDGET_USD}).")
    args = p.parse_args(argv)

    api_key = (os.environ.get("SARVAM_API_KEY") or "").strip()
    results: list[TestResult] = []

    if not api_key:
        summary = {
            "passed": 0, "failed": 0, "skipped": True,
            "reason": "SARVAM_API_KEY not set",
            "total_cost_usd": 0.0, "cap_usd": args.budget_usd,
            "results": [],
        }
        print(json.dumps(summary, indent=2))
        return 0   # a missing secret is an expected, graceful skip — not a failure

    budget = Budget(cap_usd=args.budget_usd)
    reference_audio: Optional[bytes] = None
    budget_hit = False

    try:
        # TTS tier first: generates the reference audio the STT tier replays.
        reference_audio = run_tts_short(api_key, PHRASE_1, "tts_short_1_phrase1", budget, results,
                                         codec="wav")
        run_tts_short(api_key, PHRASE_2, "tts_short_2_phrase2", budget, results, codec="mp3")
        run_tts_long(api_key, budget, results)

        # STT tier: two real-audio replays + one silence.
        if reference_audio:
            run_stt_real(api_key, reference_audio, "stt_roundtrip_1", budget, results)
            run_stt_real(api_key, reference_audio, "stt_roundtrip_2", budget, results)
        else:
            _record(results, "stt_roundtrip_1", "skipped", 0, 0.0, "no reference audio from TTS tier")
            _record(results, "stt_roundtrip_2", "skipped", 0, 0.0, "no reference audio from TTS tier")
        run_stt_silence(api_key, budget, results)

        # E2E tier.
        run_e2e(api_key, reference_audio, budget, results)

    except BudgetExceeded:
        budget_hit = True
        # Anything below the current test does not run — record them as skipped so the summary is clean.
        already = {r.test for r in results}
        for label in ("tts_short_1_phrase1", "tts_short_2_phrase2", "tts_long_600_chars",
                      "stt_roundtrip_1", "stt_roundtrip_2", "stt_silence_graceful",
                      "e2e_stt_mock_llm_tts"):
            if label not in already:
                _record(results, label, "skipped", 0, 0.0, "budget cap hit; not run")

    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    summary = {
        "passed": passed,
        "failed": failed,
        "skipped": False,
        "total_cost_usd": round(budget.total_usd, 6),
        "cap_usd": args.budget_usd,
        "budget_exceeded": budget_hit,
        "results": [asdict(r) for r in results],
    }
    print(json.dumps(summary, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
