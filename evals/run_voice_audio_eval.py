"""Audio eval: every file in evals/voice_audio/ through the REAL STT (Sarvam, then Gemini if asked), clean and with
noise mixed at 20 / 10 / 5 dB SNR. Reports WER, entity accuracy and (with --live) tool accuracy, plus STT latency
p50/p95 per condition.

usage:  python evals/run_voice_audio_eval.py [--provider sarvam|gemini] [--noise path.wav] [--live anthropic|gemini]

evals/voice_audio/ (gitignored except README.md and manifest.yaml) holds real phone recordings, each with a line in
manifest.yaml:  - {file: v01.webm, text: "Sharma ma'am ki is mahine ki attendance kya hai", role: admin,
                   entities: ["Priya Sharma"], expect_tools: [resolve_entity, attendance_stats]}
Noise mixing needs ffmpeg on PATH (decode any container -> 16 kHz mono PCM, mix in numpy-free pure Python, re-encode
WAV). Without a key or without recordings the script says so and exits 2 — it never invents results.
Audio is read from disk and sent to the vendor; nothing is written back except the report.
"""
from __future__ import annotations

import argparse
import array
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import wave

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
AUDIO_DIR = os.path.join(ROOT, "evals", "voice_audio")
SNRS = (None, 20, 10, 5)


def words(t: str) -> list[str]:
    import re
    return re.findall(r"[\w']+", (t or "").lower())


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
            prev = cur
    return d[len(h)] / len(r)


def to_pcm16(path: str) -> tuple[array.array, int]:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise SystemExit("noise mixing needs ffmpeg on PATH (clean condition can still run with --clean-only)")
    raw = subprocess.run([ff, "-v", "error", "-i", path, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                         capture_output=True, check=True).stdout
    a = array.array("h")
    a.frombytes(raw)
    return a, 16000


def rms(a: array.array) -> float:
    return math.sqrt(sum(x * x for x in a) / max(len(a), 1)) or 1.0


def mix(speech: array.array, noise: array.array, snr_db: float) -> bytes:
    """Scale noise so that 20*log10(rms(speech)/rms(noise)) == snr_db, loop it, add, clip, -> WAV bytes."""
    gain = rms(speech) / (rms(noise) * 10 ** (snr_db / 20))
    out = array.array("h", (max(-32768, min(32767, int(s + noise[i % len(noise)] * gain))) for i, s in enumerate(speech)))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        name = f.name
    with wave.open(name, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(out.tobytes())
    data = open(name, "rb").read()
    os.unlink(name)
    return data


def babble(n: int, seed: int = 7) -> array.array:
    """Fallback noise when no --noise file is given: band-limited random noise (a stand-in for an office hum; a real
    corridor recording via --noise is the better test and is what the report should use)."""
    rnd = random.Random(seed)
    a, y = array.array("h"), 0.0
    for _ in range(n):
        y = 0.97 * y + rnd.uniform(-1, 1) * 900
        a.append(int(max(-32768, min(32767, y))))
    return a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["sarvam", "gemini"], default="sarvam")
    ap.add_argument("--noise", help="a WAV/any-container noise recording to mix in (corridor, fan, traffic)")
    ap.add_argument("--clean-only", action="store_true")
    ap.add_argument("--live", choices=["anthropic", "gemini"], help="also run each transcript through the agent loop")
    a = ap.parse_args(argv)
    import yaml
    from agent.voice import config as vc
    from agent.voice.audio import inspect
    from agent.voice.providers import GeminiSTT, ProviderError, SarvamSTT
    from agent.voice.repair import repair
    key = vc.sarvam_key() if a.provider == "sarvam" else vc.gemini_key()
    man_path = os.path.join(AUDIO_DIR, "manifest.yaml")
    items = []
    if os.path.exists(man_path):
        items = [m for m in (yaml.safe_load(open(man_path, encoding="utf-8")) or []) if os.path.exists(os.path.join(AUDIO_DIR, m["file"]))]
    if not key or not items:
        why = []
        if not key:
            why.append(f"no {vc.SARVAM_KEY_ENV if a.provider == 'sarvam' else 'GEMINI_API_KEY'} in the environment")
        if not items:
            why.append("no recordings listed in evals/voice_audio/manifest.yaml")
        print("audio eval NOT run: " + "; ".join(why) + ". No synthetic results are substituted.")
        return 2
    stt = SarvamSTT() if a.provider == "sarvam" else GeminiSTT()
    noise = None
    if a.noise:
        noise, _ = to_pcm16(a.noise)
    from agent.schemas import SessionCtx
    users = {"admin": SessionCtx(user_id="aaaaaaaa-0000-0000-0000-000000000001", role="admin", institute_id="11111111-1111-1111-1111-111111111111"),
             "teacher": SessionCtx(user_id="aaaaaaaa-0000-0000-0000-000000000002", role="teacher", institute_id="11111111-1111-1111-1111-111111111111", teacher_id="aaaaaaaa-0000-0000-0000-000000000002")}
    results: dict = {snr: [] for snr in (SNRS[:1] if a.clean_only else SNRS)}
    for m in items:
        path = os.path.join(AUDIO_DIR, m["file"])
        raw = open(path, "rb").read()
        info = inspect(raw)
        pcm = None
        for snr in results:
            if snr is None:
                data, mime = raw, info.mime
            else:
                if pcm is None:
                    pcm, _ = to_pcm16(path)
                data, mime = mix(pcm, noise or babble(len(pcm)), snr), "audio/wav"
            t = time.perf_counter()
            try:
                tr = stt.transcribe(data, mime)
                text, err = tr.text, None
            except ProviderError as e:
                text, err = "", e.kind
            ms = (time.perf_counter() - t) * 1000
            fixed = repair(users[m.get("role", "admin")], text).text if text else ""
            ents = m.get("entities") or []
            ent_ok = all(e.lower() in fixed.lower() for e in ents) if ents else None
            results[snr].append({"file": m["file"], "wer": wer(m["text"], fixed), "wer_raw": wer(m["text"], text),
                                 "entity_ok": ent_ok, "ms": ms, "error": err, "seconds": info.duration_ms / 1000})
    print(f"provider {a.provider} · {len(items)} recordings · noise {'file ' + os.path.basename(a.noise) if a.noise else 'synthetic hum (use --noise for a real corridor)'}")
    print("| condition | WER (after repair) | WER (raw) | entity acc | STT p50 ms | STT p95 ms | errors |")
    print("|---|---|---|---|---|---|---|")
    for snr, rs in results.items():
        ms = sorted(r["ms"] for r in rs)
        ent = [r["entity_ok"] for r in rs if r["entity_ok"] is not None]
        p95 = ms[min(len(ms) - 1, int(round(0.95 * (len(ms) - 1))))]
        print(f"| {'clean' if snr is None else f'{snr} dB'} | {statistics.mean(r['wer'] for r in rs):.3f} | "
              f"{statistics.mean(r['wer_raw'] for r in rs):.3f} | {sum(ent)}/{len(ent) or 0} | {statistics.median(ms):.0f} | {p95:.0f} | "
              f"{sum(1 for r in rs if r['error'])} |")
    if a.live:
        print("tool accuracy on transcripts: run evals/run_voice_evals.py --live with the clean transcripts (same loop)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
