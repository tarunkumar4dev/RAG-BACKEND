# Voice eval results — 10 Nov 2026

Environment: Windows 11, embedded Postgres seed (`AGENT_DB=embedded`), **no LLM key, no Sarvam key, no recordings**
in `evals/voice_audio/`. Everything marked "not run" needs those; nothing was simulated in its place.

```powershell
$env:AGENT_DB="embedded"; agent\.devdata\venv\Scripts\python evals\run_voice_evals.py                 # offline layers
$env:AGENT_DB="embedded"; agent\.devdata\venv\Scripts\python evals\run_voice_evals.py --live anthropic # + full loop
$env:AGENT_DB="embedded"; agent\.devdata\venv\Scripts\python evals\run_voice_audio_eval.py --provider sarvam --noise C:\path\corridor.wav
```

## Report table

| metric | result | how measured |
|---|---|---|
| tool accuracy (40-utterance set, full loop) | **not run** | needs `A4AI_ANTHROPIC_API_KEY` or `GEMINI_API_KEY` |
| entity accuracy, text (misheard names → caller's scope) | **4/4** | offline, real repair code on the seed DB (v02, v03, v19, v20) |
| entity accuracy, audio (WER / names at clean, 20, 10, 5 dB) | **not run** | needs `SARVAM_API_KEY` + recordings |
| out-of-scope refusal rate | **not run** (model) | the refusal shape + outcome classifier are unit-tested |
| unsafe-action rate | **0/4 approve-by-voice + 1 injection reach the model** (deterministic guard); full-loop rate **not run** | offline guard on the set; `test_voice_phase2.py` checks no `pending_approval` |
| guard false positives (questions blocked) | **0/35** | offline |
| latency: decode | p50 < 1 ms | `tests/agent/test_voice_phase5.py` (9 s WebM) |
| latency: entity repair | p50 6.0 ms, p95 28.2 ms (first call loads the 5-min entity cache) | offline set, seed DB |
| latency: mic stop → transcript (4G, Sarvam) | **not measured** | needs a key and a phone on 4G |
| latency: spoken answer starts | **not measured** | same |
| cost per query | **₹0.30 estimated** (STT ₹0.065 + TTS ₹0.12 + Haiku ≈ ₹0.10) | prices from docs.sarvam.ai on 10 Nov 2026; trace records `cost_inr` per request |

## Latency targets vs what can be said today
Targets (4G, ≤ 10 s utterance): mic stop → transcript p50 ≤ 1.5 s / p95 ≤ 3 s; spoken answer starts p95 ≤ 7 s.
The in-process stages measured here (decode < 1 ms, repair ≤ 30 ms) are not where the time goes. The vendor stages
(upload ~0.3–0.8 s, Sarvam STT, the LLM turn, TTS first byte) are unmeasured. The trace's `stages_ms` and the
`agent_voice_stage_seconds` histogram give the breakdown as soon as real traffic runs. If p95 is missed, the top 3
cuts are in `docs/VOICE_PLAN.md` §5.
