# Voice deploy — Mumbai, and how to switch it off in seconds

Voice is part of the agent process (`agent/voice/`, mounted in `agent/api.py`). Same container, same env file as
`docs/README_DEPLOY.md`. Today nothing serves `/agent/*` publicly; Vercel (`vercel.json`) only runs the
test-generator backend.

## Where to run it
- One small VM or container host in **AWS ap-south-1 (Mumbai)**, next to the Supabase project (also ap-south-1).
  Sarvam's API is India-hosted, so STT/TTS round trips stay in the country.
- 2 vCPU / 2 GB is enough for ~300 institutes at the expected rate (voice adds one STT call and one TTS call per
  question; the CPU work in-process is < 30 ms). Gunicorn: `WEB_CONCURRENCY=2` (see `agent/gunicorn.conf.py`).
  Limits and caches are per process: two workers allow up to 2× the per-user voice limit. That is acceptable for
  the pilot.
- HTTPS in front (the browser only allows the mic on HTTPS). Set `AGENT_CORS_ORIGINS` to the exact app origin.

## Environment variables (voice)

| variable | example | notes |
|---|---|---|
| `VOICE_ENABLED` | `1` | **kill switch**. Anything else = voice off; the mic disappears on the next page load. |
| `VOICE_INSTITUTE_ALLOWLIST` | `<uuid>,<uuid>` | pilot institutes; empty = all |
| `SARVAM_API_KEY` | (secret) | STT + TTS. Name can be changed with `VOICE_SARVAM_KEY_ENV`. |
| `GEMINI_API_KEY` | (secret) | STT fallback (already used by the agent) |
| `VOICE_STT_MODEL` | `saaras:v3` | default; `mode=translit` gives Roman Hinglish. `saaras:v4` = native script + name keyterms |
| `VOICE_STT_MODE` | `translit` | v3 only |
| `VOICE_TTS_MODEL` | `bulbul:v4-flash` | ₹1 / 1000 chars; `bulbul:v3` is ₹3 |
| `VOICE_TTS` | `none` | optional: no server TTS, phones use their own speech engine |
| `VOICE_USER_PER_10MIN` | `30` | per-user voice requests per 10 min |
| `VOICE_USER_DAILY` | `100` | per-user daily cap (IST); friendly message, typing still works |
| `VOICE_MAX_SECONDS` / `VOICE_MAX_BYTES` | `30` / `1048576` | cannot be raised above 30 s (Sarvam REST limit) |

Keys go in the host's secret store or `agent.env` (never in git). Nothing prints them.

## Switch voice off in seconds
1. Set `VOICE_ENABLED=0` (or remove it) in the env file / secret store.
2. Restart the process: `docker restart a4ai-agent` (a few seconds). Every voice endpoint answers 503 "Voice abhi
   band hai. Likh kar poochhiye.", `GET /agent/voice/config` says `enabled: false`, and the mic is hidden. Typed
   chat keeps working.
3. Only one institute misbehaving: remove its id from `VOICE_INSTITUTE_ALLOWLIST` and restart.
4. Vendor outage without a deploy: nothing to do. The server falls back from Sarvam to Gemini and then to
   "Likh kar poochhiye". A provider that fails 3 times in a row is skipped for 60 s.

## Check after deploy
```bash
curl -s https://<agent-host>/healthz
curl -s https://<agent-host>/agent/metrics | grep agent_voice_
```
Watch `agent_voice_requests_total{outcome="failed"}` and `agent_voice_stage_seconds{stage="stt"}`. The trace
(`AGENT_TRACE_PATH`) has one `kind: voice` line per request with stage timings and estimated ₹ cost, ids only.
