# Deploying the a4ai Agent

## Why not the current Vercel serverless function
`RAG-BACKEND/vercel.json` routes everything to one `@vercel/python` function (`api/index.py`, region bom1).
That is wrong for the agent:
- **Long runs.** A test-paper turn is blueprint → batch generation → independent solver → up to 3 regenerate rounds:
  37–96 s measured locally, budget `max_wall_s = 150`. Serverless functions have a hard max duration (plan dependent)
  and are billed for the whole wait on the LLM; a timeout kills the turn mid-generation with no partial state.
- **SSE.** `/agent/chat` streams `AgentEvent`s for the whole turn. Python serverless responses are buffered or cut at the
  duration limit, so the user sees nothing until the end (or an error), and the "status line" UX is lost.
- **Threads.** Generation runs parallel section batches in a thread pool; serverless instances freeze between
  invocations, so background work is not safe there.
- **Per-instance state.** In-memory rate limits/caches reset on every cold start (already a known issue in CLAUDE.md).

## Option 1 — long-running container (recommended)
Fly.io / Railway / Render / Cloud Run (min-instances 1, region Mumbai): `uvicorn agent.api:app --workers 2`
(or mount `agent.api.router` into `app.main`). Set request timeout ≥ 180 s, disable proxy buffering for
`text/event-stream` (`X-Accel-Buffering: no` is already sent). Point `AGENT_DATABASE_URL` / `AGENT_DATABASE_RO_URL`
at Supabase Postgres (RO URL = a `SELECT`-only role with `default_transaction_read_only=on`). Frontend calls it via
`backendUrl()` with a separate base URL for `/agent/*`.

## Option 2 — keep Vercel for the HTTP edge, move the run to a queue worker
`POST /agent/chat` (Vercel) only validates the JWT, inserts an `agent_runs` row and returns `run_id` in < 1 s.
A worker (same container image as option 1, or Supabase Edge Function + pg_net → worker, reusing the existing
`check_jobs` claim/finish/sweep pattern from migration 18) runs the loop and appends events to `agent_events`.
The browser streams them with Supabase Realtime (RLS: owner or same-institute admin) or short polling
`GET /agent/runs/{id}/events?after=<seq>`. Survives deploys and timeouts; `seq` makes resume exact.

Either way: secrets (`A4AI_ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) only as env vars on the runtime, never in the repo,
and `AGENT_DEV_AUTH` must be unset in production (it is also refused whenever `VERCEL` is set).
