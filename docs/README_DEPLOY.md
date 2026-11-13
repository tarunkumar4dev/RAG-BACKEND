# a4ai Agent — Deployment Guide (MSIT Trial)

## Prerequisites

- Docker 24+ on the deploy host
- Supabase project (Mumbai region recommended)
- Anthropic API key (AGENT_ANTHROPIC_API_KEY) and/or Google Gemini key (A4AI_GEMINI_API_KEY)
- The deploy host must reach the Supabase Postgres on port 5432 (or pooler on 6543)

## Environment Variables

### Required

| Variable | Example | Notes |
|---|---|---|
| `AGENT_DB_URL` | `postgresql://agent_ro@db.xxx.supabase.co:5432/postgres` | Read-only role (`agent_ro`). |
| `AGENT_SUPABASE_JWT_SECRET` | (from Supabase dashboard → Settings → API) | HS256 secret for verifying JWTs. |
| `AGENT_ANTHROPIC_API_KEY` | `sk-ant-…` | Sonnet for paper gen, Haiku for insights. |
| `A4AI_GEMINI_API_KEY` | `AIza…` | Fallback / alternative provider. |
| `AGENT_CORS_ORIGINS` | `https://app.a4ai.co` | Comma-separated; **no wildcard**. |
| `AGENT_STORE_URL` | `postgresql://agent_rw@db.xxx.supabase.co:5432/postgres` | Store for sessions/drafts (needs write). |

### Optional

| Variable | Default | Notes |
|---|---|---|
| `AGENT_STORE` | `sqlite` | `postgres` for multi-worker. |
| `AGENT_MODEL` | `claude-sonnet-5-5` | Main model (paper gen). |
| `AGENT_FAST_MODEL` | `claude-haiku-5-5` | Fast model (insights). |
| `AGENT_ROUTING` | `1` | `0` = all turns use main model. |
| `AGENT_PROMPT_CACHE` | `1` | `0` to disable Anthropic prompt caching. |
| `AGENT_USER_TURNS_PER_10MIN` | `20` | Per-user rate limit; `0` = unlimited. |
| `AGENT_INSTITUTE_DAILY_TOKENS` | (unset = unlimited) | Daily token budget per institute. |
| `AGENT_ENABLED_INSTITUTES` | (unset = all) | Comma-separated institute UUIDs. |
| `WEB_CONCURRENCY` | `2` | Gunicorn workers. |
| `AGENT_THREADPOOL` | `200` | anyio worker thread limit per worker. |
| `AGENT_LOG_FORMAT` | `json` (set in Dockerfile) | `text` for readable dev logs. |
| `PORT` | `8000` | Listen port. |
| `ENVIRONMENT` | `production` (set in Dockerfile) | Affects logging, dev mode checks. |

## Build & Run

```bash
# From the repo root (the Dockerfile is at agent/Dockerfile and COPYs from root context)
docker build -f agent/Dockerfile -t a4ai-agent:latest .

docker run -d --name a4ai-agent \
  -p 8000:8000 \
  --env-file agent.env \
  --restart unless-stopped \
  a4ai-agent:latest
```

## Health Checks

- `GET /healthz` — always 200 (process is alive)
- `GET /readyz` — 200 when DB is reachable, 503 otherwise
- `GET /agent/metrics` — Prometheus text exposition (no PII)

## Database Setup

1. Create the `agent_ro` role (read-only, no RLS bypass):
   ```sql
   CREATE ROLE agent_ro LOGIN PASSWORD '…';
   GRANT USAGE ON SCHEMA public TO agent_ro;
   GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_ro;
   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO agent_ro;
   ```

2. For the session store (`AGENT_STORE=postgres`), create the `agent` schema using `agent/sql/agent_store_supabase.sql`.

3. The agent never runs migrations. All DDL is managed externally.

## Mumbai Region Notes

- Supabase project in `ap-south-1` keeps DB latency under 5 ms from a Mumbai deploy host.
- The Anthropic API endpoint is global; Gemini has a Mumbai-proximate endpoint.
- Set `AGENT_DB_POOL_TIMEOUT=5` (default) — if the pool can't get a connection in 5 s, the tool returns `db_unavailable`.

## Monitoring

- `/agent/metrics` exposes counters (turns, tools, tokens, cost, errors, rate limits) and histograms (latency).
- No PII in metrics by construction: labels are from fixed vocabularies only.
- Wire Prometheus to scrape `/agent/metrics` and alert on `agent_errors_total{error="provider_failed"}`.

## Rollback

The agent is stateless except for the session store (Postgres or SQLite). To roll back:
1. Deploy the previous Docker image tag.
2. No DB migration rollback needed (the agent never writes DDL).
