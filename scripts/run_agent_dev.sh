#!/usr/bin/env bash
# Local dev for the a4ai Agent: fake seeded Postgres + uvicorn on :8000 with AGENT_DEV_MODE=1.
# Never points at production: the agent DB URLs are pinned to the local socket and other DB URL vars are unset
# for this process (the app also refuses to start in dev mode if a non-local DB URL is configured).
# Needs: PostgreSQL binaries (initdb/pg_ctl/psql), python deps from agent/README.md, A4AI_ANTHROPIC_API_KEY in env.
set -euo pipefail
cd "$(dirname "$0")/.."

PGPORT="${AGENT_PG_PORT:-54329}"
PGSOCK="${AGENT_PG_SOCKET_DIR:-/var/tmp}"
PGDATA="${AGENT_PGDATA:-/var/tmp/a4pg}"
PGBIN="${AGENT_PG_BIN:-$(dirname "$(command -v pg_ctl 2>/dev/null || ls -d /usr/lib/postgresql/*/bin/pg_ctl 2>/dev/null | head -1)")}"
as_pg() { if [ "$(id -u)" = "0" ]; then su postgres -c "$*"; else bash -c "$*"; fi; }

if ! psql -h "$PGSOCK" -p "$PGPORT" -U postgres -tAc "select 1" postgres >/dev/null 2>&1; then
  echo "Starting local Postgres in $PGDATA (port $PGPORT)"
  if [ "$(id -u)" = "0" ] && ! id postgres >/dev/null 2>&1; then useradd -m postgres; fi
  if [ ! -f "$PGDATA/PG_VERSION" ]; then
    mkdir -p "$PGDATA"; [ "$(id -u)" = "0" ] && chown postgres "$PGDATA"
    as_pg "$PGBIN/initdb -D $PGDATA -A trust -U postgres >/dev/null"
  fi
  as_pg "$PGBIN/pg_ctl -D $PGDATA -o '-p $PGPORT -k $PGSOCK' -l $PGDATA/server.log -w start >/dev/null"
fi
psql -h "$PGSOCK" -p "$PGPORT" -U postgres -tAc "select 1 from pg_database where datname='a4agent'" postgres | grep -q 1 \
  || createdb -h "$PGSOCK" -p "$PGPORT" -U postgres a4agent

unset DATABASE_URL SUPABASE_DB_URL SUPABASE_URL POSTGRES_URL POSTGRES_PRISMA_URL VERCEL
export AGENT_DATABASE_URL="postgresql://postgres@/a4agent?host=$PGSOCK&port=$PGPORT"
export AGENT_DATABASE_RO_URL="postgresql://agent_ro@/a4agent?host=$PGSOCK&port=$PGPORT"
export AGENT_DEV_MODE=1
export AGENT_CORS_ORIGINS="${AGENT_CORS_ORIGINS:-http://localhost:5173}"

echo "Seeding fake data (schema agent_demo)"
python3 -c "from agent.db import reset_and_seed; reset_and_seed()"
echo "a4ai Agent dev server: http://localhost:8000  (CORS: $AGENT_CORS_ORIGINS, X-Dev-Role: admin|teacher)"
exec python3 -m uvicorn agent.api:app --host 127.0.0.1 --port "${AGENT_PORT:-8000}"
