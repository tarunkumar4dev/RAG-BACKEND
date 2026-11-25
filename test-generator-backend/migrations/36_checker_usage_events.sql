-- 36_checker_usage_events.sql — service-only, append-only usage log for every checker AI call unit
--
-- Covers cases rubric.build and check_jobs.usage can't:
--   • failed first-time rubric builds on the sync route (no rubric row to attach build to)
--   • failed direct copy checks (no usage column on check_sheets; ai_result is unsafe)
--   • a durable, queryable log of every unit's cost (rubric / copy / chunk / shadow)
--
-- No student names or content. RLS enabled with zero client policies; anon and authenticated
-- are revoked so only the service role (backend) can read or write.
-- Ships independently of 35_ai_unreadable.sql.
--
-- Additive and idempotent. Rollback: 36_checker_usage_events_rollback.sql.

create table if not exists public.checker_usage_events (
  id              uuid primary key default gen_random_uuid(),
  created_at      timestamptz not null default now(),
  -- what was done
  kind            text not null check (kind in ('rubric', 'copy', 'chunk', 'shadow')),
  route           text not null check (route in ('job', 'sync')),
  -- links (all nullable: a sync rubric build has no job_id; a failed first build has no sheet)
  session_id      uuid,                -- references check_sessions(id); no FK to avoid blocking deletes
  sheet_id        uuid,                -- references check_sheets(id)
  job_id          uuid,                -- references check_jobs(id)
  -- model and tokens
  model           text,                -- the model that answered (null if no reply came back)
  input_tokens    bigint not null default 0,
  output_tokens   bigint not null default 0,
  thinking_tokens bigint not null default 0,
  gemini_calls    int not null default 0,
  billed_calls    int not null default 0,
  est_cost_inr    numeric(10,4) not null default 0,
  -- outcome
  ok              boolean not null,
  error_code      text                 -- the job_worker error code or a sync-route label; null on success
);

-- Index for cost queries: by session, by day, by kind
create index if not exists checker_usage_events_session_idx
  on public.checker_usage_events (session_id) where session_id is not null;
create index if not exists checker_usage_events_created_idx
  on public.checker_usage_events (created_at);

-- Service-only: no client can read or write
alter table public.checker_usage_events enable row level security;
revoke all on public.checker_usage_events from anon, authenticated;
-- service_role bypasses RLS by default; no policy needed

comment on table public.checker_usage_events is
  'Append-only log of every checker AI call unit (rubric / copy / chunk / shadow). '
  'Service-role only; no student names or content. 36_checker_usage_events.sql.';

-- ── verify (run by hand after applying) ──
-- select relrowsecurity from pg_class where relname = 'checker_usage_events';  -- expect true
-- select has_table_privilege('authenticated', 'public.checker_usage_events', 'select');  -- expect false
-- select has_table_privilege('anon', 'public.checker_usage_events', 'select');  -- expect false
