-- a4ai Agent store on Supabase — FUTURE production deploy. NOT APPLIED anywhere. Review before running.
--
-- Same tables as agent/store.py (SQLite), in a separate `agent` schema so the a4ai tables stay untouched.
-- The agent's read-only role (agent_ro) gets NO access here: the agent backend would write with a separate role
-- (or with the caller's JWT through PostgREST, where RLS below applies). Every row is owned by auth.uid().
--
-- Apply (owner only, after review):  psql "$OWNER_URL" -f agent/sql/agent_store_supabase.sql
-- Retention (90 days) is the agent's startup purge today; in production schedule agent.purge_old() with pg_cron.

begin;

create schema if not exists agent;
revoke all on schema agent from public;
grant usage on schema agent to authenticated;

create table if not exists agent.sessions (
  id uuid primary key,
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  institute_id uuid not null references public.institutes(id) on delete cascade,
  role text not null check (role in ('admin', 'hod', 'teacher')),
  title text not null check (char_length(title) <= 80),
  title_source text not null default 'auto' check (title_source in ('auto', 'user')),
  pinned boolean not null default false,
  created_at timestamptz not null default now(),
  last_message_at timestamptz not null default now()
);
create index if not exists sessions_owner on agent.sessions (user_id, institute_id, last_message_at desc, id desc);

create table if not exists agent.messages (
  session_id uuid not null references agent.sessions(id) on delete cascade,
  seq integer not null check (seq > 0),
  user_id uuid not null default auth.uid(),
  institute_id uuid not null,
  role text not null check (role in ('user', 'assistant')),
  text text,
  events jsonb,   -- AgentEvent[] streamed for the turn (assistant)
  llm jsonb,      -- provider-neutral messages of the turn (assistant), for history replay
  created_at timestamptz not null default now(),
  primary key (session_id, seq)
);
create index if not exists messages_owner on agent.messages (user_id, institute_id, session_id, seq);

create table if not exists agent.queries (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null default auth.uid() references auth.users(id) on delete cascade,
  institute_id uuid not null references public.institutes(id) on delete cascade,
  norm text not null,          -- lower(trim(collapsed whitespace))
  text text not null,
  last_used_at timestamptz not null default now(),
  use_count integer not null default 1,
  unique (user_id, institute_id, norm)
);

create table if not exists agent.feedback (
  user_id uuid not null default auth.uid(),
  institute_id uuid not null,
  session_id uuid not null references agent.sessions(id) on delete cascade,
  message_seq integer not null,
  rating text not null check (rating in ('up', 'down')),
  note text check (char_length(note) <= 1000),
  created_at timestamptz not null default now(),
  primary key (user_id, institute_id, session_id, message_seq),
  foreign key (session_id, message_seq) references agent.messages(session_id, seq) on delete cascade
);

-- RLS: a row is visible / writable only by its owner. institute_id must be an institute the user is active staff of
-- (same rule as agent/real_identity.py), so a forged institute_id cannot be inserted.
alter table agent.sessions enable row level security;
alter table agent.messages enable row level security;
alter table agent.queries  enable row level security;
alter table agent.feedback enable row level security;

create or replace function agent.is_staff_of(inst uuid) returns boolean
language sql stable security definer set search_path = public as $$
  select exists (select 1 from public.institutes i where i.id = inst and i.owner_id = auth.uid())
      or exists (select 1 from public.institute_members m where m.institute_id = inst and m.user_id = auth.uid()
                 and m.status = 'active' and m.role in ('admin', 'institute', 'hod', 'proctor', 'teacher'));
$$;
revoke all on function agent.is_staff_of(uuid) from public;
grant execute on function agent.is_staff_of(uuid) to authenticated;

do $$
declare t text;
begin
  foreach t in array array['sessions', 'messages', 'queries', 'feedback'] loop
    execute format('drop policy if exists owner_all on agent.%I', t);
    execute format('create policy owner_all on agent.%I for all to authenticated '
                   'using (auth.uid() = user_id) '
                   'with check (auth.uid() = user_id and agent.is_staff_of(institute_id))', t);
  end loop;
end $$;

grant select, insert, update, delete on agent.sessions, agent.messages, agent.queries, agent.feedback to authenticated;

-- messages must belong to a session of the same owner + institute
create or replace function agent.check_message_owner() returns trigger language plpgsql as $$
begin
  if not exists (select 1 from agent.sessions s where s.id = new.session_id and s.user_id = new.user_id
                 and s.institute_id = new.institute_id) then
    raise exception 'session not owned by this user';
  end if;
  return new;
end $$;
drop trigger if exists messages_owner_check on agent.messages;
create trigger messages_owner_check before insert or update on agent.messages
  for each row execute function agent.check_message_owner();

create or replace function agent.purge_old(days integer default 90) returns void
language sql security definer set search_path = agent as $$
  delete from agent.sessions where last_message_at < now() - make_interval(days => days);
  delete from agent.queries  where last_used_at   < now() - make_interval(days => days);
$$;
revoke all on function agent.purge_old(integer) from public;

commit;

-- VERIFY (read-only):
--   select tablename, rowsecurity from pg_tables where schemaname = 'agent';           -- 4 rows, all true
--   select has_schema_privilege('agent_ro', 'agent', 'usage');                         -- false
