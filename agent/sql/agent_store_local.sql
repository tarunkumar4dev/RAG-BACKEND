-- Agent store tables for LOCAL dev / embedded pgserver.  No RLS, no auth.uid(), no FK references.
-- Functionally identical to agent_store_supabase.sql but portable.
-- Applied from tests / scripts only:  psql "$URL" -f agent/sql/agent_store_local.sql

CREATE SCHEMA IF NOT EXISTS agent;

CREATE TABLE IF NOT EXISTS agent.sessions (
  id text PRIMARY KEY,
  user_id text NOT NULL,
  institute_id text NOT NULL,
  role text NOT NULL CHECK (role IN ('admin', 'hod', 'teacher')),
  title text NOT NULL,
  title_source text NOT NULL DEFAULT 'auto' CHECK (title_source IN ('auto', 'user')),
  pinned boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now(),
  last_message_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS sessions_owner ON agent.sessions (user_id, institute_id, last_message_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS agent.messages (
  session_id text NOT NULL REFERENCES agent.sessions(id) ON DELETE CASCADE,
  seq integer NOT NULL CHECK (seq > 0),
  user_id text NOT NULL,
  institute_id text NOT NULL,
  role text NOT NULL CHECK (role IN ('user', 'assistant')),
  text text,
  events jsonb,
  llm jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS messages_owner ON agent.messages (user_id, institute_id, session_id, seq);

CREATE TABLE IF NOT EXISTS agent.queries (
  id text PRIMARY KEY,
  user_id text NOT NULL,
  institute_id text NOT NULL,
  norm text NOT NULL,
  text text NOT NULL,
  last_used_at timestamptz NOT NULL DEFAULT now(),
  use_count integer NOT NULL DEFAULT 1,
  UNIQUE (user_id, institute_id, norm)
);

CREATE TABLE IF NOT EXISTS agent.feedback (
  user_id text NOT NULL,
  institute_id text NOT NULL,
  session_id text NOT NULL REFERENCES agent.sessions(id) ON DELETE CASCADE,
  message_seq integer NOT NULL,
  rating text NOT NULL CHECK (rating IN ('up', 'down')),
  note text,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, institute_id, session_id, message_seq)
);

-- voice usage (voice upgrade phase 3): the count that limits a user's voice questions, per IST day and per month.
-- Written by the backend only (service role / agent store role); no policies -> no client access.
create table if not exists agent.voice_usage (
  institute_id text not null,
  user_id text not null,
  period text not null,                         -- 'YYYY-MM-DD' (day, IST) or 'YYYY-MM' (month)
  count integer not null default 0,
  updated_at timestamptz not null default now(),
  primary key (institute_id, user_id, period)
);
