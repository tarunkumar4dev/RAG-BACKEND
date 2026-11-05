-- a4ai Agent demo schema (LOCAL Postgres only, never production).
drop schema if exists agent_demo cascade;
create schema agent_demo;
set search_path = agent_demo;

create table institutes (id uuid primary key, name text not null);

create table institute_members (
  id uuid primary key,
  institute_id uuid not null references institutes(id),
  user_id uuid not null,
  role text not null check (role in ('admin','teacher')),
  status text not null default 'active',
  user_name text not null
);

create table subjects (id uuid primary key, institute_id uuid not null references institutes(id), name text not null);

create table batches (
  id uuid primary key,
  institute_id uuid not null references institutes(id),
  name text not null,
  class_level text not null
);

create table teaching_assignments (
  id uuid primary key,
  institute_id uuid not null references institutes(id),
  teacher_id uuid not null,            -- auth user_id
  batch_id uuid not null references batches(id),
  subject_id uuid not null references subjects(id),
  is_active boolean not null default true
);

create table students (
  id uuid primary key,
  institute_id uuid not null references institutes(id),
  batch_id uuid not null references batches(id),
  name text not null,
  roll_no text,
  is_active boolean not null default true
);

create table class_sessions (
  id uuid primary key,
  institute_id uuid not null references institutes(id),
  batch_id uuid not null references batches(id),
  subject_id uuid not null references subjects(id),
  teacher_id uuid not null,
  session_date date not null,
  status text not null default 'held'
);

create table attendance_records (
  id uuid primary key,
  institute_id uuid not null references institutes(id),
  session_id uuid not null references class_sessions(id) on delete cascade,
  student_id uuid not null references students(id),
  status text not null check (status in ('present','absent','late','leave')),
  marked_by uuid,
  marked_at timestamptz,
  unique (session_id, student_id)
);

create table tests (
  id uuid primary key,
  institute_id uuid not null references institutes(id),
  batch_id uuid not null references batches(id),
  subject_id uuid not null references subjects(id),
  teacher_id uuid not null,
  name text not null,
  test_date date not null,
  max_marks numeric not null
);

create table test_marks (
  id uuid primary key,
  test_id uuid not null references tests(id) on delete cascade,
  student_id uuid not null references students(id),
  marks_obtained numeric null,
  unique (test_id, student_id)
);

-- agent tables
create table agent_sessions (
  id uuid primary key,
  user_id uuid not null,
  institute_id uuid not null,
  role text not null,
  created_at timestamptz not null default now()
);

create table agent_messages (
  id bigserial primary key,
  session_id uuid not null references agent_sessions(id) on delete cascade,
  seq int not null,
  role text not null,
  content jsonb not null,
  created_at timestamptz not null default now()
);

create table agent_approvals (
  id uuid primary key,
  session_id uuid not null references agent_sessions(id) on delete cascade,
  user_id uuid not null,
  action text not null,
  args_sha256 text not null,
  args jsonb not null,
  expires_at timestamptz not null,
  used_at timestamptz,
  result jsonb
);

create table agent_drafts (
  id uuid primary key,
  session_id uuid references agent_sessions(id) on delete cascade,
  institute_id uuid not null,
  user_id uuid not null,
  paper jsonb not null,
  status text not null default 'draft',
  created_at timestamptz not null default now()
);

create index on attendance_records (student_id);
create index on attendance_records (session_id);
create index on class_sessions (batch_id, session_date);
create index on students (institute_id, batch_id);

-- read-only role for insight tools
do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'agent_ro') then
    create role agent_ro login;
  end if;
end $$;
grant usage on schema agent_demo to agent_ro;
grant select on all tables in schema agent_demo to agent_ro;
alter role agent_ro set default_transaction_read_only = on;
