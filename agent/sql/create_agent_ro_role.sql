-- a4ai Agent: dedicated READ-ONLY database role `agent_ro`.
-- Run ONCE in the Supabase SQL editor (project owner). Re-running is safe (idempotent).
-- Before running: replace the placeholder password below with a long random one (>= 24 chars). Store it only in
-- your local env var AGENT_REAL_DB_URL; never commit it.
--
-- What it allows: SELECT on the listed COLUMNS of 17 tables, nothing else.
-- What it never allows: INSERT/UPDATE/DELETE/TRUNCATE anywhere, auth.*, storage.*, vault, payments/plans/subscriptions,
-- api keys, profiles/teacher_profiles, contests, any other table or column (PII such as student phone/parent phone/
-- email is not granted at column level).
-- Row level security stays ON: the role gets one SELECT policy per table (`agent_ro_read`); the agent backend
-- scopes every query to the caller's institute / department / classes in code (agent/real/schema_adapter.py).
-- The role is NOT bypassrls, NOT superuser; the backend refuses to start if it can write (agent/db_real.py).

begin;

do $$
declare
  pw text := '<SET-A-LONG-RANDOM-PASSWORD>';
begin
  if pw like '<%' or length(pw) < 24 then
    raise exception 'Set a long random password (>= 24 chars) in this script before running it';
  end if;
  if not exists (select 1 from pg_roles where rolname = 'agent_ro') then
    execute format('create role agent_ro login password %L nosuperuser nocreatedb nocreaterole noinherit nobypassrls noreplication connection limit 10', pw);
  else
    execute format('alter role agent_ro login password %L nosuperuser nocreatedb nocreaterole noinherit nobypassrls noreplication connection limit 10', pw);
  end if;
end $$;

alter role agent_ro set default_transaction_read_only = on;
alter role agent_ro set statement_timeout = '8s';
alter role agent_ro set idle_in_transaction_session_timeout = '30s';

-- start from nothing (re-run safe)
revoke all on all tables in schema public from agent_ro;
revoke all on all sequences in schema public from agent_ro;
revoke all on schema public from agent_ro;
do $$
declare s text;
begin
  foreach s in array array['auth', 'storage', 'vault', 'extensions', 'graphql', 'graphql_public', 'realtime',
                            'supabase_functions', 'net', 'cron', 'pgsodium']
  loop
    if exists (select 1 from pg_namespace where nspname = s) then
      execute format('revoke all on all tables in schema %I from agent_ro', s);
      execute format('revoke all on schema %I from agent_ro', s);
    end if;
  end loop;
end $$;

grant usage on schema public to agent_ro;

-- institutes: institute name; owner_id = admin check. NOT join_code, plan, limits, logo.
grant select (id, name, owner_id) on public.institutes to agent_ro;
-- institute_members: role / department / active status of staff, teacher display name. NOT user_email.
grant select (id, institute_id, user_id, role, status, department_id, section_id, user_name) on public.institute_members to agent_ro;
-- departments: HOD scope, names.
grant select (id, institute_id, name, code) on public.departments to agent_ro;
-- sections: proctor scope, section names.
grant select (id, department_id, name, year) on public.sections to agent_ro;
-- batches: section/batch names + department/section links.
grant select (id, institute_id, name, department_id, section_id, is_lab, is_active) on public.batches to agent_ro;
-- students: name + roll no for answers, batch links for scope. NOT phone, parent_name, parent_phone, email, gender, user_id.
grant select (id, institute_id, batch_id, section_id, department_id, lab_batch_id, name, roll_no, is_active) on public.students to agent_ro;
-- subjects: subject names for marks filters.
grant select (id, institute_id, department_id, name, code, is_active) on public.subjects to agent_ro;
-- teaching_assignments: subject-teacher scope.
grant select (id, institute_id, teacher_id, batch_id, subject_id, is_active) on public.teaching_assignments to agent_ro;
-- timetable_slots: subject-teacher scope (timetable).
grant select (id, institute_id, batch_id, subject_id, teacher_id, is_active) on public.timetable_slots to agent_ro;
-- proctor_assignments: proctor scope.
grant select (id, institute_id, teacher_id, section_id, department_id, is_active) on public.proctor_assignments to agent_ro;
-- class_sessions: held classes (status, date, teacher, batch, subject).
grant select (id, institute_id, batch_id, subject_id, teacher_id, session_date, status) on public.class_sessions to agent_ro;
-- attendance_records: attendance status per student.
grant select (id, institute_id, session_id, student_id, status) on public.attendance_records to agent_ro;
-- assignments: marks source 1 (title, max marks, batch, creator).
grant select (id, institute_id, batch_id, created_by, title, deadline, max_marks, status, created_at) on public.assignments to agent_ro;
-- submissions: marks source 1 (grade). NOT files, text answers, feedback.
grant select (id, assignment_id, student_id, grade, status, graded_at) on public.submissions to agent_ro;
-- check_sessions: marks source 2 (check copies).
grant select (id, teacher_id, title, test_id, batch_id, subject, total_marks, created_at) on public.check_sessions to agent_ro;
-- check_sheets: marks source 2 (final marks). NOT pages, AI output, student_name.
grant select (id, session_id, student_id, final_marks, max_marks, status) on public.check_sheets to agent_ro;
-- tests: tests created by a teacher (title, class, subject, marks, status). NOT request_payload.
grant select (id, teacher_id, exam_title, class_grade, subject, total_marks, total_questions, status, created_at) on public.tests to agent_ro;

-- RLS: one read policy per granted table, for agent_ro only (other roles unaffected)
drop policy if exists agent_ro_read on public.institutes;
create policy agent_ro_read on public.institutes for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.institute_members;
create policy agent_ro_read on public.institute_members for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.departments;
create policy agent_ro_read on public.departments for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.sections;
create policy agent_ro_read on public.sections for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.batches;
create policy agent_ro_read on public.batches for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.students;
create policy agent_ro_read on public.students for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.subjects;
create policy agent_ro_read on public.subjects for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.teaching_assignments;
create policy agent_ro_read on public.teaching_assignments for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.timetable_slots;
create policy agent_ro_read on public.timetable_slots for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.proctor_assignments;
create policy agent_ro_read on public.proctor_assignments for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.class_sessions;
create policy agent_ro_read on public.class_sessions for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.attendance_records;
create policy agent_ro_read on public.attendance_records for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.assignments;
create policy agent_ro_read on public.assignments for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.submissions;
create policy agent_ro_read on public.submissions for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.check_sessions;
create policy agent_ro_read on public.check_sessions for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.check_sheets;
create policy agent_ro_read on public.check_sheets for select to agent_ro using (true);
drop policy if exists agent_ro_read on public.tests;
create policy agent_ro_read on public.tests for select to agent_ro using (true);

commit;

-- ─── VERIFY (run after the script; every query must return 0 rows) ───────────────────────────────
-- 1) no write privilege anywhere
select n.nspname, c.relname, p.priv
from pg_class c join pg_namespace n on n.oid = c.relnamespace
cross join (values ('INSERT'), ('UPDATE'), ('DELETE'), ('TRUNCATE')) p(priv)
where c.relkind in ('r','p','v','m','f') and n.nspname not in ('pg_catalog','information_schema')
  and has_table_privilege('agent_ro', c.oid, p.priv);
-- 2) nothing readable in auth / storage / vault
select n.nspname, c.relname from pg_class c join pg_namespace n on n.oid = c.relnamespace
where n.nspname in ('auth','storage','vault') and c.relkind in ('r','p','v','m')
  and has_table_privilege('agent_ro', c.oid, 'SELECT');
-- 3) no PII columns
select table_name, column_name from information_schema.column_privileges
where grantee = 'agent_ro' and column_name in ('phone','parent_phone','parent_name','email','user_email','gender',
  'join_code','request_payload','text_answer','file_url','pages','ai_result','student_name');
-- 4) role flags
select rolname from pg_roles where rolname = 'agent_ro' and (rolsuper or rolcreaterole or rolcreatedb or rolbypassrls);
-- 5) (should return 17) tables with at least one granted column
select count(distinct table_name) from information_schema.column_privileges where grantee = 'agent_ro';
