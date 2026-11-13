-- a4ai Agent: remove the read-only role `agent_ro` (undo create_agent_ro_role.sql). Safe to re-run.
begin;
drop policy if exists agent_ro_read on public.institutes;
drop policy if exists agent_ro_read on public.institute_members;
drop policy if exists agent_ro_read on public.departments;
drop policy if exists agent_ro_read on public.sections;
drop policy if exists agent_ro_read on public.batches;
drop policy if exists agent_ro_read on public.students;
drop policy if exists agent_ro_read on public.subjects;
drop policy if exists agent_ro_read on public.teaching_assignments;
drop policy if exists agent_ro_read on public.timetable_slots;
drop policy if exists agent_ro_read on public.proctor_assignments;
drop policy if exists agent_ro_read on public.class_sessions;
drop policy if exists agent_ro_read on public.attendance_records;
drop policy if exists agent_ro_read on public.assignments;
drop policy if exists agent_ro_read on public.submissions;
drop policy if exists agent_ro_read on public.check_sessions;
drop policy if exists agent_ro_read on public.check_sheets;
drop policy if exists agent_ro_read on public.tests;
do $$
begin
  if exists (select 1 from pg_roles where rolname = 'agent_ro') then
    execute 'revoke all on all tables in schema public from agent_ro';
    execute 'revoke all on schema public from agent_ro';
    execute 'drop owned by agent_ro';
    execute 'drop role agent_ro';
  end if;
end $$;
commit;
-- verify: 0 rows
select rolname from pg_roles where rolname = 'agent_ro';
