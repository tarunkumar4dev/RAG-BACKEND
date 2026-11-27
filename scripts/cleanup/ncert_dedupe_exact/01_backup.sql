-- STEP 1 — full backup. DO NOT RUN without GO (Tarun + Nitin). Plan: docs/audits/ncert-dedupe-cleanup-plan.md
-- Creates schema a4_backup_ncert_dedupe and copies every table the cleanup touches. Fails if the schema exists
-- (so it can never overwrite an earlier backup). Also take an offline dump of the same tables (see the plan).
begin;
create schema a4_backup_ncert_dedupe;
create table a4_backup_ncert_dedupe.ncert_questions      as table public.ncert_questions;
create table a4_backup_ncert_dedupe.ncert_practice_marks as table public.ncert_practice_marks;
create table a4_backup_ncert_dedupe.ncert_question_fixes as table public.ncert_question_fixes;
revoke all on schema a4_backup_ncert_dedupe from public, anon, authenticated;

-- Fingerprints to compare the copy with the source (and later with a restore).
create table a4_backup_ncert_dedupe.fingerprint as
select 'ncert_questions' as tbl, count(*) as n,
       md5(string_agg(id::text || ':' || md5(coalesce(question_text, '')) || ':' || coalesce(quality_status, ''), ',' order by id)) as fp
from public.ncert_questions
union all
select 'ncert_practice_marks', count(*), md5(coalesce(string_agg(user_id::text || ':' || question_id || ':' || result, ',' order by user_id, question_id), ''))
from public.ncert_practice_marks
union all
select 'ncert_question_fixes', count(*), md5(coalesce(string_agg(id::text || ':' || question_id, ',' order by id), ''))
from public.ncert_question_fixes;

do $$
begin
  if (select count(*) from a4_backup_ncert_dedupe.ncert_questions) <> (select count(*) from public.ncert_questions)
     or (select count(*) from a4_backup_ncert_dedupe.ncert_practice_marks) <> (select count(*) from public.ncert_practice_marks)
     or (select count(*) from a4_backup_ncert_dedupe.ncert_question_fixes) <> (select count(*) from public.ncert_question_fixes) then
    raise exception 'backup row counts differ from the source';
  end if;
end $$;
commit;

select * from a4_backup_ncert_dedupe.fingerprint;
