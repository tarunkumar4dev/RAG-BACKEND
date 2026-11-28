-- 37_ncert_pyq_rollback.sql — undo 37_ncert_pyq.sql.
--
-- DELETES every PYQ row. Builder papers saved in teachers' browsers keep only question ids; PYQ ids in them will
-- silently drop out of the restored paper (by-ids returns only rows that still exist).

delete from public.ncert_questions where source = 'pyq';

drop index if exists public.ncert_questions_pyq_lookup_idx;
drop index if exists public.ncert_questions_pyq_pair_lang_uq;

alter table public.ncert_questions
  drop constraint if exists ncert_questions_pyq_fields_check,
  drop constraint if exists ncert_questions_language_check,
  drop constraint if exists ncert_questions_source_check;

alter table public.ncert_questions
  drop column if exists pair_id,
  drop column if exists pyq_papers,
  drop column if exists pyq_exam,
  drop column if exists pyq_year,
  drop column if exists language,
  drop column if exists source;
