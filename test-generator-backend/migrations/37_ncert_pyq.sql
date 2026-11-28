-- 37_ncert_pyq.sql — CBSE previous-year questions (PYQs) in the Test Builder library.
--
-- PYQs live in ncert_questions next to the textbook questions, so a PYQ id works everywhere a question id
-- already works (builder paper, /ncert-questions/by-ids, save, export). New columns:
--   source      'ncert' (every existing row) | 'pyq'
--   language    'en' | 'hi' | NULL — NULL = untagged textbook row, read as English
--   pyq_year    year of the board exam, e.g. 2026
--   pyq_exam    e.g. 'CBSE'
--   pyq_papers  every Q.P. code the question appeared in, e.g. {'67/5/1','67/5/2'}
--   pair_id     same value on the English and Hindi version of one PYQ; with language it identifies a PYQ row,
--               so scripts/pyq_load.py can re-run without duplicating rows or changing their ids
--
-- Deploy order does not matter: the backend probes for these columns and serves the old library until they
-- exist (app/services/ncert_library.py). Load PYQ rows only after this migration (scripts/pyq_load.py).
--
-- Additive and idempotent. Rollback: 37_ncert_pyq_rollback.sql (deletes the PYQ rows).

alter table public.ncert_questions
  add column if not exists source     text not null default 'ncert',
  add column if not exists language   text,
  add column if not exists pyq_year   smallint,
  add column if not exists pyq_exam   text,
  add column if not exists pyq_papers text[],
  add column if not exists pair_id    text;

do $$
begin
  if not exists (select 1 from pg_constraint where conname = 'ncert_questions_source_check') then
    alter table public.ncert_questions
      add constraint ncert_questions_source_check check (source in ('ncert', 'pyq'));
  end if;
  if not exists (select 1 from pg_constraint where conname = 'ncert_questions_language_check') then
    alter table public.ncert_questions
      add constraint ncert_questions_language_check check (language is null or language in ('en', 'hi'));
  end if;
  if not exists (select 1 from pg_constraint where conname = 'ncert_questions_pyq_fields_check') then
    alter table public.ncert_questions
      add constraint ncert_questions_pyq_fields_check
      check (source <> 'pyq' or (pyq_year is not null and pair_id is not null and language is not null));
  end if;
end $$;

-- one row per PYQ version; also what the loader matches on
create unique index if not exists ncert_questions_pyq_pair_lang_uq
  on public.ncert_questions (pair_id, language) where source = 'pyq';

-- the builder's PYQ slice: class + subject + source (+ year)
create index if not exists ncert_questions_pyq_lookup_idx
  on public.ncert_questions (class_grade, subject, pyq_year) where source = 'pyq';
