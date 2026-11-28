-- STEP 4 — only AFTER step 3 is committed and verified. DO NOT RUN without GO.
-- CONCURRENTLY cannot run inside a transaction block: run this file alone (psql / SQL editor without BEGIN).
-- If it fails (a duplicate slipped in), Postgres leaves an INVALID index: drop it, dedupe, retry.
-- Partial: back-matter hidden by step 1c is outside the index (its "Index" pages hold 20 exact copies that are kept,
-- hidden, not deleted). An ingest using ON CONFLICT must repeat this WHERE clause to infer the index.
create unique index concurrently if not exists ncert_questions_chapter_text_uniq
  on public.ncert_questions (class_grade, subject, chapter, md5(lower(btrim(regexp_replace(question_text, '\s+', ' ', 'g')))))
  where (quality_flags->>'hidden_reason') is distinct from 'back_matter';

-- must return valid = true
select indexrelid::regclass as index, indisvalid as valid, pg_get_expr(indpred, indrelid) as predicate
from pg_index where indexrelid = 'public.ncert_questions_chapter_text_uniq'::regclass;
