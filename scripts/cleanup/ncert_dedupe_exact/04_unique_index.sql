-- STEP 4 — only AFTER step 3 is committed and verified. DO NOT RUN without GO.
-- CONCURRENTLY cannot run inside a transaction block: run this file alone (psql / SQL editor without BEGIN).
-- If it fails (a duplicate slipped in), Postgres leaves an INVALID index: drop it, dedupe, retry.
create unique index concurrently if not exists ncert_questions_chapter_text_uniq
  on public.ncert_questions (class_grade, subject, chapter, md5(lower(btrim(regexp_replace(question_text, '\s+', ' ', 'g')))));

-- must return valid = true
select indexrelid::regclass as index, indisvalid as valid
from pg_index where indexrelid = 'public.ncert_questions_chapter_text_uniq'::regclass;
