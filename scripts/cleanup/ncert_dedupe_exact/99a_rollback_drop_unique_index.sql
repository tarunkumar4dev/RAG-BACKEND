-- ROLLBACK part 1 (only if step 4 ran). CONCURRENTLY cannot run in a transaction: run this file ALONE.
drop index concurrently if exists public.ncert_questions_chapter_text_uniq;
