-- STEP 1c — hide Class 11 Physics textbook back-matter. DO NOT RUN without GO. Requires steps 1 and 1b.
-- "Answers" / "ANSWERS" (262 rows: exercise-answer stubs such as "Question 2.10" + answer text) and "Index"
-- (41 rows: activity / in-text prompts pulled out of their chapters) are back-matter pages of the Physics Part I/II
-- PDFs, not chapters (evidence: scripts/audit/22a-22e). They are HIDDEN, not deleted:
--   quality_status = 'junk'               -> every teacher-facing read already filters it out (_NCERT_VISIBLE_SQL),
--                                            so the builder's chapter list no longer shows "Answers" / "Index"
--   quality_flags.hidden_reason = 'back_matter' -> steps 2-4 skip these rows (never mapped, never deleted, outside the
--                                            unique index), and a later pass can tell them from G4.5 junk
-- The chapter strings stay as they are (after 1b: "Answers", "Index") so the rows remain traceable.
begin;
do $$
begin
  if to_regclass('a4_backup_ncert_dedupe.chapter_alias') is null then
    raise exception 'run step 1b first';
  end if;
end $$;

create table a4_backup_ncert_dedupe.back_matter as
select id, class_grade, subject, chapter, quality_status, quality_flags, duplicate_of
from public.ncert_questions
where class_grade = '11' and subject = 'Physics' and lower(chapter) in ('answers', 'index');
alter table a4_backup_ncert_dedupe.back_matter add primary key (id);

do $$
declare expected bigint := (select count(*) from a4_backup_ncert_dedupe.ncert_questions
                            where class_grade = '11' and subject in ('Physics-I', 'Physics-II', 'Physics')
                              and lower(chapter) in ('answers', 'index'));
begin
  if (select count(*) from a4_backup_ncert_dedupe.back_matter) <> expected then
    raise exception 'back-matter rows: % found, % in the step-1 backup', (select count(*) from a4_backup_ncert_dedupe.back_matter), expected;
  end if;
end $$;

-- duplicate_of must be NULL for a non-'duplicate' status (CHECK ncert_questions_duplicate_of_check); the old value
-- is kept in back_matter and in the step-1 backup.
update public.ncert_questions q
set quality_status = 'junk', duplicate_of = null,
    quality_flags = coalesce(q.quality_flags, '{}'::jsonb) || '{"hidden_reason": "back_matter"}'::jsonb
from a4_backup_ncert_dedupe.back_matter b where b.id = q.id;

do $$
begin
  if (select count(*) from public.ncert_questions where quality_flags->>'hidden_reason' = 'back_matter')
     <> (select count(*) from a4_backup_ncert_dedupe.back_matter) then
    raise exception 'hidden_reason count does not match the back-matter list';
  end if;
  if exists (select 1 from public.ncert_questions where class_grade = '11' and subject = 'Physics'
             and lower(chapter) in ('answers', 'index') and quality_status is distinct from 'junk') then
    raise exception 'a back-matter row is still visible';
  end if;
  if (select count(*) from public.ncert_questions) <> (select count(*) from a4_backup_ncert_dedupe.ncert_questions) then
    raise exception 'row count changed';
  end if;
end $$;
commit;

select chapter, count(*) as hidden_rows,
       (select count(*) from public.ncert_practice_marks m where m.question_id in (select id from a4_backup_ncert_dedupe.back_matter)) as practice_marks_on_them
from a4_backup_ncert_dedupe.back_matter group by chapter order by chapter;
