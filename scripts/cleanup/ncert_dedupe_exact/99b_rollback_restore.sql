-- ROLLBACK part 2 of steps 3-4 from the step-1 backup (run 99a first if step 4 ran). ids are preserved.
-- Before step 3 commits: just ROLLBACK the transaction instead.

begin;
insert into public.ncert_questions
select b.* from a4_backup_ncert_dedupe.ncert_questions b
where not exists (select 1 from public.ncert_questions q where q.id = b.id);

update public.ncert_questions q set duplicate_of = b.duplicate_of          -- undo step 3.2
from a4_backup_ncert_dedupe.ncert_questions b where b.id = q.id and q.duplicate_of is distinct from b.duplicate_of;

delete from public.ncert_practice_marks;                                     -- undo step 3.1 (exact copy)
insert into public.ncert_practice_marks select * from a4_backup_ncert_dedupe.ncert_practice_marks;
-- marks students recorded AFTER step 1 are lost by this line: compare counts first and merge them by hand if any.

insert into public.ncert_question_fixes                                     -- undo step 3.3
select b.* from a4_backup_ncert_dedupe.ncert_question_fixes b
where not exists (select 1 from public.ncert_question_fixes f where f.id = b.id);

drop table if exists public.ncert_question_redirects;                        -- undo step 3.4
commit;

select (select count(*) from public.ncert_questions) as now_rows,
       (select n from a4_backup_ncert_dedupe.fingerprint where tbl = 'ncert_questions') as backup_rows;
