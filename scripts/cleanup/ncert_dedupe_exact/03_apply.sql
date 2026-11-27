-- STEP 3 — remap student records, then delete the duplicates. ONE transaction. DO NOT RUN without GO.
-- Requires steps 1 and 2 (backup + reviewed dedupe_map) from the same day, and migration 41.
-- Every check raises an exception, which rolls the whole transaction back.
begin;
set local lock_timeout = '5s';
set local statement_timeout = '5min';
lock table public.ncert_questions, public.ncert_practice_marks, public.ncert_question_fixes in share row exclusive mode;

-- 0. The map must still describe the live table (nothing inserted/edited since step 2).
do $$
begin
  if exists (select 1 from a4_backup_ncert_dedupe.dedupe_map m left join public.ncert_questions q on q.id = m.dup_id where q.id is null)
     or exists (select 1 from a4_backup_ncert_dedupe.dedupe_map m left join public.ncert_questions q on q.id = m.canonical_id where q.id is null) then
    raise exception 'dedupe_map is stale: a mapped row no longer exists';
  end if;
  if (select count(*) from public.ncert_practice_marks) <> (select count(*) from a4_backup_ncert_dedupe.ncert_practice_marks) then
    raise exception 'practice marks changed since the backup: redo steps 1-2';
  end if;
end $$;

-- 1. Student practice marks: one mark per (student, canonical question); the newest one wins
--    (record_practice_mark upserts with updated_at, so the newest is the student's latest intent).
with target as (
  select m.user_id, m.question_id, m.updated_at, coalesce(d.canonical_id, m.question_id) as canonical_id
  from public.ncert_practice_marks m left join a4_backup_ncert_dedupe.dedupe_map d on d.dup_id = m.question_id),
ranked as (
  select *, row_number() over (partition by user_id, canonical_id order by updated_at desc, question_id) as rn from target)
delete from public.ncert_practice_marks m
using ranked r where r.rn > 1 and m.user_id = r.user_id and m.question_id = r.question_id;

update public.ncert_practice_marks m set question_id = d.canonical_id
from a4_backup_ncert_dedupe.dedupe_map d where m.question_id = d.dup_id;

-- a mark moved to a canonical in another chapter (English poem rule) carries that chapter's names
update public.ncert_practice_marks m set class_grade = q.class_grade, subject = q.subject, chapter = q.chapter
from public.ncert_questions q
where q.id = m.question_id and (m.class_grade, m.subject, m.chapter) is distinct from (q.class_grade, q.subject, q.chapter);

-- 2. G4.5 pointers: anything pointing at a row about to go points at its canonical instead.
update public.ncert_questions q set duplicate_of = d.canonical_id
from a4_backup_ncert_dedupe.dedupe_map d
where q.duplicate_of = d.dup_id and q.id not in (select dup_id from a4_backup_ncert_dedupe.dedupe_map);

-- 3. Fix log rows of deleted copies: kept in a4_backup_ncert_dedupe.ncert_question_fixes (step 1); removed explicitly
--    here instead of silently by the ON DELETE CASCADE.
delete from public.ncert_question_fixes f using a4_backup_ncert_dedupe.dedupe_map d where f.question_id = d.dup_id;

-- 4. Redirects, so a paper saved in a teacher's browser (localStorage ids) can still be restored.
create table if not exists public.ncert_question_redirects (
  old_id bigint primary key,
  new_id bigint not null references public.ncert_questions(id) on delete cascade
);
revoke all on public.ncert_question_redirects from anon, authenticated;   -- default privileges would grant ALL
grant select on public.ncert_question_redirects to anon, authenticated;
alter table public.ncert_question_redirects enable row level security;
drop policy if exists ncert_question_redirects_select on public.ncert_question_redirects;
create policy ncert_question_redirects_select on public.ncert_question_redirects for select using (true);
insert into public.ncert_question_redirects (old_id, new_id)
select dup_id, canonical_id from a4_backup_ncert_dedupe.dedupe_map;

-- 5. Nothing may still point at a doomed row, then delete.
do $$
begin
  if exists (select 1 from public.ncert_practice_marks m join a4_backup_ncert_dedupe.dedupe_map d on d.dup_id = m.question_id) then
    raise exception 'practice marks still reference rows to delete';
  end if;
end $$;

create temp table _before on commit drop as select count(*) as marks from public.ncert_practice_marks;

delete from public.ncert_questions q using a4_backup_ncert_dedupe.dedupe_map d where q.id = d.dup_id;

do $$
declare expected bigint := (select count(*) from a4_backup_ncert_dedupe.ncert_questions) - (select count(*) from a4_backup_ncert_dedupe.dedupe_map);
begin
  if (select count(*) from public.ncert_questions) <> expected then
    raise exception 'ncert_questions has % rows, expected %', (select count(*) from public.ncert_questions), expected;
  end if;
  if (select count(*) from public.ncert_practice_marks) <> (select marks from _before) then
    raise exception 'the delete cascaded into ncert_practice_marks';
  end if;
  if exists (select 1 from public.ncert_questions
             group by class_grade, subject, chapter, md5(lower(btrim(regexp_replace(question_text, '\s+', ' ', 'g'))))
             having count(*) > 1) then
    raise exception 'exact duplicates remain';
  end if;
end $$;

commit;
