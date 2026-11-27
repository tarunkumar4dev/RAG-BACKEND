-- STEP 2 — build the duplicate -> canonical map (writes only to the backup schema). DO NOT RUN without GO.
-- Same grouping and priority as scripts/audit/16d_dedupe_dryrun.sql. Export the map as CSV for review before step 3.
-- Group: same class_grade + subject + chapter + md5(lower(trim(collapse-whitespace(question_text)))).
-- Canonical: 1 most student practice marks, 2 visible (not G4.5 duplicate/junk), 3 lowest quality_rank,
--            4 has image_url, 5 has answer, 6 has options, 7 has question_table, 8 lowest id.
begin;
create table a4_backup_ncert_dedupe.dedupe_map (
  dup_id       bigint primary key,
  canonical_id bigint not null,
  group_size   int    not null,
  class_grade  text   not null,
  subject      text   not null,
  chapter      text   not null,
  dup_marks    int    not null,
  check (dup_id <> canonical_id)
);

insert into a4_backup_ncert_dedupe.dedupe_map
with q as (
  select n.id, n.class_grade, n.subject, n.chapter, n.quality_status, n.quality_rank,
         n.image_url is not null as has_img, coalesce(btrim(n.answer), '') <> '' as has_ans,
         coalesce(n.options #>> '{}', '[]') not in ('[]', '') as has_opts, n.question_table is not null as has_tbl,
         md5(lower(btrim(regexp_replace(n.question_text, '\s+', ' ', 'g')))) as h,
         (select count(*) from public.ncert_practice_marks m where m.question_id = n.id) as marks
  from public.ncert_questions n),
r as (
  select q.*, count(*) over g as group_size,
         first_value(id) over (g order by marks desc,
                                   (quality_status is not null and quality_status in ('duplicate', 'junk')),
                                   coalesce(quality_rank, 0), not has_img, not has_ans, not has_opts, not has_tbl, id) as canonical_id
  from q window g as (partition by class_grade, subject, chapter, h))
select id, canonical_id, group_size, class_grade, subject, chapter, marks from r where id <> canonical_id;

do $$
begin
  if exists (select 1 from a4_backup_ncert_dedupe.dedupe_map m join a4_backup_ncert_dedupe.dedupe_map c on c.dup_id = m.canonical_id) then
    raise exception 'a canonical row is itself marked as a duplicate';
  end if;
end $$;
commit;

select count(*) as rows_to_delete, count(distinct canonical_id) as groups, sum(dup_marks) as marks_to_remap
from a4_backup_ncert_dedupe.dedupe_map;
-- Review export:  npx supabase db query --remote --read-only -o csv
--   "select m.*, left(q.question_text, 120) from a4_backup_ncert_dedupe.dedupe_map m join ncert_questions q on q.id = m.dup_id"
