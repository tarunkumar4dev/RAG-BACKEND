-- STEP 2 — build the duplicate -> canonical map (writes only to the backup schema). DO NOT RUN without GO.
-- Same grouping and priority as scripts/audit/16d_dedupe_dryrun.sql. Export the map as CSV for review before step 3.
-- Group: same class_grade + subject + chapter + md5(lower(trim(collapse-whitespace(question_text)))).
-- Canonical: 1 most student practice marks, 2 visible (not G4.5 duplicate/junk), 3 lowest quality_rank,
--            4 has image_url, 5 has answer, 6 has options, 7 has question_table, 8 lowest id.
-- Back-matter hidden by step 1c (quality_flags.hidden_reason = 'back_matter') is left out: never mapped, never deleted.
begin;
do $$
begin
  if to_regclass('a4_backup_ncert_dedupe.back_matter') is null then
    raise exception 'run step 1c first';
  end if;
end $$;
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
  from public.ncert_questions n
  where (n.quality_flags->>'hidden_reason') is distinct from 'back_matter'),
r as (
  select q.*, count(*) over g as group_size,
         first_value(id) over (g order by marks desc,
                                   (quality_status is not null and quality_status in ('duplicate', 'junk')),
                                   coalesce(quality_rank, 0), not has_img, not has_ans, not has_opts, not has_tbl, id) as canonical_id
  from q window g as (partition by class_grade, subject, chapter, h))
select id, canonical_id, group_size, class_grade, subject, chapter, marks from r where id <> canonical_id;

-- Cross-chapter rule (Class 10 English): a poem is printed inside the preceding prose chapter's PDF, so its questions
-- were extracted twice. A text present in both the prose chapter and its paired poem keeps only the poem copy.
-- Pairs from scripts/audit/19a_chapter_alias_overlap.sql (83-100% of each poem's texts also sit in the prose chapter).
create temp table _poem(cls text, prose text, poem text) on commit drop;
insert into _poem values
  ('10', 'A Letter to God', 'Fire and Ice'), ('10', 'Nelson Mandela - Long Walk to Freedom', 'A Tiger in the Zoo'),
  ('10', 'Stories About Flying', 'How to Tell Wild Animals'), ('10', 'Stories About Flying', 'The Ball Poem'),
  ('10', 'From the Diary of Anne Frank', 'Amanda'), ('10', 'Glimpses of India', 'The Trees'),
  ('10', 'Mijbil the Otter', 'Fog'), ('10', 'Madam Rides the Bus', 'The Tale of Custard Dragon'),
  ('10', 'The Sermon at Benares', 'For Anne Gregory');

create temp table _cross on commit drop as
with k as (
  select n.id, n.class_grade, n.subject, n.chapter, md5(lower(btrim(regexp_replace(n.question_text, '\s+', ' ', 'g')))) as h
  from public.ncert_questions n
  where n.class_grade = '10' and n.subject = 'English'
    and not exists (select 1 from a4_backup_ncert_dedupe.dedupe_map d where d.dup_id = n.id))   -- canonical rows only
select distinct on (pr.id) pr.id as prose_id, po.id as poem_id
from k pr join _poem p on p.cls = pr.class_grade and p.prose = pr.chapter
join k po on po.class_grade = pr.class_grade and po.chapter = p.poem and po.h = pr.h
order by pr.id, po.id;

-- copies that pointed at a prose row now point at the poem row, then the prose row itself becomes a duplicate
update a4_backup_ncert_dedupe.dedupe_map d set canonical_id = c.poem_id from _cross c where d.canonical_id = c.prose_id;
insert into a4_backup_ncert_dedupe.dedupe_map (dup_id, canonical_id, group_size, class_grade, subject, chapter, dup_marks)
select c.prose_id, c.poem_id, 0, n.class_grade, n.subject, n.chapter,
       (select count(*) from public.ncert_practice_marks m where m.question_id = c.prose_id)
from _cross c join public.ncert_questions n on n.id = c.prose_id;

do $$
begin
  if exists (select 1 from a4_backup_ncert_dedupe.dedupe_map m join a4_backup_ncert_dedupe.dedupe_map c on c.dup_id = m.canonical_id) then
    raise exception 'a canonical row is itself marked as a duplicate';
  end if;
  if exists (select 1 from a4_backup_ncert_dedupe.dedupe_map m join a4_backup_ncert_dedupe.back_matter b
             on b.id in (m.dup_id, m.canonical_id)) then
    raise exception 'a back-matter row is in the map';
  end if;
end $$;
commit;

select count(*) as rows_to_delete, count(distinct canonical_id) as groups, sum(dup_marks) as marks_to_remap
from a4_backup_ncert_dedupe.dedupe_map;
-- Review export:  npx supabase db query --remote --read-only -o csv
--   "select m.*, left(q.question_text, 120) from a4_backup_ncert_dedupe.dedupe_map m join ncert_questions q on q.id = m.dup_id"
