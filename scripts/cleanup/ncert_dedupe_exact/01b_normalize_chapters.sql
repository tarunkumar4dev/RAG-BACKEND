-- STEP 1b — normalise subject / chapter strings BEFORE the duplicate map is built. DO NOT RUN without GO.
-- Requires step 1 (backup). Rules (evidence: scripts/audit/19a_chapter_alias_overlap.sql, 19b_history_section_content.sql):
--   subject family: Physics-I/II -> Physics, Chemistry-I/II -> Chemistry, Political_Science -> Political Science
--   chapter: case/punctuation-insensitive match inside class + family, plus the explicit renames below; the kept spelling
--            is the explicit name, else the most-used raw spelling of the target chapter.
-- Rewrites ncert_questions.subject/chapter and the class/subject/chapter copies in ncert_practice_marks.
-- ncert_chunks is NOT touched (RAG + /chapters keep their own names until the chapters table exists).
-- Same rules as the read-only dry run scripts/audit/16g_dedupe_dryrun_normalised.sql.
begin;
create table a4_backup_ncert_dedupe.chapter_alias as
with ov(cls, fam, from_norm, to_norm, to_name) as (values
  ('12', 'Physics',   'alternatingcurrent',               'alternatingcurrentac', null),
  ('12', 'Physics',   'currentandelectricity',            'currentelectricity', null),
  ('12', 'Physics',   'electromagneticinduction',         'electromagneticinductionemi', null),
  ('12', 'Physics',   'electromagneticwave',              'electromagneticwavesemwaves', null),
  ('12', 'Physics',   'electrostaticpotential',           'electrostaticpotentialandcapacitance', null),
  ('12', 'Physics',   'rayoptics',                        'rayopticsandopticalinstruments', null),
  ('12', 'Physics',   'semiconductors',                   'semiconductorelectronicsmaterialsdevicesandsimplecircuits', null),
  ('12', 'Chemistry', 'chemicalcoordination',             'coordinationcompounds', null),
  ('10', 'History',   'livelihoodseconomiesandsocieties', 'themakingofaglobalworld', 'The Making of a Global World'),
  ('10', 'History',   'everydaylifecultureandpolitics',   'printcultureandthemodernworld', 'Print Culture and the Modern World')),
raw as (
  select class_grade, subject, chapter, count(*) as n,
         case when subject ~ '^Political' then 'Political Science' else regexp_replace(subject, '-I+$', '') end as fam,
         regexp_replace(lower(chapter), '[^a-z0-9]', '', 'g') as norm
  from public.ncert_questions group by 1, 2, 3),
q as (select raw.*, coalesce(ov.to_norm, raw.norm) as norm2, ov.to_name
      from raw left join ov on ov.cls = raw.class_grade and ov.fam = raw.fam and ov.from_norm = raw.norm),
names as (select distinct on (class_grade, fam, norm2) class_grade, fam, norm2, coalesce(to_name, chapter) as new_chapter
          from q order by class_grade, fam, norm2, (to_name is not null) desc, (norm = norm2) desc, n desc, chapter)
select q.class_grade, q.subject as old_subject, q.chapter as old_chapter, q.fam as new_subject, nm.new_chapter, q.n as rows
from q join names nm using (class_grade, fam, norm2);

alter table a4_backup_ncert_dedupe.chapter_alias add primary key (class_grade, old_subject, old_chapter);

update public.ncert_questions q set subject = a.new_subject, chapter = a.new_chapter
from a4_backup_ncert_dedupe.chapter_alias a
where q.class_grade = a.class_grade and q.subject = a.old_subject and q.chapter = a.old_chapter
  and (a.old_subject, a.old_chapter) is distinct from (a.new_subject, a.new_chapter);

update public.ncert_practice_marks m set subject = a.new_subject, chapter = a.new_chapter
from a4_backup_ncert_dedupe.chapter_alias a
where m.class_grade = a.class_grade and m.subject = a.old_subject and m.chapter = a.old_chapter
  and (a.old_subject, a.old_chapter) is distinct from (a.new_subject, a.new_chapter);

do $$
begin
  if (select count(*) from public.ncert_questions) <> (select count(*) from a4_backup_ncert_dedupe.ncert_questions) then
    raise exception 'row count changed during normalisation';
  end if;
  if exists (select 1 from public.ncert_questions where subject ~ '(-I+|_)') then
    raise exception 'subject variants remain';
  end if;
end $$;
commit;

select count(*) filter (where (old_subject, old_chapter) is distinct from (new_subject, new_chapter)) as chapter_strings_renamed,
       sum(rows) filter (where (old_subject, old_chapter) is distinct from (new_subject, new_chapter)) as rows_rewritten,
       count(distinct (class_grade, new_subject, new_chapter)) as chapters_after
from a4_backup_ncert_dedupe.chapter_alias;
