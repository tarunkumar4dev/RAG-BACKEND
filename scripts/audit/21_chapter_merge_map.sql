-- Read-only: the full step-1b mapping (same rules as scripts/cleanup/ncert_dedupe_exact/01b_normalize_chapters.sql)
-- plus the 9 Class 10 English prose -> poem pairs of step 2. Export with -o csv to docs/audits/samples/chapter_merge_map.csv.
-- rule:  unchanged | subject_family (only the subject string changes) | spelling (case/punctuation variant merged)
--        | explicit_merge (different name, merged on shared-text evidence) | explicit_rename (new name, nothing merged)
--        | english_poem_pair (rows move from the prose chapter to the poem; chapter names unchanged)
--        | back_matter_hidden (Class 11 Physics Answers/Index: subject folded by 1b, rows hidden by 1c, never deleted,
--          not a chapter teachers can pick)
-- note:  evidence for the non-obvious rows (scripts/audit/22a-22e back-matter, 23a-23d Chemical Coordination)
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
poem(cls, prose, poem) as (values
  ('10', 'A Letter to God', 'Fire and Ice'), ('10', 'Nelson Mandela - Long Walk to Freedom', 'A Tiger in the Zoo'),
  ('10', 'Stories About Flying', 'How to Tell Wild Animals'), ('10', 'Stories About Flying', 'The Ball Poem'),
  ('10', 'From the Diary of Anne Frank', 'Amanda'), ('10', 'Glimpses of India', 'The Trees'),
  ('10', 'Mijbil the Otter', 'Fog'), ('10', 'Madam Rides the Bus', 'The Tale of Custard Dragon'),
  ('10', 'The Sermon at Benares', 'For Anne Gregory')),
raw as (
  select class_grade, subject, chapter, count(*) as n,
         case when subject ~ '^Political' then 'Political Science' else regexp_replace(subject, '-I+$', '') end as fam,
         regexp_replace(lower(chapter), '[^a-z0-9]', '', 'g') as norm
  from ncert_questions group by 1, 2, 3),
q as (select raw.*, coalesce(ov.to_norm, raw.norm) as norm2, ov.to_name, ov.from_norm is not null as explicit
      from raw left join ov on ov.cls = raw.class_grade and ov.fam = raw.fam and ov.from_norm = raw.norm),
names as (select distinct on (class_grade, fam, norm2) class_grade, fam, norm2, coalesce(to_name, chapter) as new_chapter
          from q order by class_grade, fam, norm2, (to_name is not null) desc, (norm = norm2) desc, n desc, chapter),
m as (select q.*, nm.new_chapter from q join names nm using (class_grade, fam, norm2)),
k as (select n.id, n.class_grade, n.subject, n.chapter, md5(lower(btrim(regexp_replace(n.question_text, '\s+', ' ', 'g')))) as h
      from ncert_questions n where n.class_grade = '10' and n.subject = 'English')
select class_grade, subject as old_subject, chapter as old_chapter, n as rows, fam as new_subject, new_chapter,
       case when class_grade = '11' and subject ~ '^Physics' and chapter ~* '^(answers|index)$' then 'back_matter_hidden'
            when explicit and to_name is not null then 'explicit_rename'
            when explicit then 'explicit_merge'
            when subject = fam and chapter = new_chapter then 'unchanged'
            when chapter = new_chapter then 'subject_family'
            else 'spelling' end as rule,
       null::bigint as rows_moved_to_poem,
       case when class_grade = '11' and subject ~ '^Physics' and chapter ~* '^answers$'
              then 'textbook answer pages (exercise-answer stubs); quality_status=junk, hidden_reason=back_matter; kept, not deleted'
            when class_grade = '11' and subject ~ '^Physics' and chapter ~* '^index$'
              then 'activity/in-text prompts extracted from the back pages; quality_status=junk, hidden_reason=back_matter; kept, not deleted'
            when class_grade = '12' and norm = 'chemicalcoordination'
              then 'all 46 rows are coordination chemistry (Unit 5, sections 5.1-5.5): 25 of 36 texts also in Coordination Compounds, 0 in Class 11 Biology Chemical Coordination and Integration'
       end as note
from m
union all
select p.cls, 'English', p.prose, (select count(*) from k where k.chapter = p.prose), 'English', p.poem, 'english_poem_pair',
       (select count(*) from k pr where pr.chapter = p.prose and exists (select 1 from k po where po.chapter = p.poem and po.h = pr.h)),
       'poem printed inside the prose chapter PDF; shared texts keep only the poem copy'
from poem p
order by 1, 5, 6, 7, 2, 3;
