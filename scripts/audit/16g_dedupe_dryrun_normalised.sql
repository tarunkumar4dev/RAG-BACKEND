-- Read-only DRY RUN v2: the exact-duplicate cleanup AFTER chapter-name normalisation (nothing is changed).
-- Normalisation (same rules as scripts/cleanup/ncert_dedupe_exact/01b_normalize_chapters.sql):
--   subject family: Physics-I/II -> Physics, Chemistry-I/II -> Chemistry, Political_Science -> Political Science
--   chapter: compared case/punctuation-insensitive inside class + family, plus the explicit renames in `ov`
--            (evidence: scripts/audit/19a_chapter_alias_overlap.sql, 19b_history_section_content.sql)
-- Cross-chapter rule: Class 10 English poems printed inside a prose chapter's PDF were extracted twice; a text found
-- in both the prose chapter and its paired poem keeps only the poem-chapter copy.
with ov(cls, fam, from_norm, to_norm, to_name) as (values
  ('12', 'Physics',   'alternatingcurrent',              'alternatingcurrentac', null),
  ('12', 'Physics',   'currentandelectricity',           'currentelectricity', null),
  ('12', 'Physics',   'electromagneticinduction',        'electromagneticinductionemi', null),
  ('12', 'Physics',   'electromagneticwave',             'electromagneticwavesemwaves', null),
  ('12', 'Physics',   'electrostaticpotential',          'electrostaticpotentialandcapacitance', null),
  ('12', 'Physics',   'rayoptics',                       'rayopticsandopticalinstruments', null),
  ('12', 'Physics',   'semiconductors',                  'semiconductorelectronicsmaterialsdevicesandsimplecircuits', null),
  ('12', 'Chemistry', 'chemicalcoordination',            'coordinationcompounds', null),
  ('10', 'History',   'livelihoodseconomiesandsocieties', 'themakingofaglobalworld', 'The Making of a Global World'),
  ('10', 'History',   'everydaylifecultureandpolitics',  'printcultureandthemodernworld', 'Print Culture and the Modern World')),
poem(cls, prose, poem) as (values
  ('10', 'A Letter to God', 'Fire and Ice'), ('10', 'Nelson Mandela - Long Walk to Freedom', 'A Tiger in the Zoo'),
  ('10', 'Stories About Flying', 'How to Tell Wild Animals'), ('10', 'Stories About Flying', 'The Ball Poem'),
  ('10', 'From the Diary of Anne Frank', 'Amanda'), ('10', 'Glimpses of India', 'The Trees'),
  ('10', 'Mijbil the Otter', 'Fog'), ('10', 'Madam Rides the Bus', 'The Tale of Custard Dragon'),
  ('10', 'The Sermon at Benares', 'For Anne Gregory')),
b as (
  select n.id, n.class_grade, n.subject, n.chapter, n.quality_status, n.quality_rank,
         case when n.subject ~ '^Political' then 'Political Science' else regexp_replace(n.subject, '-I+$', '') end as fam,
         regexp_replace(lower(n.chapter), '[^a-z0-9]', '', 'g') as norm,
         n.image_url is not null as has_img, coalesce(btrim(n.answer), '') <> '' as has_ans,
         coalesce(n.options #>> '{}', '[]') not in ('[]', '') as has_opts, n.question_table is not null as has_tbl,
         md5(lower(btrim(regexp_replace(n.question_text, '\s+', ' ', 'g')))) as h,
         (select count(*) from ncert_practice_marks m where m.question_id = n.id) as marks
  from ncert_questions n),
q as (select b.*, coalesce(ov.to_norm, b.norm) as norm2, ov.to_name
      from b left join ov on ov.cls = b.class_grade and ov.fam = b.fam and ov.from_norm = b.norm),
-- canonical chapter name per (class, family, norm2): explicit name, else the most-used raw spelling of the target
names as (select distinct on (class_grade, fam, norm2) class_grade, fam, norm2, coalesce(to_name, chapter) as new_chapter
          from (select class_grade, fam, norm2, to_name, chapter, norm, count(*) n from q group by 1,2,3,4,5,6) x
          order by class_grade, fam, norm2, (to_name is not null) desc, (norm = norm2) desc, n desc, chapter),
k as (select q.*, nm.new_chapter from q join names nm using (class_grade, fam, norm2)),
r as (select k.*, count(*) over g as group_size,
             first_value(id) over (g order by marks desc, (quality_status is not null and quality_status in ('duplicate', 'junk')),
                                       coalesce(quality_rank, 0), not has_img, not has_ans, not has_opts, not has_tbl, id) as canonical_id
      from k window g as (partition by class_grade, fam, new_chapter, h)),
-- English: canonical rows of a prose chapter whose text also exists in its paired poem chapter
cross_ch as (select pr.id as prose_canonical, po.canonical_id as poem_canonical
             from r pr join poem p on p.cls = pr.class_grade and p.prose = pr.new_chapter
             join r po on po.class_grade = pr.class_grade and po.fam = pr.fam and po.new_chapter = p.poem and po.h = pr.h and po.id = po.canonical_id
             where pr.fam = 'English' and pr.id = pr.canonical_id),
final as (select r.id, coalesce(c.poem_canonical, r.canonical_id) as canonical_id
          from r left join cross_ch c on c.prose_canonical = r.canonical_id),
base as (select count(*) as n from (
           select id, first_value(id) over (partition by class_grade, subject, chapter, h order by id) f from b) x where id <> f)
select 1 as ord, 'rows deleted, raw chapter key (previous plan)' as metric, (select n from base)::text as value
union all select 2, 'rows deleted, normalised key + English poem rule', count(*)::text from final where id <> canonical_id
union all select 3, '  of which: extra from subject/chapter normalisation', ((select count(*) from r where id <> canonical_id) - (select n from base))::text
union all select 4, '  of which: extra from the English poem rule', (select count(*) from cross_ch)::text
union all select 5, 'rows kept', count(*) filter (where id = canonical_id)::text from final
union all select 6, 'chapter strings before -> after normalisation', (select count(distinct (class_grade, subject, chapter)) from b)::text || ' -> ' || (select count(distinct (class_grade, fam, new_chapter)) from k)::text
union all select 7, 'rows whose subject and/or chapter string is rewritten', count(*) filter (where subject <> fam or chapter <> new_chapter)::text from k
union all select 8, 'practice marks on deleted rows (remap) / students', (select count(*) || ' / ' || count(distinct m.user_id) from ncert_practice_marks m join final f on f.id = m.question_id where f.id <> f.canonical_id)
union all select 9, 'practice marks whose class/subject/chapter copy is rewritten', (select count(*) from ncert_practice_marks m join k on k.id = m.question_id where k.subject <> k.fam or k.chapter <> k.new_chapter)::text
union all select 10, 'groups keeping a hidden row although a visible copy exists (must be 0)', count(*)::text from (
    select 1 from r where group_size > 1 group by class_grade, fam, new_chapter, h
    having bool_or(id = canonical_id and quality_status in ('duplicate', 'junk')) and bool_or(quality_status is null or quality_status not in ('duplicate', 'junk'))) x
union all select 11, 'ncert_question_fixes rows on deleted rows', (select count(*) from ncert_question_fixes fx join final f on f.id = fx.question_id where f.id <> f.canonical_id)::text
union all select 12, 'unique key collisions left after cleanup (must be 0)', count(*)::text from (
    select 1 from k join final f on f.id = k.id where f.id = f.canonical_id group by k.class_grade, k.fam, k.new_chapter, k.h having count(*) > 1) x
order by ord;
