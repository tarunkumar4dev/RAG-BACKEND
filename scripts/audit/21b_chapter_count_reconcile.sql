-- Read-only: reconcile the audit's "431 real chapters" with the plan's "425 chapters after step 1b".
--   431 = distinct (class, subject family, case/punctuation-normalised chapter), Answers/Index/Appendices EXCLUDED
--   425 = distinct (class, subject family, new chapter name after 01b), Answers/Index INCLUDED as strings
--   423 = chapters teachers can pick after 01c hides the Answers/Index back-matter
-- so 425 = 431 - (explicit merges of differently-named chapters) + (non-chapter strings 01b keeps), and 423 = 431 - merges.
with ov(cls, fam, from_norm, to_norm) as (values
  ('12', 'Physics', 'alternatingcurrent', 'alternatingcurrentac'), ('12', 'Physics', 'currentandelectricity', 'currentelectricity'),
  ('12', 'Physics', 'electromagneticinduction', 'electromagneticinductionemi'), ('12', 'Physics', 'electromagneticwave', 'electromagneticwavesemwaves'),
  ('12', 'Physics', 'electrostaticpotential', 'electrostaticpotentialandcapacitance'), ('12', 'Physics', 'rayoptics', 'rayopticsandopticalinstruments'),
  ('12', 'Physics', 'semiconductors', 'semiconductorelectronicsmaterialsdevicesandsimplecircuits'), ('12', 'Chemistry', 'chemicalcoordination', 'coordinationcompounds'),
  ('10', 'History', 'livelihoodseconomiesandsocieties', 'themakingofaglobalworld'), ('10', 'History', 'everydaylifecultureandpolitics', 'printcultureandthemodernworld')),
c as (select distinct class_grade,
             case when subject ~ '^Political' then 'Political Science' else regexp_replace(subject, '-I+$', '') end as fam,
             regexp_replace(lower(chapter), '[^a-z0-9]', '', 'g') as norm, chapter
      from ncert_questions),
c2 as (select c.*, coalesce(ov.to_norm, c.norm) as norm2, ov.from_norm is not null as explicit,
              exists (select 1 from c t where t.class_grade = c.class_grade and t.fam = c.fam and t.norm = ov.to_norm) as target_exists
       from c left join ov on ov.cls = c.class_grade and ov.fam = c.fam and ov.from_norm = c.norm)
select 1 as ord, 'A  real chapters in the audit (Answers/Index excluded, spelling-normalised)' as step,
       count(distinct (class_grade, fam, norm)) filter (where norm !~ '^(answers|index|appendices)$')::text as value from c2
union all select 2, 'B  explicit merges into an EXISTING differently-named chapter (each removes 1)',
       count(distinct (class_grade, fam, norm)) filter (where explicit and target_exists)::text || ': ' ||
       string_agg(distinct class_grade || ' ' || fam || ': ' || chapter, ' | ') filter (where explicit and target_exists) from c2
union all select 3, 'C  explicit renames with no existing target (count unchanged)',
       count(distinct (class_grade, fam, norm)) filter (where explicit and not target_exists)::text || ': ' ||
       string_agg(distinct class_grade || ' ' || fam || ': ' || chapter, ' | ') filter (where explicit and not target_exists) from c2
union all select 4, 'D  non-chapter strings that 01b keeps and 01c hides (each adds 1 string, 0 pickable chapters)',
       count(distinct (class_grade, fam, norm)) filter (where norm ~ '^(answers|index)$')::text || ': ' ||
       string_agg(distinct class_grade || ' ' || fam || ': ' || chapter, ' | ') filter (where norm ~ '^(answers|index)$') from c2
union all select 5, 'E  chapter strings after 01b (A - B + D)', count(distinct (class_grade, fam, norm2))::text from c2
union all select 6, 'F  chapters teachers can pick after 01c (A - B)',
       count(distinct (class_grade, fam, norm2)) filter (where not (class_grade = '11' and fam = 'Physics' and norm ~ '^(answers|index)$'))::text from c2
order by ord;
