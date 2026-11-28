-- Read-only: every Class 12 "Chemical Coordination" row classified by vocabulary (biology endocrine terms vs
-- coordination-chemistry terms) to rule out Class 11 Biology "Chemical Coordination and Integration".
select case
         when question_text ~* '(hormone|gland|endocrine|insulin|thyroid|pituitary|adrenal|hypothalam|pancrea)' then 'biology_endocrine'
         when question_text ~* '(complex|ligand|coordinat|isomer|crystal field|valence bond|\[[A-Z][a-z]?\(|chelat|werner|carbonyl|magnetic moment|oxidation state|iupac)' then 'coordination_chemistry'
         else 'other' end as kind,
       count(*) as n, string_agg(id::text, ', ' order by id) filter (where true) as ids
from ncert_questions
where class_grade = '12' and subject ~ '^Chemistry' and lower(chapter) = 'chemical coordination'
group by 1 order by 1;
