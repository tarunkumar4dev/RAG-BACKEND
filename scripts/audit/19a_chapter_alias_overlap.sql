-- Read-only: evidence for a chapter alias map. For every pair of DIFFERENT chapter strings in the same class and
-- subject family (Physics-I/II -> Physics, Chemistry-I/II -> Chemistry, Political_Science -> Political Science),
-- how many distinct question texts they share. High overlap = same NCERT chapter stored under two names.
with q as (
  select class_grade, subject, chapter,
         regexp_replace(replace(subject, '_', ' '), '-I+$', '') as fam,
         md5(lower(btrim(regexp_replace(question_text, '\s+', ' ', 'g')))) as h
  from ncert_questions),
c as (select class_grade, fam, subject, chapter, count(distinct h) as texts from q group by 1, 2, 3, 4),
pair as (
  select a.class_grade, a.fam, a.subject as s1, a.chapter as ch1, b.subject as s2, b.chapter as ch2, count(distinct a.h) as shared
  from (select distinct class_grade, fam, subject, chapter, h from q) a
  join (select distinct class_grade, fam, subject, chapter, h from q) b
    on a.class_grade = b.class_grade and a.fam = b.fam and a.h = b.h and (a.subject, a.chapter) < (b.subject, b.chapter)
  group by 1, 2, 3, 4, 5, 6)
select p.class_grade as cls, p.fam, p.s1 || ' / ' || p.ch1 as chapter_a, c1.texts as a_texts,
       p.s2 || ' / ' || p.ch2 as chapter_b, c2.texts as b_texts, p.shared,
       round(100.0 * p.shared / least(c1.texts, c2.texts), 0) as pct_of_smaller,
       regexp_replace(lower(p.ch1), '[^a-z0-9]', '', 'g') = regexp_replace(lower(p.ch2), '[^a-z0-9]', '', 'g') as same_name_normalised
from pair p
join c c1 on (c1.class_grade, c1.fam, c1.subject, c1.chapter) = (p.class_grade, p.fam, p.s1, p.ch1)
join c c2 on (c2.class_grade, c2.fam, c2.subject, c2.chapter) = (p.class_grade, p.fam, p.s2, p.ch2)
where p.shared >= 3
order by same_name_normalised, pct_of_smaller desc limit 200;
