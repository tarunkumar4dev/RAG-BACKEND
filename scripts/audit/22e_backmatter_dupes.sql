-- Read-only: exact duplicates among the back-matter rows after subject-family normalisation (Physics-I/II -> Physics,
-- Answers/ANSWERS -> one string), and matches of their texts in real Class 11 Physics chapters.
with b as (
  select id, lower(chapter) as ch, md5(lower(btrim(regexp_replace(question_text, '\s+', ' ', 'g')))) as h
  from ncert_questions where class_grade = '11' and subject in ('Physics-I', 'Physics-II') and lower(chapter) in ('answers', 'index')),
real as (
  select distinct md5(lower(btrim(regexp_replace(question_text, '\s+', ' ', 'g')))) as h
  from ncert_questions where class_grade = '11' and subject ~ '^Physics' and lower(chapter) not in ('answers', 'index'))
select ch, count(*) as rows, count(distinct h) as distinct_texts, count(*) - count(distinct h) as exact_copies,
       count(*) filter (where h in (select h from real)) as text_also_in_a_real_chapter
from b group by ch order by ch;
