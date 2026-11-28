-- Read-only: 10 random samples each from Class 11 "Answers" and "Index" (fixed seed for reproducibility).
select setseed(0.42);
select chapter, id, subject, question_type, coalesce(quality_status, '') as qs,
       left(regexp_replace(question_text, '\s+', ' ', 'g'), 160) as text,
       left(regexp_replace(coalesce(answer, ''), '\s+', ' ', 'g'), 60) as answer
from (select *, row_number() over (partition by lower(chapter) order by random()) as rn
      from ncert_questions where class_grade = '11' and lower(chapter) in ('answers', 'index')) x
where rn <= 10 order by chapter, id;
