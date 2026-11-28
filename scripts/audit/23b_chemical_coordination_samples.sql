-- Read-only: 15 random samples of Class 12 Chemistry "Chemical Coordination" (fixed seed).
select setseed(0.42);
select id, subject, question_type, section, left(regexp_replace(question_text, '\s+', ' ', 'g'), 170) as text
from (select *, row_number() over (order by random()) rn from ncert_questions
      where class_grade = '12' and subject ~ '^Chemistry' and lower(chapter) = 'chemical coordination') x
where rn <= 15 order by id;
