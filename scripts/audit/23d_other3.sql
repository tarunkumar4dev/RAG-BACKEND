select id, question_type, section, left(regexp_replace(question_text, '\s+', ' ', 'g'), 200) as text
from ncert_questions where id in (51501, 51504, 75377) order by id;
