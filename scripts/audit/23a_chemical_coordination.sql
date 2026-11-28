-- Read-only: what is Class 12 Chemistry "Chemical Coordination"? Rows by subject string + where its texts also occur.
with cc as (
  select id, subject, md5(lower(btrim(regexp_replace(question_text, '\s+', ' ', 'g')))) as h
  from ncert_questions where class_grade = '12' and subject ~ '^Chemistry' and lower(chapter) = 'chemical coordination'),
other as (
  select class_grade, subject, chapter, md5(lower(btrim(regexp_replace(question_text, '\s+', ' ', 'g')))) as h
  from ncert_questions where not (class_grade = '12' and subject ~ '^Chemistry' and lower(chapter) = 'chemical coordination'))
select 'rows' as k, (select count(*) from cc)::text as v, string_agg(distinct subject, ', ') as detail from cc
union all
select 'distinct texts', count(distinct h)::text, null from cc
union all
select 'shared texts with ' || o.class_grade || ' / ' || o.subject || ' / ' || o.chapter, count(distinct o.h)::text, null
from other o where o.h in (select h from cc) group by o.class_grade, o.subject, o.chapter
union all
select 'chunks: ' || class_grade || ' / ' || subject || ' / ' || chapter, count(*)::text, string_agg(distinct coalesce(book,''), ', ')
from ncert_chunks where lower(chapter) like '%coordination%' group by class_grade, subject, chapter
union all
select 'questions chapter like %coordination%: ' || class_grade || ' / ' || subject || ' / ' || chapter, count(*)::text, null
from ncert_questions where lower(chapter) like '%coordination%' group by class_grade, subject, chapter;
