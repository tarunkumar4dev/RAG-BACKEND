-- Read-only: Class 11 Physics back-matter rows stored as chapters "Answers" / "Index" (textbook appendix pages,
-- not NCERT chapters). Counts by subject string and quality_status, plus the shape of quality_status/quality_flags.
select class_grade, subject, chapter, coalesce(quality_status, '(null)') as quality_status, count(*) as n,
       count(*) filter (where image_url is not null) as with_image,
       count(*) filter (where coalesce(btrim(answer), '') <> '') as with_answer,
       min(id) as min_id, max(id) as max_id
from ncert_questions
where class_grade = '11' and lower(chapter) in ('answers', 'index')
group by 1, 2, 3, 4 order by 1, 2, 3, 4;
