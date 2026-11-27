-- Read-only: exact chapter strings per subject string, Class 11-12 Physics/Chemistry/Pol Sci families.
select class_grade, subject, chapter, count(*) n from ncert_questions where class_grade in ('11','12') and subject ~ '^(Physics|Chemistry|Political)' group by 1,2,3 order by 1, regexp_replace(lower(chapter),'[^a-z]','','g'), 2;
