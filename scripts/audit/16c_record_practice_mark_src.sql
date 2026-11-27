-- Read-only: source of the only function that reads ncert_questions.
select p.oid::regprocedure as fn, p.prosecdef as security_definer, p.prosrc as body
from pg_proc p where p.proname = 'record_practice_mark';
