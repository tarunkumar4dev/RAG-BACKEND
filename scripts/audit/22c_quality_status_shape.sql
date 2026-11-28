-- Read-only: allowed values of quality_status (CHECK constraints) and the column types used for hiding rows.
select conname, pg_get_constraintdef(oid) as def from pg_constraint
where conrelid = 'public.ncert_questions'::regclass and contype = 'c';
