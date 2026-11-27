-- Read-only: effective anon privileges on the tables the backend's public client reads.
select t as tbl, has_table_privilege('anon', 'public.' || t, 'SELECT') as anon_select,
       has_table_privilege('anon', 'public.' || t, 'INSERT') as anon_insert
from unnest(array['ncert_questions', 'ncert_chunks', 'plans']) t;
