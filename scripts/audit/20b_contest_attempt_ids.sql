-- Read-only: what the id-like keys in contest_attempts.answers / progress point at (contest_questions vs ncert_questions).
with v as (
  select a.id as attempt, 'answers' as col, m[1] as key, m[2] as val
  from contest_attempts a, regexp_matches(a.answers::text, '"([a-z_]*(?:ncert|question|source)[a-z_]*_?ids?)"\s*:\s*\[?\s*"?([0-9a-f-]{1,36})', 'gi') m
  union all
  select a.id, 'progress', m[1], m[2]
  from contest_attempts a, regexp_matches(a.progress::text, '"([a-z_]*(?:ncert|question|source)[a-z_]*_?ids?)"\s*:\s*\[?\s*"?([0-9a-f-]{1,36})', 'gi') m)
select col, key, count(*) as vals,
       count(*) filter (where val ~ '^[0-9]+$') as numeric_vals,
       count(*) filter (where exists (select 1 from contest_questions c where c.id::text = v.val)) as match_contest_questions,
       count(*) filter (where val ~ '^[0-9]{1,9}$' and exists (select 1 from ncert_questions q where q.id = v.val::bigint)) as match_ncert_ids,
       (select data_type from information_schema.columns where table_name = 'contest_questions' and column_name = 'id') as contest_q_id_type,
       (select string_agg(column_name, ',') from information_schema.columns where table_name = 'contest_questions' and column_name ~* 'ncert|source') as contest_q_source_cols
from v group by 1, 2 order by 1, 2;
