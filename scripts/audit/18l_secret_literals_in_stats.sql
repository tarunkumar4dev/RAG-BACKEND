-- Read-only: statements kept in pg_stat_statements that contain a secret-looking literal (DO-block / DDL literals are
-- not normalised to $n). Prints the role, the kind of hint, the literal's length and a MASKED preview only.
with s as (
  select r.rolname as role, s.calls, s.query,
         (regexp_match(s.query, '(?i)(password|passwd|pw\s+text|secret|token|api[_-]?key|service_role|bearer)\s*(?::=|=|:|\s)\s*''([^'']{12,})''')) as m
  from extensions.pg_stat_statements s join pg_roles r on r.oid = s.userid
  where s.query !~* 'cli_login_postgres'),  -- CLI temp login role: SCRAM hashes, expired
j as (select role, calls, query, (regexp_match(query, '(eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|sb_secret_[A-Za-z0-9_-]{10,}|AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9_-]{20,})'))[1] as tok
      from s)
select 'assigned literal' as kind, role, calls, lower(m[1]) as hint, length(m[2]) as len,
       left(m[2], 2) || repeat('*', greatest(length(m[2]) - 4, 0)) || right(m[2], 2) as masked,
       left(regexp_replace(regexp_replace(query, '''[^'']{12,}''', '''<redacted>''', 'g'), '\s+', ' ', 'g'), 110) as context
from s where m is not null
union all
select 'key-shaped token', role, calls, left(tok, 4), length(tok), left(tok, 4) || '…', left(regexp_replace(replace(query, tok, '<redacted>'), '\s+', ' ', 'g'), 110)
from j where tok is not null
order by 1, 2 limit 100;
