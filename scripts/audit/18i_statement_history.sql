-- Read-only: exfiltration / tampering patterns in pg_stat_statements (since its last reset), grouped by role.
-- Query text is cut to 140 chars; parameters are already normalised ($1) by pg_stat_statements.
select (select stats_reset::date from extensions.pg_stat_statements_info) as stats_since,
       r.rolname as role, s.calls, s.rows,
       case when s.query ~* '^\s*copy\M' then 'COPY'
            when s.query ~* 'pg_dump|pg_catalog\.pg_get_.*def.*from pg_proc' then 'DUMP-LIKE'
            when s.query ~* 'from\s+"?auth"?\."?users"?' and s.query !~* '\mwhere\M' then 'AUTH.USERS FULL SCAN'
            when s.query ~* '(alter|create)\s+(role|user)|grant\s+.*\s+to\s' then 'ROLE/GRANT DDL'
            when s.query ~* 'create\s+(or\s+replace\s+)?(function|trigger)|create\s+event\s+trigger' then 'FUNC/TRIGGER DDL'
            when s.query ~* 'vault\.decrypted_secrets' then 'VAULT READ'
            when s.query ~* 'net\.http_(post|get)' and s.query !~* 'a4_checker' then 'HTTP OUT'
       end as pattern,
       left(regexp_replace(s.query, '\s+', ' ', 'g'), 140) as query
from extensions.pg_stat_statements s join pg_roles r on r.oid = s.userid
where s.query ~* '^\s*copy\M|pg_dump|from\s+"?auth"?\."?users"?|(alter|create)\s+(role|user)|grant\s+.*\s+to\s|create\s+(or\s+replace\s+)?(function|trigger)|create\s+event\s+trigger|vault\.decrypted_secrets|net\.http_(post|get)'
  and r.rolname not in ('supabase_admin', 'supabase_auth_admin', 'supabase_storage_admin', 'supabase_realtime_admin', 'pgbouncer')
order by pattern, s.calls desc limit 80;
