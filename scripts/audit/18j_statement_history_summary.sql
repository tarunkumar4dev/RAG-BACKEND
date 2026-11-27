-- Read-only: pg_stat_statements pattern counts per role, then every ROLE/GRANT, VAULT, HTTP, COPY, DUMP row.
with s as (
  select r.rolname as role, s.calls, left(regexp_replace(s.query, '\s+', ' ', 'g'), 150) as q,
         case when s.query ~* '^\s*copy\M' then 'COPY'
              when s.query ~* 'pg_dump' then 'DUMP'
              when s.query ~* '\m(create|alter|drop)\s+(role|user)\M' then 'ROLE DDL'
              when s.query ~* '\mgrant\M[^;]*\mto\M' then 'GRANT'
              when s.query ~* 'vault\.decrypted_secrets|vault\.create_secret' then 'VAULT'
              when s.query ~* 'net\.http_(post|get)' then 'HTTP OUT'
              when s.query ~* 'create\s+(or\s+replace\s+)?(function|trigger)|create\s+event\s+trigger' then 'FUNC/TRIGGER DDL'
              when s.query ~* 'from\s+"?auth"?\."?users"?' then 'READS auth.users' end as pattern
  from extensions.pg_stat_statements s join pg_roles r on r.oid = s.userid)
select 'summary' as part, role, pattern, sum(calls)::text as calls, count(*)::text as statements from s where pattern is not null group by 2, 3
union all
select 'detail', role, pattern, calls::text, q from s where pattern in ('COPY', 'DUMP', 'ROLE DDL', 'GRANT', 'VAULT', 'HTTP OUT') and role <> 'supabase_admin'
order by 1 desc, 2, 3 limit 150;
