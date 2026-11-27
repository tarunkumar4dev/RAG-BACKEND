-- Read-only follow-ups: role DDL other than the Supabase CLI login role, vault statements, and the live grants on the
-- contest tables that one historical GRANT gave ALL privileges to anon.
select 'role ddl (not cli_login_postgres)' as what, r.rolname as by_role, s.calls::text as n, left(regexp_replace(s.query, '\s+', ' ', 'g'), 200) as detail
from extensions.pg_stat_statements s join pg_roles r on r.oid = s.userid
where s.query ~* '\m(create|alter|drop)\s+(role|user)\M' and s.query !~* 'cli_login_postgres' and r.rolname <> 'supabase_admin'
union all
select 'vault statement', r.rolname, s.calls::text, left(regexp_replace(s.query, '\s+', ' ', 'g'), 160)
from extensions.pg_stat_statements s join pg_roles r on r.oid = s.userid where s.query ~* 'vault\.(decrypted_secrets|create_secret)'
union all
select 'live grants: ' || t, g.role,
       concat_ws(',', case when has_table_privilege(g.role, 'public.' || t, 'SELECT') then 'SELECT' end, case when has_table_privilege(g.role, 'public.' || t, 'INSERT') then 'INSERT' end,
                 case when has_table_privilege(g.role, 'public.' || t, 'UPDATE') then 'UPDATE' end, case when has_table_privilege(g.role, 'public.' || t, 'DELETE') then 'DELETE' end,
                 case when has_table_privilege(g.role, 'public.' || t, 'TRUNCATE') then 'TRUNCATE' end),
       (select 'rls=' || relrowsecurity || ' policies=' || (select count(*) from pg_policies p where p.tablename = t) from pg_class where oid = ('public.' || t)::regclass)
from unnest(array['contests', 'contest_questions', 'contest_attempts']) t, (values ('anon'), ('authenticated')) g(role)
order by 1, 2 limit 100;
