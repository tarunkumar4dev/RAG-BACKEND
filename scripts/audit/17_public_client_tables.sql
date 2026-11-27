-- Read-only: for each table the backend's public client touches, RLS state, anon grants and policies that apply to anon.
with t(tbl) as (values ('ncert_questions'), ('ncert_chunks'), ('plans'), ('quizzes'))
select t.tbl, c.oid is not null as exists_, c.relrowsecurity as rls,
       (select string_agg(privilege_type, ',' order by privilege_type) from information_schema.role_table_grants g
         where g.table_schema = 'public' and g.table_name = t.tbl and g.grantee = 'anon') as anon_grants,
       (select string_agg(p.policyname || ' [' || p.cmd || ' to ' || array_to_string(p.roles, ',') || '] using ' || coalesce(p.qual, '-') || ' check ' || coalesce(p.with_check, '-'), ' | ')
          from pg_policies p where p.schemaname = 'public' and p.tablename = t.tbl) as policies,
       (select count(*) from pg_class k where k.oid = c.oid) as n
from t left join pg_class c on c.relname = t.tbl and c.relnamespace = 'public'::regnamespace
order by 1;
