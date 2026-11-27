-- Read-only: every table privilege on ncert_questions / ncert_chunks, per grantee (from relacl), plus the
-- schema default privileges that new tables in public inherit.
select c.relname as tbl, case when a.grantee = 0 then 'PUBLIC' else a.grantee::regrole::text end as grantee,
       string_agg(a.privilege_type, ', ' order by a.privilege_type) as privileges,
       bool_or(a.is_grantable) as with_grant_option
from pg_class c cross join lateral aclexplode(c.relacl) a
where c.oid in ('public.ncert_questions'::regclass, 'public.ncert_chunks'::regclass)
group by 1, 2
union all
select 'DEFAULT (' || pg_get_userbyid(d.defaclrole) || ', ' || coalesce(n.nspname, '*') || ', ' || d.defaclobjtype::text || ')',
       case when a.grantee = 0 then 'PUBLIC' else a.grantee::regrole::text end, string_agg(a.privilege_type, ', ' order by a.privilege_type), bool_or(a.is_grantable)
from pg_default_acl d left join pg_namespace n on n.oid = d.defaclnamespace cross join lateral aclexplode(d.defaclacl) a
where d.defaclobjtype = 'r' and coalesce(n.nspname, 'public') = 'public' and a.grantee <> 0 and a.grantee::regrole::text in ('anon', 'authenticated')
group by 1, 2
order by 1, 2 limit 50;
