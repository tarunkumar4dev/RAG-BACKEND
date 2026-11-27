-- Read-only: bodies of the post-Aug SECURITY DEFINER / helper functions that no repo file references, and the
-- live RLS policies that call them (a policy reference explains why a helper exists).
select 'body' as what, p.proname as name, p.prosecdef::text as secdef, regexp_replace(p.prosrc, '\s+', ' ', 'g') as text
from pg_proc p where p.pronamespace = 'public'::regnamespace
  and p.proname in ('fill_member_identity', 'is_proctor_of', 'is_hod_of_dept', 'is_within_edit_window', 'has_institute_role', 'search_module_chunks', 'update_modules_updated_at')
union all
select 'used_by_policy', f.name, p.tablename, p.policyname || ' [' || p.cmd || ']'
from (values ('is_proctor_of'), ('is_hod_of_dept'), ('is_within_edit_window'), ('has_institute_role'), ('is_session_teacher'), ('my_institute_ids')) f(name)
join pg_policies p on coalesce(p.qual, '') || ' ' || coalesce(p.with_check, '') ilike '%' || f.name || '(%'
order by 1, 2 limit 200;
