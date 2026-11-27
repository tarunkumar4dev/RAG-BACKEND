-- 42_default_privileges_read_only_rollback.sql — restores the Supabase default (ALL on new public tables).
begin;
alter default privileges for role postgres in schema public
  grant insert, update, delete, truncate, trigger, references on tables to anon, authenticated;
do $$
begin
  if pg_has_role(current_user, 'supabase_admin', 'MEMBER') then
    execute 'alter default privileges for role supabase_admin in schema public
               grant insert, update, delete, truncate, trigger, references on tables to anon, authenticated';
  end if;
end $$;
commit;
