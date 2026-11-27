-- 42_default_privileges_read_only.sql
-- New tables in schema public stop granting write-type privileges to the API roles by default.
--
-- Before (live, 27 Nov 2026, scripts/audit/15_ncert_grants.sql), pg_default_acl for tables in schema public:
--   owner postgres       -> anon, authenticated : SELECT, INSERT, UPDATE, DELETE, TRUNCATE, TRIGGER, REFERENCES
--   owner supabase_admin -> anon, authenticated : SELECT, INSERT, UPDATE, DELETE, TRUNCATE, TRIGGER, REFERENCES
-- After (for tables CREATED from now on; existing tables are not touched):
--   anon, authenticated : SELECT only (rows still filtered by RLS)
--
-- CONSEQUENCE FOR FUTURE MIGRATIONS: a new table that the browser writes to through RLS (as migrations 06 and 09
-- did, relying on these defaults) now needs an explicit
--   grant insert, update, delete on public.<table> to authenticated;
-- next to its policies. Without it the insert fails with "permission denied", not silently.
-- service_role is not listed in these defaults and keeps its own grants.
--
-- supabase_admin's defaults can only be changed by supabase_admin (or a member). Migrations normally run as
-- postgres, which is not, so that part is attempted only when allowed and otherwise reported with a NOTICE.
-- Tables created from the dashboard / migrations are owned by postgres, so the postgres defaults are the ones
-- that matter in practice.
-- Rollback: 42_default_privileges_read_only_rollback.sql.

begin;

alter default privileges for role postgres in schema public
  revoke insert, update, delete, truncate, trigger, references on tables from anon, authenticated;

do $$
begin
  if pg_has_role(current_user, 'supabase_admin', 'MEMBER') then
    execute 'alter default privileges for role supabase_admin in schema public
               revoke insert, update, delete, truncate, trigger, references on tables from anon, authenticated';
  else
    raise notice 'skipped supabase_admin default privileges: % is not a member of supabase_admin', current_user;
  end if;
end $$;

-- Verify: postgres' defaults for public tables now give anon/authenticated SELECT and nothing else.
do $$
declare bad text;
begin
  select string_agg(distinct a.grantee::regrole::text || ':' || a.privilege_type, ', ') into bad
  from pg_default_acl d cross join lateral aclexplode(d.defaclacl) a
  where d.defaclrole = 'postgres'::regrole and d.defaclobjtype = 'r'
    and d.defaclnamespace = 'public'::regnamespace
    and a.grantee in ('anon'::regrole, 'authenticated'::regrole) and a.privilege_type <> 'SELECT';
  if bad is not null then
    raise exception 'default privileges still grant: %', bad;
  end if;
end $$;

commit;
