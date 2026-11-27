-- Read-only: every non-system object a leaked DB/service key could have been used to create.
-- Postgres keeps no creation time for these (track_commit_timestamp = off), so each name is later matched against
-- the DDL in both repos' git history.
with sys(n) as (values ('pg_catalog'), ('information_schema'), ('auth'), ('storage'), ('realtime'), ('extensions'),
                       ('graphql'), ('graphql_public'), ('pgsodium'), ('pgsodium_masks'), ('vault'), ('net'), ('cron'),
                       ('supabase_functions'), ('supabase_migrations'), ('pgbouncer'), ('pgtle'), ('_realtime'), ('_analytics'), ('pg_toast'))
select 'role' as kind, '' as schema_, rolname as name,
       concat_ws(' ', case when rolsuper then 'SUPERUSER' end, case when rolcanlogin then 'LOGIN' end, case when rolbypassrls then 'BYPASSRLS' end,
                 case when rolcreaterole then 'CREATEROLE' end, 'member_of=' || (select string_agg(b.rolname, ',') from pg_auth_members m join pg_roles b on b.oid = m.roleid where m.member = r.oid)) as detail
from pg_roles r
where rolname !~ '^pg_' and rolname not in ('postgres', 'anon', 'authenticated', 'service_role', 'authenticator', 'dashboard_user', 'pgbouncer',
  'supabase_admin', 'supabase_auth_admin', 'supabase_storage_admin', 'supabase_functions_admin', 'supabase_realtime_admin',
  'supabase_replication_admin', 'supabase_read_only_user', 'supabase_etl_admin', 'pgsodium_keyholder', 'pgsodium_keyiduser',
  'pgsodium_keymaker', 'pgtle_admin', 'supabase_privileged_role')
union all
select 'table', n.nspname, c.relname, c.relkind::text || ' rows~' || c.reltuples::bigint || ' owner=' || pg_get_userbyid(c.relowner) || ' rls=' || c.relrowsecurity
from pg_class c join pg_namespace n on n.oid = c.relnamespace
where c.relkind in ('r', 'p', 'v', 'm', 'f') and n.nspname not in (select n from sys) and n.nspname !~ '^pg_temp'
union all
select 'function', n.nspname, p.proname, 'secdef=' || p.prosecdef || ' owner=' || pg_get_userbyid(p.proowner) || ' anon_exec=' || has_function_privilege('anon', p.oid, 'EXECUTE')
from pg_proc p join pg_namespace n on n.oid = p.pronamespace
where n.nspname not in (select n from sys) and not exists (select 1 from pg_depend d where d.objid = p.oid and d.deptype = 'e')
union all
select 'trigger', n.nspname, t.tgname, c.relname || ' -> ' || t.tgfoid::regproc::text
from pg_trigger t join pg_class c on c.oid = t.tgrelid join pg_namespace n on n.oid = c.relnamespace
where not t.tgisinternal and n.nspname not in ('auth', 'storage', 'realtime', 'supabase_functions', 'cron', 'net', 'vault', 'pgsodium')
union all
select 'trigger(system schema)', n.nspname, t.tgname, c.relname || ' -> ' || t.tgfoid::regproc::text
from pg_trigger t join pg_class c on c.oid = t.tgrelid join pg_namespace n on n.oid = c.relnamespace
where not t.tgisinternal and n.nspname in ('auth', 'storage')
union all
select 'event_trigger', '', evtname, evtevent || ' -> ' || evtfoid::regproc::text || ' owner=' || pg_get_userbyid(evtowner) from pg_event_trigger
union all
select 'extension', n.nspname, e.extname, e.extversion from pg_extension e join pg_namespace n on n.oid = e.extnamespace
union all
select 'publication', '', p.pubname, 'all_tables=' || p.puballtables || ' tables=' || coalesce((select string_agg(schemaname || '.' || tablename, ',') from pg_publication_tables pt where pt.pubname = p.pubname), '')
from pg_publication p
union all
select 'cron_job', '', coalesce(j.jobname, j.jobid::text), j.schedule || ' | ' || left(regexp_replace(j.command, '\s+', ' ', 'g'), 120) || ' | user=' || j.username || ' active=' || j.active from cron.job j
union all
select 'storage_bucket', '', b.id, 'public=' || b.public || ' created=' || b.created_at::date from storage.buckets b
union all
select 'vault_secret', '', coalesce(s.name, '(unnamed)'), 'created=' || s.created_at::date from vault.secrets s
union all
select 'schema', '', n.nspname, 'owner=' || pg_get_userbyid(n.nspowner) from pg_namespace n
where n.nspname not in (select n from sys) and n.nspname !~ '^pg_'
order by 1, 2, 3 limit 2000;
