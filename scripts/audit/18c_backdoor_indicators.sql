-- Read-only: scan every non-extension function body, cron job, event trigger and auth/storage hook for
-- exfiltration / persistence patterns. Returns indicator flags and URL hosts, not full bodies.
with f as (
  select 'function' as kind, n.nspname || '.' || p.proname || '(' || pg_get_function_identity_arguments(p.oid) || ')' as obj,
         p.prosecdef as secdef, has_function_privilege('anon', p.oid, 'EXECUTE') as anon_exec, lower(p.prosrc) as body
  from pg_proc p join pg_namespace n on n.oid = p.pronamespace
  where n.nspname not in ('pg_catalog', 'information_schema', 'extensions', 'graphql', 'graphql_public', 'pgsodium', 'vault',
                          'net', 'cron', 'realtime', 'storage', 'supabase_functions', 'pgbouncer', '_realtime', 'auth')
    and not exists (select 1 from pg_depend d where d.objid = p.oid and d.deptype = 'e')
  union all
  select 'cron_job', coalesce(jobname, jobid::text), null, null, lower(command) from cron.job
  union all
  select 'auth_function', n.nspname || '.' || p.proname, p.prosecdef, null, lower(p.prosrc)
  from pg_proc p join pg_namespace n on n.oid = p.pronamespace
  where n.nspname = 'auth' and p.proowner <> (select oid from pg_roles where rolname = 'supabase_auth_admin')
),
x as (
  select kind, obj, secdef, anon_exec,
    array_remove(array[
      case when body ~ 'net\.http_(post|get|request)|http_post|http_get|extensions\.http' then 'HTTP_CALL' end,
      case when body ~ 'dblink|postgres_fdw' then 'DBLINK' end,
      case when body ~ '\m(alter|create)\s+(role|user)\M' then 'ROLE_DDL' end,
      case when body ~ '\mgrant\M' then 'GRANT' end,
      case when body ~ '\mset\s+(local\s+)?role\M' then 'SET_ROLE' end,
      case when body ~ '(insert\s+into|update|delete\s+from)\s+auth\.' then 'WRITES_AUTH' end,
      case when body ~ '\m(copy|lo_export|lo_import|pg_read_file|pg_read_binary_file|pg_ls_dir)\M' then 'FILE_IO' end,
      case when body ~ 'execute\s+(format\s*\(|[^;]*\|\|)' then 'DYNAMIC_SQL' end,
      case when body ~ 'service_role|supabase_admin' then 'MENTIONS_PRIV_ROLE' end,
      case when body ~ 'vault\.|decrypted_secrets' then 'READS_VAULT' end,
      case when body ~ '(drop|truncate)\s+(table|schema)' then 'DROP_TRUNCATE' end
    ], null) as flags,
    (select string_agg(distinct m[1], ',') from regexp_matches(body, 'https?://([a-z0-9.-]+)', 'g') m) as url_hosts
  from f)
select kind, obj, secdef, anon_exec, array_to_string(flags, ' ') as flags, url_hosts
from x where cardinality(flags) > 0 or url_hosts is not null
order by kind, obj limit 200;
