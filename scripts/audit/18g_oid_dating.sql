-- Read-only: date objects by OID order (OIDs are handed out in increasing order). Tables with a created_at column
-- give anchor dates (earliest row); functions/triggers/tables without DDL in the repos are placed between anchors.
with t as (
  select c.oid::bigint as oid, 'table' as kind, n.nspname || '.' || c.relname as obj,
         case when exists (select 1 from pg_attribute a where a.attrelid = c.oid and a.attname = 'created_at' and not a.attisdropped)
              then (xpath('/row/m/text()', query_to_xml(format('select min(created_at) as m from %I.%I', n.nspname, c.relname), false, true, '')))[1]::text end as first_row
  from pg_class c join pg_namespace n on n.oid = c.relnamespace
  where c.relkind in ('r', 'p') and n.nspname in ('public', 'practice') or (c.relkind = 'r' and n.nspname like 'a4_backup%')),
f as (select p.oid::bigint, 'function', n.nspname || '.' || p.proname, null::text
      from pg_proc p join pg_namespace n on n.oid = p.pronamespace
      where n.nspname in ('public', 'practice') and not exists (select 1 from pg_depend d where d.objid = p.oid and d.deptype = 'e')),
g as (select t.oid::bigint, 'trigger', c.relname || '.' || t.tgname, null::text
      from pg_trigger t join pg_class c on c.oid = t.tgrelid join pg_namespace n on n.oid = c.relnamespace
      where not t.tgisinternal and n.nspname in ('public', 'practice', 'auth')),
x as (select oid::bigint, 'extension', extname, null::text from pg_extension
      union all select oid::bigint, 'role', rolname, null::text from pg_roles where rolname in ('agent_ro', 'cli_login_postgres'))
select * from t union all select * from f union all select * from g union all select * from x
order by oid limit 3000;
