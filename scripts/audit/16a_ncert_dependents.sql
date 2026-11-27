-- Read-only: everything that depends on ncert_questions ids (FKs, views, functions, look-alike id columns).
select 'fk' as kind, conrelid::regclass::text as obj, conname || ': ' || pg_get_constraintdef(oid) as detail
from pg_constraint where confrelid = 'public.ncert_questions'::regclass
union all
select 'view', v.oid::regclass::text, 'depends on ncert_questions'
from pg_depend d join pg_rewrite r on r.oid = d.objid join pg_class v on v.oid = r.ev_class
where d.refobjid = 'public.ncert_questions'::regclass and v.oid <> 'public.ncert_questions'::regclass
union all
select 'function', n.nspname || '.' || p.proname, 'body mentions ncert_questions'
from pg_proc p join pg_namespace n on n.oid = p.pronamespace
where p.prosrc ilike '%ncert_questions%' and n.nspname not in ('pg_catalog', 'information_schema')
union all
select 'column (no FK?)', table_schema || '.' || table_name, column_name || ' ' || data_type
from information_schema.columns
where table_schema not in ('pg_catalog', 'information_schema', 'auth', 'storage') and table_schema not like 'a4_backup%'
  and (column_name ilike '%ncert%' or (column_name ilike '%question_id%' and data_type in ('bigint', 'integer')))
order by 1, 2 limit 100;
