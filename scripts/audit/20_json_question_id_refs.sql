-- Read-only: do any JSON / JSONB / array columns hold ncert_questions ids that a dedupe delete would orphan?
-- For every json/jsonb column in the app schemas: rows, rows with an id-like key (…ncert…id / …question…id(s) /
-- source_id), how many numeric values under such keys exist, and how many of those are live ncert_questions ids.
-- Array columns of integer types are checked the same way.
with cols as (
  select c.table_schema as s, c.table_name as t, c.column_name as col, c.data_type as dt, c.udt_name as udt
  from information_schema.columns c join information_schema.tables tb on tb.table_schema = c.table_schema and tb.table_name = c.table_name
  where c.table_schema in ('public', 'practice') and tb.table_type = 'BASE TABLE'
    and (c.data_type in ('json', 'jsonb') or (c.data_type = 'ARRAY' and c.udt_name in ('_int2', '_int4', '_int8', '_numeric', '_text'))))
select s || '.' || t as tbl, col, dt,
       (xpath('/row/n/text()', query_to_xml(format('select count(*) as n from %I.%I', s, t), false, true, '')))[1]::text::bigint as table_rows,
       case when dt in ('json', 'jsonb') then
         (xpath('/row/n/text()', query_to_xml(format(
           $q$select count(*) as n from %I.%I where %I::text ~* '"[a-z_]*(ncert|question|source)[a-z_]*_?ids?"\s*:'$q$, s, t, col), false, true, '')))[1]::text::bigint end as rows_with_id_key,
       case when dt in ('json', 'jsonb') then
         (xpath('/row/n/text()', query_to_xml(format(
           $q$select count(*) as n from (select (m[1])::bigint as v from %I.%I x,
                regexp_matches(x.%I::text, '"[a-z_]*(?:ncert|question|source)[a-z_]*_?ids?"\s*:\s*\[?\s*"?([0-9]{1,9})(?![0-9a-f-])', 'gi') m) v
              where exists (select 1 from public.ncert_questions q where q.id = v.v)$q$, s, t, col), false, true, '')))[1]::text::bigint
       else
         (xpath('/row/n/text()', query_to_xml(format(
           $q$select count(*) as n from (select unnest(%I)::text as v from %I.%I) a where a.v ~ '^[0-9]{1,9}$' and exists (select 1 from public.ncert_questions q where q.id = a.v::bigint)$q$, col, s, t), false, true, '')))[1]::text::bigint
       end as values_matching_ncert_ids
from cols
order by values_matching_ncert_ids desc nulls last, rows_with_id_key desc nulls last, tbl, col
limit 300;
