-- Read-only: is ncert_practice_marks really empty, or is RLS hiding rows from this session?
select current_user as role, (select rolbypassrls or rolsuper from pg_roles where rolname = current_user) as bypasses_rls,
       (select relowner::regrole::text from pg_class where oid = 'public.ncert_practice_marks'::regclass) as table_owner,
       (select relforcerowsecurity from pg_class where oid = 'public.ncert_practice_marks'::regclass) as force_rls,
       (select count(*) from ncert_practice_marks) as visible_rows,
       (select n_live_tup from pg_stat_user_tables where relid = 'public.ncert_practice_marks'::regclass) as n_live_tup,
       (select n_tup_ins from pg_stat_user_tables where relid = 'public.ncert_practice_marks'::regclass) as inserts_ever_since_stats_reset,
       (select n_tup_del from pg_stat_user_tables where relid = 'public.ncert_practice_marks'::regclass) as deletes_since_stats_reset,
       pg_size_pretty(pg_total_relation_size('public.ncert_practice_marks')) as size;
