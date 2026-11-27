-- Read-only: can object creation times be recovered? (needs track_commit_timestamp=on) + Supabase migration history table.
select current_setting('track_commit_timestamp') as track_commit_timestamp,
       (select count(*) from information_schema.tables where table_schema = 'supabase_migrations') as migration_tables,
       (select string_agg(version || ':' || coalesce(name, ''), ', ' order by version) from (select * from supabase_migrations.schema_migrations order by version desc limit 15) m) as last_cli_migrations;
