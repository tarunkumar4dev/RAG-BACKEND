-- Read-only: types of the quality columns, quality_status distribution, most common quality_flags shapes.
select column_name, data_type, column_default from information_schema.columns
where table_schema = 'public' and table_name = 'ncert_questions'
  and column_name in ('quality_status', 'quality_flags', 'quality_rank', 'duplicate_of', 'hash_version');
select coalesce(quality_status, '(null)') as quality_status, count(*) from ncert_questions group by 1 order by 2 desc;
select quality_flags, count(*) from ncert_questions where quality_flags is not null group by 1 order by 2 desc limit 15;
