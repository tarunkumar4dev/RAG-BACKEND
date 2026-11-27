-- Read-only: shape and size of ncert_practice_marks (student practice records).
select column_name, data_type, is_nullable, column_default from information_schema.columns
where table_schema = 'public' and table_name = 'ncert_practice_marks' order by ordinal_position;
