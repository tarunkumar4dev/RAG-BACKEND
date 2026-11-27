-- Read-only DRY RUN of the exact-duplicate cleanup (nothing is changed). One row per metric.
-- Group  = same class_grade + subject + chapter + md5(lower(trim(collapse-whitespace(question_text)))).
-- Canonical row per group, in priority order:
--   1. most student practice marks (ncert_practice_marks)      <- keeps students' records where they already are
--   2. visible today (quality_status NULL or not duplicate/junk) <- never resurrects a G4.5-hidden copy
--   3. lowest quality_rank (G4.5 clean/fixed first)
--   4. has image_url, 5. has answer, 6. has options, 7. has question_table, 8. lowest id
with q as (
  select n.id, n.class_grade, n.subject, n.chapter, n.quality_status, n.duplicate_of, n.quality_rank,
         n.image_url is not null as has_img, coalesce(btrim(n.answer), '') <> '' as has_ans,
         coalesce(n.options #>> '{}', '[]') not in ('[]', '') as has_opts, n.question_table is not null as has_tbl,
         md5(lower(btrim(regexp_replace(n.question_text, '\s+', ' ', 'g')))) as h,
         (select count(*) from ncert_practice_marks m where m.question_id = n.id) as marks
  from ncert_questions n),
r as (
  select q.*, count(*) over g as group_size,
         first_value(id) over (g order by marks desc,
                                   (quality_status is not null and quality_status in ('duplicate', 'junk')),
                                   coalesce(quality_rank, 0), not has_img, not has_ans, not has_opts, not has_tbl, id) as canonical_id
  from q window g as (partition by class_grade, subject, chapter, h)),
doomed as (select id, canonical_id from r where id <> canonical_id),
pm as (select m.*, d.canonical_id from ncert_practice_marks m left join doomed d on d.id = m.question_id),
remap as (select * from pm where canonical_id is not null),
-- after remapping, a student may hold 2+ marks for the same canonical question: keep the newest, drop the rest
coll as (select user_id, coalesce(canonical_id, question_id) as target, count(*) as n
         from pm group by 1, 2 having count(*) > 1)
select 1 as ord, 'duplicate groups (exact, same chapter)' as metric, count(distinct canonical_id)::text as value from r where group_size > 1
union all select 2, 'rows that would be deleted', count(*)::text from doomed
union all select 3, 'rows kept', (count(*) - (select count(*) from doomed))::text from r
union all select 4, 'ncert_practice_marks rows (total)', count(*)::text from ncert_practice_marks
union all select 5, 'students with any practice mark', count(distinct user_id)::text from ncert_practice_marks
union all select 6, 'practice marks on a row that would be deleted (must be remapped)', count(*)::text from remap
union all select 7, 'students affected by a remap', count(distinct user_id)::text from remap
union all select 8, 'marks dropped after remap (student already has a newer/other mark on the canonical)', coalesce(sum(n - 1), 0)::text from coll
union all select 9, 'students losing a mark to that collision rule', count(distinct user_id)::text from coll
union all select 10, 'marks currently on G4.5-hidden rows (duplicate/junk)', count(*)::text from ncert_practice_marks m join ncert_questions n on n.id = m.question_id where n.quality_status in ('duplicate', 'junk')
union all select 11, 'groups whose canonical is chosen because of marks (would differ without rule 1)', count(*)::text from (
    select canonical_id from r where group_size > 1 group by class_grade, subject, chapter, h, canonical_id
    having bool_or(marks > 0) and canonical_id <> min(id) filter (where quality_status is null or quality_status not in ('duplicate', 'junk'))) x
union all select 12, 'groups where 2+ copies have marks', count(*)::text from (select 1 from r group by class_grade, subject, chapter, h having count(*) filter (where marks > 0) > 1) x
union all select 13, 'groups whose canonical is hidden although a visible copy exists (must be 0)', count(*)::text from (
    select 1 from r where group_size > 1 group by class_grade, subject, chapter, h
    having bool_or(id = canonical_id and quality_status in ('duplicate', 'junk'))
       and bool_or(quality_status is null or quality_status not in ('duplicate', 'junk'))) x
union all select 14, 'ncert_question_fixes rows on rows that would be deleted (cascade)', count(*)::text from ncert_question_fixes f join doomed d on d.id = f.question_id
union all select 15, 'kept rows whose duplicate_of points at a deleted row (repoint first)', count(*)::text from r join doomed d on d.id = r.duplicate_of where r.id = r.canonical_id
union all select 16, 'kept rows still marked duplicate after cleanup (G4.5 near-dups, stay hidden)', count(*)::text from r where id = canonical_id and quality_status = 'duplicate'
union all select 17, 'groups where every copy is hidden (kept row stays hidden, its duplicate_of target survives)', count(*)::text from r where id = canonical_id and quality_status in ('duplicate', 'junk') and group_size > 1
union all select 19, 'visible (non-hidden) rows deleted', count(*)::text from r where id <> canonical_id and (quality_status is null or quality_status not in ('duplicate', 'junk'))
union all select 20, 'hidden (G4.5 duplicate/junk) rows deleted', count(*)::text from r where id <> canonical_id and quality_status in ('duplicate', 'junk')
order by ord;
