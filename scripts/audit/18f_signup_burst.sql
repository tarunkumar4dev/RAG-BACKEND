-- Read-only, aggregated: what the busiest signup hour (2026-10-25 15:00 UTC) consisted of, and the example.com accounts.
select 'burst hour' as what, lower(split_part(email, '@', 2)) as domain, coalesce(raw_app_meta_data->>'provider', '?') as provider, count(*) as n,
       count(*) filter (where last_sign_in_at is null) as never_signed_in, min(created_at)::time as first, max(created_at)::time as last
from auth.users where created_at >= '2026-10-25 15:00' and created_at < '2026-10-25 16:00' group by 2, 3
union all
select 'example.com', 'example.com', coalesce(raw_app_meta_data->>'provider', '?'), count(*), count(*) filter (where last_sign_in_at is null), min(created_at)::time, max(created_at)::time
from auth.users where email ilike '%@example.com' group by 3 order by 1, 4 desc;
