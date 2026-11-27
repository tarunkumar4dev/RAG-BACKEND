-- Read-only, aggregated (no emails/ids printed): auth users created on/after 1 Aug 2026 and anything privileged.
with u as (select * from auth.users where created_at >= '2026-08-01')
select 'users created since 2026-08-01' as metric, count(*)::text as value from u
union all select 'by month', string_agg(m || '=' || n, ', ' order by m) from (select to_char(created_at, 'YYYY-MM') m, count(*) n from u group by 1) x
union all select 'by provider', string_agg(p || '=' || n, ', ' order by n desc) from (select coalesce(raw_app_meta_data->>'provider', '(none)') p, count(*) n from u group by 1) x
union all select 'max signups in one hour', coalesce(max(n), 0)::text from (select date_trunc('hour', created_at), count(*) n from u group by 1) x
union all select 'busiest hour', (select to_char(h, 'YYYY-MM-DD HH24:00') || ' (' || n || ')' from (select date_trunc('hour', created_at) h, count(*) n from u group by 1 order by 2 desc limit 1) x)
union all select 'email domains (count>=1)', string_agg(dom || '=' || n, ', ' order by n desc) from (select lower(split_part(email, '@', 2)) dom, count(*) n from u group by 1) x
union all select 'unconfirmed email', count(*) filter (where email_confirmed_at is null and phone_confirmed_at is null)::text from u
union all select 'never signed in', count(*) filter (where last_sign_in_at is null)::text from u
union all select 'anonymous users', count(*) filter (where is_anonymous)::text from u
-- privilege anomalies across ALL users, any date
union all select 'ALL users: role <> authenticated', coalesce(string_agg(distinct coalesce(role, '(null)'), ','), '') || ' n=' || count(*) filter (where coalesce(role, '') <> 'authenticated') from auth.users where coalesce(role, '') <> 'authenticated'
union all select 'ALL users: is_super_admin', count(*) filter (where is_super_admin)::text from auth.users
union all select 'ALL users: app_metadata keys beyond provider/providers', coalesce(string_agg(distinct k, ','), '(none)') from auth.users, jsonb_object_keys(coalesce(raw_app_meta_data, '{}')) k where k not in ('provider', 'providers')
union all select 'ALL users: app_metadata role/claims values', coalesce(string_agg(distinct coalesce(raw_app_meta_data->>'role', raw_app_meta_data->>'claims_admin', ''), ','), '(none)') from auth.users where raw_app_meta_data ? 'role' or raw_app_meta_data ? 'claims_admin'
union all select 'ALL users: identities without a matching user / users without identity', (select count(*) from auth.identities i where not exists (select 1 from auth.users x where x.id = i.user_id))::text || ' / ' || (select count(*) from auth.users x where not exists (select 1 from auth.identities i where i.user_id = x.id))::text
union all select 'users created since Aug with no identity (admin-API created?)', count(*)::text from u where not exists (select 1 from auth.identities i where i.user_id = u.id)
union all select 'sso providers / saml', (select count(*) from auth.sso_providers)::text
union all select 'mfa factors created since Aug', (select count(*) from auth.mfa_factors where created_at >= '2026-08-01')::text;
