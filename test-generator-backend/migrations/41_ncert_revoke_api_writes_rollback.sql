-- 41_ncert_revoke_api_writes_rollback.sql — restores the grants exactly as they were before migration 41
-- (anon/authenticated: REFERENCES, SELECT, TRIGGER, TRUNCATE; INSERT/UPDATE/DELETE were already absent).
begin;
grant truncate, trigger, references on table public.ncert_questions to anon, authenticated;
grant truncate, trigger, references on table public.ncert_chunks   to anon, authenticated;
commit;
