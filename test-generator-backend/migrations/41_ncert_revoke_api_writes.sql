-- 41_ncert_revoke_api_writes.sql
-- P0 security: the public API roles (anon, authenticated) must only READ the NCERT tables.
--
-- Before (live, 27 Nov 2026, scripts/audit/15_ncert_grants.sql):
--   ncert_questions  anon, authenticated : REFERENCES, SELECT, TRIGGER, TRUNCATE
--   ncert_chunks     anon, authenticated : REFERENCES, SELECT, TRIGGER, TRUNCATE
--   (INSERT / UPDATE / DELETE were already revoked; service_role and postgres keep ALL)
-- After this migration:
--   ncert_questions  anon, authenticated : REFERENCES, SELECT, TRIGGER
--   ncert_chunks     anon, authenticated : REFERENCES, SELECT, TRIGGER
--
-- Why: RLS does not apply to TRUNCATE, so the existing "SELECT USING (true)" policy gives no protection against
-- it. A plain TRUNCATE ncert_questions fails on the FKs from ncert_practice_marks / ncert_question_fixes, but
-- TRUNCATE ... CASCADE would empty those tables too, i.e. wipe students' practice marks.
-- PostgREST does not expose TRUNCATE, but any SECURITY INVOKER function callable by anon would.
--
-- Safe to re-run: REVOKE of a privilege that is not held is a no-op. SELECT is untouched, so the Test Builder,
-- /chapters, RAG retrieval and the student portal keep working. The backend writes with service_role (unchanged).
-- Rollback: 41_ncert_revoke_api_writes_rollback.sql.

begin;

revoke truncate, delete, insert, update on table public.ncert_questions from anon, authenticated;
revoke truncate, delete, insert, update on table public.ncert_chunks   from anon, authenticated;

-- OPTIONAL (recommended, not requested): TRIGGER and REFERENCES are not needed for reads either.
-- TRIGGER lets a role attach triggers, REFERENCES lets it create foreign keys pointing at these tables.
-- Uncomment to leave anon/authenticated with SELECT only. Also add the matching GRANT lines to the rollback.
-- revoke trigger, references on table public.ncert_questions from anon, authenticated;
-- revoke trigger, references on table public.ncert_chunks   from anon, authenticated;

-- Verify inside the transaction; aborts (and rolls back) if anything is still writable or SELECT was lost.
do $$
declare r record;
begin
  for r in select t.tbl, g.role
           from (values ('public.ncert_questions'), ('public.ncert_chunks')) t(tbl),
                (values ('anon'), ('authenticated')) g(role)
  loop
    if not has_table_privilege(r.role, r.tbl, 'SELECT') then
      raise exception '% lost SELECT on %', r.role, r.tbl;
    end if;
    if has_table_privilege(r.role, r.tbl, 'TRUNCATE') or has_table_privilege(r.role, r.tbl, 'DELETE')
       or has_table_privilege(r.role, r.tbl, 'INSERT') or has_table_privilege(r.role, r.tbl, 'UPDATE') then
      raise exception '% can still write to % (granted via PUBLIC or a parent role?)', r.role, r.tbl;
    end if;
  end loop;
end $$;

commit;

-- NOT part of this migration (separate decision): new tables in schema public are created with ALL privileges for
-- anon and authenticated (pg_default_acl for owners postgres and supabase_admin). Any new PYQ table will start out
-- with DELETE/TRUNCATE for the API roles unless its migration revokes them, or the defaults are changed with
--   alter default privileges for role postgres in schema public revoke truncate, delete, insert, update on tables from anon, authenticated;
-- which affects every future table (check the frontend migrations first — some may rely on the defaults + RLS).
