-- 41_ncert_revoke_api_writes.sql
-- P0 security: the public API roles (anon, authenticated) may only READ the NCERT tables.
--
-- Before (live, 27 Nov 2026, scripts/audit/15_ncert_grants.sql):
--   ncert_questions  anon, authenticated : REFERENCES, SELECT, TRIGGER, TRUNCATE
--   ncert_chunks     anon, authenticated : REFERENCES, SELECT, TRIGGER, TRUNCATE
--   (INSERT / UPDATE / DELETE were already revoked; service_role and postgres keep ALL)
-- After this migration:
--   ncert_questions  anon, authenticated : SELECT
--   ncert_chunks     anon, authenticated : SELECT
--
-- Why: RLS does not apply to TRUNCATE, so the "SELECT USING (true)" policy gives no protection against it. A plain
-- TRUNCATE ncert_questions fails on the FKs from ncert_practice_marks / ncert_question_fixes, but
-- TRUNCATE ... CASCADE would empty those tables too, i.e. wipe students' practice marks. TRIGGER would let the role
-- attach triggers, REFERENCES would let it create foreign keys pointing at these tables; reads need neither.
-- PostgREST exposes none of these, but any SECURITY INVOKER function callable by anon would.
--
-- Safe to re-run: REVOKE of a privilege that is not held is a no-op. SELECT is untouched, so the Test Builder,
-- /chapters, RAG retrieval and the student portal keep working. The backend writes with service_role (unchanged).
-- Rollback: 41_ncert_revoke_api_writes_rollback.sql.

begin;

revoke insert, update, delete, truncate, trigger, references on table public.ncert_questions from anon, authenticated;
revoke insert, update, delete, truncate, trigger, references on table public.ncert_chunks   from anon, authenticated;

-- Verify inside the transaction; aborts (and rolls back) if anything but SELECT is still held or SELECT was lost.
do $$
declare r record; p text;
begin
  for r in select t.tbl, g.role
           from (values ('public.ncert_questions'), ('public.ncert_chunks')) t(tbl),
                (values ('anon'), ('authenticated')) g(role)
  loop
    if not has_table_privilege(r.role, r.tbl, 'SELECT') then
      raise exception '% lost SELECT on %', r.role, r.tbl;
    end if;
    foreach p in array array['INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'TRIGGER', 'REFERENCES'] loop
      if has_table_privilege(r.role, r.tbl, p) then
        raise exception '% still has % on % (granted via PUBLIC or a parent role?)', r.role, p, r.tbl;
      end if;
    end loop;
  end loop;
end $$;

commit;

-- New tables: see 42_default_privileges_read_only.sql.
