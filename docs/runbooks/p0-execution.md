# P0 execution runbook (credentials → grants → anon key → dedupe)

Do the steps in order; don't start a step until the previous one verifies. "SQL editor" = Supabase Dashboard → SQL
Editor (runs as `postgres`). Every SQL file named here is on branch `sec/p0-credentials`. Checks marked *(RO)* are
read-only, e.g. `npx supabase db query --remote --read-only -f <file>`.

## 1. Credential rotation (you)
Details for each credential: `docs/security/credential-inventory-2026-11-27.md`.

| # | Do | Where | Verify | Undo |
|---|---|---|---|---|
| 1a | Reset the DB password | Dashboard → Project Settings → Database → Reset database password | Login works with the new password | Reset again |
| 1b | Put the new DB password into `DATABASE_URL`, `DB_PASSWORD`, `DATABASE_PASSWORD` (Prod, Preview, Dev), then **Redeploy** prod | Vercel → RAG-BACKEND → Settings → Environment Variables, then Deployments → Redeploy | `GET /api/v1/test-generator/ncert-question-stats?subject=Science&class_grade=10` is fast (<1 s warm). A Worksheet/Modules action works | Restore the old values and redeploy (only works until 1a is redone) |
| 1c | Revoke the 2 leaked Gemini keys. If either is still `GEMINI_API_KEY` / `CHECKER_GEMINI_API_KEY`, create a new key, update Vercel, redeploy **first** | Google AI Studio / Cloud Console → Credentials | One AI paper + one copy check succeed | New key (an old key can't be un-revoked) |
| 1d | Regenerate the Razorpay **test** keys | Razorpay Dashboard (Test mode) | Test checkout locally | n/a |
| 1e | If Vercel `WHATSAPP_VERIFY_TOKEN` equals the old hardcoded fallback: change it in Vercel **and** in Meta App → WhatsApp → Webhook at the same time | Vercel + Meta | Meta "Verify and save" succeeds | Set both back |
| 1f | Rotate `agent_ro`: run `ALTER ROLE agent_ro PASSWORD '<SCRAM verifier>';` (given in chat, never saved to a file), **then** `select extensions.pg_stat_statements_reset();` | SQL editor | *(RO)* `scripts/audit/18l_secret_literals_in_stats.sql` returns no `agent_ro` rows; the agent connects with the new password | Re-run with a new verifier |
| 1g | Delete `LOADTEST_BYPASS_TOKEN` (Preview) | Vercel env | Gone from the list | n/a |

## 2. Migration 41: NCERT tables SELECT-only for anon/authenticated (you, SQL editor)
- **Run:** `test-generator-backend/migrations/41_ncert_revoke_api_writes.sql`.
- **Expect:** `COMMIT`. The built-in check raises an error and rolls back if anything except SELECT remains.
- **Verify** *(RO)* with `scripts/audit/15_ncert_grants.sql`: anon/authenticated show `SELECT` only on both tables.
  The Test Builder still lists questions.
- **Undo:** `41_ncert_revoke_api_writes_rollback.sql`.

## 3. Migration 42: new tables default to SELECT-only (you, SQL editor)
- **Run:** `42_default_privileges_read_only.sql`.
- **Expect:** `COMMIT`, maybe with the NOTICE "skipped supabase_admin default privileges" (fine: tables are created
  by `postgres`).
- **Verify** *(RO)* with `15_ncert_grants.sql`: the `DEFAULT (postgres, public, r)` rows show `SELECT` only.
- **Undo:** `42_default_privileges_read_only_rollback.sql`.
- **From now on:** see `test-generator-backend/migrations/README.md` (new writable tables need an explicit GRANT).

## 4. Backend `SUPABASE_ANON_KEY`: Preview first (you, Vercel)
- **Do:**
  - Add `SUPABASE_ANON_KEY` = the project's **anon** key, Preview only (Dashboard → Project Settings → API).
  - Redeploy the `sec/p0-credentials` preview (it also removes `/quiz/create`).
- **Smoke test** the preview URL with `curl -i` (expect 200 and `"ok": true` unless noted). Compare
  `x-process-time-ms` with prod:
  1. `GET /health`
  2. `GET /api/v1/test-generator/chapters?subject=Science&class_grade=10`
  3. `GET /api/v1/test-generator/ncert-question-stats?subject=Science&class_grade=10`
  4. `GET /api/v1/test-generator/ncert-questions?subject=Science&class_grade=10&chapter=Electricity&limit=30`
  5. same + `&search=current`, plus `subject=Physics&class_grade=11` without a chapter. This is the heaviest
     anon query; it must finish well under anon's 3 s statement timeout.
  6. `GET /api/v1/test-generator/ncert-questions/by-ids?ids=<3 ids from 4>`
  7. `GET /api/v1/payment/plans`
  8. `GET /api/v1/payment/plan-status/<your user id>` with `Authorization: Bearer <your teacher JWT>`. This
     checks token verification on the anon key.
  9. One Smart Assignment page, which uses the user-JWT client.
  10. `POST /api/v1/test-generator/quiz/create` → **404**.
  11. Optional, costs one paper: one AI generation, which exercises RAG over `ncert_chunks`.
- **Watch for:** any 401/403/500, `ok:false`, or `canceling statement due to statement timeout` in the Vercel logs.
- **Undo:** remove the variable and redeploy (the backend falls back to `SUPABASE_KEY`).

## 5. Same for Production (you, Vercel)
Merge the PR first (it contains the `/quiz/create` removal). Then add `SUPABASE_ANON_KEY` for Production, redeploy,
and repeat smoke tests 1–10 on the production host. The "identical to the anon key" warning should stop appearing in
the logs. Undo as in step 4.

## 6. Dedupe (you, SQL editor; Nitin GO before 1b and before 3). Plan: `docs/audits/ncert-dedupe-cleanup-plan.md`

| Step | Run (`scripts/cleanup/ncert_dedupe_exact/`) | Expected | Verify *(RO)* | Undo |
|---|---|---|---|---|
| 0 | `scripts/audit/16g_dedupe_dryrun_normalised.sql` the same day | ≈54,902 deleted, 0 practice marks, 0 collisions | Numbers close to the plan; investigate any marks > 0 | n/a |
| 1 | `01_backup.sql`, plus an offline `pg_dump` of the 3 tables | Fingerprint rows printed | `select count(*) from a4_backup_ncert_dedupe.ncert_questions` = live count | `drop schema a4_backup_ncert_dedupe cascade` (only before 1b) |
| 1b | `01b_normalize_chapters.sql` | 4,967 rows rewritten, 425 chapters | `select count(distinct (class_grade, subject, chapter)) from ncert_questions` = 425; no `-I`/`_` subjects; builder chapter lists look right | `99b_rollback_restore.sql` |
| 2 | `02_build_map.sql`, then export the map to CSV and spot-check about 50 groups | ≈54,902 map rows | Map count ≈ dry run; no canonical is itself a duplicate (the script checks) | `drop table a4_backup_ncert_dedupe.dedupe_map` |
| 3 | `03_apply.sql` (one transaction, seconds) | `COMMIT`; any failed check = full rollback | `ncert_questions` count = backup − map; `scripts/audit/04a_exact_dupes_summary.sql` → 0 extra rows; `ncert_question_redirects` count = map count; stats endpoint totals ≈ pre-run snapshot | `99b_rollback_restore.sql` |
| 4 | `04_unique_index.sql`: the `CREATE UNIQUE INDEX CONCURRENTLY` line **alone** (not in a transaction), then its check query | `valid = true` | Inserting a copy fails with a unique violation | `99a_rollback_drop_unique_index.sql` (alone), then `99b` if needed |

Rollback order after step 4: `99a` (alone), then `99b`. `99b` restores the rows, the subject/chapter names, marks,
fix log and `duplicate_of`, and drops the redirects. Marks recorded after step 1 need a manual merge.
