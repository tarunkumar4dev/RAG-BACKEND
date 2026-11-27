# Backend public Supabase client + post-leak DB check (28 Nov 2026, read-only)

## 1. What uses the backend's "public" client, and does it work as anon?

Context:
- `get_supabase()` (`app/core/database.py`) uses `SUPABASE_ANON_KEY`.
- Vercel has no `SUPABASE_ANON_KEY`, so `config.py:50` falls back to `SUPABASE_KEY`. That is most likely the
  service_role key (`database.py:95` warning).

Live RLS/grants: `scripts/audit/17_*.sql`.

| Route / service (file:line) | What it does with the public client | Works as anon today? |
|---|---|---|
| `GET /chapters` (`test_generator.py:949`) | `SELECT` `ncert_chunks` | **Yes**: anon SELECT + policy `USING (true)` |
| `GET /ncert-questions` (`:1129`), `/ncert-questions/by-ids` (`:1687`), `/ncert-question-stats` fallback (`:1243`) | `SELECT` `ncert_questions` | **Yes**: same. The stats main path uses direct psycopg2, not this client |
| RAG retrieval: `rag_service._resolve_chapters` (`:69`, `:91`), `keyword_search` (`:193`) | `SELECT` `ncert_chunks` | **Yes** |
| Auth: `core/auth.py:107` `auth.get_user(token)` | GoTrue validates the *user's* JWT; any project key works as `apikey` | **Yes** |
| `GET /payment/plans` (`payment.py:176`) | `SELECT plans where is_active` | **Yes**: policy "Anyone can view plans" |
| `POST /quiz/create` (`test_generator.py:1592`) | **INSERT** into `quizzes`, **no auth**, `teacher_id` taken from the request body | **Broken either way**: table `quizzes` does not exist (500 today). If it existed, this unauthenticated write would only succeed on service_role. No frontend caller. **Delete the route.** |
| `get_supabase_for_user(jwt)` (`database.py:74`, every user-scoped route) | anon-key `apikey` + the user's JWT → role `authenticated` | Unchanged: RLS already applies |

Not the public client:
- `module_service.get_supabase()` builds its own client from `SUPABASE_SERVICE_KEY` (Modules, Worksheet storage),
  so it is service-role by design.
- Payment helpers receive `get_supabase_admin()` from their callers.

**Verdict:** no working route silently depends on service_role. Switching `SUPABASE_ANON_KEY` to the real anon key
should not break anything. Two caveats:
- `anon` has `statement_timeout=3s`. The current path (service_role via authenticator) gets 8 s. Today's NCERT
  reads are well under that (index scans, ≤1,000-row pages), but watch `/ncert-questions` search and
  `/ncert-question-stats` fallback latency after the switch.
- Test `/quiz/create` removal and the switch on a preview deployment first. No Vercel env was changed in this task.

## 2. Post-leak check — objects created after 1 Aug 2026 that our migrations don't explain

Method:
- Postgres stores no creation time for roles, functions, triggers or tables (`track_commit_timestamp` is off),
  and the Supabase CLI migration-history table is empty.
- So I inventoried 383 non-system objects (`18b`) and matched each name against the DDL in every commit of both
  repos plus untracked local SQL (`match_inventory.py`, about 59 MB of history).
- Unmatched objects were dated by OID order against tables whose first row has a known date (`18g`).
- Every function body, cron job and auth hook was scanned for exfiltration/persistence patterns (`18c`).
- `pg_stat_statements` (collecting since 23 Jul 2025) was mined for who ran what (`18i`–`18l`).

| Area | Result |
|---|---|
| Roles | `agent_ro` (our migration). `cli_login_postgres` = the Supabase CLI's temporary login role: created by the CLI's own `DO` block (257 statements in history), SCRAM password, `rolvaliduntil` 2026-11-24 (expired). Nothing else |
| Extensions | Supabase defaults + `vector`, `pg_trgm`, `pg_cron`, `pg_net`, `moddatetime`. `moddatetime` has no DDL in the repos; it is used by trigger `trg_profiles_updated_at` (pre-Aug by OID) |
| Event triggers / publications | Only Supabase defaults (`pgrst_*`, `issue_*`; `supabase_realtime`, `supabase_realtime_messages_publication`) |
| Cron jobs | `a4-checker-sweep`, `a4-orphan-upload-cleanup`: both ours |
| Vault secrets | `checker_job_run_url`, `checker_job_secret` (22 Oct, checker migration 26) |
| Function bodies | No backdoor patterns. The only flagged bodies are ours: `a4_checker_kick` (pg_net + vault), plus keyword false positives. The only URL host is our own project |
| Tables / functions / triggers without repo DDL | 96 names; 48 are dated before 1 Aug by OID. The 29 possibly-later ones form two coherent clusters: (a) the **Modules** feature (`modules`, `module_chunks`, `search_module_chunks`, `update_modules_updated_at`) + 2 institute helpers, created ≤ 22–26 Aug; (b) the **MSIT attendance / proctor / HOD** layer (`mark_attendance`, `edit_attendance`, `is_proctor_of`, `is_hod_of_dept`, `is_within_edit_window`, `fill_member_identity`, `get_section_*`, …), created ≤ 11 Oct. Bodies were read and are benign (`auth.uid()`-scoped checks, `updated_at`, vector search; `fill_member_identity` copies the member's own name/email). `pg_stat_statements` shows them created **as `postgres` from the SQL editor** with the team's own comments. The frontend `CLAUDE.md` names `attendance_sessions` / `batch_teachers` as pre-existing legacy tables (30 Sep) |
| GRANT history | One historical `grant select, insert, update, delete, truncate, references, trigger on contests, contest_questions, contest_attempts to anon, authenticated`. Live grants are now SELECT-only (later lockdown), so OK |
| COPY / dump / outbound HTTP by non-system roles | none |
| Auth users since 1 Aug | 127 (Aug 8, Sep 18, Oct 74, Nov 27); email 66 / google 53 / phone 8; no non-`authenticated` role, no super-admin, no custom app_metadata claims, no identity-less (admin-API) users, no SSO, no MFA. Busiest hour: 25 Oct 20:31–21:24 IST, 18 Google sign-ins + 1 email, all later signed in (same day the contest leaderboard shipped) |

### Unexplained / needs your confirmation
1. **`agent_ro` passwords sit in plain text in `pg_stat_statements`** (6 distinct values). Two come from the
   setup script's `DO` block; four are `alter role agent_ro password '…'` run as `supabase_admin` (dashboard).
   Anyone with stats access (`postgres`, `pg_read_all_stats`, `supabase_read_only_user`) can read them. **Rotate
   `agent_ro`** with a pre-hashed SCRAM verifier (a plain `ALTER ROLE … PASSWORD 'x'` is logged again), then run
   `select extensions.pg_stat_statements_reset();`. The postgres password reset went through the platform
   (SCRAM-hashed), so it is fine.
2. 4 `example.com` accounts (all signed in): test accounts? Who created them?
3. The 25 Oct burst: a live contest with students?
4. Bucket `contest-submissions` (created 2025-08-10, private): not referenced in either repo; still used?
5. Cluster (a) (Modules, Aug) and cluster (b) (MSIT attendance, Sep–Oct) were built in the SQL editor, not in
   migrations. Recommend exporting their DDL into migration files so the next audit can match them.

Limits: this cannot see changes to *existing* function bodies or policies made by a privileged attacker. It also
cannot see dashboard-level changes (API keys, auth settings, Edge Function deploys), and reads leave no trace beyond
`pg_stat_statements`. Check the Supabase audit log (Organization → Audit logs) for 1 Aug onward.
