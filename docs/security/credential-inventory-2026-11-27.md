# Credential inventory — both a4ai repos, all refs, full history (27 Nov 2026)

How: every commit on every ref (`git log --all -p`, incl. stashes) of `RAG-BACKEND` (395 commits) and `a4ai`
(frontend), scanned for JWTs, `sb_secret_`, Google `AIza…`, `sk-…`, Razorpay, Meta tokens, AWS keys, Postgres URIs,
password/secret literals, `getenv(…, "<default>")` and unquoted `NAME=value` env lines. Values are never printed:
each credential is identified by `sha256[:10]` and length. Both repos were public until 22 Oct 2026, so anything on
a pushed branch must be treated as exposed (forks of a public repo survive the switch to private).

## Rotate

| # | Credential | Where (file — commits) | On GitHub? | What breaks when rotated |
|---|---|---|---|---|
| A | **Supabase Postgres password, role `postgres`** (direct host `db.<ref>.supabase.co:5432`). fp `c6751e49e4`, len 17 | backend `check_table.py:7` — `e1c1443` (2026-02-03), `5eaefc2` (2026-06-20). Still in `main`'s tree; removed on branch `sec/p0-credentials` | yes (49 remote branches) | Backend Vercel env `DATABASE_URL` (Prod/Preview/Dev; holds the password), `DB_PASSWORD` (Prod/Preview), `DATABASE_PASSWORD` (Prod/Preview/Dev) → update all, then **redeploy** (env changes apply only to new deployments). Until then: Modules + Worksheet fail (direct Postgres only); `/ncert-question-stats` falls back to the slower Supabase-client path. Local: backend `.env`, `ncert-ingestion/.env`, DB GUI clients, `supabase db push/dump` (prompt for the DB password). **Not affected:** anon/service_role API keys, frontend, Edge Functions, `supabase db query --remote`, the `agent_ro` role (own password). Rotate in Dashboard → Project Settings → Database → Reset database password |
| A2 | Older DB passwords (same role, presumably) | backend `test-generator-backend/app/core/db_pool.py` getenv default, len 18 — added `c206252` (2026-09-23), removed `5d1be9e` / `b2ddba8`; frontend `scripts/ncert-ingestion/rag_system.py` getenv default, len 30, fp `e26bfb837f` — `95a9c9c` (2025-12-15), not in `main` now | yes | Nothing extra: the 18-char one failed auth on 22 Oct; both are covered by rotating A |
| B | **Two Google / Gemini API keys** (`AIza…`), fp `abaeae5bf0`, `4ee7445a17` | backend `work/list_models.py`, `work/test_checker_pdf.py` — `5eaefc2` (2026-06-20); removed from the tree in `94b2b73` (2026-11-21), still in history | yes | Only if one of them is still in use: compare with Vercel `GEMINI_API_KEY` (all envs) and `CHECKER_GEMINI_API_KEY` (Prod) and local `.env`s. If it is: generation / checker / voice stop until the new key is set and redeployed. Revoke in Google AI Studio or Cloud Console → Credentials |
| C | **Razorpay TEST key secrets ×2**, fp `799c1c14fa`, `30c5f9bc85` (+ their `rzp_test_` key ids) | frontend `Backend/server.js` — `67e1983` (2025-09-11); `Backend/.env.local` — `cce969b` (2025-10-04). Both files gone from `main` | yes | Test mode only (no money). Breaks local/dev test checkouts using them. Regenerate in Razorpay Dashboard → Test mode → API keys. Confirm Vercel `RAZORPAY_KEY_SECRET` is a different (live) secret. Note: one of these values was printed in plain text in Claude's local tool output during this scan (a masking gap); it left the machine only in that session transcript |
| A3 | **`agent_ro` DB role password** (read-only agent role) | Not in git. 6 plaintext values in the live DB's `pg_stat_statements` (setup `DO` block + `alter role … password` from the dashboard); found 28 Nov | n/a (readable by any role with stats access) | Agent real-DB mode (`AGENT_REAL_DB_URL`) until updated. Rotate with a pre-hashed SCRAM verifier, then `select extensions.pg_stat_statements_reset();`. Details in `public-client-and-post-leak-check-2026-11-28.md`. 28 Nov: the first generated replacement was shown in chat, copied elsewhere and is **burned** (never applied, so never live); a fresh one was generated. Its plain value lives only in a local file outside every repo (path in `docs/runbooks/p0-execution.md` 1f), and its verifier was given in chat |
| D | WhatsApp webhook **verify token** (hardcoded fallback), len 11 | backend `test-generator-backend/app/api/v1/endpoints/whatsapp.py` — `19be1f6` (2026-09-29); no longer in `main` | yes | Only if Vercel `WHATSAPP_VERIFY_TOKEN` still equals that fallback: change it in Vercel **and** in Meta App → WhatsApp → Configuration at the same time (only used for the webhook subscription handshake). Low severity |

## Local only (never pushed) — clean up, rotation optional

| # | Credential | Where | Action |
|---|---|---|---|
| E | `VERCEL_PROTECTION_BYPASS` (len 32), `LOADTEST_BYPASS_TOKEN` (len 43) | backend `loadtest/.env.loadtest` — `4495a56` (2026-10-22), local ref `wip-loadtest` only (with a committed `loadtest/.venv`) | `LOADTEST_BYPASS_TOKEN` is **still set on Vercel Preview (feat/loadtest)** — delete it there. Regenerate the Vercel protection-bypass secret if it is not used elsewhere. Don't push `wip-loadtest` |

## Not secrets (checked)
Test fixtures: `test-generator-backend/tests/unit/test_fanout.py` `SECRET`, `tests/agent/test_real_mode.py` +
`scripts/agent_fixture_env.py` JWT secret, `tests/agent/real_fixture/build.py` `FIXTURE_PASSWORD` (local pgserver),
`tests/agent/test_log_scrub.py` sample strings. `scripts/run_agent_real.bat` prompts for the password (batch
variable). Frontend anon JWT (`role=anon`, public by design) in `.env`, `.env.local`,
`src/integrations/supabase/client.ts` history. Razorpay key **ids** (public). Library constants inside the
committed `loadtest/.venv`. Getenv defaults that are model names, env-var names or hosts.

## Related findings (no leak, but fix)
- **Frontend production bundle is clean**: 309 chunks from www.a4ai.in contain no `service_role` / `sb_secret_`; the
  only embedded key decodes to `role=anon`. Service-role is used only in Edge Functions (`send-whatsapp-otp`,
  `submit-contact`) via `Deno.env`.
- **Backend probably runs its "anon" client on the service_role key.** Vercel has no `SUPABASE_ANON_KEY`;
  `config.py:50` falls back to `SUPABASE_KEY`, and `database.py:95` says production uses one key for both clients.
  Payments (admin writes) work, so that key is most likely service_role → `get_supabase()` bypasses RLS. Fix: add
  `SUPABASE_ANON_KEY` = the anon key in Vercel (all envs), redeploy, check that NCERT/chapters/plans reads still work.
- History purge (planned 22 Oct, not run): rotate first, then purge with `git filter-repo`, then force-push and
  ask GitHub support to drop cached views; forks must be checked separately.
