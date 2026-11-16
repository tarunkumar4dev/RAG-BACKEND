# PR: a4ai Agent + Voice (contract 1.5) and the Gate 2 cost fixes

Branch `claude/youthful-archimedes-tw0p0s` → `main`. 62 commits, 188 files. Nothing in here is deployed yet.

## What this changes in PRODUCTION (`test-generator-backend`, served by Vercel)

These change behaviour for live users the moment `main` deploys:

| # | Change | Risk | Needs |
|---|---|---|---|
| 6 | Module summaries / OCR: one Gemini key, one call; no key rotation, no key prefixes in logs | none | — |
| 1 | `/worksheet/generate-direct` and `/modules/{id}/generate-worksheet` require a login token, count against the plan quota, 5/min per user, 320k-char input cap | **frontend must deploy first** (it now sends the token); the old UI gets 401 on worksheets | product PR |
| 4 | Paid-call cap: at most 3 provider calls per failing unit of work (paper batch, copy, agent turn); no top-up rounds; a paper that fails says so in Hindi/English with a 503 | none on the happy path; a paper with an unusable reply now fails fast instead of being retried up to 10× | — |
| 5 | Quota: reserve → consume/release, idempotency key on papers (resend = 409, not a second paper); voice counted on success | none until migration 32 is applied (falls back to the old check/record pair with one warning) | migration 32 (product repo), optional |
| 7 | Checker cost accounting uses the real gemini-3.8-flash list price (switches 1 Jan 2027); Gemini audio priced at the audio rate | accounting only | **delete** `CHECKER_PRICE_IN_PER_M_INR` / `CHECKER_PRICE_OUT_PER_M_INR` in Vercel |
| 8 | Tokens + feature + institute_id logged for papers (`AI_USAGE` log line, `tests.request_payload.ai_usage`) and worksheets | none | — |
| 10 | Dead paid paths deleted: `/api/v1/test-checker/*`, `/verify-answers`, `validation_service.py`, root `api_server.py` / `rag_system.py` / `module_service.py` / `fix_indent.py` | none (no caller) | — |

Not in this PR (separate branches, waiting for sign-off): `checker-bundle` (3.6-flash fallback, feedback reasons, job-run deadline) and `hotfix/gate2-security` (the same fixes cherry-picked onto `main` without the agent code — **merge either this PR or that hotfix, not both**).

## What this adds but does NOT deploy (`agent/`)

The agent is a separate FastAPI app (`uvicorn agent.api:app`), not served by `api/index.py`, so merging changes nothing live:

- Agent API v2/v3 (sessions, history, suggestions, cancel, exports, Postgres store, rate limits, metrics), voice layer (Sarvam STT/TTS, Gemini backup, entity repair, kill switch), and now **1.5**: the SPEECH line is spoken while the model writes it (`audio` events), barge-in (`cancel_current`), usage counts in the store table `voice_usage` (shared across processes), institute + feature on every cost record, `GET /agent/admin/voice-usage`, keep-alive Sarvam client + startup warmup, prompt-cache prefix test.
- Deploy guide: `docs/VOICE_DEPLOY.md`, `agent/README.md`. SQL for Supabase (not applied): `agent/sql/agent_store_supabase.sql` (incl. `agent.voice_usage`).

## Deploy order

1. Merge + deploy the **product** PR first (worksheet auth header, idempotency key, contract 1.5 types).
2. Set Vercel env: delete `CHECKER_PRICE_IN_PER_M_INR` and `CHECKER_PRICE_OUT_PER_M_INR`; check `LLM_PROVIDER` is **not** `deepseek` (the local `.env` has it; the DeepSeek account is at 402).
3. Merge this PR → Vercel deploys `main`.
4. Later, when you want it: apply migration 32 (product repo), fill `plans.voice_monthly_limit`.

## Tests

- `test-generator-backend`: 389 passed, 1 skipped, 1 xfail (strict, documents the remaining model-id spread). The CORS preflight test fails only in the full run and passes alone — same on `main`.
- `tests/agent`: 407 passed, 13 skipped.
- Paid calls: none in CI; `scripts/cache_probe.py` and the 40-utterance eval need keys.

## Rollback

`git revert -m 1 <merge commit>` on `main`, or redeploy the previous Vercel deployment. No migration is applied by this PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
