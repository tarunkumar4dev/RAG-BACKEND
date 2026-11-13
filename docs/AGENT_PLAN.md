# a4ai Agent — plan (5 Nov 2026, timeboxed 22 min)

## Recon (file:line)
- Auth/session (backend): `test-generator-backend/app/core/auth.py:89` verify_token, `:139` require_user (Supabase JWT → AuthUser).
- Auth/session (frontend): `a4ai/src/context/AccessProvider.tsx` → `get_my_access()` RPC (`a4ai/supabase/migrations/01_access_helpers.sql`).
- Attendance tables (live): `class_sessions`, `attendance_records` (`a4ai/supabase/tests/upload_flow/01_baseline_from_audit.sql:173,192`),
  scope tables `batches:44`, `institute_members:62`, `teaching_assignments:106`, `students:136`.
  RPCs: `get_monthly_attendance` (`07_security_hardening.sql:120`), save_class_attendance, get_proctor_section_*.
- Marks/tests: no single marks table on live — grades live in `submissions.grade` (11_assignment_ai_check.sql) and
  `check_sheets` (12_copy_checker.sql). Agent seed adds `tests` + `test_marks` (local only).
- NCERT RAG: `app/services/rag_service.py:307` retrieve_context(chapters, topics, subject, class_grade).
- Gemini generation: `app/services/test_generator_service.py:2694` generate_questions(request, context_chunks).
- PDF export: `app/services/export_service.py:1593` generate_pdf, `:3223` generate_answer_key_pdf.

## Provider check (minute 1)
- Anthropic `claude-sonnet-5-5` 1-token call: **200 OK**. Gemini `gemini-2.5-flash` 1-token call: **200 OK**.
- Both behind `agent/llm.py` LLMProvider (AnthropicProvider default, GeminiProvider fallback). Key env var name read
  from config (`AGENT_ANTHROPIC_KEY_ENV`, default `A4AI_ANTHROPIC_API_KEY`). Keys never logged.

## Design
- New package `RAG-BACKEND/agent/` (no edits to existing endpoints). Mount with `app.include_router(agent.api.router)` later.
- Contract frozen in `agent/schemas.py` (AgentEvent discriminated union).
- Data: local Postgres seed `agent/seed/seed.sql` (schema `agent_demo`, fake data). Read tools use a read-only
  connection (`default_transaction_read_only=on`).
- Security: SessionCtx frozen from JWT/session dependency; tools are `tool(ctx, args)`; ctx never in tool schemas.
- Verification gate deterministic (sympy + rules) + independent solver pass without key; rubric checks; manual_review.
- Approval: single-use token bound to session + sha256(args), 15 min TTL, replay returns the stored result.

## Status (end of session, 5 Nov)
- Provider used for the whole session: **Anthropic `claude-sonnet-5-5`** (ping OK). Gemini path implemented
  (`GeminiProvider`); only its schema sanitiser is unit-tested — `GeminiProvider.complete` was not exercised in this
  session (no fallback was needed: Anthropic worked for the demo and evals).
- Files: `agent/{schemas,config,llm,core,verify,approval,trace,auth,api,export,db,text}.py`, `agent/tools/{insights,paper,registry}.py`,
  `agent/seed/*.sql`, `tests/agent/*`, `evals/*`, frontend `a4ai/src/{pages,components,lib}/agent/*` (not routed).
- Trig demo: 3 live runs; verifier bugs found by the demo fixed with regression tests (see `evals/sample_trigonometry_review.md`).

---
# Real data + premium UI (6 Nov 2026)

## Real schema map (source: `a4ai/supabase/tests/upload_flow/01_baseline_from_audit.sql` = live audit, + migrations)
| Table | Columns the agent uses | Notes |
|---|---|---|
| institutes | id, name, owner_id | admin = `owner_id = auth.uid()` (a4_is_inst_admin) |
| institute_members | id, institute_id, user_id, role, status, department_id, section_id, user_name | role ∈ admin/institute (admin), hod, proctor, teacher; only `status='active'` |
| teacher_profiles | id (= auth user id), full_name | also phone/email/role/plan_id — never read |
| departments | id, institute_id, name, code, hod_id | HOD scope = `institute_members.role='hod'` + `department_id` (a4_my_hod_dept_ids) |
| sections | id, department_id, name, year | no institute_id: reach via departments |
| batches | id, institute_id, name, department_id, section_id, is_lab (05) | main batch "ECE-2", lab batches "ECE-2 A/B/C" |
| students | id, institute_id, batch_id, section_id, department_id, lab_batch_id (05), name, roll_no, is_active | **never read**: parent_name, parent_phone, phone (06a), email, gender, user_id |
| subjects | id, institute_id, department_id, name, code | |
| teaching_assignments | teacher_id (= auth user id), batch_id, subject_id, is_active | subject-teacher scope |
| timetable_slots | teacher_id (= auth user id), batch_id, subject_id, is_active | also counts as teaching scope (a4_my_teaching_batch_ids) |
| proctor_assignments | teacher_id (= **institute_members.id**), section_id, department_id, is_active | proctor scope = all batches of the section |
| class_sessions | id, institute_id, batch_id, subject_id, teacher_id, session_date, status | held class = `status='conducted'` (get_proctor_section_status) |
| attendance_records | session_id, student_id, status, institute_id | status ∈ present/absent/late/leave; % = (present+late)/marked (same as get_monthly_attendance) |
| tests | id, teacher_id, exam_title, class_grade, subject, total_marks, total_questions, status, created_at | **no institute_id** → scoped through institute_members of the creator |
| assignments / submissions | assignments(id, institute_id, batch_id, created_by, title, max_marks, deadline); submissions(assignment_id, student_id, grade, status, graded_at) | marks source 1 (grade numeric after 11) |
| check_sessions / check_sheets | sessions(id, teacher_id, title, test_id, batch_id, subject, total_marks, created_at); sheets(session_id, student_id, final_marks, max_marks, status) | marks source 2 (status checked/approved, final_marks) |
How the app identifies the user: Supabase Auth JWT (`sub` = auth.users.id); frontend `AccessProvider` → `get_my_access()` RPC;
backend `app/core/auth.py:verify_token` calls Supabase `/auth/v1/user`.

## Design system map (a4ai)
- Tokens: shadcn HSL CSS variables in `src/index.css` (`--background`, `--primary` 265 80% 52% purple, `--muted`, `--border`,
  `--radius` 0.5rem, sidebar tokens); dark mode = `.dark` class on `<html>` via `src/context/ThemeContext.tsx`. No success/
  warning tokens → use `emerald-*`/`amber-*` with `dark:` pairs only for the data-source pill/verification (documented exception).
- Fonts: Plus Jakarta Sans self-hosted (`font-halenoir` maps to it), body `font-sans`.
- Components: full shadcn set in `src/components/ui` (button, card, badge, sheet, popover, dropdown-menu, tooltip, table,
  skeleton, scroll-area, collapsible, chart (recharts wrapper), sonner/toaster, avatar, separator, textarea, tabs).
- Libraries present: recharts 2.15, lucide-react, framer-motion (but lite mode: `src/styles/lite.css` + `MotionConfig
  reducedMotion="always"` → no animations; transitions allowed only where lite.css permits), sonner, katex, dompurify.
  Missing: react-markdown, remark-gfm (to add).
- Staff shell: `src/pages/TeacherDashboardPage.tsx` (own `<aside>` sidebar, `navItems` at ~L1369, `?tab=` routing at
  ~L1263, tab bodies in a `<Suspense>`). Admin/HOD also may use /dashboard (RoleRoute admin/hod/proctor/teacher).
- Role: `useAccess().access.primary_role` (admin | hod | proctor | teacher | none); students never get staff access.

## Decisions
1. **Real mode = `AGENT_REAL_DB_MODE=1`**, mutually exclusive with `AGENT_DEV_MODE=1` (refuse to start if both).
2. **JWT verified locally**: JWKS (`SUPABASE_URL/auth/v1/.well-known/jwks.json`, ES256/RS256) or HS256 with
   `SUPABASE_JWT_SECRET`; checks `exp`, `iss = SUPABASE_URL/auth/v1`, `aud = authenticated`, `role = authenticated`.
   Role/institute/department come only from DB rows for `sub`.
3. **Scope** (mirrors `01_access_helpers.sql`): admin (owner or member admin/institute) = institute; HOD = own department
   (+ own classes); proctor = proctor sections' batches (+ own classes); teacher = teaching_assignments ∪ timetable_slots
   batches (subject-level for sessions they teach). Multi-role = union.
4. **Marks** = graded `submissions.grade` (assignments) ∪ `check_sheets.final_marks` (check copies). No unified marks table exists.
5. **Real DB access** only through `agent_ro` (SELECT on 16 tables, no auth/storage/payments), `default_transaction_read_only`
   + `statement_timeout=8s` per connection, startup probe refuses if any INSERT/UPDATE/DELETE privilege exists.
6. **Writes in real mode**: save_draft / finalize_test / assign_to_section return "Abhi read-only mode hai"; drafts stay in
   memory for the preview only. Agent sessions/messages are kept in-process (no write to Supabase) in real mode.
7. **/agent UI** lives inside the existing staff shell as `/dashboard?tab=agent` (one nav item + one tab body in
   TeacherDashboardPage = the "layout slot"); `/agent` redirects there.

## API contract additions (1.1, backward compatible)
- `stat_card.period`, `stat_card.delta` (optional strings; delta only when a tool computed the previous period).
- `thinking_status` per tool: `key = "tool.<tool_name>"`, `params = {"phase": "start"}` then
  `{"phase": "done", "ms": <int>, "ok": <bool>}`. Paper pipeline keeps `status.generating|verifying|regenerating`.
- `GET /agent/meta` → `{"data_source": "live"|"demo", "read_only": bool, "role": "admin"|"hod"|"teacher",
  "first_name": str|null, "institute_name": str|null}`.
- `GET /agent/sessions` → `{"sessions": [{"session_id", "title", "created_at", "updated_at"}]}` (own sessions, newest first,
  max 50; title = first user message, 60 chars).
- `POST /agent/chat` in real mode never writes to Supabase; sessions are kept in the agent process (lost on restart).

## Status (6 Nov 2026)
- Phase 0 done (maps above). Phase 1 done: `AGENT_REAL_DB_MODE`, `agent/jwt_auth.py`, `agent/real_identity.py`,
  `agent/db_real.py`, `agent/sql/create_agent_ro_role.sql` (+ drop), `agent/real/schema_adapter.py`, live-mode write
  block, contract 1.1. Tests: 98 passed + 1 skipped (system PG), 99 passed (pgserver); 19 real-mode tests on the
  schema-faithful fixture `tests/agent/real_fixture/` (values in `EXPECTED.md`).
- Phase 2 done (a4ai, see `a4ai/docs/AGENT_UI_NOTES.md`): workspace at `/dashboard?tab=agent` + "Ask a4ai" sheet;
  vitest 327, Playwright agent spec 4, 28 screenshots in `a4ai/docs/screenshots/`.
- Decision 8: admins/HODs are redirected away from /dashboard by the existing home-route rule, so for them the agent is
  the "Ask a4ai" sheet on /institute and /hod. Letting them open the workspace needs a one-line change in that
  redirect (outside the allowed mount points) — left for the owner to approve.
- Not run here: anything against production Supabase (role script, JWKS fetch, pooler connection, real rows).

---
# API v2 + agent store (9 Nov 2026)

## Recon (before)
- Events: 12-type union (contract 1.1). Real mode kept sessions / LLM history / events / drafts in `MemoryStore`
  (process memory, lost on restart); seed mode wrote `agent_sessions` + `agent_messages` in the local demo Postgres.
- No titles editable, no pins, no query history, no feedback, no cancel. `GET /agent/sessions` = `{sessions: [...]}`.

## Decisions
9. **Contract v2 is additive (1.2)**: `session` event (seq 0), `final.cancelled`, `clarify.chips[].hint`; `ChatRequest`
   gains `regenerate_from_seq` / `edit_of_seq`. `GET /agent/sessions` returns `items` + `next_cursor` **and** the 1.1
   `sessions` key (deprecated alias) so the current UI keeps working. Spec: `docs/AGENT_API_V2.md`.
10. **One agent store for both modes** (`agent/store.py`): SQLite `agent/.devdata/agent_store.db` (WAL), one
    connection behind a lock. The "memory" implementation is the same SQL on `:memory:` (tests) so both behave
    identically. Every row carries user_id + institute_id; every statement filters by both. Seed mode still inserts
    the local demo `agent_sessions` row (drafts / approvals reference it) but no longer writes `agent_messages`.
11. **History replay**: each assistant message stores its exact event list (UI replay) and its provider-neutral LLM
    messages (`llm`), so the next turn's history is rebuilt from the store. A cancelled turn drops dangling tool calls.
12. **Regenerate / edit truncate**: delete from the target seq onward (feedback too), new messages get max+1.
13. **Cancel**: process-local registry keyed by request_id + owner; LLM calls run in the worker pool when a cancel
    event exists so the stream can stop within ~0.2 s (the provider call itself is abandoned, not interrupted).
14. **Privacy**: no endpoint logs content; uvicorn access lines for `/agent/*` lose their query string
    (`RedactAgentQueryStrings`); traces unchanged (ids, tool names, arg keys).
15. **Legacy `GET /agent/sessions/{id}`** is owner-only for store sessions (1.1 let same-institute admins read real-mode
    sessions). The seed Postgres fallback keeps the 1.1 admin rule for old demo rows.
16. Real-mode paper drafts stay in process memory (`DRAFTS`), never on disk: they are previews of a read-only mode.
17. **Identity cache** (`AGENT_IDENTITY_CACHE_S`, default 60 s): building `SessionCtx` is a DB round trip (~60 ms
    locally) and alone breaks the suggestions p95 budget. Tokens are still verified on every request; refusals are
    never cached; a removed staff member keeps access for at most the TTL. Set `0` to disable.
18. **Tool routing**: `my_scope` for "meri classes / mere students / my batches"; `section_comparison` for
    "which class has the lowest/highest attendance" (never `low_attendance_list`). Ties are reported, not broken:
    `lowest` / `highest` list every tied class. Batches with no data get rank null and are listed under `no_data`.
19. **Near-duplicates**: ambiguous `resolve_entity` matches carry `hint` (department · student count, "0 students
    (empty)"); when two labels normalise to the same key ("ECE 1" / "ECE1") the chip value is `"<label> (id <uuid>)"`
    and the prompt tells the model to use that id directly (otherwise the next turn is ambiguous again).
20. Takeaway rule relaxed from "no digits" to "no statistics" so class names like 10A / ECE-2 may be named.
21. `/agent/chat` without an LLM key -> 503 before anything is stored (was a 500 after the session was half-written).

## Status (9 Nov 2026)
- Commits: phase 1 contract `1ff49b1`, phase 2 store `317d98e`, phase 3 tools `1ae6a34`, phase 4 tests `728848d`,
  finish (evals, docs, 503, provider-error code) — see git log. Branch `claude/youthful-archimedes-tw0p0s`.
- Tests: **147 passed** (`AGENT_DB=embedded`, ~90 s): 104 before + 43 new (31 in `test_agent_v2.py` incl. store tests
  run on both implementations, 6 real-fixture tool tests R15–R20, 2 no-free-text checks for the new arg models).
- Evals: 6 new routing cases, Gemini 2.5 Flash, 6/6 tools, 3/3 oracle, 0 unsafe, $0.007. Anthropic not run (no key).
- Dev flow checked on Windows PowerShell: `python -m agent.dev_server` + `Invoke-RestMethod` on sessions, suggestions,
  history, messages, PATCH, DELETE, CORS preflight for PATCH; cross-user read -> 404.
- Not verified: real Supabase (fixture only), the frontend against v2, Anthropic routing on the new cases,
  `agent_store_supabase.sql` (never applied), multi-worker deployments (store / cancel registry are per process).

# v3: production trial readiness (10 Nov 2026) — MSIT 1-month trial

## Decisions
22. **Contract 1.3 is additive and committed first** (Phase 0). `docs/agent_openapi_v2.json` is generated from the
    app (`python scripts/export_openapi.py`, same document at `/openapi.json`) and `tests/agent/test_contract.py` fails
    when it drifts. *Why:* the frontend generates its TS types from that file; one source of truth, no hand edits.
    It now types every SSE frame (`AgentEvent` = `oneOf` + `type` discriminator), the new `file` event, typed views of
    the my_scope / section_comparison tables and chart, `/agent/meta` (incl. `scope`), and the 429 body.
23. **1.3 additions decided up front**, so the frontend regenerates once: `file` event; `error.retryable` + codes
    `rate_limited` / `db_unavailable` / `tool_timeout`; `pending_approval.status/result_ref/decided_at`;
    `ApproveRequest.decision` (approve | cancel); `/agent/meta` → `MetaResponse` (enabled, scope, features, limits);
    `GET /agent/files/{id}`; `/healthz`, `/readyz`, `/agent/metrics`. *Why:* additive fields with defaults keep 1.2
    clients working; new routes cost nothing until used.
24. **Trial switch** `AGENT_ENABLED_INSTITUTES` (comma list; unset = all): other institutes get 403 everywhere except
    `/agent/meta`, which answers `enabled: false` (no 403) so the UI can simply hide the agent. *Why:* MSIT first,
    no frontend deploy needed to widen it.
25. **psycopg_pool for all read paths** (min 1 / max 5, AGENT_DB_POOL_TIMEOUT 5 s). Each checkout runs
    `BEGIN READ ONLY; SET LOCAL statement_timeout; SET LOCAL search_path … ROLLBACK`; `default_transaction_read_only=on`
    is set once via the configure callback. The pool replaces per-call TCP+TLS+auth (≈75 ms embedded → ~8 ms pooled).
    *Why:* tool p50 dropped from ~100 ms to ~10 ms; turn p50 from 143 ms to 24 ms (10-rep mock bench).
26. **Prompt caching** (`AGENT_PROMPT_CACHE`, default on): Anthropic `cache_control` on the last tool def + system +
    last message block; Gemini reports implicit `cached_content_token_count`. Cache read 0.1× input price, write 1.25×.
    *Why:* same system prompt + tool schemas reused across turns and users of the same role.
27. **Model routing** (`AGENT_ROUTING`, default on): insight turns → AGENT_FAST_MODEL (default `claude-haiku-5-5`);
    paper intent / paper-in-progress → AGENT_MODEL (default `claude-sonnet-5-5`). Deterministic regex classifier:
    paper verbs × paper nouns (Hindi + English + Devanagari), PAPER_WORDS, paper_in_progress (last 2 user turns).
    "kitne tests bane" is excluded (it's a stats query). Escalation: fast model asks for a non-read tool → rest of
    turn runs on main. *Why:* Haiku 5.5 is 20× cheaper than Sonnet for insight turns that only call read tools.
28. **Parallel independent read tools** (ThreadPoolExecutor, bounded by AGENT_TOOL_CONCURRENCY 4). Separate
    `_LLM_POOL` (64 threads) and `_POOL` (16 threads) so 50 concurrent streams never starve tool I/O. *Why:* a model
    response with two `resolve_entity` calls runs them simultaneously instead of serially.
29. **No-cancel fast path** in `_wait`: when `cancel_event is None` the future is awaited with a single blocking
    `fut.result(timeout=…)` instead of a 50 ms polling loop. *Why:* the polling added ≈50 ms overhead per tool call;
    removing it brought turn p50 from 60 ms back down to 24 ms.
30. **Approval persistence** (Phase 2): POST /approve with `decision=cancel` stores "cancelled" without executing;
    `decision=approve` (default) executes then stores "approved" + result_ref in the message events. On replay
    (`list_messages`, `all_messages`), pending_approval events past `expires_at` are returned as "expired". The card
    in the frontend never shows a stale "Approve" button. *Why:* spec says "replay never shows a stale pending card".
31. **Retries already in Phase 1**: `_llm` generator retries 429/5xx/connection errors with jittered backoff (max 2),
    then yields a `provider_error` event with `retryable=True`. Cancel aborts the in-flight stream (LLMAborted).
    Per-tool timeouts (`tool_timeout_s`) and DB-down (`OperationalError → db_unavailable`) are caught in `_call`.
32. **PostgresStore** (`AGENT_STORE=postgres AGENT_STORE_URL=...`): same interface as SQLiteStore, using psycopg_pool.
    DDL in `agent/sql/agent_store_local.sql` (dev) and `agent_store_supabase.sql` (production with RLS). The store
    is NOT applied from code — migrations are external. SQLite stays the dev default. *Why:* multi-worker deploys
    need a shared store; SQLite WAL doesn't survive across processes on networked filesystems.
33. **Identity cache per-key invalidation** (`invalidate_identity(user_id)`): evict one user without clearing all.
    Refusals (build() raises) are never cached (existing test + new test). *Why:* a removed user should lose access
    within one TTL, but invalidating all caches on every admin action is wasteful.
34. **Rate limits** (Phase 4): per-user (AGENT_USER_TURNS_PER_10MIN, default 20) and per-institute daily token budget
    (AGENT_INSTITUTE_DAILY_TOKENS, unset = unlimited). In-process counters (a dict + lock). 429 response with
    `RateLimitDetail` body and `Retry-After` header, Hinglish message. Token recording after each turn.
    *Why:* MSIT trial needs a safety net; in-process is OK for WEB_CONCURRENCY=2.
35. **Prometheus metrics** at GET /agent/metrics: counters (requests, turns, tools, LLM calls, tokens, cost, cache
    hits, rate-limit hits, errors, retries, aborts) and histograms (tool, LLM, turn, HTTP latency). No PII by
    construction: label values from fixed vocabularies only (route templates, registered tool names, model ids).
    Unknown tools → "unknown". DB pool gauges from psycopg_pool stats.
36. **Structured JSON logging** (`AGENT_LOG_FORMAT=json`): one JSON object per line with ts, level, logger, message,
    request_id (when available). No PII. Default in production; text format in dev.
37. **Export tools** (Phase 5): `defaulters_export` and `proctor_month_report` build .xlsx via openpyxl and store in
    the in-process file store (`agent/files.py`). Both are registered as "read" kind INSIGHT_TOOLS (fast-model route,
    real-mode replaceable). The file store is per-process, 1-hour expiry, owner-only download at
    `GET /agent/files/{id}`. The `_ui_events` emits a `file` event with download_path when the tool result
    contains file_id. *Why:* teachers/proctors ask "75% se kam wale ki list do" — the agent should answer with a
    downloadable spreadsheet, not just text. openpyxl is lightweight; no external service needed.
38. **active_tools graceful fallback** for real mode: `hasattr(real_tools, n)` guard so export tools that don't have
    real-mode adapters yet use their seed-mode functions. *Why:* the real `schema_adapter` only has insight query
    tools; export tools reuse the same DB queries, so the seed-mode code works on live data too (read-only connect).
39. **Dockerfile** (Phase 6): python:3.12-slim, non-root `agent` user, pip install from requirements.txt + gunicorn.
    HEALTHCHECK pings `/healthz`. `preload_app=True` in gunicorn.conf.py so the pool + routing is shared across
    workers. `max_requests=1000` recycles workers to guard against memory leaks. *Why:* slim image (~180 MB), no
    build tools in prod, non-root per security baseline.
40. **Gunicorn conf** (`agent/gunicorn.conf.py`): UvicornWorker, 2 workers (WEB_CONCURRENCY), 300 s timeout (paper
    gen takes 4+ min), keepalive 5 s, preload_app. *Why:* uvicorn workers handle async lifespan + sync endpoints
    via anyio; 2 workers is enough for the 50-user MSIT trial.
41. **CORS** via `AGENT_CORS_ORIGINS` (comma-separated, no wildcard, validated at startup). The Dockerfile sets it
    empty; the deploy env file must provide the real origin. *Why:* wildcard CORS + credentials is a security hole;
    exact origins are the only safe option.
