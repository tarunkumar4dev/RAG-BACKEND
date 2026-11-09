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
