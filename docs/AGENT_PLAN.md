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
  (`GeminiProvider`, schema sanitiser) and unit-tested; not used for the demo/evals, so no Gemini fallback was needed.
- Files: `agent/{schemas,config,llm,core,verify,approval,trace,auth,api,export,db,text}.py`, `agent/tools/{insights,paper,registry}.py`,
  `agent/seed/*.sql`, `tests/agent/*`, `evals/*`, frontend `a4ai/src/{pages,components,lib}/agent/*` (not routed).
- Trig demo: 3 live runs; verifier bugs found by the demo fixed with regression tests (see `evals/sample_trigonometry_review.md`).
