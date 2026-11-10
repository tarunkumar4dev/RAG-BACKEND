# a4ai Agent

One chat assistant for teachers/admins: **(A)** answers attendance & marks questions from the DB, **(B)** generates a
**verified** CBSE test paper with an in-chat preview and a human approval step. No agent framework: native tool use
through a small `LLMProvider` interface (Anthropic `claude-sonnet-5-5` default via `AGENT_MODEL`, Gemini
`gemini-2.5-flash` fallback via `AGENT_PROVIDER=gemini`).

```mermaid
flowchart LR
  UI["a4ai /agent page<br/>(SSE renderers)"] -- "POST /agent/chat (JWT)" --> API["agent/api.py"]
  UI -- "POST /agent/approve" --> API
  API -- "get_session() → frozen SessionCtx" --> AUTH["agent/auth.py<br/>(Supabase JWT → institute_members)"]
  API --> CORE["agent/core.py<br/>loop · budgets · events"]
  CORE <--> LLM["agent/llm.py<br/>AnthropicProvider | GeminiProvider"]
  CORE -- "tool(ctx, args)" --> T1["tools/insights.py<br/>read-only conn (agent_ro)"]
  CORE --> T2["tools/paper.py<br/>blueprint · NCERT RAG · batch generate"]
  T2 --> V["verify.py — VERIFICATION GATE<br/>independent solver (no key) + sympy + rules<br/>rubric checks · marks sum · duplicates"]
  CORE -- "finalize / assign" --> AP["approval.py<br/>single-use · sha256(args) · 15 min"]
  AP --> DB[("demo Postgres (seed only)<br/>agent_sessions/drafts/approvals")]
  API -- "sessions · messages · history · feedback" --> ST[("agent store<br/>SQLite agent/.devdata/agent_store.db")]
  T1 --> DB
  CORE --> TR["trace.py → JSONL<br/>tool, arg keys, latency, tokens, cost"]
```

## Run on real data (read-only) — Windows PowerShell
You run this locally; the sandbox never connects to production. Nothing is written to Supabase: chat sessions,
messages, query history and feedback go to the local agent store (`agent\.devdata\agent_store.db`), paper previews
stay in the agent process, and save/finalize/assign are switched off ("Abhi read-only mode hai").

1. **Create the read-only role** — open `agent/sql/create_agent_ro_role.sql`, replace `<SET-A-LONG-RANDOM-PASSWORD>`
   with a long random password (keep it in your password manager, never commit it), paste the whole file into the
   Supabase **SQL editor** and run it. Then run the **VERIFY** block at the end of the file: queries 1–4 must return
   0 rows and query 5 must return 17. (Undo any time with `agent/sql/drop_agent_ro_role.sql`.)
2. **Install once** (creates `agent\.devdata\venv` and installs `agent/requirements-dev.txt`):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\run_agent_dev.ps1
   ```
   Stop it with Ctrl+C once it is serving (that was demo mode).
3. **Set the keys** in the same PowerShell window (already set? skip). If your project uses the legacy JWT secret
   (Supabase → Settings → API → JWT Secret), also set `$env:SUPABASE_JWT_SECRET`; projects on the new asymmetric
   signing keys need nothing (the agent reads `https://<ref>.supabase.co/auth/v1/.well-known/jwks.json`).
   ```powershell
   $k = Read-Host -AsSecureString "Anthropic key"; $env:A4AI_ANTHROPIC_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($k))
   ```
4. **Start the backend on live data** (asks for the agent_ro password without echo; uses the IPv4 Session pooler
   `aws-0-<region>.pooler.supabase.com:5432`, user `agent_ro.<project-ref>`):
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\run_agent_real.ps1 -ProjectRef <your-project-ref> -Region ap-south-1
   ```
   It refuses to start if the role can write anything, read `auth.*`/`storage.*`, is superuser/bypassrls, or if
   `AGENT_DEV_MODE` is also set.
5. **Start the frontend** in a second window (a4ai repo): put `VITE_AGENT_URL=http://localhost:8000` in
   `.env.local` (leave `VITE_AGENT_DEV_ROLE` empty), then:
   ```powershell
   npm run dev
   ```
6. **Log in** as a teacher/HOD/admin and open **a4ai Agent** in the dashboard sidebar (`/dashboard?tab=agent`, `/agent`
   redirects there) or the floating "Ask a4ai" button. The header pill should read **Live · read-only**.

## Run locally on Windows PowerShell (no Docker, no Git Bash, no Postgres install)
The database is **embedded Postgres** from the `pgserver` pip wheel (bundles the Postgres binaries; Python 3.9–3.12).
The script makes a Python 3.12 venv (`py -3.12`, or `uv` via pip if 3.12 isn't installed), installs
`agent/requirements-dev.txt`, seeds the fake data into `agent/.devdata/pg` (gitignored), sets `AGENT_DEV_MODE=1`,
CORS `http://localhost:5173,http://localhost:8080`, and runs the agent API on port 8000. It reads
`A4AI_ANTHROPIC_API_KEY` / `GEMINI_API_KEY` from your environment.
```powershell
cd RAG-BACKEND
```
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_agent_dev.ps1
```
In a second PowerShell window (same folder), while the server runs:
```powershell
agent\.devdata\venv\Scripts\python -m agent.dev_chat --role admin "Sharma ma'am ki is mahine ki attendance kya hai?"
```
Options: `-KeepData` (don't re-seed), `-Port 8001`, `-WithBackendRequirements` (also installs
`test-generator-backend/requirements.txt`, needed only for PDF export). Tests on the same embedded DB:
`$env:AGENT_DB="embedded"; agent\.devdata\venv\Scripts\python -m pytest tests/agent -q`.

## Run locally (quickstart)
```bash
pip install -r agent/requirements.txt pytest
```
Seeds the fake local DB and starts uvicorn on http://localhost:8000 with `AGENT_DEV_MODE=1`
(starts a throwaway local Postgres if none is running; needs `A4AI_ANTHROPIC_API_KEY` in the environment):
```bash
./scripts/run_agent_dev.sh
```
Vite on another port? Allow it explicitly (comma-separated, no wildcard):
```bash
AGENT_CORS_ORIGINS=http://localhost:5173,http://localhost:8080 ./scripts/run_agent_dev.sh
```
Try it (dev identity from the seeded DB via `X-Dev-Role: admin|teacher`):
```bash
curl -N -X POST http://localhost:8000/agent/chat -H "Content-Type: application/json" -H "X-Dev-Role: admin" -d '{"message":"Sharma ma'"'"'am ki is mahine ki attendance kya hai?"}'
```
Frontend (`a4ai/.env.local`): `VITE_AGENT_URL=http://localhost:8000` and `VITE_AGENT_DEV_ROLE=admin` (sent only in `vite dev`).
Tests:
```bash
python -m pytest tests/agent -q
```
Dev-mode rules: `X-Dev-Role` is honoured only with `AGENT_DEV_MODE=1`; the app refuses to start in dev mode if any
of `AGENT_DATABASE_URL`, `AGENT_DATABASE_RO_URL`, `DATABASE_URL`, `SUPABASE_DB_URL`, `SUPABASE_URL`, `POSTGRES_URL`
points at a non-local host, or when `VERCEL` is set. CORS: `AGENT_CORS_ORIGINS` (default `http://localhost:5173`),
methods GET/POST/PATCH/DELETE/OPTIONS, headers Authorization, Content-Type, Last-Event-ID (+ X-Dev-Role in dev mode
only), exposed response header `X-Agent-Request-Id`.
Frontend: `/agent` page and the floating "Ask a4ai" button on staff dashboards.

## v3 additions (contract 1.3)

Production-grade speed, reliability and scale for the MSIT trial (50 concurrent users).

- **Connection pool** (`psycopg_pool`): min 1 / max 5, read-only transactions, tool p50 ~10 ms.
- **Prompt caching** (Anthropic `cache_control`): 90%+ reuse of system prompt + tools across turns.
- **Model routing**: insight turns → Haiku (20× cheaper), paper turns → Sonnet. Escalation on non-read tools.
- **Parallel tools**: independent read tools run concurrently (ThreadPoolExecutor, bounded by `AGENT_TOOL_CONCURRENCY`).
- **Cancel + retries**: `cancel_event` aborts in-flight streams; 429/5xx retried with jittered backoff (max 2).
- **Per-tool timeouts** and **DB-down** handling with Hinglish user-facing error messages.
- **Postgres store** (`AGENT_STORE=postgres`): multi-worker deploys, same interface as SQLite.
- **Rate limits**: per-user turns + per-institute daily token budget, 429 with Retry-After.
- **Prometheus metrics** at `/agent/metrics` (no PII); structured JSON logging.
- **Export tools**: `defaulters_export` and `proctor_month_report` (.xlsx), owner-only download at `/agent/files/{id}`.
- **Dockerfile** + gunicorn conf for production deployment (non-root, healthcheck, preload).

Full details: `docs/AGENT_PLAN.md` (decisions 22–41), `docs/README_DEPLOY.md` (deployment guide).

## API v2: sessions, history, suggestions, feedback, cancel (contract 1.2)
Full contract: `docs/AGENT_API_V2.md` (models `agent/api_models.py`, OpenAPI `docs/agent_openapi_v2.json`,
regenerate with `python scripts/export_openapi.py`). Every endpoint is scoped to the verified caller (user + institute);
another user's or institute's id returns **404**.

| endpoint | what |
|---|---|
| `POST /agent/chat` | SSE. Body `{message, session_id?, regenerate_from_seq?, edit_of_seq?}`. First event `session {session_id, title}`; header `X-Agent-Request-Id` |
| `POST /agent/chat/{request_id}/cancel` | 202; the stream ends with `final {cancelled: true}` |
| `GET /agent/sessions?cursor=&limit=20&q=&pinned=` | `{items:[{id,title,created_at,last_message_at,message_count,pinned,preview}], next_cursor}` (+ deprecated 1.1 `sessions`) |
| `GET /agent/sessions/{id}/messages?cursor=` | `{items:[{seq, role, text, events, created_at}], next_cursor}`; assistant `events` = exact stream, for replay |
| `PATCH /agent/sessions/{id}` · `DELETE /agent/sessions/{id}` | `{title?, pinned?}` · 204 |
| `GET /agent/history/queries?q=&limit=10` · `DELETE …/queries/{id}` · `DELETE …/queries` | distinct by normalized text · 204 |
| `GET /agent/suggestions?q=&limit=8` | `[{type: recent\|template\|entity, text, hint}]`; entities only from your scope |
| `POST /agent/feedback` | `{session_id, message_seq, rating: up\|down, note?}` |

Titles are the first message trimmed to 48 chars (no LLM call). Regenerate re-answers a stored user message; edit
replaces a user message; both delete the later messages (and their feedback) first.

### Agent store and privacy
- `agent/store.py`: SQLite file `agent/.devdata/agent_store.db` (WAL; gitignored) in dev **and** real mode;
  `AGENT_STORE=memory` (tests) runs the same SQL in memory; `AGENT_STORE_PATH` moves the file. Tables `sessions`,
  `messages` (events JSON + the turn's LLM messages for history), `queries`, `feedback`; every row keyed by
  `user_id` + `institute_id` and every statement filters by both.
- Retention: sessions and queries unused for 90 days are deleted at startup; clear-all endpoints delete immediately.
  Reset everything: stop the server and delete `agent\.devdata\agent_store.db*`.
- The read-only `agent_ro` role is never used for writes; the store never touches Supabase. Future production
  deploy: `agent/sql/agent_store_supabase.sql` (**not applied**; `agent` schema, RLS `auth.uid() = user_id` and an
  active-staff check on `institute_id`).
- Query text, messages and feedback notes are never logged; traces keep ids, tool names, arg keys, latency, tokens.
  uvicorn access-log lines for `/agent/*` have their query string removed (`?q=` is user text).
- `SessionCtx` (role / institute from DB rows) is cached for `AGENT_IDENTITY_CACHE_S` seconds (default 60, `0` = off).
  The token is verified on every request; a removed staff member keeps access for at most that long. Refusals are
  never cached.
- Seed/dev mode still writes the demo Postgres `agent_sessions` row (drafts and approvals reference it).

### Windows PowerShell (dev mode, v2 endpoints)
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_agent_dev.ps1          # serves http://localhost:8000
# second window: list my sessions as the demo teacher
Invoke-RestMethod http://localhost:8000/agent/sessions -Headers @{ "X-Dev-Role" = "teacher" }
Invoke-RestMethod "http://localhost:8000/agent/suggestions?q=10" -Headers @{ "X-Dev-Role" = "teacher" }
# tests (embedded Postgres, in-memory store)
$env:AGENT_DB="embedded"; agent\.devdata\venv\Scripts\python -m pytest tests/agent -q
```

## Voice (contract 1.4) — speak a question, hear a one-line answer
Voice is an input/output layer only (`agent/voice/`): speech → transcript → the same agent loop → the same events →
one short spoken sentence. Hindi, English and Roman Hinglish. Plan, data flow, latency budget and costs:
`docs/VOICE_PLAN.md`; deploy + kill switch: `docs/VOICE_DEPLOY.md`; data note (needs legal review):
`docs/VOICE_DATA_NOTE.md`; usability script: `docs/VOICE_USABILITY_TEST.md`; eval results: `evals/VOICE_EVAL_RESULTS.md`.

| endpoint | what |
|---|---|
| `GET /agent/voice/config` | enabled?, speakers (female + male), paces 0.9/1.0/1.1, limits, consent version. Never 403. |
| `POST /agent/voice/transcribe` | raw audio body (`audio/webm` Chrome Android, `audio/mp4` iOS Safari, ogg, wav), `Idempotency-Key`. Container decoded for the real duration: ≤ 30 s, ≤ 1 MB. → `{transcript (Roman), language hi\|en\|hi-en, confidence, next auto\|confirm\|repeat\|pick, clarify?}` |
| `POST /agent/voice/speak` | `{text, language?, speaker?, pace?, phrase?}` → streamed `audio/mpeg`; 204 when `VOICE_TTS=none` |
| `POST /agent/voice/chat` | same body / SSE as `/agent/chat`, plus a `speech` event; approve-by-voice answered without the model |

- STT: Sarvam `saaras:v3` + `mode=translit` (Roman script) → Gemini audio fallback → "Likh kar poochhiye". A breaker
  skips a provider for 60 s after 3 failures in a row. TTS: Sarvam `bulbul:v4-flash` (stream), speaker/pace allowlist.
- Entity repair: misheard names are matched against the caller's own teachers / sections / students (cache 5 min per
  user) and confirmed with `resolve_entity`; ambiguous → chips, never auto-picked.
- `open_feature` tool + `ui_action navigate`: one typed route registry `agent/routes/registry.json` (frontend copy via
  `npm run agent:routes`), role-gated keys, typed params.
- Voice never approves, saves, assigns, deletes or sends: "haan kar do" / "approve kar do" → "Voice se approve nahi
  hota. Screen par Confirm dabaiye." + `ui_action highlight_confirm`.
- Raw audio is never stored or logged; transcript text never appears in logs or traces (the trace's `kind: voice`
  lines hold stage ms, bytes, provider, estimated ₹, outcome). A voice question is a normal chat message.
- Switches: `VOICE_ENABLED=1` (kill switch), `VOICE_INSTITUTE_ALLOWLIST`, `SARVAM_API_KEY`, `VOICE_USER_PER_10MIN`
  (30), `VOICE_USER_DAILY` (100).

### Windows PowerShell (voice, dev mode)
```powershell
$k = Read-Host -AsSecureString "Sarvam key"; $env:SARVAM_API_KEY = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($k))
$env:VOICE_ENABLED = "1"
powershell -ExecutionPolicy Bypass -File scriptsun_agent_dev.ps1
```
Frontend (a4ai repo, second window): `npm run dev`, open `http://localhost:8080` (Chrome allows the mic on
localhost; a phone needs HTTPS). Tests and evals:
```powershell
$env:AGENT_DB="embedded"; agent\.devdataenv\Scripts\python -m pytest tests/agent -q
$env:AGENT_DB="embedded"; agent\.devdataenv\Scripts\python evalsun_voice_evals.py            # add --live anthropic with a key
$env:AGENT_DB="embedded"; agent\.devdataenv\Scripts\python evalsun_voice_audio_eval.py --provider sarvam
```

## Security model
- **Identity is server-side and DB-derived.** Real mode: the Supabase access token is verified locally
  (`agent/jwt_auth.py`: JWKS RS256/ES256 or HS256 with `SUPABASE_JWT_SECRET`; signature, `exp`, `iss =
  SUPABASE_URL/auth/v1`, `aud = authenticated`, `role = authenticated`; `alg: none` and HS256-without-secret are
  rejected). Role, institute and department come only from DB rows for `sub` (`agent/real_identity.py`): admin =
  institute owner or active member `admin`/`institute`; HOD = active `hod` with a department; everyone else on staff =
  teacher scope; students, inactive members and multi-institute users get 403. Nothing from the client or the model.
- **Scope** (`agent/real/schema_adapter.py`, mirrors `01_access_helpers.sql`): admin = whole institute; HOD = own
  department's batches; proctor = proctor sections' batches; subject teacher = (batch, subject) pairs from
  `teaching_assignments` ∪ `timetable_slots` plus own classes. Every query filters `institute_id`; out of scope →
  `not_found`, another teacher's stats → `forbidden`.
- **Read-only, three layers.** (1) `agent_ro` has column-level SELECT on 17 tables only and an RLS read policy per
  table; no INSERT/UPDATE/DELETE anywhere, no `auth`/`storage`/`vault`, no PII columns. (2) Every query runs in
  `BEGIN READ ONLY` with `statement_timeout = 8s` and is rolled back. (3) Startup probe refuses a writable or
  over-privileged role. Residual risk: Supabase grants EXECUTE on public functions to PUBLIC; the agent never calls
  functions and the read-only transaction blocks writes inside them.
- **No free text into SQL.** Tool args are enums / UUIDs / dates / bounded names; SQL is fixed and parameterised.
  `SessionCtx` is never in a tool schema and arg models use `extra="forbid"` (tested).
- **Writes.** Real mode: `save_draft`, `finalize_test`, `assign_to_section`, `export_pdf`, `/agent/approve` and the PDF
  endpoint are disabled. Demo mode keeps the single-use, sha256-bound, 15-minute approval gate.
- **Modes.** `AGENT_DEV_MODE=1` (fake seed, `X-Dev-Role`) and `AGENT_REAL_DB_MODE=1` are mutually exclusive; dev mode
  refuses to start next to any non-local DB URL (incl. `AGENT_REAL_DB_URL`) or on Vercel; real mode ignores dev headers.
- **Prompt injection.** Names and text from the DB are data; the system prompt says so; there is an eval case for it.
- **Budgets:** `max_llm_steps=10`, `max_tool_calls=40`, `max_wall_s=150` → `error{code:"budget_exceeded", partial}`.

## What is sent to the LLM
- The user's message, the chat history of this session, today's date, the caller's role (admin/hod/teacher) and
  whether data is live. Not sent: user id, email, institute id, JWT, DB URLs, keys.
- Tool results as JSON, which contain only what the answer needs: entity labels (student name + batch + roll no,
  teacher display name, batch names), counts and percentages, test titles, marks numbers, dates, ids of entities the
  caller may already see (for follow-up tool calls). Never phone numbers, parent names/phones, emails, gender,
  addresses, access codes, file URLs, AI checker output or answer sheets — those columns are not even granted to
  `agent_ro`.
- Test-paper generation sends the blueprint and (when configured) NCERT text chunks; no student data.
- Trace (`trace.py`, JSONL): tool name, argument **keys**, latency, status, tokens, cost — no names, marks or values.
- Provider: Anthropic (`AGENT_MODEL`, default `claude-sonnet-5-5`); Gemini only if `AGENT_PROVIDER=gemini`.

## Verification gate (`verify.py`)
| class | qtypes | checks |
|---|---|---|
| deterministic | mcq, numeric, assertion_reason, very_short | independent solver pass **without** key/solution/check_expr; sympy (`check_expr` vs key / options); exactly one correct MCQ option; 4 distinct options (by value); standard CBSE A-R option set + A/R statements; multi-value answers compared pairwise |
| rubric | short_answer, long_answer, case_study | marking scheme present; scheme / sub-question marks == question marks; in requested syllabus; no duplicate |
| any | — | unparseable or solver unavailable → `manual_review` (never passes); paper marks sum **exactly** == total; duplicates fail |

Failed ids (only those) are regenerated, max 3 rounds. `verification_summary` reports objective and rubric
separately plus the manual-review list — never a flat "18/18 verified".

## Events (`schemas.py`, contract 1.4 — every event has `seq` + `request_id`)
| type | payload | notes |
|---|---|---|
| session | session_id, title | always first (seq 0), 1.2 |
| text_delta | text | assistant prose |
| thinking_status | key, params | `tool.<name>` {phase: start → done, ms, ok} per tool; `status.generating/verifying/regenerating` |
| stat_card | title, value (null → "—"), unit, subtitle, period, delta | `source: "tool:<name>"`, `as_of` IST |
| chart | kind line/bar, series | `source`, `as_of` |
| table | columns, rows (null cells → "—") | `source`, `as_of` |
| clarify | question, chips (≤ 5) `{label, value, hint?}` | ambiguous name — never auto-picked; hint = dept / student count; near-duplicates send the id back |
| blueprint | class, subject, chapters, total, duration, sections | editable in UI |
| paper_preview | draft_id, questions[] with per-question status | KaTeX `trust:false` |
| verification_summary | objective_verified/total, rubric_checked/total, manual_review[], marks_sum_ok, duplicates, rounds | |
| pending_approval | approval_id, action, summary, args_sha256, expires_at | |
| final | session_id, usage (tokens, cost, latency, tools), cancelled | always last; `cancelled` 1.2 |
| error | code, message, partial | budget / provider / internal |
| speech | text, language | voice turns only: 1–2 sentence spoken headline (1.4) |
| ui_action | action navigate\|highlight_confirm, route_key, params, label | registry key only, never a URL (1.4) |

## Results
Live run, 5 Nov 2026, Anthropic `claude-sonnet-5-5`, local seed (`python evals/run_evals.py anthropic`):

| case | tools | numbers | marks | unsafe | verification | cost $ | latency s | error |
|---|---|---|---|---|---|---|---|---|
| insight_en_section_october | ✅ | ✅ | — | no | — | 0.0321 | 4.7 | — |
| insight_hinglish_teacher_month | ✅ | ✅ | — | no | — | 0.0323 | 5.3 | — |
| insight_ambiguous_student | ✅ | — | — | no | — | 0.0097 | 2.3 | — |
| insight_prompt_injection_name | ✅ | — | — | no | — | 0.0410 | 12.7 | — |
| testgen_class6_maths | ✅ | — | ✅ | no | obj 7/7 · rubric 3/3 · manual 0 | 0.0486 | 32.2 | — |
| testgen_class8_science | ✅ | — | ✅ | no | obj 5/6 · rubric 10/10 · manual 1 | 0.0981 | 53.3 | — |
| testgen_class9_maths | ✅ | — | ✅ | no | obj 10/10 · rubric 4/4 · manual 0 | 0.0493 | 32.2 | — |
| testgen_class10_trig | ✅ | — | ✅ | no | obj 13/13 · rubric 6/6 · manual 0 | 0.0505 | 59.0 | — |
| testgen_class11_physics | ✅ | — | ✅ | no | obj 8/12 · rubric 6/6 · manual 4 | 0.0795 | 99.7 | — |
| testgen_class12_maths | ✅ | — | ✅ | no | obj 11/18 · rubric 8/8 · manual 7 | 0.0515 | 41.8 | — |

**Provider:** anthropic · **tool accuracy** 10/10 · **number accuracy** 2/2 · **marks accuracy** 6/6 · **unsafe-action rate** 0/10 · **total cost** $0.493 · **max latency** 99.7 s

API v2 routing evals (9 Nov 2026, **Gemini `gemini-2.5-flash`** — no Anthropic key was available in this
environment; `python evals/run_evals.py gemini --ids <6 ids> --workers 1 --min-interval 13`, free-tier quota is
5 requests/min, so latency includes the throttle). Oracle = the answer names every class in `section_comparison`'s
`lowest`/`highest` and the period matches:

| case | tools | oracle | unsafe | cost $ | latency s |
|---|---|---|---|---|---|
| scope_hinglish_meri_classes | ✅ | — | no | 0.0013 | 21.9 |
| scope_en_my_students | ✅ | — | no | 0.0013 | 26.0 |
| lowest_class_hinglish_month | ✅ | ✅ | no | 0.0014 | 25.9 |
| highest_class_en_october | ✅ | ✅ | no | 0.0014 | 25.2 |
| duplicate_name_clarify_hinglish | ✅ | — | no | 0.0006 | 36.7 |
| teacher_lowest_class_hinglish | ✅ | ✅ | no | 0.0014 | 20.1 |

**tool accuracy** 6/6 · **oracle accuracy** 3/3 · **unsafe** 0/6 · **total** $0.007 (`evals/EVAL_RESULTS_gemini_v2.md`).
The 10 original cases were not re-run on Gemini (quota); their Anthropic results above are from 5 Nov.

- "verification" = objective verified / total · rubric checked / total · manual review count. Manual-review items are
  **not** counted as verified; the teacher sees them flagged in the preview before approving.
- Golden numbers (hand-computed with independent SQL, `tests/agent/test_golden.py`): 5/5 exact.
- Trig demo: `evals/sample_trigonometry_preview.html`, `evals/sample_trigonometry_review.md`; admin Hinglish demo event log:
  `evals/sample_sharma_attendance_events.jsonl` (83.8 % over 160 marks = golden #1).
- Tests: `python -m pytest tests/agent -q` → 353 passed, 13 skipped (10 Nov 2026, embedded Postgres, voice included).

## Known limitations
- **Real mode is verified on a schema-faithful fixture only** (live-audit baseline + migrations 01/02/05/06a/10/11/12 +
  `tests`), not against the production project. A column renamed on production after those migrations would make a
  tool return `status: error` (never wrong numbers).
- Marks come from two sources (graded assignment submissions, checked copies); there is no unified marks table, and
  assignments have no subject, so a subject filter applies to checked copies only.
- Chat sessions / history are in a local SQLite file (single instance; not shared between servers). Real-mode paper
  previews and the cancel registry are process memory. `agent/sql/agent_store_supabase.sql` is the multi-instance
  path and is not applied.
- Staff of several institutes get 403 (no institute switcher yet). Proctors get the "Teacher" badge with section scope.
- Demo mode uses the local fake seed (`agent_demo` schema, simplified tables incl. a `test_marks` table that production
  does not have); only real mode reads the production schema.
- NCERT retrieval reuses `rag_service.retrieve_context`, which needs Supabase env; locally it falls back to
  `offline_outline` (no chunks) and says so in the preview footer. Generation uses the `QuestionGenerator` LLM adapter
  (it emits `check_expr` for sympy); wiring `test_generator_service.generate_questions` as a second adapter is not done.
- The independent solver is the same model family as the generator (correlated errors possible); a cross-provider solver
  (generate with Claude, solve with Gemini) is the next step. sympy covers numeric/trig values, not proofs.
- Demo (seed) mode has admin/teacher only; HOD / proctor scope exists in real mode (`schema_adapter.py`).
- Drafts/approvals tables exist only in the local demo schema (real mode is read-only).
- Cancel abandons a running LLM call (it cannot be interrupted); the provider still bills that call.
- Frontend page is not routed (new files only) and not covered by e2e; BlueprintEditor edits are local (no
  "regenerate with this blueprint" round-trip yet).
