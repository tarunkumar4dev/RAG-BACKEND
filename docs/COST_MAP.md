# a4ai cost map: Gate 1 audit (12 Nov 2026)

**Scope.** Every paid model call in both repos:
- RAG-BACKEND, branch `claude/youthful-archimedes-tw0p0s`. Production serves `test-generator-backend/app`, through `api/index.py`. The agent and voice code in `agent/` is **not deployed**.
- The frontend repo (`product`). Its only paid path is the `generate-test` edge function.

**Labels:** every number is one of these three.
- **MEASURED** comes from logs or the database.
- **ESTIMATED** is computed from a price list (source and date given) plus a stated token assumption.
- **TEST** is counted by the mocked tests added in this commit.

**Sources.**
- **Production DB**, read-only (`npx supabase db query --remote --read-only`, 12 Nov):
  - `check_jobs.usage` for checker tokens and cost.
  - `usage_records` for AI papers.
  - `ai_chat_usage` for chat messages.
  - `modules` and `contest_import_jobs` for counts.
- **Prices:** aistudio.google.com/docs/pricing (12 Nov 2026), docs.sarvam.ai pricing (10 Nov 2026, as recorded in `agent/voice/config.py`) and Anthropic prices from `agent/config.py`. Currency rate is ₹84/$, from `agent/voice/config.py`.

**What is not recorded anywhere.** Test-generator calls write their tokens only to a log line (`PAPER STATS`) on Vercel, and nothing keeps it. Worksheet, module and contest-import calls don't record usage at all. All papers, worksheets and imports below are therefore ESTIMATED.

## 1. Paid calls, sorted by estimated monthly spend at current usage

"Current usage" is October 2026, MEASURED: 21 AI papers, 23 graded copies, 2 contest imports, 2 modules, 8 chat messages. November so far has 0 papers and 0 copies.

| # | file:line | feature | model id (where set) | runs per | typical tokens / chars | cost per call | est. ₹ / month |
|---|---|---|---|---|---|---|---|
| 1 | `test-generator-backend/app/services/test_generator_service.py:2278` (`_call_gemini_model`) | Test paper (`/generate-frontend`) | `gemini-3.5-flash-lite` (`GEMINI_GEN_MODEL`, config.py:61). Falls back to `gemini-3.6-flash` (config.py:62). | **4 calls** for a custom 2-chapter paper, **14 calls** for a CBSE 2-chapter paper (TEST) | In: ≈1.1k static (TEST) + ≤1.5k NCERT context (8 × 750 chars). Out: 1–2.5k (**assumption**, not recorded) | ₹0.28–0.59 (ESTIMATED: $0.30 in / $2.50 out) | **₹25–175 ESTIMATED** (21 papers × ₹1.1–8.3 a paper) |
| 2 | `.../smart_assignment_service.py:258` (`_generate`, tag GRADE) and `:571` (RUBRIC) | Copy check, one per copy, plus a rubric once per assignment and key version | `gemini-3.8-flash` (`CHECKER_GEMINI_MODEL`, :49, set in prod). Falls back to `gemini-2.5-flash`, then `gemini-3-flash-preview` (:99). | 1 GRADE per copy (+ fallbacks; 10 of 23 copies needed 3 calls) | **MEASURED, 23 copies:** in 5.1k–13.6k (Σ 192k); out 1.4k–4.6k; thinking 0–9.8k (Σ 88k, **51 % of billed output**) | **MEASURED ₹1.29–5.65, p50 ₹3.50, p90 ₹4.80** | **₹76.42 MEASURED** (Oct). Tokens are measured; ₹ uses the price production is configured with (see §5 R4) |
| 3 | `.../contest_import_gemini.py:141` (REST) | Contest PDF import | `gemini-3-flash-preview`, then `gemini-3.8-flash` (`CONTEST_IMPORT_MODELS`, contest_import_service.py:52) | 1 per import (+1 fallback) | ~10-page PDF ≈ 2.6k in + 2k prompt; out ≈ 8k (assumption); thinking 0 | ≈ ₹2.2 (ESTIMATED: preview at $0.50 / $3.00) | ₹2–10 ESTIMATED |
| 4 | `.../module_service.py:90` (summary) and `:314` (OCR of scanned PDFs) | Module processing | `gemini-3.5-flash-lite` (`GEMINI_MODEL`, read again with its own default at module_service.py:57) | 1–2 per module | Full chapter text ≈ 15k in (assumption), out ≈ 2k | ≈ ₹0.8 | ₹2 ESTIMATED |
| 5 | `test-generator-backend/app/api/v1/endpoints/modules.py:417` | Worksheet from a PDF (`/worksheet/generate-direct`) | `gemini-3.5-flash-lite` (`GEMINI_MODEL`, read at :416) | 1 per worksheet | **Up to 400,000 chars of PDF text (≈100k tokens)** + 1.5k static (TEST). Out ≈ 3k | ≈ ₹1 typical (15k in); **₹3.2 at the cap** | **unknown, volume not recorded** |
| 6 | `.../worksheet_service.py:356` | Worksheet from a module | same, `GEMINI_MODEL` read at :28 | 1 per worksheet | up to 200,000 chars (≈50k tokens) | ≈ ₹1–1.7 | unknown, not recorded |
| 7 | `test-generator-backend/app/api/v1/endpoints/chat.py` (OpenRouter) | Sarthi chat | `deepseek/deepseek-chat`, falls back to `google/gemini-2.5-flash` (:47-48) | 1 per message | short | price not checked | < ₹1 (8 messages a month, MEASURED) |
| 8 | `.../community_quiz_generator.py:269` | Community quiz generation | `gemini-flash-latest` (an **alias**, hardcoded :43) | 1 per quiz | | | ₹0 (0 since Sep, MEASURED) |
| 9 | `agent/llm.py:139/141` (Anthropic), `:223/225` (Gemini) | Agent text question | `claude-sonnet-5-5` / `claude-haiku-5-5` (agent/config.py:5-6), or `gemini-2.5-flash` (:7) | 2 calls on the happy path (TEST) | ≈3.6k static per call (system prompt + tools, TEST) | Sonnet **$0.030 median, $0.057 at the 90th percentile**: MEASURED, but on local dev traces from 9–11 Nov, not production. Gemini $0.0012 MEASURED (local evals). Haiku ≈ ₹0.08 ESTIMATED | ₹0 (not deployed) |
| 10 | `agent/voice/providers.py:137` (Sarvam STT), `:184/200` (Sarvam TTS), `:241` (Gemini STT fallback) | Voice query | `saaras:v3`, `bulbul:v4-flash` (agent/voice/config.py:20-22); Gemini STT uses `agent_config.GEMINI_MODEL` | 1 STT + 2 LLM + 1 TTS (TEST) | about 8 s of audio; spoken answer about 120 chars | STT ₹0.065, TTS ₹0.12, LLM (Haiku) ≈ ₹0.08: **≈ ₹0.27 per query** ESTIMATED | ₹0 (not deployed) |
| 11 | `.../test_checker_service.py:235/423/680`, `.../validation_service.py:79`, `verify_answer_key` (test_generator_service.py:2851) | Old test checker, validation, answer-key verify | `gemini-3.8-flash`, `GEMINI_VAL_MODEL` | | | | ₹0. **No caller in the UI** (`/dashboard/test-checker` redirects; `/verify-answers` is not called) |
| 12 | product `supabase/functions/generate-test/index.ts:231, :473` | Old edge-function generator | `text-embedding-004` (**shut down 14 Jan 2026**), `gpt-4o-mini`, `deepseek-chat` | | | | ₹0 (the UI imports only `downloadTestFile`, never `generate-test`) |
| 13 | `api_server.py`, `rag_system.py`, root `module_service.py`, `fix_indent.py` | Legacy root scripts | `Config.MODEL_NAME` etc. | | | | ₹0 (not deployed) |

**Total at current usage:** about **₹110–270 a month**. ₹76 of that is MEASURED (the checker). Everything else is ESTIMATED, plus the worksheet volume, which is unknown.

### Where the money actually is
1. The **copy checker** is the only MEASURED cost: about ₹3.50 per copy, ₹76 in October. Thinking tokens are about 40 % of it, and production can't cap them because it runs the google-genai 0.8.0 SDK.
2. **Test papers** are probably about the same size (₹25–175 a month), but nobody records their tokens, so we can't tell.
3. **Worksheets** are an unmeasured risk. One call can send up to 100k input tokens. The endpoint doesn't check a login token or a plan quota (§3, R6).
4. **Voice and agent cost ₹0 today.** At launch, plan for about ₹0.27 per voice query with Haiku. With Sonnet the LLM part alone is about ₹2.5 per query (MEASURED, dev traces).
5. **Failure paths can multiply cost.** One teacher click on a CBSE paper can bill up to 100 calls if Gemini keeps returning unusable output (§2).

## 2. Retry + fallback amplification

The scenario: the provider returns **429, then 500**, then keeps failing (the worst case). Failed 429/5xx requests aren't billed by Gemini or DeepSeek. A reply that comes back **200 but unusable** (empty or bad JSON) **is billed**. That is the paid worst case.

| Layer | Test paper | Copy check (one copy) | Agent question / voice |
|---|---|---|---|
| Frontend | 1 POST, no retry (`src/lib/api.ts` `apiFetch`). React Query `retry:1` applies to queries only. | Upload, then the job is queued (server-side retries) | `retry:false` (`src/lib/agent/queries.ts:266`). The user can press "Dobara try karein". Voice uses an `Idempotency-Key`. |
| Backend | Per batch: each model × `MAX_RETRIES=4` (test_generator_service.py:49). Models: primary + `GEMINI_FALLBACK_MODEL`. Up to 3 top-up rounds per chapter. With `LLM_PROVIDER=deepseek`, each model step first retries DeepSeek 4×. | `_generate_json`: ≤2 attempts × ≤3 models (503 only). Cap `MAX_GEMINI_CALLS_PER_SHEET=12`. Job retried up to `max_retries=3`. | `LLM_RETRIES=2` per step (agent/config.py:44), `MAX_LLM_STEPS=10`. STT chain Sarvam → Gemini (1 each, plus a breaker). TTS 1 (REST only on 404/405). |
| SDK | google-genai: no retry on `generate_content` by default (0.8.0 / 1.14.0 retry only file uploads; 2.34.0 defaults to "never retry"). Anthropic: `max_retries=0` (agent/llm.py:79). | same | same |
| **Worst-case requests** (TEST) | **40 per chapter** (5 batch runs × 2 models × 4). With DeepSeek, 40 DeepSeek + 40 Gemini. | 6 per run, 18 per copy | 3 per step, up to 30 per question |
| **Worst-case PAID calls** (TEST) | **10 per chapter; 100 per 2-chapter CBSE paper** ⚠ | **6 grading calls per copy** (2 per run × 3 runs, ≈ ₹21) ⚠ | **10 per question** (every step billed) ⚠. Paper tools inside the agent add generator and solver calls. Voice adds ≤ 2 STT + 1 TTS. |

Everything above 3 is flagged ⚠. `module_service._call_gemini_with_fallback` (module_service.py:78) also **rotates through every configured API key** (`GEMINI_API_KEY`, `GEMINI_API_KEY_1..9`, …) on 429 or 403. That multiplies the attempts and works around quota, and it **logs the first 10 characters of each key** (:98).

Tests: `test-generator-backend/tests/unit/test_audit_retry_amplification.py`, `tests/agent/test_audit_gate1.py`. They pin today's numbers exactly. The `xfail(strict=True)` tests state the target of ≤ 3 paid calls and will fail, which forces an update, once a cap lands.

## 3. Quota correctness

**Paper limit.** The flow is `check_usage` (read only) → generate (20–60 s) → `record_usage`, where the `record_usage` SQL has no limit check (live definition read 12 Nov). Tests are in `test_audit_quota.py`.

| Case | Today | Wrong? |
|---|---|---|
| Job fails after the quota check | Not recorded | ✅ correct |
| Two concurrent requests with 1 left (two serverless instances) | Both pass and both are recorded, so the count ends at 11/10 | ⚠ **over-limit.** The user isn't overcharged, but we pay for a paper the plan doesn't cover. |
| Client disconnects mid-generation | Generation runs on (it's synchronous inside `async def`, with no disconnect check). It's counted, and the paper is saved to `tests`. | ✅ acceptable: the paper is in the teacher's history |
| Retry of an already-counted request (response lost on 4G, teacher clicks again) | No idempotency key, so it's a new paper and counted twice | ❌ **charged twice** |
| `record_usage` fails (DB timeout) | Swallowed ("non-fatal"), so the paper is free | ⚠ revenue leak |
| Partial paper (e.g. 3 of 10 questions) | Counted as a full paper | ❌ arguably charged wrongly (not tested) |

**Voice allowance.** There is **no per-plan voice allowance at all**. `agent/voice/limits.py` gives every user 100 a day (`USER_DAILY_DEFAULT`) plus a 10-minute rate, and keeps the counters in process memory. Tests are in `tests/agent/test_audit_gate1.py`.

| Case | Today | Wrong? |
|---|---|---|
| Fails after the quota check (STT down, 413 too big, 415/422 bad audio) | **Counted**: `check_user` runs before validation and before STT (voice/api.py:224) | ❌ **charged wrongly** |
| Two concurrent requests with 1 left | One process: a lock gives exactly 1 ✅. Two workers or instances: each has its own counter. | ⚠ the limit is per process; a restart resets it |
| Client disconnects mid-transcription | Counted (same reason as the first row) | ❌ |
| Retry with the same `Idempotency-Key` after **success** | Cached, not counted again | ✅ |
| Retry with the same key after **failure** | Counted again | ❌ **charged twice** |
| `/agent/voice/speak` (TTS, paid per character) | **Never counted** | ⚠ unlimited TTS per user |

## 4. Cache isolation

| Cache | Where | Key | TTL | Includes institute / user? | Stale after a direct DB write? |
|---|---|---|---|---|---|
| Auth token → user | test-generator-backend/app/core/auth.py:44 | sha256(token) | 60 s (≤ token exp) | user only (the value is id/email/role claim). OK. | Sign-out or revoke still works for ≤ 60 s |
| NCERT question list / stats | test_generator.py:1563 / :1597 (`api_cache`) | all query params | 600 s / 1800 s | No. **Safe only because the NCERT bank is global** (pinned by a test). | Yes: `ncert_question_fixes` edited directly in Supabase stay hidden ≤ 10/30 min per instance. The browser gets `private, no-store`. |
| Contest board | contest_service.py:910 | contest id | 10 s | per contest. The value is the **full teacher view** (names, e-mails, answers), filtered per viewer after the cache. | ≤ 10 s |
| Chapter list | rag_service.py:26 | (subject, class) | 1 h | global data | yes, ≤ 1 h |
| Route-template lru | core/rate_limit.py:108 | path | process | n/a | n/a |
| Checker rubric | `check_sessions.rubric` (job_worker.py:398) | assignment + answer-key `version` | until the key version changes | per assignment | Only if a key changes without a version bump (not verified) |
| Agent identity | agent/auth.py:72 | ("live", sub, fn) | 60 s | user → institute | **Yes**: a teacher removed in the frontend (direct write to `institute_members`) keeps agent access ≤ 60 s. `invalidate_identity()` exists but nothing calls it. |
| Agent entities / scope / voice name-repair | agent/suggest.py:35, meta.py:10, voice/repair.py:22 | (user, institute, role, data_source) | 300 s | ✅ (tested: A never sees B, and the same user in another institute never shares) | A new student is unknown for ≤ 5 min |
| Voice idempotency | agent/voice/limits.py:21 | (user, key) | 600 s | ✅ per user (tested) | n/a |
| Voice static phrases | agent/voice/api.py:241 | (phrase, lang, speaker, pace, model) | process | static text only. Data answers are never cached. | n/a |
| Frontend React Query | product src/App.tsx:187 | agent keys include uid | 5 min stale | ✅. Cleared on `SIGNED_OUT`. | Yes, ≤ 5 min |
| Frontend agent localStorage | product src/lib/agent/persist.ts | `a4ai.agent.<uid>` prefix | until sign-out | ✅. No message text is stored. | n/a |
| Provider prompt cache (Anthropic) | agent/llm.py:107-114 | exact prefix (provider side) | 5 min | the prefix includes the caller's context, so there's no cross-tenant hit | n/a. **0 cache reads measured** on dev traces (§6 F6). |

No cache can serve institute A's value to institute B. The remaining risk is staleness after writes that bypass the backend: the agent identity (60 s), entities (300 s) and NCERT (10–30 min). Tests are in `test_audit_cache_isolation.py` and `tests/agent/test_audit_gate1.py`.

## 5. Model config

The ids in use and where each is set are listed in §1. Findings:

| # | Risk | Detail |
|---|---|---|
| R1 | **Deprecated id** | `gemini-2.5-flash` **shuts down 1 Dec 2026**, after which requests are re-routed to `gemini-3.5-flash-lite` (aistudio.google.com/docs/deprecations, 12 Nov 2026). It's used as the checker's 1st fallback (smart_assignment_service.py:99), the agent's Gemini model and voice STT fallback (agent/config.py:7), and the chat fallback (chat.py:48). **That is a silent provider-side model change** for grading. |
| R2 | Preview / alias ids | The checker's 2nd fallback and contest-import primary is `gemini-3-flash-preview` (preview, no SLA). Community quiz uses `gemini-flash-latest` (an alias; the model under it changes without notice). |
| R3 | Silent fallbacks to a different model | Generator: `gemini-3.5-flash-lite` → `gemini-3.6-flash` (2.5× the input price and 1.5× the output price), and DeepSeek → Gemini. Checker: 3.8 → 2.5-flash → 3-flash-preview (MEASURED: 10 of 23 October copies were graded by the preview model). Chat: deepseek → gemini-2.5-flash. Agent: Sonnet ↔ Haiku routing (by design). |
| R4 | Wrong price constants | The checker's cost estimate defaults are ₹26/₹213 per million tokens (job_worker.py:62-63), which are **2.5-flash prices**. The real `gemini-3.8-flash` price is $0.75/$3.75 (₹63/₹315) until 31 Dec, then **$1.50/$7.50 from 1 Jan 2027**. The recorded rows back-solve to ≈ ₹73/₹362, so production overrides the env vars. If those were unset, the cost estimate and the ₹150 daily budget guard would undercount about 1.5×. Agent Gemini STT cost uses the text price ($0.30) for audio ($1.00), a 3.3× undercount (voice/config.py:139). |
| R5 | Scattered config | Ids are hardcoded in 13 files outside the 3 config modules. `GEMINI_MODEL` is read in 4 places, each with its own default (config.py, modules.py:416, module_service.py:57, worksheet_service.py:28), and `CHECKER_GEMINI_MODEL` in 2 (smart_assignment_service.py:49, test_checker_service.py:37). `CHECKER_MODEL` (config.py:80) is a different setting from `CHECKER_GEMINI_MODEL`. `GEMINI_THINKING_BUDGET` (config.py:64) is never passed to Gemini. |
| R6 | Stale comment | Code says "gemini-3.6-flash retires 19 Nov 2026" (smart_assignment_service.py:48, test_checker_service.py:37). Google's page lists **no shutdown date** for 3.6-flash. |
| R7 | Local env | The local `.env` has `LLM_PROVIDER=deepseek`, and the DeepSeek account returns **402 Insufficient Balance**. If production has the same setting, every generator batch makes a failed DeepSeek call first. **Needs a check of the production env** (I did not read it). |

Anthropic ids are active: `claude-sonnet-5-5` not before 28 Sep 2027 and `claude-haiku-5-5` not before 7 Oct 2027 (platform.claude.com/docs/en/about-claude/model-deprecations, 12 Nov 2026). Note that `temperature` is removed in API version `2026-10-21`. The agent's Anthropic path doesn't send it; its Gemini path sends `temperature=0`, which is fine for Gemini.

Test: `test_audit_model_config.py`. It fails on any **new** hardcoded id outside the inventory, pins the deprecated and preview ids in use, and states the single-config target as a strict xfail.

## 6. Cost guard

`tests/cost_guard_baseline.json` holds, per operation, the provider calls on the happy path and the static prompt size in approximate tokens (chars/4, measured with mocked providers). A test fails if either grows more than 20 % unless the baseline is updated in the same commit (`COST_GUARD_WRITE=1`).

| Operation | Calls | Static prompt tokens |
|---|---|---|
| test paper, custom, 2 chapters | 4 | 4,454 |
| test paper, CBSE, 2 chapters | 14 | 16,402 |
| worksheet (direct) | 1 | 1,459 |
| copy check (rubric + one copy) | 2 | 4,813 (text only; page images are not counted) |
| agent question | 2 | 7,180 (≈3.6k system prompt + tools per call) |
| voice query | 4 (STT 1, LLM 2, TTS 1) | 7,604 |

## 7. Attribution: cost record → feature + institute

| Call path | Cost/usage record | Feature? | Institute? | Student names? |
|---|---|---|---|---|
| Copy check | `check_jobs.usage` (+ `smart_assignment_checks.result_json.usage`) | ✅ `kind` / stage | ✅ `check_jobs.institute_id` | ✅ none (logs use `mask_name`) |
| Test paper | `PAPER STATS` log line only (not persisted). `usage_records` counts papers but has no tokens. | log only | ❌ | ✅ none |
| Worksheet (direct and module) | **none** | ❌ | ❌ | n/a |
| Module processing | `modules.token_count` (input size only) | ❌ | ❌ (teacher_id only) | n/a |
| Contest import | `contest_import_jobs.result` has no usage | ❌ | ❌ | n/a |
| Sarthi chat | `ai_chat_usage` (counts, no tokens) | implicit | ❌ (user only) | n/a |
| Community quiz | none | ❌ | ❌ | n/a |
| Agent question | JSONL trace (`agent/trace.py`): `kind=llm`, `cost_usd`, `request_id` | partial (`kind` / `route`) | ❌ (**no institute_id or user**; tested) | ✅ none (tested) |
| Voice | the same trace, `kind=voice`, `cost_inr` | ✅ `endpoint` | ❌ | ✅ none |

## How these numbers were produced and what was not done
- I made no paid calls on purpose, did not deploy, made no production DB writes, and changed no plans or prices.
- **One incident.** An early local experiment script imported the generator with the local `.env` (`LLM_PROVIDER=deepseek`) and reached `api.deepseek.com` 4–6 times. Every request returned **402 Insufficient Balance**, so nothing was billed. All audit tests now force Gemini, blank the DeepSeek key and block outbound HTTP (`tests/unit/audit_support.py`).
- I used no LLM output comparisons, and quality is not claimed anywhere.
- **Pre-existing test failure:** `tests/unit/test_cors_attempt_header.py::test_contest_submit_preflight_allows_attempt_id` returns 400 locally. It depends on the local CORS env and is unrelated to this audit.
