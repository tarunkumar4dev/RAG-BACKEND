# Checker audit — Phase 0 inventory (read-only)

Date: 2026-10-26 · backend `main` @ dbac670 · product `origin/main` (migrations 00–24) · live DB read with SELECT only.
Evidence classes: `[EVIDENCE]` = read from the cited file / query · `[INFERRED FROM x]` · `[HYPOTHESIS]` · `[ASSUMPTION]`.

## 0. Corrections to the brief (read these first)

| Brief says | Reality | Class |
|---|---|---|
| Harness `eval_checker.py` (stdlib, wide/long CSV, `IGNORED_COLS`, `--compare`, `--usage`) is the starting base | That file is `C:\Users\login\Downloads\eval_checker.py` (25 Oct), **not in either repo**. The repo's `test-generator-backend/scripts/eval_checker.py` is a different tool: it runs the real grader on `Downloads/golden_set.json` and scores per question. | [EVIDENCE] both files |
| Golden set = "Rahul Sir's 30 copies" with a 10–12 mark gap | No such data on disk. `golden_set.json` = 3 synthetic Trigo copies (40 marks). `eval_cases/smart_checker/science10_rahul` = 1 synthetic sheet "typed in a handwriting font" (its `expected.json` says so). `golden_teacher_template.csv` = header + MAX row + one blank student. No 10–12 mark gap is recorded anywhere. | [EVIDENCE] |
| `scripts/export_ai_marks.py`, `checker_stage_usage` table, `ai_marks.csv`, `run0/summary.json` exist | None exist (searched Downloads, OneDrive\Desktop, both repos, migrations 00–24). | [EVIDENCE] |
| `CHECKER_DAILY_BUDGET_INR=1000` | Code default is **150** (`job_worker.py:61`). The prod value isn't readable from here. | [EVIDENCE] code / [ASSUMPTION] prod |
| Queue starts at L0 (pg_cron → Vercel per job) | Already at **L1**: `a4_claim_check_job` = advisory lock + `FOR UPDATE SKIP LOCKED`, pg_net kick per INSERT statement, pg_cron sweep every minute, exponential backoff, idempotency keys. | [EVIDENCE] `18_upload_flow.sql:757-850` |
| LLM computes marks/totals ("arithmetic" risk) | The model returns **verdicts only** (`full/mostly/partial/none` per point). All marks, caps, OR-groups and totals are computed in `_score` (`smart_assignment_service.py:924-998`). An arithmetic error in a published total can only come from our code. | [EVIDENCE] |

Consequence: most of §2 of the brief ("known defects in the current harness") is about the Downloads script, not the system, so it can't be "carried forward as P0" against the repo.

## 1. Deployment

- One Vercel project `rag-backend`, region `bom1`, everything routed to `api/index.py` → FastAPI `app.main` (`vercel.json`). Hobby plan: 300 s per invocation (`job_worker.py:47`). [EVIDENCE]
- Prod installs `api/requirements.txt` (google-genai 0.8.0 per memory of 16 Oct; not re-verified today). [HYPOTHESIS — re-check `api/requirements.txt` before relying on thinking-level features]
- Frontend: React/Vite in `OneDrive\Desktop\product`, talks to Supabase directly + this backend for AI work.
- **No CI in either repo** (no `.github/`). Tests run by hand. [EVIDENCE]

## 2. Two grading paths

| | A. Check Copies (teacher uploads photos) | B. Assignment link (student uploads, `/s/<token>`) |
|---|---|---|
| Entry | `POST /copy-checker/sheets/{id}/check` (`copy_checker_router.py:263`) | `create_submission` RPC → `a4_enqueue_submission` → `check_jobs` row → pg_net kick → `POST /internal/jobs/run` |
| Execution | **Synchronous inside the HTTP request** | Background after 202, budget 270 s, claims while ≥200 s left (`job_worker.py:49-50`) |
| Rubric | Built inline if missing (`_check_sheet`) | Own job per (assignment, key version), idempotency key `rubric:<session>:k<v>`; copies park and are woken |
| Grade | `copy_check.grade_copy` (same core) | Same; >12 pages → chunk jobs, merged by `portal_result.merge_chunks` |
| Marks written | `check_sheets` via caller's JWT (RLS owner-only) + separate audit insert (`_save_to_db`) | `JobStore.save_result`: sheet + audit + checkpoint `graded` **in one transaction** (`job_store.py:205-221`) |
| Review flags | Per-question `needs_review` in `ai_result` only | `portal_result.review` → `needs_review_reasons` enum + `confidence` |
| Override | `PATCH …/questions/{q}/marks` (Python, read-modify-write) | `override_marks` RPC (SQL, `FOR UPDATE`) |
| Publish | Teacher "approve" in UI | `publish_results` / auto-publish trigger; unpublish requires reason |

## 3. LLM call sites (checker only)

| Stage | Function | Model | Timeout | Retry | Output contract |
|---|---|---|---|---|---|
| Rubric from paper+key | `build_rubric` → `_build_rubric_from_files` (`smart_assignment_service.py:560`) | `CHECKER_GEMINI_MODEL` default `gemini-3.8-flash`, fallbacks `gemini-2.5-flash,gemini-3-flash-preview` (`:49, :98`) | job: 180 s (`CHECKER_RUBRIC_TIMEOUT_S`); sync path: 60 s global (`:52`) | ≤1 JSON/busy retry (`_generate_json`, `:287`); re-ask once if marks sum ≠ stated max (`:578`) | `_validate_rubric` normalises points, computes total |
| Rubric from saved test | `rubric_from_test_fast` (no LLM) / `build_rubric_from_test` | same | same | same | same |
| Grade copy | `grade_sheet` (`:1029`) via `copy_check.grade_copy` | same chain | job: 150 s; sync: 60 s | same | JSON verdicts → `_score`; unknown point ids → `none`; marks clamped per leaf and per question; total capped at paper max |
| Decoding | `_generate` (`:258-262`) | — | — | one call per model in the chain; 404/503 → next model; 429 → `GeminiBusy` | `temperature 0, seed 7`, `response_mime_type json` |

Call cap: `MAX_GEMINI_CALLS_PER_SHEET=12` (`:54`). Prompt-injection line in the grading prompt (`:784-785`). No output schema object is passed to Gemini; validation is by the code that reads it. [EVIDENCE]

Other Gemini callers (generation, contests, chat) are out of scope.

## 4. Data model (checker tables, from `18_upload_flow.sql` + `23_open_link.sql`)

- `submissions` — unique `idempotency_key`; unique `(assignment_id, student_id, attempt_no)`; `files jsonb` (client-declared paths, sha256); status mirrored from job state by trigger `a4_check_job_state_sync`.
- `check_sessions` — one per assignment (`check_sessions_assignment_uq`); holds `rubric jsonb`, `rubric_status`.
- `check_sheets` — unique `submission_id`; `ai_result jsonb` (all per-question/per-point detail), `ai_marks`, `final_marks`, `needs_review_reasons a4_review_reason[]`, `confidence`, `model_version`.
- `check_jobs` — unique `idempotency_key` (`check:<sheet>:k<keyver>`, `rubric:<session>:k<v>`, `chunk:<parent>:<n>`), `retries/max_retries=3`, `next_run_at`, `locked_at`, `checkpoint`, `usage jsonb` (tokens, est ₹, stage timings), `last_error_code` enum-by-check.
- `smart_assignment_checks` — audit/history: one row per AI check plus `teacher_override | publish | auto_publish | unpublish | regrade` rows (`result_json.type`).
- `assignment_ai_keys` — answer key path + `version` bumped by trigger when the path changes.
- Per-question / per-criterion results are **not normalised**: they live in `check_sheets.ai_result.graded_questions[].points[]` (verdict, evidence quote, marks). `a4_project_per_question` projects a summary. [EVIDENCE]
- RLS enabled on `check_jobs` (teacher or institute admin read), portal read policies on sessions/sheets, storage policies on `submissions` bucket (`18_upload_flow.sql:1632-1650`). Full policy list per table: not enumerated in this pass. [HYPOTHESIS — Phase 1 to list every policy from the live catalog]

## 5. Upload path (B)

1. Browser → Storage `submissions/{inst}/{asg}/{student_id | guest/submitter_key}/…` (insert policy `a4_submissions_insert`, path check `a4_valid_submission_path`).
2. `create_submission` RPC: access-code gate (`a4_code_gate`, IP-hash lockout), deadline + 2 min grace, 60 s per-student throttle, attempt number under lock.
3. Worker trusts only `submission_id`: downloads only paths listed in `check_sheets.pages`, inside the owner folder, no `..` (`job_worker.py:329-350`); `upload_validation` sniffs magic bytes, opens PDF/photo, page cap (`assignments.max_pages`, hard 30), strips EXIF, sha256 for duplicate flags. [EVIDENCE]
4. Page count vs `page_count` declared by the client: validation reports per-file problems; a missing object is dropped from `blobs` and then reported by `validate` as missing. [INFERRED FROM `job_worker.py:341` + `upload_validation.validate` signature — Phase 1 to confirm with a test]

## 6. Where marks are written / read

Written: `JobStore.save_result` (B), `_check_sheet` (A), `_override_marks` PATCH (A), `override_marks` RPC (B), `a4_publish_submission` (`final_marks`, `submissions.grade`), `unpublish_results` (clears grade), `merge_chunks` (in-memory, then `save_result`).
Read: student `my_submissions` / `a4_submission_json` (only after publish), teacher ReviewDrawer, Excel export (`checkerExport.ts`), analytics (`CheckSessionAnalytics.tsx`), eval script.

## 7. One assignment-link submission, end to end

```
student ──upload──▶ Storage(submissions/…)
student ──create_submission RPC──▶ submissions(status=uploaded)
        a4_enqueue_submission ─▶ check_sheets(upsert by submission_id) ─▶ check_jobs(insert, key check:<sheet>:k<v>)
        trigger check_jobs_kick ─▶ pg_net POST /internal/jobs/run (X-Job-Secret)      [also pg_cron sweep / min]
worker.run(): claim (SKIP LOCKED, max 5 running) ─▶ daily limit ─▶ key? ─▶ rubric ready?
        no  → ensure rubric job, park copy 60 s (retry given back)
        yes → download+validate ─▶ (>12 pages → chunk jobs, park 600 s) ─▶ grade_copy (1 Gemini call)
            ─▶ save pages to answer-sheets/…/norm ─▶ review() ─▶ save_result (TX: sheet + audit + checkpoint)
            ─▶ a4_finish_check_job(ok) ─▶ trigger: submissions = needs_review | checked (→ auto-publish?)
errors ─▶ a4_finish_check_job(code): 429/5XX/TIMEOUT/UNKNOWN → backoff 30·2^(n-1) s ≤ 600 s, max 3 runs;
          QUOTA → next IST day; BAD_FILE → rejected; KEY_MISSING → waiting_key; else failed
sweep: running > 6 min → requeue/fail
teacher ─▶ override_marks / publish_results / unpublish_results / POST /internal/regrade
```

## 8. Live numbers (prod DB, 2026-10-26, SELECT only)

| Item | Value | Class |
|---|---|---|
| Assignment-link copies ever | 23 (1 assignment, Class XII Economics, 30 marks), all `published` | [EVIDENCE] |
| Check Copies sheets | 5 (3 approved, 1 failed, 1 pending) | [EVIDENCE] |
| Teacher overrides | **0** (no `teacher_override` audit rows; `final_marks = ai_marks` on all 23) | [EVIDENCE] |
| Flagged `needs_review` | 15 / 23 (65 %), mean confidence 0.949 — all published unchanged | [EVIDENCE] |
| Model that produced the marks | 13 × `gemini-3.8-flash` (1 call), **10 × `gemini-3-flash-preview` (3 calls = primary and 1st fallback failed)** | [EVIDENCE] `check_jobs.usage` |
| Est. cost per copy | ₹3.09 avg (3.8-flash), ₹3.63 avg (preview); total ₹76 | [EVIDENCE] `usage.est_cost_inr` (estimate, not invoice) |
| Gemini time per grade | p50 115 s / max 148 s (3.8-flash); 35 s / 42 s (preview) | [EVIDENCE] `usage.stages` |
| Submission → result | 2 copies in < 10 min; **19 copies submitted 23 Oct 21:33–22:16 IST finished 24 Oct 17:31–18:35 (~20 h)** | [EVIDENCE] timestamps |
| Cause of the 20 h stall | Grade calls timing out at the old 60 s and the rubric step, fixed by 711a501 (stage timeouts) | [INFERRED FROM `job_worker.py:51-54` comment + commit 711a501] |
| Retries | 28 runs for 23 jobs | [EVIDENCE] |
| 4xx/5xx rates, provider rate-limit events, p95/p99 per stage from logs | not collected (Vercel logs not queried) | [HYPOTHESIS — missing artifact: Vercel log export] |

## 9. Tests and gaps seen in passing (input to Phase 1, not a finding list yet)

Present: `test_job_worker_unit.py` (18), `test_job_worker_pg.py` (31, real Postgres via pgserver), `test_preprocess.py` (11), `test_gemini_calls.py` (10), `test_inline_guard.py` (5), `test_ai_busy_routes.py` (4). [EVIDENCE]

Candidates for Phase 1 (each still needs its own evidence row):
1. **Mixed-model grading.** 43 % of live marks came from `gemini-3-flash-preview`, which failed grading in the 17 Oct test (memory: MAX_TOKENS, JSON array). The fallback chain changes the grader silently within one assignment. → accuracy / fairness, likely P0.
2. **65 % review rate with 0 overrides.** Either the flags are noise or teachers publish without reviewing. Neither is measurable today. → accuracy / UX.
3. **No unit test for `_score`** (verdict → marks, OR-groups, caps), and no parity test between Python `_apply_overrides` and SQL `a4_apply_overrides`, which must agree. [EVIDENCE] grep of `tests/`
4. **Check Copies PATCH override is read-modify-write without a lock or version** (`copy_checker_router.py:318-361`): two concurrent overrides on one sheet lose one. The audit insert is outside that write. [INFERRED FROM code]
5. **Check Copies `/check` has no in-progress guard**: a double submit runs two Gemini calls and writes two audit rows (counts twice against the daily limit). [INFERRED FROM `_check_sheet`]
6. **Chunk merge keeps the higher marks** when a question appears in two chunks (flagged for review, but auto-publish could publish it). [EVIDENCE] `portal_result.py:85-93`
7. **Daily spend cap counts every job touched today** by cumulative `usage`. A job updated today with yesterday's usage counts in full, so the cap errs toward pausing too early. [INFERRED FROM `job_store.spend_since`]
8. **`_score` keys answers by `sub_id`/`q_no`; a duplicate answer entry from the model silently overwrites the first.** [EVIDENCE] `:926-932`
9. **No CI**, so nothing enforces the existing 222 tests or a golden gate.
10. **Override → few-shot loop has no data**: 0 overrides exist.
