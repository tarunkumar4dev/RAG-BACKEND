# Checker audit — Phase 1a (fixes that don't need teacher marks)

Branch `fix/checker-p0` in both repos · 2026-10-27 · evidence classes as in [inventory.md](inventory.md).
No prompt, rubric or grading-logic change. Nothing applied to production; migrations wait for an explicit OK.

## Deploy order (one way only)

1. Migration `25_review_reason_fallback.sql` (enum value, own transaction), then `26_checker_p0.sql` (product repo).
2. Backend `fix/checker-p0`. It writes `failed_retryable`, `GEMINI_CONFIG` and `fallback_model`, so it needs 25 and 26 first.
3. Frontend `fix/checker-p0`. It selects `check_sheets.review_approved_at` and calls `approve_review`, so it needs 26 first.

Rollback runs in reverse: frontend, then backend, then `26_checker_p0_rollback.sql`. The rollback was tested on a scratch Postgres
(pgserver, all migrations 18–26). 26 applies twice without error. The rollback removes every new object, and the five changed
functions come back byte-identical to the live `pg_get_functiondef` of 26 Oct. 26 re-applies cleanly after the rollback.

## 1. A backup model is never silent (P0)

- `_generate` (`smart_assignment_service.py`) calls the **primary model up to 3 times** on 429 / 500 / 503 / 504 / timeout,
  waiting 2 s then 5 s (for 429: Retry-After, capped at 30 s). It only retries while a whole call (its stage timeout) still fits
  before the job's deadline. The worker passes `deadline = run start + 270 s`.
- Errors 400 / 401 / 403 / 404 (bad or revoked key, model gone) raise `GeminiConfigError` at once. The worker logs
  `ALERT … check CHECKER_GEMINI_API_KEY / CHECKER_GEMINI_MODEL` at ERROR, and the job is retried later (code `GEMINI_CONFIG`).
- **Fallback is off** unless `CHECKER_ALLOW_FALLBACK=true`. When it is on and used, the meter says `fallback_used`, the copy gets
  the sheet flag `FALLBACK_MODEL` and the review reason `fallback_model`, and it is therefore never auto-published.
  A marking scheme built by a backup model flags every copy graded with it. `ai_result.meta.models = {rubric, grade[]}`.
- If the primary still fails and fallback is off, the job uses its 3 runs, then becomes **`failed_retryable`**. The submission
  stays `queued`, so nothing is published. The pg_cron sweep queues the job again after 30 min with fresh retries, for up to
  24 rounds (about 12 h). Then it becomes `failed`, and the teacher's "Check again" works as before.
- Teacher tab: an `AI: <model>` chip on every copy whose `model_version` isn't the primary. This also covers the 10 copies of
  24 Oct, which predate the reason. The new "Backup AI model" reason label is added too.
- Check Copies (the synchronous path) adds the same sheet flag. It has no review-reason column.

## 2. Flagged copies stay hidden until the teacher reviews them (P0)

[EVIDENCE] All 23 Economics copies were published together at **24 Oct 19:09 IST** with **Publish all**: 15 of them had
review reasons, 2 of them at 0/30 (one with `empty_answer; blank_pages; bad_scan`). Auto-publish was off for this assignment,
so the trigger was not the cause. `publish_results` published `needs_review` exactly like `checked`.

- `publish_results` (both Publish all and a row's Publish) now blocks a `needs_review` copy until
  `check_sheets.review_approved_at` is set. The blocked entry carries `reason: needs_review`.
- `approve_review(submission_id)` (new RPC, teacher of the assignment only) stamps the approval and publishes the copy (audited
  as `publish`). An edit in the drawer (`override_marks`) also counts as the review.
- Tab: a **Needs review: N** chip at the top (unapproved flagged copies; clicking it filters to them), "Review & approve" in
  place of Publish on those rows, and toasts that say why copies weren't published. Drawer: an "Approve & publish" bar.
- Clean copies auto-publish as before. **The 23 published copies are not touched.**

## 3. Shadow re-grade (cost gate)

`scripts/shadow_regrade.py` grades the same copies, rubric and key version again on the primary model only (temperature 0,
seed 7), the way the worker does: validation, chunks above 12 pages, review reasons. It only inserts into
`check_shadow_results`, which has RLS on, no policies, and no grants to anon or authenticated. There are no writes to
sheets, submissions, jobs or Storage, and no notifications. A re-run skips copies already stored under the label.
`export_ai_marks.py --run-label` exports `prod` or any shadow run. `compare_runs.py` diffs two exports per copy and question,
split by original model (primary or fallback).

**Dry run** (production DB, read-only, 27 Oct) for `bbd9b04b-…` (Foreign Exchange Rate – Class XII Economics):

```
copies: 23 checked, 23 with this rubric's key version
to grade: 23 copies, 81 uploaded pages/files
estimated cost: ₹73.88 (13 from their own past primary cost, 10 at the primary's ₹0.91/page)
primary avg ₹3.09/copy · what these copies cost originally (all models): ₹76.42
```

The real run needs migration 26 (the table) and your go.

## 4. Latency: the 23–24 Oct stall

**711a501 did not prevent it.** [EVIDENCE] 711a501 reached production with PR #4, whose merge deployed at
23 Oct 20:30 IST. The stalled copies were submitted 21:33–22:16 IST, after that. There was no backend commit on main between
that deploy and the 24 Oct deploys at 13:46, 17:27, 18:13 and 18:25. The copies finished 17:31–17:43 and 18:26–18:35, minutes
after the 17:27 and 18:13/18:25 redeploys. Two new copies submitted 24 Oct 18:00 and 18:05 also waited until just after 18:25.
The rubric was ready from 23 Oct 17:12 and the key never changed, so it wasn't the rubric.

**Most likely cause** [INFERRED FROM `vercel env ls`: `CHECKER_GEMINI_API_KEY` (Production) re-created "2d ago" = 24 Oct, and
the redeploy timing above]: the production Gemini key stopped working (invalid, revoked or out of quota) and was replaced on
24 Oct. The failure path in the code that was live:

1. Gemini answers 400/401/403 (key problem) or 429 (quota). `_generate` re-raises a 4xx (only 404/503 went to another model,
   and only 429/500/504 became `GeminiBusy`).
2. The worker classifies it as `UNKNOWN` or `GEMINI_429`. `a4_finish_check_job` requeues it with 30 s and 60 s backoff, and
   after the 3rd run sets `failed`. So the copy was failed within about 2 minutes of being claimed.
3. Submission `failed` → it waits for a human **Check again** (`retry_checks`), which resets `retries` to 0. That matches the
   final `retries = 1` on most of these jobs. Nothing alerted.

[HYPOTHESIS — not confirmable: Vercel Hobby runtime logs and `net._http_response` for 23–24 Oct have expired.] Two things
would settle it. Did you rotate the key on 24 Oct because of errors? And did you or Rahul Sir press Check again after the
deploys?

What's new against it: provider and key errors never end in `failed` by themselves (`failed_retryable`, §1). Key and model
errors log `ALERT` at ERROR. The view **`checker_job_health`** lists unfinished jobs with age, runs, rounds and last error, and
RLS applies, so a teacher sees only their own. Every sweep (pg_cron, every minute) raises a `WARNING` per job unfinished
15+ min after creation, with the job id and assignment id. `WARNING` is the highest level that doesn't abort the sweep; it
shows in the Supabase Postgres logs. Every worker run logs the same at **ERROR** in Vercel.

Remaining risk [EVIDENCE `check_jobs.usage`]: the 3.8-flash grade call took up to 148 s against the 150 s grade timeout. A
longer copy can time out every time. It now retries in rounds instead of failing, but it would keep timing out. That needs
a per-page timeout, which is a Phase 1b item and not changed here.

## 5. Upload audit (Economics link)

- `docs/audit/econ_submissions.csv` (git-ignored: real names) has, per attempt: roll, name, submission_id, created_at (IST),
  page_count, storage_object_count, submission/job state, job runs, model, review reasons, published_at (IST), marks, attempt_no.
- [EVIDENCE] 23 attempts by 20 students (17 × 1 attempt, 3 × 2). There are 81 files, all 81 present in Storage. 0 rejected,
  0 failed.
- Orphan uploads (objects under the link with no submission): **0 today**. But the hourly retention job deleted **13**
  unattached uploads on 24 Oct (17:47: 2, 20:47: 4, 21:47: 3, 22:47: 4). Retention only deletes files over 24 h old, so these
  were uploaded on 23 Oct in this link's hours (it was the only active link). [INFERRED] They are most likely students who
  uploaded and never completed `create_submission`, or replaced pages. Their names can't be recovered: the files are gone and
  orphans have no row.
- Upload-endpoint errors (429, size/page rejects, sha256 mismatch) for the window: **not available**. Uploads go
  browser → Storage directly, and `create_submission` is an RPC whose refusals aren't stored. Vercel Hobby runtime logs and
  Supabase API logs for 23–24 Oct have expired, and `access_code_attempts` keeps only the last day (12 rows now). None of the
  23 attempts has a reject reason.
  → Phase 1b proposal: an `upload_events` table (code, assignment, ip hash, at) written by `create_submission` refusals and the
  worker's validation, kept 30 days.

## Not done here (needs your decision or the teacher marks)

- The production `CHECKER_DAILY_BUDGET_INR` exists (created 22 Oct), but its value can only be read with your
  `vercel env pull` check.
- The real shadow run (≈ ₹74) needs your go after migration 26.
- The grade timeout margin above.

## Tests

Executed:
- Backend unit: 20 Gemini-call rules, 13 P0 worker, 2 compare/export, and the full `tests/unit` suite. The only failure is
  `test_cors_attempt_header::test_contest_submit_preflight_allows_attempt_id`, which also fails on `main`.
- Integration on real SQL (pgserver + migrations 18–26): 13 new tests in `test_checker_p0_pg.py`, plus the full existing suite
  (see the PR).
- Rollback test on a scratch DB.
- Frontend: vitest `src/lib/uploadLink` 72/72, eslint clean on touched files, `tsc` shows no errors in touched files (the repo
  has unrelated existing errors).

Not executed: a browser walk-through of the tab and drawer, and anything on production.
