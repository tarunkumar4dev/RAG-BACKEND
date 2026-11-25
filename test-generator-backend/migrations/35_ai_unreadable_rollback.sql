-- 35_ai_unreadable_rollback.sql — revert 35_ai_unreadable.sql
-- Any existing AI_UNREADABLE rows become UNKNOWN so the constraint passes.

update public.check_jobs set last_error_code = 'UNKNOWN' where last_error_code = 'AI_UNREADABLE';

alter table public.check_jobs drop constraint if exists check_jobs_last_error_code_check;
alter table public.check_jobs add constraint check_jobs_last_error_code_check
  check (last_error_code in (
    'GEMINI_429', 'GEMINI_5XX', 'TIMEOUT', 'BAD_FILE', 'QUOTA', 'BUDGET', 'SCHEMA', 'CALL_CAP',
    'PAUSED', 'FILES_EXPIRED', 'KEY_MISSING', 'UNKNOWN', 'GEMINI_CONFIG'
  ));

-- Restore 26_checker_p0 version of a4_finish_check_job (without AI_UNREADABLE branch).
-- Copy the function body from 26_checker_p0.sql lines 68–120.
