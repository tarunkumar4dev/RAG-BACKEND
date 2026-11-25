-- 35_ai_unreadable.sql — AI_UNREADABLE is a final error code for check_jobs
--
-- The backend (job_worker._classify) already maps UnreadableReply to AI_UNREADABLE, but the
-- DB constraint on last_error_code doesn't include it, so a4_finish_check_job silently maps it
-- to UNKNOWN and re-queues the job (up to 24 retry rounds ≈ 12 h), burning billed calls each time.
-- After this migration AI_UNREADABLE is recognised and treated as final: the job goes straight to
-- state=failed (no re-queue), and the trigger sets the submission to status=failed.
--
-- Additive and idempotent. Rollback: 35_ai_unreadable_rollback.sql.
-- Needs: 26_checker_p0.sql (check_jobs_state_check includes failed_retryable, retry_rounds exists).

-- ── 1. Add AI_UNREADABLE to the last_error_code constraint ──
alter table public.check_jobs drop constraint if exists check_jobs_last_error_code_check;
alter table public.check_jobs add constraint check_jobs_last_error_code_check
  check (last_error_code in (
    'GEMINI_429', 'GEMINI_5XX', 'TIMEOUT', 'BAD_FILE', 'QUOTA', 'BUDGET', 'SCHEMA', 'CALL_CAP',
    'PAUSED', 'FILES_EXPIRED', 'KEY_MISSING', 'UNKNOWN', 'GEMINI_CONFIG',
    'AI_UNREADABLE'
  ));

-- ── 2. a4_finish_check_job: recognise AI_UNREADABLE as a valid code and treat it as final ──
-- The function body is the live 26_checker_p0 version with two changes marked NEW (35).
CREATE OR REPLACE FUNCTION public.a4_finish_check_job(
  p_job_id uuid, p_ok boolean,
  p_error_code text DEFAULT NULL::text, p_error text DEFAULT NULL::text,
  p_usage jsonb DEFAULT NULL::jsonb
)
 RETURNS check_jobs
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public'
AS $function$
declare
  j      check_jobs%rowtype;
  v_code text := case when p_ok then null
                      when p_error_code in ('GEMINI_429', 'GEMINI_5XX', 'TIMEOUT', 'BAD_FILE', 'QUOTA', 'BUDGET',
                                            'SCHEMA', 'CALL_CAP', 'PAUSED', 'FILES_EXPIRED', 'KEY_MISSING',
                                            'GEMINI_CONFIG',
                                            'AI_UNREADABLE')           -- NEW (35)
                           then p_error_code
                      else 'UNKNOWN' end;
begin
  select * into j from check_jobs where id = p_job_id for update;
  if not found or j.state <> 'running' then
    raise exception 'check job % is not running', p_job_id using errcode = '55000';
  end if;

  if p_ok then
    update check_jobs set state = 'done', finished_at = now(), locked_at = null, last_error_code = null,
                          error = null, usage = coalesce(p_usage, usage), updated_at = now()
     where id = j.id returning * into j;
  elsif v_code = 'QUOTA' then
    update check_jobs set state = 'blocked_quota', locked_at = null, retries = greatest(retries - 1, 0),
                          next_run_at = ((now() at time zone 'Asia/Kolkata')::date + 1)::timestamp at time zone 'Asia/Kolkata',
                          last_error_code = v_code, error = left(p_error, 300), updated_at = now()
     where id = j.id returning * into j;
  elsif v_code in ('PAUSED', 'BUDGET') then
    update check_jobs set state = 'paused', locked_at = null, retries = greatest(retries - 1, 0),
                          last_error_code = v_code, error = left(p_error, 300), updated_at = now()
     where id = j.id returning * into j;
  -- NEW (35): AI_UNREADABLE is always final — the model's reply was not JSON twice in a row; retrying
  -- the same copy wastes billed calls and won't produce a different answer.
  elsif v_code = 'AI_UNREADABLE' then
    update check_jobs set state = 'failed', finished_at = now(), locked_at = null,
                          last_error_code = v_code, error = left(p_error, 300),
                          usage = coalesce(p_usage, usage), updated_at = now()
     where id = j.id returning * into j;
  elsif v_code in ('GEMINI_429', 'GEMINI_5XX', 'TIMEOUT', 'GEMINI_CONFIG', 'UNKNOWN') and j.retries < j.max_retries then
    update check_jobs set state = 'queued', locked_at = null, next_run_at = now() + public.a4_job_backoff(j.retries),
                          last_error_code = v_code, error = left(p_error, 300), usage = coalesce(p_usage, usage),
                          updated_at = now()
     where id = j.id returning * into j;
  elsif v_code in ('GEMINI_429', 'GEMINI_5XX', 'TIMEOUT', 'GEMINI_CONFIG') and j.retry_rounds < public.a4_retry_rounds_max() then
    update check_jobs set state = 'failed_retryable', locked_at = null,
                          next_run_at = now() + public.a4_retry_round_wait(),
                          last_error_code = v_code, error = left(p_error, 300), usage = coalesce(p_usage, usage),
                          updated_at = now()
     where id = j.id returning * into j;
  else
    update check_jobs set state = 'failed', finished_at = now(), locked_at = null,
                          last_error_code = v_code, error = left(p_error, 300), usage = coalesce(p_usage, usage),
                          updated_at = now()
     where id = j.id returning * into j;
  end if;
  return j;
end $function$;

-- ── 3. verify (run by hand after applying) ──
-- select pg_get_constraintdef(oid) like '%AI_UNREADABLE%'
--   from pg_constraint where conname = 'check_jobs_last_error_code_check';
-- select public.a4_finish_check_job('00000000-0000-0000-0000-000000000000'::uuid, false, 'AI_UNREADABLE', 'test');
--   ^ expect: ERROR "check job 00000000-… is not running" (proves the code recognises the value)
