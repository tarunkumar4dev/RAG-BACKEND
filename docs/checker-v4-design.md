# Copy Checker v4 — design (Gate 0)

Status: G0 done, 21 Oct 2026. Source plan: "a4ai Copy Checker v4 — Build Plan (HLD + LLD)".
This file records what was verified on the live systems, where the plan had to be adapted, the final DDL,
and the font spike. Branch `feat/checker-v4` (from `origin/main` 4ccc147).

---

## 1. Verified facts (read-only checks, 21 Oct)

| Item | Finding | Source |
|---|---|---|
| Vercel plan | **Hobby**, team `taruns-projects-5d2bf9e6`, project `rag-backend` | `vercel api /v2/teams/…` |
| Function config | Fluid compute on, default timeout **300 s** (Hobby max), memory "standard" (2 GB / 1 vCPU), region was **iad1** (US East) → **bom1 since 21 Oct** (7.2) | `vercel api /v9/projects/rag-backend` |
| Python bundle limit | 500 MB uncompressed (other runtimes 250 MB) | Vercel docs, functions/limitations |
| Request body | 4.5 MB per function request/response | Vercel docs |
| PyMuPDF | **Correction (22 Oct):** 2.0 is 115 MB itself AND pulls onnxruntime + numpy (~200 MB) → prod bundle **351.63 MB** ("exceeds the standard size; optimizing dependencies" in the build log) → dependencies installed at runtime on cold start (8–90 s, /tmp full under burst). Pinned to **1.28.2** (the version all code and tests run on) | build log dpl_4bDL…, fix/deps-region c2af51a |
| rapidfuzz wheel | 3.14.6: 3 MB wheel, 12 MB unzipped (new dependency) | `pip download` |
| Supabase | project `a4ai.in`, region **ap-south-1** (Mumbai), Postgres 15.8 | `supabase projects list` |
| pg_net / pg_cron / pgmq | available, **not installed** | `pg_available_extensions` |
| supabase_vault | installed (0.3.1) — kick URL + secret go here | same |
| Storage used | ~1.03 GB total (Modules 944 MB, answer-sheets 2.3 MB / 18 files) | `storage.objects` |
| Upload path | Already client → Storage direct (`copyCheckerApi.uploadPage`, images compressed in browser). No file bytes through a function | code |
| answer-sheets bucket | private, 15 MB limit, jpeg/png/webp/pdf. Storage RLS: `foldername[1] = auth.uid()` (insert/read/delete) | `pg_policies` |
| check_sessions / check_sheets | owner-only RLS `teacher_id = auth.uid()`. **No institute_id column.** 12 sessions (3 teachers), 5 sheets, 3 checked, 4 with PDF pages | `information_schema`, counts |
| Roster | `check_sheets.student_id` → `students` (guard trigger + unique `(session_id, student_id)` from 17) | migration 17 |
| Parent phone | `students.parent_phone` exists, **0 / 43 filled**, no format constraint | live |
| Institute membership | max 1 active `institute_members` row per user; `institutes.plan_id` exists | live |
| Rate limiter | `main.py` middleware, **per IP + exact path**, in-memory per instance, 300/min default | code |
| Auth | `require_user` → `auth.get_user(token)` (network call to Mumbai per request) | `app/core/auth.py` |
| Realtime publication | institute_members, assignments, submissions, calendar_events, attendance_legacy, class_sessions, attendance_records — no checker table | `pg_publication_tables` |
| Gemini tier | **not verifiable from here** (limits are per project in AI Studio). Model from env `CHECKER_GEMINI_MODEL`, default `gemini-3.8-flash` | — |
| Mobile app | Play Store WebView wrapper; wrapper source is not in this repo or on this machine. Web code has no JS bridge | grep |

### Consequences
1. **Hobby plan.** 300 s max and 2 GB match the design (240 s work budget per kick). Open risks, need the owner's decision:
   Vercel's terms limit Hobby to non-commercial use, and Hobby has monthly caps on active CPU / provisioned memory
   / invocations. Check the usage page after G2's load test; the 500 copies/hour target may need Pro.
2. **Region mismatch.** Functions run in iad1 and the DB and Storage are in Mumbai. That is ~200–250 ms per
   round trip, and one sheet does ~20–30 DB/Storage calls plus page image transfers, i.e. several seconds per
   sheet. Proposal: set the function region to `bom1` (prod config change → owner).
3. **Pin PyMuPDF to 1.28.2** (corrected 22 Oct — 2.0 drags onnxruntime + numpy and blew the bundle to 351 MB).
   Add `rapidfuzz==3.14.*` later only after measuring the bundle.
4. **pg_net + pg_cron must be enabled** before G2 (owner action).
5. **Storage growth.** Per copy ≈ norm pages 7 × ~200 KB + marked PDF ≤ 1.5 MB + summary ~0.3 MB ≈ 3–3.5 MB, so
   500 copies ≈ 1.7 GB. The project already holds ~1 GB. Check the Supabase plan (Free = 1 GB). Rendered files are
   a cache: delete renders older than 30 days in the sweeper and re-render on demand.
6. **Mobile download.** `jsPDF.save()` produces a `blob:` download, which Android WebView only handles if the
   wrapper implements it. Signed https URLs with `download=<filename>` (Content-Disposition) work with a standard
   `DownloadListener` → `DownloadManager`. Need confirmation of what the wrapper does.
7. **Parent phones are empty** (0/43). WhatsApp links will mostly be the generic `wa.me/?text=` until rosters are
   filled. Untracked frontend migration `20261020000000_add_parent_phone.sql` runs `add column parent_phone`
   **without `if not exists`** on a column that already exists → it fails if run. Don't run it as is.

---

## 2. Changes to the plan (adapted to what exists)

| # | Plan said | Decision | Why |
|---|---|---|---|
| D1 | Teachers read/write their institute's sessions/sheets/jobs | **Owner scope stays** (`teacher_id = auth.uid()`). `institute_id` is stored (nullable) on jobs for quota and reporting only | Individual teachers have no institute, existing RLS and storage RLS are owner-based; institute-shared checks are a separate feature |
| D2 | Storage `{institute_id}/{session}/{sheet}/…` | `{teacher_uid}/{session_id}/{sheet_id}/{source,norm,rendered}/…`; class files under `{teacher_uid}/{session_id}/rendered/…` | Storage RLS is `foldername[1] = auth.uid()`; old v1 paths (`{uid}/{session}/{folder}/…`) stay readable |
| D3 | `check_sheets.roster_student_id` | Reuse **`check_sheets.student_id`** | Already the roster FK with a guard trigger + unique index (17) |
| D4 | `check_overrides` new | New table **+ backfill** from `ai_result.graded_questions[].teacher_marks/teacher_note` | Overrides live inside `ai_result` today |
| D5 | pg_net webhook on INSERT | **Statement-level** trigger (one kick per insert batch, not per row); URL + secret from Vault | A 40-file class upload = 1 kick, not 40 |
| D6 | Daily cap per institute | `checker_usage_daily(scope_id, scope_kind)` — institute if the teacher has an active membership, else the teacher; reserved atomically at enqueue | Individual teachers need a cap too |
| D7 | Per-user rate limit | Middleware keys on the JWT `sub` **decoded without verification** (bucket key only), IP when there is no or a bad token; routes still verify. `/internal/*` exempt, `/r/*` per IP 30/min. Also switch the key from exact path to route template (today every sheet id gets its own bucket) | Verifying in middleware = extra Mumbai round trip per request; a forged `sub` only fails later with 401 |
| D8 | Claim RPC `returns check_jobs` | `returns setof check_jobs` (empty = nothing to do), `security definer`, executable by `service_role` only. Worker claims a new job only while ≥ 120 s of its budget is left | PostgREST returns an all-null object for a null composite; no job started that can't finish |
| D9 | Realtime on check_jobs (status columns only) | Publication **column list** `(id, teacher_id, session_id, sheet_id, status, stage, last_error_code, updated_at)` + owner SELECT policy. Verify on the test project; fallback = 10 s poll (J12) | PG15 supports column lists; Supabase Realtime support for them is unverified |
| D10 | Total top-right of page 1 | Total + not-attempted list go in the **margin strip header**, not on the scan | Spike: top-right of the scan is where students write |
| D11 | Load test on a Supabase branch | Use the paused second project `zshgoopqjdoilkwiuupu` (free, ap-south-1) restored as test project, or a branch if the plan allows | ₹0, never prod |
| D12 | Fonts Noto → DejaVu fallback | Caveat Bold (marks) + Noto Sans Regular/Bold (remarks), both OFL, bundled in `app/fonts/`. MuPDF's built-in Noto Math/Symbols already cover √ ✓ ✗ (verified); DejaVu stays as the explicit CSS fallback | Spike result, section 5 |

Calls per sheet (for the Gemini RPM estimate): today 1 grade call (+ up to ~10 on retries). v4 target: B = 1
(single call) or 7 (per page), C = 0–1, D = 1–2 → **3–4 or 9–10**. G3's experiment decides.

---

## 3. Final DDL (migration `18_checker_v4.sql`, frontend repo; written in G2, run by the owner)

```sql
-- Requires 12_copy_checker.sql + 17. Extensions enabled by the owner first:
--   create extension if not exists pg_net;  create extension if not exists pg_cron;
-- Vault secrets (owner): select vault.create_secret('<worker url>/api/v1/internal/jobs/kick', 'checker_kick_url');
--                        select vault.create_secret('<random 32 bytes>', 'checker_kick_secret');

-- ── queue ──────────────────────────────────────────────────────────────
create table public.check_jobs (
  id              uuid primary key default gen_random_uuid(),
  kind            text not null check (kind in ('check','export_zip','retention')),
  teacher_id      uuid references auth.users(id) on delete cascade,             -- null only for retention
  institute_id    uuid references public.institutes(id) on delete set null,
  session_id      uuid references public.check_sessions(id) on delete cascade,  -- null for retention
  sheet_id        uuid references public.check_sheets(id) on delete cascade,    -- null for export_zip / retention
  status          text not null default 'queued'
                  check (status in ('queued','running','done','failed','blocked_quota','cancelled','paused')),
  attempts        int  not null default 0,
  max_attempts    int  not null default 3,
  run_after       timestamptz not null default now(),
  locked_at       timestamptz,
  finished_at     timestamptz,
  stage           text check (stage in ('A','B','C','D','E')),
  checkpoint      jsonb,
  idempotency_key text unique,          -- sha256(session_id + sorted file sha256s + mode)
  last_error_code text check (last_error_code in
                  ('GEMINI_429','GEMINI_5XX','TIMEOUT','BAD_FILE','QUOTA','BUDGET','SCHEMA','CALL_CAP','PAUSED',
                   'FILES_EXPIRED','UNKNOWN')),
  last_error      text,                 -- teacher-safe message only
  usage           jsonb not null default '{}'::jsonb,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),
  check ((kind = 'check' and teacher_id is not null and sheet_id is not null and session_id is not null)
      or (kind = 'export_zip' and teacher_id is not null and session_id is not null)
      or (kind = 'retention' and teacher_id is null and session_id is null and sheet_id is null))
);
create index check_jobs_runnable on public.check_jobs (run_after) where status = 'queued';
create index check_jobs_running  on public.check_jobs (locked_at) where status = 'running';
create index check_jobs_session  on public.check_jobs (session_id);
alter table public.check_jobs enable row level security;
create policy "own jobs read" on public.check_jobs for select using (teacher_id = auth.uid());
-- inserts/updates: backend (service role) only — no write policy

-- ── sheets ─────────────────────────────────────────────────────────────
alter table public.check_sheets
  add column if not exists file_sha256    text,
  add column if not exists schema_version int not null default 1,
  add column if not exists result_version int not null default 0,
  add column if not exists source_paths   text[],
  add column if not exists norm_pages     jsonb,      -- PageInfo[]
  add column if not exists student_match  jsonb,      -- {detected_name, detected_roll, confidence, confirmed_by, confirmed_at}
  add column if not exists is_stale       boolean not null default false,
  add column if not exists files_expired_at timestamptz;  -- source + norm pages deleted by retention → marked copy "expired"
create index if not exists check_sheets_sha on public.check_sheets (teacher_id, file_sha256) where file_sha256 is not null;

-- ── teacher overrides (separate from AI output) ────────────────────────
create table public.check_overrides (
  sheet_id   uuid not null references public.check_sheets(id) on delete cascade,
  q_no       text not null check (length(q_no) <= 20),
  teacher_id uuid not null references auth.users(id) on delete cascade,
  marks      numeric(5,1) not null check (marks >= 0 and marks * 2 = trunc(marks * 2)),
  note       text check (length(note) <= 500),
  edited_at  timestamptz not null default now(),
  primary key (sheet_id, q_no)
);
alter table public.check_overrides enable row level security;
create policy "own overrides" on public.check_overrides for all
  using (teacher_id = auth.uid())
  with check (teacher_id = auth.uid()
              and sheet_id in (select id from public.check_sheets where teacher_id = auth.uid()));
-- backfill from v1 ai_result (max_marks check stays in the API)
insert into public.check_overrides (sheet_id, q_no, teacher_id, marks, note, edited_at)
select s.id, gq->>'q_no', s.teacher_id, round((gq->>'teacher_marks')::numeric * 2) / 2,
       left(gq->>'teacher_note', 500), coalesce((gq->>'overridden_at')::timestamptz, s.updated_at)
from public.check_sheets s, jsonb_array_elements(s.ai_result->'graded_questions') gq
where gq->>'teacher_marks' is not null
on conflict do nothing;

-- ── render cache (service role only) ───────────────────────────────────
create table public.rendered_files (
  id               uuid primary key default gen_random_uuid(),
  teacher_id       uuid not null,
  sheet_id         uuid references public.check_sheets(id) on delete cascade,
  session_id       uuid not null references public.check_sessions(id) on delete cascade,
  kind             text not null check (kind in ('marked','summary','class_summary','zip')),
  result_version   int  not null,
  renderer_version text not null,
  path             text not null,
  bytes            int,
  created_at       timestamptz not null default now()
);
create unique index rendered_files_key
  on public.rendered_files (coalesce(sheet_id, session_id), kind, result_version, renderer_version);
alter table public.rendered_files enable row level security;   -- no policies

-- ── share links (service role only; API checks ownership) ──────────────
create table public.share_links (
  id             uuid primary key default gen_random_uuid(),
  token_hash     text unique not null,      -- sha256(token), token never stored
  sheet_id       uuid not null references public.check_sheets(id) on delete cascade,
  teacher_id     uuid not null,
  expires_at     timestamptz not null,
  revoked_at     timestamptz,
  views          int not null default 0,
  last_viewed_at timestamptz,
  created_at     timestamptz not null default now()
);
create index share_links_sheet on public.share_links (sheet_id);
alter table public.share_links enable row level security;      -- no policies

-- ── usage / quota ──────────────────────────────────────────────────────
create table public.checker_usage_daily (
  scope_id     uuid not null,                       -- institute id, or teacher id for individual teachers
  scope_kind   text not null check (scope_kind in ('institute','teacher')),
  day          date not null,                       -- Asia/Kolkata
  sheets       int not null default 0,
  gemini_calls int not null default 0,
  tokens       bigint not null default 0,
  est_cost_inr numeric(10,2) not null default 0,   -- from token counts × CHECKER_PRICE_* env (estimate, not the bill)
  primary key (scope_id, day)
);
create index checker_usage_daily_day on public.checker_usage_daily (day);

-- month-to-date estimated spend (all scopes), checked at enqueue against CHECKER_MONTHLY_BUDGET_INR
create or replace function public.checker_month_spend_inr() returns numeric
language sql stable security definer set search_path = public as $$
  select coalesce(sum(est_cost_inr), 0) from checker_usage_daily
   where day >= date_trunc('month', now() at time zone 'Asia/Kolkata')::date
$$;
alter table public.checker_usage_daily enable row level security;  -- no policies

create or replace function public.reserve_checker_quota(p_scope uuid, p_kind text, p_sheets int, p_cap int)
returns boolean language plpgsql security definer set search_path = public as $$
declare d date := (now() at time zone 'Asia/Kolkata')::date; n int;
begin
  insert into checker_usage_daily (scope_id, scope_kind, day, sheets) values (p_scope, p_kind, d, 0)
  on conflict (scope_id, day) do nothing;
  update checker_usage_daily set sheets = sheets + p_sheets
   where scope_id = p_scope and day = d and sheets + p_sheets <= p_cap
  returning sheets into n;
  return n is not null;
end $$;

-- ── claim (atomic, global cap) ─────────────────────────────────────────
create or replace function public.claim_check_job(p_max_running int)
returns setof public.check_jobs language plpgsql security definer set search_path = public as $$
begin
  perform pg_advisory_xact_lock(424242);
  if (select count(*) from check_jobs where status = 'running') >= p_max_running then return; end if;
  return query
  update check_jobs set status = 'running', locked_at = now(), attempts = attempts + 1, updated_at = now()
   where id = (select id from check_jobs
                where status = 'queued' and run_after <= now()
                order by run_after, created_at for update skip locked limit 1)
  returning *;
end $$;

-- ── kick (statement-level) + sweeper ───────────────────────────────────
create or replace function public.checker_kick() returns void
language plpgsql security definer set search_path = public as $$
begin
  perform net.http_post(
    url     := (select decrypted_secret from vault.decrypted_secrets where name = 'checker_kick_url'),
    headers := jsonb_build_object('Content-Type','application/json',
               'X-Internal-Secret', (select decrypted_secret from vault.decrypted_secrets where name = 'checker_kick_secret')),
    body    := '{}'::jsonb, timeout_milliseconds := 2000);
end $$;

create or replace function public.checker_kick_on_insert() returns trigger
language plpgsql security definer set search_path = public as $$
begin perform checker_kick(); return null; end $$;
create trigger check_jobs_kick after insert on public.check_jobs
  for each statement execute function public.checker_kick_on_insert();

create or replace function public.checker_sweep() returns void
language plpgsql security definer set search_path = public as $$
begin
  update check_jobs set status = 'queued', run_after = now() + interval '30 seconds', locked_at = null,
         last_error_code = 'TIMEOUT', updated_at = now()
   where status = 'running' and locked_at < now() - interval '6 minutes' and attempts < max_attempts;
  update check_jobs set status = 'failed', finished_at = now(), last_error_code = 'TIMEOUT',
         last_error = 'Checking took too long. Please retry.', updated_at = now()
   where status = 'running' and locked_at < now() - interval '6 minutes' and attempts >= max_attempts;
  -- retention: one job per hour; the worker deletes Storage objects (SQL can't), then the rows
  if not exists (select 1 from check_jobs where kind = 'retention'
                  and (status in ('queued','running') or created_at > now() - interval '1 hour')) then
    insert into check_jobs (kind) values ('retention');
  end if;
  if exists (select 1 from check_jobs where status = 'queued' and run_after <= now()) then
    perform checker_kick();
  end if;
end $$;
select cron.schedule('checker-sweeper', '* * * * *', $$select public.checker_sweep()$$);

revoke all on function public.claim_check_job(int), public.reserve_checker_quota(uuid,text,int,int),
  public.checker_kick(), public.checker_sweep(), public.checker_month_spend_inr() from public, anon, authenticated;
grant execute on function public.claim_check_job(int), public.reserve_checker_quota(uuid,text,int,int),
  public.checker_month_spend_inr() to service_role;

-- ── realtime (status only) ─────────────────────────────────────────────
alter publication supabase_realtime add table
  public.check_jobs (id, teacher_id, session_id, sheet_id, status, stage, last_error_code, updated_at);
```

Notes:
- Retention rules: section 7.4. Budget / spend: section 7.5.
- `MAX_RUNNING_JOBS` (env, default 10) is passed to `claim_check_job`. Kill switch `CHECKER_ENABLED=0` → enqueue
  writes `paused`; daily cap → `blocked_quota`.

---

## 4. Schema v2

`test-generator-backend/app/services/answer_checker/schema.py` — Pydantic models from the plan (section 4), plus the
Gemini response models (`Transcription`, `Grading`) kept flat for `response_schema`. Differences from the plan:
`SheetResult.student` is a typed `StudentInfo` (with `student_id` = `check_sheets.student_id`), and `Segment.lines`
defaults to `[]` so v1 rows adapt cleanly. The TS mirror `src/types/checker.ts` is written in G3, when the frontend
first reads v2.

---

## 5. Font spike (G0)

Script: `test-generator-backend/scripts/spikes/font_spike.py` (PyMuPDF 2.0 = the version prod installs).
Output: `docs/checker-v4/spike_marked_g0.png`. Synthetic page, 1400 × 1980 px scan + 420 px margin strip.

| Check | Result |
|---|---|
| Glyphs θ √ ² ³ ° ₹ × − ✓ ✗ ½ and O₂ | All render. Noto Sans lacks √ ✓ ✗ — MuPDF fell back to its built-in Noto Sans Math / Symbols / Symbols2 automatically, no □ |
| Caveat Bold | lacks θ √ ✓ ✗ → used only for marks/total (digits, `/`, `.`); ticks/crosses are vector shapes |
| Size | 1 page = 356 KB (JPEG 196 KB + subset fonts). **Before `subset_fonts()` it was 4.1 MB** → subsetting is mandatory. Fonts are shared across pages, so 7 pages ≈ 1.4–1.5 MB at JPEG q82 |
| Speed | ~600 ms for one page incl. 5 dry-run measurements (local) |
| Layout | 5 margin items, no overlap; pushed items get a red marker at the answer's height + dashed leader inside the strip |

Gotchas found (now rules for G4):
1. MuPDF's HTML engine ignores the universal selector: `* {margin:0}` does nothing, `<p>` keeps a 1 em top/bottom
   margin (that made every item ~110 px too tall). Use `p {margin:0; padding:0}`.
2. Measure with `insert_htmlbox(..., scale_low=1)` on a scratch page; `used = rect.height - spare_height`.
3. Always `doc.subset_fonts()` then `save(garbage=4, deflate=True)`.
4. The half-tick (tick + bar) reads like ✗ at small size → redraw in G4 (tick with a short bar across the long
   stroke, or "½" next to the tick).
5. The total drawn top-right lands where students write → D10.

---

## 6. Owner decisions (21 Oct)

| # | Question | Decision |
|---|---|---|
| 1 | Vercel plan | Hobby through G1–G3. Pro decided before launching to paying institutes (terms + compute). Usage estimate in 7.1 |
| 2 | Region | **Switched to bom1 on 21 Oct** (project setting + redeploy of the same commit 4ccc147). Results in 7.2 |
| 3 | pg_net + pg_cron | Owner enables them at the **start of G2** — remind then, never enable from code/CLI |
| 4 | Supabase | **Free plan**, storage 1.03 GB (over 1 GB). Cleanup proposal 7.3 (nothing deleted yet), retention 7.4 |
| 5 | Gemini | Owner sends RPM/TPM. Paid project capped at ₹5,000/month; checker must stay under `CHECKER_MONTHLY_BUDGET_INR` (default 3000) → 7.5 |
| 6 | Android downloads | Rule: every download is an **https signed URL** with `Content-Disposition: attachment` and a proper filename. **Never `blob:` URLs.** Owner will share the wrapper location |
| 7 | Load test DB | Restore `zshgoopqjdoilkwiuupu` for the G2 load test only. Never load-test prod |

---

## 7. Addendum (21 Oct)

### 7.1 Vercel usage and the 500-copy evening

Current month (1–21 Oct, whole team, almost all functions = rag-backend; `vercel api /v2/usage?type=requests`):

| Metric | Value |
|---|---|
| Function execution | **1.64 GB-h** (peak day 17 Oct: 0.66 GB-h); 0 timeouts, 0 throttles |
| Function invocations | 1,785 ok, 7 errors |
| Edge requests | 136,129 (99 % the a4ai frontend) |
| Outgoing bandwidth | 12.55 GB |

Hobby's included limits can't be read via the API (usage-limits endpoint is Pro-only) → check the dashboard Usage page.

Estimate for one 500-copy evening (v4 pipeline, ~75 s wall time per sheet, mostly waiting on Gemini, 2 GB instances):
- worst case (every job on its own instance): 500 × 75 s × 2 GB = **≈ 21 GB-h**
- expected with fluid concurrency (~4 jobs sharing an instance, `MAX_RUNNING_JOBS=10`): ≈ 2.6 instance-h × 2 GB = **≈ 5 GB-h**
- renders on demand: ~1–3 s × 2 GB each → +0.5–1 GB-h if every copy is downloaded once

So one busy evening ≈ 3–13× everything the backend used this month so far. G2's load test (mocked, on the test
project) measures the real number from the Vercel usage API before/after.

Supabase egress (Free plan has a monthly egress cap — check the dashboard): per 500 copies ≈ worker reads of
sources + norm pages (~1.4 GB) + one download of each marked copy and summary (~0.9 GB) ≈ **2.3 GB**.

### 7.2 Region switch iad1 → bom1 (21 Oct)

Side-effect-free probe `test-generator-backend/scripts/latency_probe.py` (no Gemini, no payments, no writes), 7 runs after a warm-up, from India:

| Probe | iad1 median | bom1 median |
|---|---|---|
| `/health` (no DB) | 323 ms | 328 ms |
| `GET /test-generator/chapters` (DB read) | 955 ms | **177 ms** |
| `POST /copy-checker/sheets/…/check` with an invalid JWT (Supabase Auth round trip → 401) | 942 ms | **152 ms** |

Deployment `rag-backend-9iqow8gr9` (regions `['bom1']`, commit 4ccc147) is aliased to rag-backend-kappa-three.vercel.app.
Authenticated smoke tests (login, test generation, payment order creation, checker) need a real session → owner,
in the browser.

### 7.3 Storage (Free plan, 1,027.9 MB) — cleanup proposal (NOT executed)

| Bucket | Files | MB |
|---|---|---|
| Modules (private) | 1,277 | 944.3 |
| ncert-diagrams | 448 | 53.8 |
| faiss-index | 2 | 9.0 |
| notes | 11 | 6.8 |
| assignments | 15 | 5.3 |
| institute-assets | 6 | 5.2 |
| answer-sheets | 18 | 2.3 |
| others (submissions, resumes, question-images, branding) | 19 | 1.2 |

Modules bucket by kind (vs `public.modules.storage_path`, 52 rows):

| Kind | Files | MB | Read again after use? |
|---|---|---|---|
| `{uid}/worksheets/*` — worksheet generator uploads | 85 | 333.8 | **No.** `/worksheet/generate-direct` downloads once, nothing stores the path |
| module source referenced by a `modules` row | 52 | 304.1 | Yes (keep) |
| module source **not** referenced by any `modules` row (failed / abandoned creates) | 76 | 283.3 | **No** |
| `{uid}/{module}/images/*` | 1,064 | 23.1 | Yes (keep) |

Exact duplicates by eTag: 143 groups, 846 MB of extra copies (same PDFs uploaded again and again).

**Proposal (owner approves, then a one-off script deletes through the Storage API):**
- B: all 85 worksheet uploads (333.8 MB)
- C: all 76 unreferenced module sources (283.3 MB) — only those older than 7 days at deletion time, so an upload
  in progress is never touched
- → 161 files / **617 MB**, storage ≈ 1,028 → **≈ 411 MB**
- Full list: `storage_cleanup_candidates_2026-10-21.csv` (bucket, path, MB, created, reason) — shared with the owner, not committed.

Code follow-ups (outside the checker, separate PRs): worksheet generator deletes its upload after generation;
ModuleCreator deletes the upload when `/modules/create` fails; dedupe module uploads by hash.

### 7.4 Retention policy (checker files)

| What | Kept | Then |
|---|---|---|
| Source uploads (`…/{sheet}/source/*`) | `CHECKER_RETENTION_DAYS` (default **90**) after the sheet's `checked_at` (or `created_at` if never checked) | deleted |
| Normalised pages (`…/{sheet}/norm/*`) | same | deleted; `check_sheets.files_expired_at = now()` |
| Marked copy render (`rendered/…_marked.pdf`) | cache, 30 days after creation | deleted; re-rendered on demand from norm pages while they exist; after `files_expired_at` the download API returns `FILES_EXPIRED` and the UI shows "Marked copy expired (photos are deleted after 90 days)" |
| Student summary render (`…_summary.pdf`) | **kept** (no images, ~50–300 KB) | — |
| Class summary + ZIP | class summary kept; ZIP 7 days | ZIP deleted (re-export is a new job) |
| Result JSON, overrides, marks | kept | — |
| Old v1 photos (`{uid}/{session}/{folder}/…`) | same 90-day rule, by `check_sheets.created_at` | deleted, `files_expired_at` set |

Implementation (G2): the sweeper enqueues one `retention` job per hour; the worker deletes up to 500 objects per
run through the Storage API (SQL can't delete Storage objects), then updates the rows. Deletions are logged by
count only (no paths of minors' files in logs). Env: `CHECKER_RETENTION_DAYS`, `CHECKER_RENDER_CACHE_DAYS=30`,
`CHECKER_ZIP_DAYS=7`.

### 7.5 Spend control (`CHECKER_MONTHLY_BUDGET_INR`, default 3000)

- After every Gemini call the worker adds `gemini_calls`, `tokens` and `est_cost_inr` to the scope's
  `checker_usage_daily` row. Price per 1M input/output tokens comes from env (`CHECKER_PRICE_IN_PER_M_INR`,
  `CHECKER_PRICE_OUT_PER_M_INR`) so a model/price change is config, not code.
- At enqueue: if `checker_month_spend_inr() + est_cost_per_sheet × n > CHECKER_MONTHLY_BUDGET_INR` → jobs are
  created as `paused` with `BUDGET` ("Monthly checking budget reached"). `est_cost_per_sheet` = rolling average of
  the last 50 sheets (fallback: env `CHECKER_EST_COST_PER_SHEET_INR`).
- The worker re-checks before every Gemini call; at ≥ 100 % it stops after the current stage and re-queues.
- Daily per-scope cap (`reserve_checker_quota`, default 200 sheets/day) stays as the second guard.
- The ₹5,000 project cap in Google stays the last line of defence; the checker budget is below it so test
  generation and chat keep working.

### 7.6 Parent phone

- `students.parent_phone` exists (0 / 43 filled). The untracked frontend migration
  `20261020000000_add_parent_phone.sql` was **deleted** (21 Oct): its `add column` (no `if not exists`) would fail,
  and its format check `^\+91[6-9]\d{9}$` would break BulkStudentUpload and the teacher "Add student" form,
  which save the number as typed (only StudentProfilePage saves `+91XXXXXXXXXX`).
- **G5 task:** parent phone in the roster form + CSV import with Indian-mobile validation and normalisation to
  `+91XXXXXXXXXX` in every writer (StudentProfilePage, TeacherStudentsTab add/edit, BulkStudentUpload); report
  invalid rows instead of failing the import; then a migration adds the check as `NOT VALID`, normalises existing
  rows, and `VALIDATE CONSTRAINT`.

### 7.7 Reminders by gate

- **G1:** rate limiter app-wide (route template + user id; IP only for public routes; `/internal/*` exempt; every
  route + limit in the PR description).
- **G2 start:** owner enables pg_net + pg_cron; restore `zshgoopqjdoilkwiuupu` for the load test; retention job.
- **G5:** parent phone task (7.6).

---

## 8. G1 — implemented (21 Oct)

- **Gemini calls (rule 2):** `_generate` = one attempt per model; fallback only on 404/503; 429/500/504 →
  `GeminiBusy` (HTTP 503 "AI service is busy…" until G2 re-queues); `_generate_json` = at most 1 retry (2 s);
  `usage_meter()` counts calls + tokens per sheet / rubric and raises `CallCapExceeded` at
  `MAX_GEMINI_CALLS_PER_SHEET` (12). Request timeout `CHECKER_GEMINI_TIMEOUT_S` (90). Files go **inline** (no
  Files API upload/delete — 14 fewer API calls for 7 pages, nothing kept at Google); Files API only above 14 MB.
  `GEMINI_MOCK=1` + `GEMINI_MOCK_DIR` = recorded replies, zero cost.
- **Stage A** (`answer_checker/preprocess.py`): PDF/photos → JPEG (150 dpi, ≤ 2000 px, q80), BadFile errors with
  no Gemini call (empty, wrong type, corrupt, password, > 16 pages), blank pages (ink < 0.02 % after removing
  ruled lines / margins / shadows) are not sent. Rotation + student page order come from the SAME grade call
  (`page_meta=True`, no extra call); pages are turned upright, reordered, answer positions remapped, flags
  `PAGES_REORDERED` / `PAGE_ROTATED` / `BLANK_PAGE`.
- **Storage:** final pages at `answer-sheets/{uid}/{session}/{sheet}/norm/p{i}.jpg`; `ai_result.norm_pages`,
  `ai_result.sheet_flags`, `ai_result.usage`. No migration in G1: these live in `ai_result` (JSON); G2's migration
  adds the `norm_pages` column and backfills it.
- **Stopgap red pen** (frontend): marks drawn on `norm_pages` (PDF uploads work, rotated pages are upright);
  delivered as `…/{sheet}/rendered/marked_<ts>.pdf` (older ones removed) + 10-min signed URL with
  `download=<file name>` — no `blob:` URL.
- **Rate limiter:** `app/core/rate_limit.py`; route list + limits in `docs/checker-v4/g1-route-limits.md`.
- **Proof:** `scripts/g1_marked_copy_demo.py` (mocked) → 35/40, all 17 questions = expected, flags raised, 1 + 1
  calls; frontend `buildMarkedCopyPdf` run in headless Chrome on that output → `docs/checker-v4/g1/*.png`.
- **Tests:** `pytest` in `test-generator-backend` (33 tests, no API calls); preprocess tests also pass on
  PyMuPDF 2.0. `api/requirements.txt` pins `PyMuPDF==1.28.2` (2.0 bloats the bundle, see §1).

### 8.1 Before deploy (21 Oct)

- **Gemini inline limit:** docs disagree — file-input page: 100 MB per request (50 MB PDFs, base64-encoded per
  the Jan 2026 blog); image-understanding page: 20 MB total request with inline images. We use the stricter:
  base64-encoded request ≤ 18 MB (`CHECKER_INLINE_BUDGET_BYTES`), 1 MB reserved for the prompt. Ladder 2000 →
  1600 → 1280 px, then `BadFile("Too many pages in one copy, max N")`, no call. 20 real photo pages at 2000 px
  = 4.2 MB encoded → the page cap (16) binds long before the size.
- **429:** one retry after Retry-After / RetryInfo (default 5 s, cap `CHECKER_MAX_RETRY_WAIT_S` = 30), then
  503 `{"detail": msg, "error": {"code": "AI_BUSY"}}` (copy checker + smart-assignment routes).
- **Student routes** (no login): student portal (access code) = Supabase RPCs only (`get_student_by_code`,
  `get_student_feed`, `submit_assignment`) + Storage upload — never the backend. Mega contests = Supabase
  tables directly. Backend: contest `info` / `start` / `leaderboard` and quiz `q/{slug}` / `start` /
  `leaderboard` → per IP 1200/min (no identity yet); contest + quiz **submit** → per `X-Attempt-Id` (30/min),
  all attempts from one IP ≤ 3000/min, IP fallback 1200/min for clients without the header.
- **Bug found:** backend CORS didn't allow `X-Attempt-Id`, so the browser preflight for contest submit got a
  400 in production (checked 21 Oct). Fixed with this change.
- **Prod env:** `RATE_LIMIT_PER_MINUTE` is not set → 60/min default. `CHECKER_GEMINI_MODEL` is a "sensitive"
  var (value not readable via `env pull`).

### 8.2 Page cap (21 Oct)

The old 16-page cap had no size basis (it dates from Check Copies P1, when a whole copy went to Gemini through the
Files API). Now: the inline budget decides (`fit_pages`: 18 MB base64, 2000 → 1600 → 1280 px), with a hard
maximum of **30 pages** (`COPY_CHECKER_MAX_PAGES`, capped at 30; frontend `MAX_PAGES` = 30). One router + one
preprocess constant. 20 phone-photo pages at 2000 px ≈ 4.2 MB → a 30-page copy fits at full size in normal cases.

**G3 task — long copies in chunks:** a copy whose pages don't fit one call's budget (or > ~12 pages, to keep the
Stage B/D replies short and the latency bounded) is split into chunks of 10–12 consecutive pages. Each chunk gets
the same rubric; Stage C merges segments across chunk boundaries (continuations, a question started in chunk 1
and finished in chunk 2) before grading, so grading still sees whole answers. Chunk results merge by question;
page indexes stay global. Calls per sheet grow with chunks → counted against `MAX_GEMINI_CALLS_PER_SHEET`.

### 8.3 Real copies test set

Real copies are **never in git** — not raw, not anonymised, not in a private repo (owner decision, 21 Oct). They
live in `REAL_COPIES_DIR` (outside every repo, backed up to the owner's Google Drive). In the repo:
`tests/fixtures/real/README.md` (layout + expected.json), `scripts/anonymise_copies.py` (page-1 header band +
redact.json boxes filled solid, EXIF dropped, identifier keys stripped, folders renamed copy_NN, mapping kept in
raw/; refuses any folder inside a git repo), `tests/real/test_real_copies.py` (reads
`$REAL_COPIES_DIR/anonymised`, skipped when unset), `.gitignore` for everything in `tests/fixtures/real/` except
the README, guard test `tests/unit/test_real_fixtures_guard.py`.
