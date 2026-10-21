# Copy Checker v4 — design (Gate 0)

Status: G0 done, 21 Oct 2026. Source plan: "a4ai Copy Checker v4 — Build Plan (HLD + LLD)".
This file records what was verified on the live systems, where the plan had to be adapted, the final DDL,
and the font spike. Branch `feat/checker-v4` (from `origin/main` 4ccc147).

---

## 1. Verified facts (read-only checks, 21 Oct)

| Item | Finding | Source |
|---|---|---|
| Vercel plan | **Hobby**, team `taruns-projects-5d2bf9e6`, project `rag-backend` | `vercel api /v2/teams/…` |
| Function config | Fluid compute on, default timeout **300 s** (Hobby max), memory "standard" (2 GB / 1 vCPU), region **iad1** (US East) | `vercel api /v9/projects/rag-backend` |
| Python bundle limit | 500 MB uncompressed (other runtimes 250 MB) | Vercel docs, functions/limitations |
| Request body | 4.5 MB per function request/response | Vercel docs |
| PyMuPDF wheel | `pymupdf-2.0` manylinux: 69 MB wheel, **115.5 MB unzipped**. `api/requirements.txt` has it **unpinned** | `pip download` |
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
3. **Pin PyMuPDF** (`pymupdf==2.0.*`): 115 MB unzipped fits the 500 MB Python limit, but an unpinned major bump
   can change rendering. Add `rapidfuzz==3.14.*`.
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
  kind            text not null check (kind in ('check','export_zip')),
  teacher_id      uuid not null references auth.users(id) on delete cascade,
  institute_id    uuid references public.institutes(id) on delete set null,
  session_id      uuid not null references public.check_sessions(id) on delete cascade,
  sheet_id        uuid references public.check_sheets(id) on delete cascade,   -- null for export_zip
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
                  ('GEMINI_429','GEMINI_5XX','TIMEOUT','BAD_FILE','QUOTA','SCHEMA','CALL_CAP','PAUSED','UNKNOWN')),
  last_error      text,                 -- teacher-safe message only
  usage           jsonb not null default '{}'::jsonb,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now(),
  check (kind = 'export_zip' or sheet_id is not null)
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
  add column if not exists is_stale       boolean not null default false;
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
  primary key (scope_id, day)
);
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
  delete from rendered_files where created_at < now() - interval '30 days';   -- storage objects: worker deletes by path
  if exists (select 1 from check_jobs where status = 'queued' and run_after <= now()) then
    perform checker_kick();
  end if;
end $$;
select cron.schedule('checker-sweeper', '* * * * *', $$select public.checker_sweep()$$);

revoke all on function public.claim_check_job(int), public.reserve_checker_quota(uuid,text,int,int),
  public.checker_kick(), public.checker_sweep() from public, anon, authenticated;
grant execute on function public.claim_check_job(int), public.reserve_checker_quota(uuid,text,int,int)
  to service_role;

-- ── realtime (status only) ─────────────────────────────────────────────
alter publication supabase_realtime add table
  public.check_jobs (id, teacher_id, session_id, sheet_id, status, stage, last_error_code, updated_at);
```

Notes:
- The rendered_files cleanup in the sweeper must remove the Storage objects too. The worker does that (it lists
  expired rows before the delete), so the SQL `delete` above moves to the worker in G4.
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

## 6. Open questions for the owner

1. Vercel Hobby (non-commercial terms + monthly caps) — stay on Hobby through G2's load test, or move to Pro?
2. Move the function region iad1 → bom1?
3. Enable pg_net + pg_cron (before G2).
4. Supabase plan / storage quota (1.03 GB used today).
5. Gemini tier: RPM/TPM for `CHECKER_GEMINI_MODEL` from AI Studio → Usage & tiers.
6. Android wrapper: does it implement `DownloadListener` (https downloads) and/or `blob:` downloads? Where is its source?
7. OK to restore the paused project `zshgoopqjdoilkwiuupu` as the G2 load-test database?
