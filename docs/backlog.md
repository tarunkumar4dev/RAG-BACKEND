# Backlog

Items outside the current gate, with a proposal each. Order = priority when the next slot opens.

## 1. "Prevent silent breakage" PR — right after the G1 deploy, before G2 (approved 21 Oct)

- **CORS contract test.** `contracts/browser_headers.json` lists every custom header the frontends send to the
  backend (`X-Attempt-Id` today). Backend test: each one is in `allow_headers` and its preflight returns 200;
  frontend check: a `fetch(` with a header not in the list fails CI.
- **Daily prod smoke test** (GitHub Action, both repos, no login, no writes): preflight per listed header (200)
  + an unknown header (400), `/health`, `GET /test-generator/chapters` (DB), invalid JWT on a checker route (401
  = Supabase Auth reachable), `https://www.a4ai.in` (200). Failure → GitHub's failure email.
- **Sentry, free plan, EU region** — backend `sentry-sdk[fastapi]` (`SENTRY_DSN`), frontend `@sentry/react`
  (`VITE_SENTRY_DSN`, source maps with `SENTRY_AUTH_TOKEN`). PII scrubbing in `before_send` / `beforeSend`:
  drop Authorization / Cookie / request bodies / query strings, `send_default_pii=False`, no user object beyond
  the id, scrub name / roll / phone patterns, session replay off; error sample 100 %, traces 5–10 %.
  Owner creates the account + 2 projects and adds the three env vars in Vercel.
- Later (after G2, with a dedicated prod test account): logged-in smoke test (login, test generation).

## 2. Contests without an end time stay "active" forever

Found 21 Oct: `contests.expires_at` is optional; `contest_service` marks a contest `ended` only when
`expires_at` has passed, so a contest created without one never ends, and an attempt that never submits stays
`in_progress` (e.g. contest `ftk5i8zd`, internal test — left as is).

Proposal:
- **Attempt deadline (always):** an attempt ends at `started_at + duration_minutes + 5 min grace`. A submit
  after that is rejected with "Time is up"; answers saved so far are scored. Closing is lazy (checked on
  submit / leaderboard / info) so it needs no cron; the G2 sweeper (pg_cron) also closes stale `in_progress`
  attempts as `timed_out` once a day.
- **Contest end (default):** on create, if the teacher sets no end, `expires_at = (scheduled_at or now) +
  7 days` (choice in the create form: 1 day / 3 days / 7 days / custom). Existing rows with `expires_at IS NULL`:
  one migration sets `created_at + 30 days` (owner decides the number before it runs).
- **Teacher "End contest" button:** `POST /contests/{id}/end` (owner only) → `status = ended`,
  `expires_at = now()`, in-progress attempts scored from saved answers, leaderboard final. The share link then
  shows "This contest has ended".

## 3. Parent phone in the roster (G5)

See `docs/checker-v4-design.md` §7.6.

## 4. Long copies in chunks (G3)

See `docs/checker-v4-design.md` §8.2.
