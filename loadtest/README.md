# NCERT Test Builder load test

Zero-cost load test for the drag-and-drop Test Builder. It only calls two public, read-only,
DB-backed endpoints:

- `GET /api/v1/test-generator/ncert-question-stats`
- `GET /api/v1/test-generator/ncert-questions`

The locustfile allowlists these two paths and GET only. Any other path, including every
Gemini / export / payment / write endpoint, raises `BLOCKED` before a request is sent.
Hosts are allowlisted: only a branch preview alias `rag-backend-git-<branch>-taruns-projects-5d2bf9e6.vercel.app`
(never `git-main`) or localhost. Every production alias and per-deploy URL is refused at start.

Target for this test: `https://rag-backend-git-feat-loadtest-taruns-projects-5d2bf9e6.vercel.app`
No login: both endpoints are public.

## Scenarios

| User class | What it does |
|---|---|
| `TeacherUser` | Realistic: stats → prefetch first 3 chapters → open chapters, paginate, filter by type. 2–6 s waits. Vercel edge cache allowed. |
| `CacheBustUser` | Small: adds a random `_lt` param so Vercel's edge cache misses and every request reaches the Python function. The in-memory cache still hits (its key ignores unknown params), so this measures function capacity, not the DB. |

Request names start with a step number (`1 stats`, `2 prefetch chapter`, ...); `[bust]` marks
cache-bust requests. The `x-vercel-cache` HIT/MISS split is printed at the end and saved as
`<csv prefix>_vercel_cache.json`.

## One-time setup (Windows, VS Code terminal / PowerShell)

```powershell
cd C:\Users\login\Downloads\RAG-BACKEND
python -m venv loadtest\.venv
loadtest\.venv\Scripts\python -m pip install locust==2.32.4
```

## Preview deployment setup (Vercel dashboard, before the test)

1. Deploy branch `feat/loadtest` as a **preview** deployment.
2. Project → Settings → Environment Variables → add `LOADTEST_BYPASS_TOKEN` (a long random
   string) for the **Preview** environment only. Redeploy the preview.
3. Make sure "Automatically expose System Environment Variables" is on, so `VERCEL_ENV` is set.
   The bypass works only when `VERCEL_ENV == "preview"` **and** the header matches.
4. **After the test:** delete `LOADTEST_BYPASS_TOKEN` from Preview.

Generate a token in PowerShell:

```powershell
-join ((48..57)+(97..122) | Get-Random -Count 40 | % {[char]$_})
```

## Run

```powershell
$env:LOADTEST_BYPASS_TOKEN = "<same token as on Vercel Preview>"
# Only if the preview has Deployment Protection: Project → Settings → Deployment Protection
# → Protection Bypass for Automation → copy the secret
$env:VERCEL_PROTECTION_BYPASS = "<automation bypass secret>"
$PREVIEW = "https://<your-preview>.vercel.app"
mkdir loadtest\results -Force

# One step (repeat with 10, 25, 50, 100, 200, 400; wait 1 minute between steps)
loadtest\.venv\Scripts\locust -f loadtest\locustfile.py TeacherUser --headless -u 10 -r 2 -t 3m --host $PREVIEW --csv loadtest\results\10u

# Small cache-bust run
loadtest\.venv\Scripts\locust -f loadtest\locustfile.py CacheBustUser --headless -u 25 -r 5 -t 2m --host $PREVIEW --csv loadtest\results\bust25u
```

Spawn rate: about users / 5 (10→2, 25→5, 50→10, 100→20, 200→40, 400→80).

**Hard stops (the locustfile quits immediately, exit code 3):**
- any **429**: with the bypass on, this means the bypass isn't working
- any **403**, an `x-vercel-mitigated` header, or a Vercel "Security Checkpoint" challenge page
- any **401** or **3xx** (redirects are not followed): Vercel protection bypass not working
- any **non-JSON response** with a 2xx/3xx/4xx status (content-type not JSON, or the body
  fails to parse)

**5xx is not a hard stop.** It counts as a failure, named `HTTP <status>: <first ~100 chars of
body>` in `_failures.csv`, so a Vercel 504 timeout (`FUNCTION_INVOCATION_TIMEOUT ...`) is easy
to tell apart from a FastAPI 500 (`{"detail":"Something went wrong..."}`). Vercel request ids
are masked as `<vercel-id>` so identical errors group into one row. Connection errors and client
timeouts are also failures, not stops (`connection error: <ExceptionType>`).

**Stop conditions checked after each step (don't go to the next step):** failure rate > 5%,
p95 > 5 s, or errors that look like DB connection exhaustion (`ok:false` on stats, or 500s
mentioning connections).

The first 5 requests print live as `[first N/5] <status> x-vercel-cache=<HIT|MISS> <name>`
and are saved in `_vercel_cache.json`.

Secrets can also go in `loadtest\.env.loadtest` (gitignored), one `KEY=value` per line; the
locustfile loads it without printing values. Env vars already set in the shell take precedence.

Results: `loadtest\results\<N>u_stats.csv` (RPS, p50/p95/p99), `_failures.csv` (error types),
`_vercel_cache.json`.

## Backlog (not now)

- **200 + `ok:false` on DB errors.** `ncert-questions` / `ncert-question-stats` return HTTP 200
  with `{"ok": false}` when the DB fails. Monitoring and caches see success. Refactor them to
  return 5xx. (The locustfile already counts `ok:false` as a failure.)
- **Save write-test.** Test Builder "Save" inserts into `tests` / `questions` directly from the
  browser via Supabase. Not load tested yet; needs a throwaway test account plus cleanup.
- **Class 8** is in the Test Builder class dropdown, but `ncert_questions` has no class 8 rows.
