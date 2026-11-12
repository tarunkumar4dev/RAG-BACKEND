# Branch `checker-bundle`: what it contains and what to change at merge

This branch starts from `claude/youthful-archimedes-tw0p0s`. It has not been merged or deployed.

| Commit | What |
|---|---|
| cherry-pick of 904e318 | Checker fallback `gemini-2.5-flash` and `gemini-3-flash-preview` replaced by **`gemini-3.6-flash`**. That model is Stable, has no shutdown date, and costs the same as `gemini-3.8-flash`: $0.75 / $3.75 per million tokens until 31 Dec 2026, then $1.50 / $7.50. |
| d7e224d | Feedback states the reason for every deduction. When an attempted answer gets less than full marks, the server puts the weakest marking point's reason first in the feedback, as one line. It's free (no prompt change, no extra call), and marks don't change. Kill switch: `CHECKER_FEEDBACK_REASON=0`. |
| dc27769 | A total deadline for each copy-check job run, so that retries, fallbacks and waits always finish inside the function's maximum duration. |

## Vercel environment variables to set at merge (Production)

| Variable | Set to | Why |
|---|---|---|
| `CHECKER_GEMINI_FALLBACK_MODELS` | `gemini-3.6-flash`, or **delete the variable** | If it's set, it overrides the new code default, and the old list (`gemini-2.5-flash,gemini-3-flash-preview`) would stay live. |
| `CHECKER_GEMINI_MODEL` | keep `gemini-3.8-flash` | This is the primary model and is unchanged. |
| `CHECKER_JOB_MAX_DURATION_S` | **the real Vercel `maxDuration` of the function in seconds** (TODO, Tarun) | The code defaults to 300. If the real limit is lower, a run can still be killed mid-call. |
| `CHECKER_JOB_RUN_MARGIN_S` | leave unset (30) | Time kept back for database writes and the response. |
| `CHECKER_GRADE_TIMEOUT_S` | leave unset (150), or keep it at or below `MAX_DURATION − 30 − 20` | Timeout for one grade call. |
| `CHECKER_GEMINI_TIMEOUT_S` | leave unset (60) | Timeout on the sync path, where a teacher is waiting. |
| `CHECKER_PRICE_IN_PER_M_INR` / `CHECKER_PRICE_OUT_PER_M_INR` | **delete both** (recommended) | If they're unset, the dated list price applies, and it switches by itself on 1 Jan 2027. October's stored costs suggest these are set to about ₹73 / ₹362. That's inferred from the stored costs; I haven't read the env. Values set by hand won't follow the January price doubling. |
| `CHECKER_FEEDBACK_REASON` | leave unset (on) | Set it to `0` only to switch the new feedback line off. |

## Open decision (not changed here)

**Agent `MAX_LLM_STEPS = 10`** (`agent/config.py`): one agent question can still be billed for up to 10 model calls. Retries are capped by `TURN_RETRY_BUDGET=2`, but the number of steps isn't. Lowering it would limit multi-tool flows: the paper flow uses blueprint → generate → verify → save. This needs a decision before agent launch (fix #9).
