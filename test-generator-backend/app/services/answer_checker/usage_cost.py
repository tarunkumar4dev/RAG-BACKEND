"""What the checker spent: token counters, the ₹ estimate and the records that go to the database.

No Gemini calls, no database, no imports from the routers or the job worker, so both of them (and the scripts)
can use it. job_worker re-exports everything here under its old names.

Where usage is stored (see docs/audit/phase1a.md):
  check_jobs.usage                  job path: every run of every job added up, failed runs included
  check_sheets.ai_result['usage']   the final successful run of a copy (both routes)
  check_sheets.model_version        the model that marked the copy (both routes since 21 Nov 2026)
  check_sessions.rubric['build']    what building the rubric cost, all attempts (rubric_build_record)
"""
import os
import time
from typing import Optional

# Gemini paid-tier list prices for the checker model, USD per 1M tokens (input, output incl. thinking).
# Source: aistudio.google.com/docs/pricing, read 12 Nov 2026 (Gate 2 #7). gemini-3.8-flash: $0.75 / $3.75
# through 31 Dec 2026, $1.50 / $7.50 from 1 Jan 2027. The old defaults (Rs 26 / Rs 213) were gemini-2.5-flash
# prices and undercounted the checker ~1.5x. CHECKER_PRICE_*_PER_M_INR env vars still override.
INR_PER_USD = 84.0
_CHECKER_PRICES_USD = (            # (first day, input $/M, output $/M)
    ("2026-01-01", 0.75, 3.75),
    ("2027-01-01", 1.50, 7.50),
)


def _price_inr(which: int, today: Optional[str] = None) -> float:
    today = today or time.strftime("%Y-%m-%d")
    row = [p for p in _CHECKER_PRICES_USD if p[0] <= today][-1]
    return row[which] * INR_PER_USD


def price_in_per_m_inr(today: Optional[str] = None) -> float:
    env = os.getenv("CHECKER_PRICE_IN_PER_M_INR")
    return float(env) if env else _price_inr(1, today)


def price_out_per_m_inr(today: Optional[str] = None) -> float:
    env = os.getenv("CHECKER_PRICE_OUT_PER_M_INR")
    return float(env) if env else _price_inr(2, today)


_COUNTERS = ("gemini_calls", "billed_calls", "input_tokens", "output_tokens", "thinking_tokens")


def est_cost_inr(u: dict) -> float:
    out = (u.get("output_tokens") or 0) + (u.get("thinking_tokens") or 0)    # thinking is billed as output
    return round(((u.get("input_tokens") or 0) * price_in_per_m_inr() + out * price_out_per_m_inr()) / 1e6, 4)


def add_usage(total: dict, part: dict) -> dict:
    out = dict(total or {})
    part = part or {}
    for k in _COUNTERS:
        out[k] = int(out.get(k) or 0) + int(part.get(k) or 0)
    if part.get("model"):
        out["model"] = part["model"]
    models = set(out.get("models") or []) | set(part.get("models") or []) | ({part["model"]} if part.get("model") else set())
    if models:
        out["models"] = sorted(models)          # every model that answered a call of this job (all runs, chunks)
    if part.get("fallback_used"):
        out["fallback_used"] = True
    out["est_cost_inr"] = est_cost_inr(out)
    return out


def has_tokens(usage: Optional[dict]) -> bool:
    """The model's reply carried token counts: a billed call happened (a 429 / 5xx / timeout has none)."""
    u = usage or {}
    return bool(u.get("input_tokens") or u.get("output_tokens") or u.get("thinking_tokens"))


def rubric_build_record(usage: dict, prior: Optional[dict] = None, source: str = "sync") -> dict:
    """check_sessions.rubric['build']: tokens, calls, ₹ estimate and models of building the rubric.
    `prior` = the record of earlier builds that this one adds to (the sync route has no job row to add them up);
    the job path passes the job's own running total with prior=None (check_jobs.usage already adds every run)."""
    rec = add_usage(prior or {}, usage or {})
    rec.pop("stages", None)
    rec["source"] = source
    return rec


def answering_model(usage: Optional[dict], primary: str) -> Optional[str]:
    """The value for check_sheets.model_version: a model other than the primary wins (the copy is then flagged
    fallback_model), else the model that answered. None when the usage names no model at all: an unknown model is
    stored as unknown, never as the primary (the job worker still falls back to the primary; the direct route does
    not claim what it cannot know)."""
    u = usage or {}
    models = set(u.get("models") or []) | set((u.get("models_by_tag") or {}).values())
    if u.get("model"):
        models.add(u["model"])
    models.discard("mock")
    other = sorted(m for m in models if m != primary)
    return other[0] if other else (u.get("model") or None)


def without_build(rubric: Optional[dict]) -> Optional[dict]:
    """The rubric as an API may show it: no 'build' (what building it cost: tokens, ₹, model). Use it in every
    serializer that returns check_sessions.rubric, and before hashing a rubric to compare runs."""
    if not isinstance(rubric, dict):
        return rubric
    return {k: v for k, v in rubric.items() if k != "build"}
