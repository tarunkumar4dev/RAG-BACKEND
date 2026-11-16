"""Cost attribution for paid AI calls that have no usage table (Gate 2 #8). No new tables.

log_usage(feature, user_id, usage) writes ONE structured log line

    AI_USAGE {"feature": "paper", "institute_id": "...", "user_id": "...", "outcome": "ok",
              "calls": 14, "input_tokens": ..., "output_tokens": ..., "est_cost_usd": ..., "models": {...}}

and returns the same dict, so callers can also store it on an existing row (papers: tests.request_payload.ai_usage).
Ids only — never names, prompts or question text. institute_id comes from institute_members (active row) and is
cached per user for 10 minutes; None for a teacher who is not in an institute.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("a4ai.usage")

_INST_TTL_S = 600
_inst_cache: dict[str, tuple[float, Optional[str]]] = {}
_lock = threading.Lock()


def _lookup_institute(user_id: str) -> Optional[str]:
    from app.core.database import get_supabase_admin
    rows = (get_supabase_admin().table("institute_members").select("institute_id")
            .eq("user_id", user_id).eq("status", "active").limit(1).execute().data) or []
    return str(rows[0]["institute_id"]) if rows and rows[0].get("institute_id") else None


lookup_institute = _lookup_institute   # replaceable in tests


def institute_for(user_id: Optional[str]) -> Optional[str]:
    if not user_id:
        return None
    now = time.monotonic()
    with _lock:
        hit = _inst_cache.get(user_id)
        if hit and now - hit[0] < _INST_TTL_S:
            return hit[1]
    try:
        inst = lookup_institute(user_id)
    except Exception as e:  # noqa: BLE001  (attribution must never break the paid call's response)
        logger.warning("institute lookup failed for usage log: %s", type(e).__name__)
        inst = None
    with _lock:
        _inst_cache[user_id] = (now, inst)
    return inst


def usage_from_response(model: str, response) -> dict:
    """{calls, input_tokens, output_tokens(+thinking), models} from one google-genai response."""
    um = getattr(response, "usage_metadata", None)
    tin = int(getattr(um, "prompt_token_count", 0) or 0) if um is not None else 0
    tout = (int(getattr(um, "candidates_token_count", 0) or 0) + int(getattr(um, "thoughts_token_count", 0) or 0)
            if um is not None else 0)
    return {"calls": 1, "input_tokens": tin, "output_tokens": tout, "models": {model: [1, tin, tout]}}


def log_usage(feature: str, user_id: Optional[str], usage: dict, outcome: str = "ok",
              est_cost_usd: Optional[float] = None, institute_id: Optional[str] = None) -> dict:
    rec = {
        "feature": feature,
        "institute_id": institute_id if institute_id is not None else institute_for(user_id),
        "user_id": user_id,
        "outcome": outcome,
        "calls": int(usage.get("calls", 0) or 0),
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "est_cost_usd": None if est_cost_usd is None else round(est_cost_usd, 6),
        "models": usage.get("models") or {},
    }
    logger.info("AI_USAGE %s", json.dumps(rec, separators=(",", ":")))
    return rec


def clear_cache() -> None:
    with _lock:
        _inst_cache.clear()
