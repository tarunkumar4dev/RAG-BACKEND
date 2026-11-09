"""GET /agent/meta: who am I, what can I see, what is switched on (MetaResponse in agent/api_models.py)."""
from __future__ import annotations

from typing import Optional

from . import features
from .schemas import CONTRACT_VERSION, SessionCtx
from .suggest import EntityCache

SCOPE_CACHE = EntityCache()  # my_scope summary per user, 5 minutes (meta is called on every page load)


def _scope_loader(ctx: SessionCtx) -> list[dict]:
    from .tools.insights import MyScopeArgs
    from .tools.registry import active_tools
    res = active_tools(ctx.data_source == "live")["my_scope"].fn(ctx, MyScopeArgs())
    return [res]


def scope_summary(ctx: SessionCtx) -> Optional[dict]:
    try:
        res = SCOPE_CACHE.get(ctx, _scope_loader)[0]
    except Exception:  # noqa: BLE001  meta must answer even when the data DB is down
        return None
    batches = res.get("batches") or []
    roles = {b.get("role") for b in batches}
    if ctx.role == "admin":
        level = "institute"
    elif not batches:
        level = "none"
    elif "hod" in roles:
        level = "department"
    elif "proctor" in roles:
        level = "section"
    else:
        level = "classes"
    return {"level": level, "batch_count": len(batches), "student_count": int(res.get("students_in_scope") or 0),
            "batches": sorted(b["name"] for b in batches)[:50]}


def institute_name(ctx: SessionCtx) -> Optional[str]:
    try:
        if ctx.data_source == "live":
            from .db_real import connect_real
            with connect_real() as c:
                r = c.execute("select name from institutes where id = %s", (str(ctx.institute_id),)).fetchone()
        else:
            from .db import connect
            with connect(readonly=True) as c:
                r = c.execute("select name from institutes where id = %s", (str(ctx.institute_id),)).fetchone()
        return r["name"] if r else None
    except Exception:  # noqa: BLE001
        return None


def build_meta(ctx: SessionCtx) -> dict:
    live = ctx.data_source == "live"
    enabled = features.institute_enabled(ctx.institute_id)
    return {"contract_version": CONTRACT_VERSION, "enabled": enabled, "data_source": ctx.data_source,
            "read_only": live, "role": ctx.role, "first_name": ctx.first_name,
            "institute_name": institute_name(ctx) if enabled else None,
            "scope": scope_summary(ctx) if enabled else None,
            "features": {"paper_generation": not live, "defaulters_export": True, "proctor_month_report": True,
                         "cancel": True},
            "limits": {"chat_turns_per_10min": features.user_turns_per_10min(),
                       "institute_daily_tokens": features.institute_daily_tokens()}}
