"""Plan quota for paid AI actions (papers, worksheets). One place, used by every endpoint that spends money.

Gate 2 #5 — reserve -> generate -> consume | release:
  reserve(user, action, request_id)  a4_reserve_usage: lock + limit check + count in ONE statement, keyed by the
                                     client's Idempotency-Key (a repeat answers 409, nothing is charged twice).
  consume(r)                         a4_consume_usage: the unit stays counted.
  release(r)                         a4_release_usage: generation failed, the unit goes back.
Until migration 32 (supabase/migrations/32_quota_reserve_voice_allowance.sql, product repo) is applied those
functions do not exist (PostgREST PGRST202); reserve() then falls back to the old pair — check_usage before,
record_usage after — and logs one warning. The legacy pair is kept below unchanged for that reason.
"""
import logging
import uuid
from typing import Optional

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import get_supabase_admin
from app.core.sanitize import sanitize_uuid

logger = logging.getLogger(__name__)

ACTION_PAPER = "test_generated"          # worksheets count as papers: the plan has one AI-paper limit
MSG_LIMIT = ("Is mahine ki AI paper limit poori ho gayi ({used}/{limit}). Plan upgrade karein. / "
             "Monthly limit reached ({used}/{limit} papers). Upgrade your plan.")
MSG_DUPLICATE = ("Yeh request pehle hi process ho chuki hai; paper History mein dekhein. / "
                 "This request was already processed; find the paper in History.")
MSG_SERVICE = "Usage check abhi nahi ho paaya, thodi der baad try karein. / Unable to verify usage. Please try again."
_legacy_warned = False


class Reservation:
    """One counted unit of quota for one request. `open` until consumed or released."""
    __slots__ = ("user_id", "action", "request_id", "legacy", "usage", "open")

    def __init__(self, user_id: str, action: str, request_id: str, legacy: bool, usage: dict):
        self.user_id, self.action, self.request_id, self.legacy, self.usage, self.open = \
            user_id, action, request_id, legacy, dict(usage or {}), True


def _missing_function(e: Exception) -> bool:
    s = str(e)
    return "PGRST202" in s or "Could not find the function" in s or "does not exist" in s


def _dev_bypass(user_id: Optional[str]) -> bool:
    return (not user_id or user_id == "00000000-0000-0000-0000-000000000000") and not settings.IS_PRODUCTION


def reserve(user_id: str, action: str = ACTION_PAPER, request_id: Optional[str] = None) -> Reservation:
    """Count one unit for `request_id` (the client's Idempotency-Key) or raise 401 / 403 / 409 / 503."""
    global _legacy_warned
    request_id = (request_id or "").strip() or uuid.uuid4().hex
    if _dev_bypass(user_id):
        logger.info("Dev mode: quota bypass for local testing")
        return Reservation(user_id or "", action, request_id, True,
                           {"allowed": True, "used": 0, "limit": 999, "remaining": 999})
    if not user_id:
        raise HTTPException(401, {"error": "unauthorized", "message": "Login required."})
    try:
        sanitize_uuid(user_id)
    except ValueError:
        raise HTTPException(400, "Invalid user ID format")
    try:
        data = get_supabase_admin().rpc("a4_reserve_usage", {
            "p_user_id": user_id, "p_action": action, "p_request_id": request_id[:80]}).execute().data
    except HTTPException:
        raise
    except Exception as e:
        if _missing_function(e):
            if not _legacy_warned:
                logger.warning("a4_reserve_usage missing (migration 32 not applied): quota is check-then-record, "
                               "not atomic")
                _legacy_warned = True
            return Reservation(user_id, action, request_id, True, check_usage(user_id))
        logger.error(f"a4_reserve_usage failed: {e}")
        raise HTTPException(503, {"error": "service_error", "message": MSG_SERVICE})
    data = data or {}
    status = data.get("status")
    if status == "duplicate":
        raise HTTPException(409, {"error": "duplicate_request", "message": MSG_DUPLICATE, "request_id": request_id})
    if status == "limit_reached" or not data.get("allowed"):
        used, limit = data.get("used"), data.get("limit")
        raise HTTPException(403, {"error": "limit_reached", "message": MSG_LIMIT.format(used=used, limit=limit),
                                  "used": used, "limit": limit, "upgrade_url": "/pricing"})
    return Reservation(user_id, action, request_id, False, data)


def consume(r: Reservation) -> dict:
    """The paid work succeeded: keep the unit. Returns the usage dict (used/limit/remaining when known)."""
    if not r.open:
        return r.usage
    r.open = False
    if r.legacy:
        if not _dev_bypass(r.user_id):
            r.usage.update(record_usage(r.user_id, r.action))
            if isinstance(r.usage.get("limit"), int) and r.usage["limit"] >= 0 and isinstance(r.usage.get("used"), int):
                r.usage["remaining"] = max(r.usage["limit"] - r.usage["used"], 0)
        return r.usage
    try:
        get_supabase_admin().rpc("a4_consume_usage", {"p_user_id": r.user_id, "p_request_id": r.request_id}).execute()
    except Exception as e:  # the unit is already counted; a failed consume only leaves the ledger row 'reserved'
        logger.error(f"a4_consume_usage failed (unit stays counted): {e}")
    return r.usage


def release(r: Reservation) -> None:
    """The paid work failed before anything was delivered: give the unit back (no-op on the legacy path, which
    only counts on success)."""
    if not r.open:
        return
    r.open = False
    if r.legacy:
        return
    try:
        get_supabase_admin().rpc("a4_release_usage", {"p_user_id": r.user_id, "p_request_id": r.request_id}).execute()
    except Exception as e:
        logger.error(f"a4_release_usage failed (user keeps one unit charged): {e}")


# ── legacy pair (pre-migration-32 path; unchanged behaviour) ─────────────────────────────────────────────────────

def check_usage(user_id: str) -> dict:
    if _dev_bypass(user_id):
        logger.info("Dev mode: Bypassing login requirement for local testing")
        return {"allowed": True, "used": 0, "limit": 999, "remaining": 999}

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "unauthorized",
                "message": "Login required. Please sign in to generate tests.",
            },
        )

    # SECURITY: Validate UUID format
    try:
        sanitize_uuid(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    try:
        supabase = get_supabase_admin()
        result = supabase.rpc("check_usage", {
            "p_user_id": user_id,
        }).execute()

        if not result.data:
            logger.error(f"Usage check returned no data for user {user_id}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_error",
                    "message": "Unable to verify usage. Please try again later.",
                },
            )

        usage = result.data

        if not usage.get("allowed"):
            logger.info(f"Usage limit reached: user={user_id}")
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "limit_reached",
                    "message": f"Monthly limit reached ({usage['used']}/{usage['limit']} tests). Upgrade your plan.",
                    "used": usage.get("used"),
                    "limit": usage.get("limit"),
                    "upgrade_url": "/pricing",
                },
            )

        return usage

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Usage check error: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_error",
                "message": "Unable to verify usage. Please try again later.",
            },
        )


def record_usage(user_id: str, action: str = ACTION_PAPER) -> dict:
    if _dev_bypass(user_id):
        return {"recorded": True}
    try:
        supabase = get_supabase_admin()
        params = {"p_user_id": user_id, "p_action": action}
        try:
            result = supabase.rpc("record_usage", params).execute()
        except Exception as e:
            # usage_records.user_id references teacher_profiles; some accounts never got a
            # row, and without it their papers were never counted against the plan.
            if "foreign key" not in str(e).lower():
                raise
            supabase.table("teacher_profiles").insert({"id": user_id}).execute()
            result = supabase.rpc("record_usage", params).execute()

        if result.data:
            logger.info(f"Usage recorded: user={user_id}")
        return result.data or {}

    except Exception as e:
        logger.error(f"Record usage failed (non-fatal): {e}")
        return {"recorded": False}
