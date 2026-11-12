"""Plan quota for paid AI actions (papers, worksheets). One place, used by every endpoint that spends money.

check_usage -> a4ai `check_usage` RPC (read): allowed / used / limit / remaining / plan.
record_usage -> a4ai `record_usage` RPC (write): +1 for the billing period.
Gate 2 #1 moved these here from endpoints/test_generator.py unchanged; #5 makes them atomic.
"""
import logging

from fastapi import HTTPException

from app.core.config import settings
from app.core.database import get_supabase_admin
from app.core.sanitize import sanitize_uuid

logger = logging.getLogger(__name__)


def check_usage(user_id: str) -> dict:
    if not user_id or user_id == "00000000-0000-0000-0000-000000000000":
        if not settings.IS_PRODUCTION:
            logger.info("Dev mode: Bypassing login requirement for local testing")
            return {"allowed": True, "used": 0, "limit": 999, "remaining": 999}
        logger.warning("Usage check blocked: no valid user_id provided")
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


def record_usage(user_id: str) -> dict:
    if not user_id or user_id == "00000000-0000-0000-0000-000000000000":
        if not settings.IS_PRODUCTION:
            return {"recorded": True}
    try:
        supabase = get_supabase_admin()
        params = {"p_user_id": user_id, "p_action": "test_generated"}
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
