"""
Payment & Subscription endpoints for Razorpay integration (Security Hardened)
v2: Added billing_cycle support (monthly/yearly with 20% yearly discount)

FIXES:
  - Added extra="forbid" on all Pydantic models
  - Added UUID validation on user_id
  - Removed str(e) from ALL error responses (was leaking DB/Razorpay internals)
  - Added validation on plan_slug and billing_cycle (enum-style)
  - Capped amount to prevent manipulation
"""

import os
import hmac
import hashlib
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

from app.core.database import get_supabase
from app.core.config import settings
from app.core.sanitize import sanitize_uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payment", tags=["payment"])

# ─── Constants ────────────────────────────────────────────
YEARLY_DISCOUNT = 0.20
YEARLY_MONTHS = 12
MAX_AMOUNT_PAISE = 50_00_000  # ₹50,000 max — safety cap

VALID_BILLING_CYCLES = {"monthly", "yearly"}
VALID_PAYMENT_METHODS = {"upi", "card", "netbanking", "wallet"}

# ─── Razorpay Client ──────────────────────────────────────
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")

razorpay_client = None
try:
    import razorpay
    if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
        razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        logger.info("Razorpay client initialized")
    else:
        logger.warning("RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set")
except ImportError:
    logger.warning("razorpay package not installed")
except Exception as e:
    logger.error(f"Razorpay init failed: {e}")


# ─── Request Models ───────────────────────────────────────
class CreateOrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    amount: int = Field(ge=100, le=MAX_AMOUNT_PAISE)  # min ₹1, max ₹50k
    payment_method: str = "upi"
    plan_slug: str = Field(max_length=50)
    billing_cycle: str = Field(default="monthly", max_length=10)
    user_id: str = Field(max_length=50)
    vpa: Optional[str] = Field(default=None, max_length=100)


class VerifyPaymentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    razorpay_order_id: str = Field(max_length=100)
    razorpay_payment_id: str = Field(max_length=100)
    razorpay_signature: str = Field(max_length=200)
    plan_slug: str = Field(max_length=50)
    billing_cycle: str = Field(default="monthly", max_length=10)
    user_id: str = Field(max_length=50)


# ─── Helpers ──────────────────────────────────────────────

def _validate_billing_cycle(cycle: str) -> str:
    if cycle not in VALID_BILLING_CYCLES:
        raise HTTPException(400, "Invalid billing cycle. Use: monthly or yearly")
    return cycle


def _validate_user_id(user_id: str) -> str:
    try:
        return sanitize_uuid(user_id)
    except ValueError:
        raise HTTPException(400, "Invalid user ID format")


def _calculate_expected_amount(monthly_paise: int, billing_cycle: str) -> int:
    if billing_cycle == "yearly":
        return round(monthly_paise * YEARLY_MONTHS * (1 - YEARLY_DISCOUNT))
    return monthly_paise


def _get_subscription_duration_days(billing_cycle: str) -> int:
    if billing_cycle == "yearly":
        return 365
    return 30


# ─── GET /plans ───────────────────────────────────────────
@router.get("/plans")
async def get_plans():
    try:
        supabase = get_supabase()
        result = supabase.table("plans").select("*").eq("is_active", True).order("sort_order").execute()
        return {"success": True, "plans": result.data or []}
    except Exception as e:
        logger.error(f"Failed to fetch plans: {e}")
        raise HTTPException(500, detail="Failed to fetch plans")


# ─── GET /plan-status/{user_id} ──────────────────────────
@router.get("/plan-status/{user_id}")
async def get_plan_status(user_id: str):
    user_id = _validate_user_id(user_id)
    try:
        supabase = get_supabase()
        result = supabase.rpc("get_user_plan_status", {"p_user_id": user_id}).execute()
        if not result.data:
            raise HTTPException(404, detail="User not found")
        return result.data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plan status error: {e}")
        raise HTTPException(500, detail="Failed to fetch plan status")


# ─── POST /create-order ──────────────────────────────────
@router.post("/create-order")
async def create_order(req: CreateOrderRequest):
    if not razorpay_client:
        raise HTTPException(503, detail="Payment service unavailable")

    user_id = _validate_user_id(req.user_id)
    billing_cycle = _validate_billing_cycle(req.billing_cycle)

    if req.payment_method not in VALID_PAYMENT_METHODS:
        raise HTTPException(400, detail="Invalid payment method")

    try:
        supabase = get_supabase()

        # Validate plan
        plan_result = supabase.table("plans").select("*").eq("slug", req.plan_slug).eq("is_active", True).execute()
        if not plan_result.data:
            raise HTTPException(400, detail="Invalid plan")

        plan = plan_result.data[0]

        # SECURITY: Validate amount server-side (never trust client amount)
        expected_amount = _calculate_expected_amount(plan["price_paise"], billing_cycle)

        if req.amount != expected_amount:
            logger.warning(
                f"Amount mismatch: got {req.amount}, expected {expected_amount} "
                f"(plan={req.plan_slug}, cycle={billing_cycle})"
            )
            raise HTTPException(400, detail="Amount mismatch. Please refresh and try again.")

        # Create Razorpay order
        order = razorpay_client.order.create({
            "amount": req.amount,
            "currency": "INR",
            "notes": {
                "plan_slug": req.plan_slug,
                "billing_cycle": billing_cycle,
                "user_id": user_id,
            },
        })

        # Record pending payment
        supabase.table("payments").insert({
            "user_id": user_id,
            "plan_id": plan["id"],
            "amount_paise": req.amount,
            "status": "created",
            "razorpay_order_id": order["id"],
            "metadata": {
                "payment_method": req.payment_method,
                "billing_cycle": billing_cycle,
            },
        }).execute()

        logger.info(f"Order created: {order['id']} | user={user_id} | plan={req.plan_slug} | cycle={billing_cycle}")
        return order

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create order failed: {e}")
        raise HTTPException(500, detail="Order creation failed. Please try again.")


# ─── POST /verify-payment ────────────────────────────────
@router.post("/verify-payment")
async def verify_payment(req: VerifyPaymentRequest):
    user_id = _validate_user_id(req.user_id)
    billing_cycle = _validate_billing_cycle(req.billing_cycle)

    if not RAZORPAY_KEY_SECRET:
        logger.error("RAZORPAY_KEY_SECRET not set — cannot verify payment")
        raise HTTPException(503, detail="Payment verification unavailable")

    try:
        supabase = get_supabase()

        # Step 1: Verify Razorpay signature
        msg = f"{req.razorpay_order_id}|{req.razorpay_payment_id}"
        expected_sig = hmac.new(
            RAZORPAY_KEY_SECRET.encode(),
            msg.encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(expected_sig, req.razorpay_signature):
            supabase.table("payments").update({"status": "failed"}).eq(
                "razorpay_order_id", req.razorpay_order_id
            ).execute()
            logger.warning(f"Signature mismatch: order={req.razorpay_order_id}")
            raise HTTPException(400, detail="Payment verification failed. Contact support if amount was deducted.")

        # Step 2: Try activate_subscription RPC
        try:
            result = supabase.rpc("activate_subscription", {
                "p_user_id": user_id,
                "p_plan_slug": req.plan_slug,
                "p_razorpay_order_id": req.razorpay_order_id,
                "p_razorpay_payment_id": req.razorpay_payment_id,
                "p_razorpay_signature": req.razorpay_signature,
            }).execute()

            if result.data and result.data.get("success"):
                if billing_cycle == "yearly" and result.data.get("subscription_id"):
                    sub_id = result.data["subscription_id"]
                    yearly_expiry = (datetime.utcnow() + timedelta(days=365)).isoformat()
                    supabase.table("subscriptions").update({
                        "expires_at": yearly_expiry,
                        "metadata": {"billing_cycle": "yearly"},
                    }).eq("id", sub_id).execute()
                    result.data["expires_at"] = yearly_expiry

                logger.info(f"Payment verified: user={user_id} | plan={req.plan_slug} | cycle={billing_cycle}")
                return {
                    "success": True,
                    "plan": req.plan_slug,
                    "billing_cycle": billing_cycle,
                    "subscription_id": result.data.get("subscription_id"),
                    "expires_at": result.data.get("expires_at"),
                }
            else:
                raise Exception("RPC returned no success")

        except Exception as rpc_err:
            logger.warning(f"activate_subscription RPC failed ({rpc_err}), using manual activation")

            duration_days = _get_subscription_duration_days(billing_cycle)
            expires_at = (datetime.utcnow() + timedelta(days=duration_days)).isoformat()

            plan_result = supabase.table("plans").select("id").eq("slug", req.plan_slug).execute()
            if not plan_result.data:
                raise HTTPException(500, detail="Activation failed. Contact support.")
            plan_id = plan_result.data[0]["id"]

            supabase.table("payments").update({
                "status": "captured",
                "razorpay_payment_id": req.razorpay_payment_id,
            }).eq("razorpay_order_id", req.razorpay_order_id).execute()

            sub_data = {
                "user_id": user_id,
                "plan_id": plan_id,
                "status": "active",
                "started_at": datetime.utcnow().isoformat(),
                "expires_at": expires_at,
                "metadata": {"billing_cycle": billing_cycle},
            }

            existing = supabase.table("subscriptions").select("id").eq("user_id", user_id).execute()
            if existing.data:
                supabase.table("subscriptions").update(sub_data).eq("user_id", user_id).execute()
                sub_id = existing.data[0]["id"]
            else:
                insert_result = supabase.table("subscriptions").insert(sub_data).execute()
                sub_id = insert_result.data[0]["id"] if insert_result.data else None

            logger.info(f"Manual activation: user={user_id} | plan={req.plan_slug} | cycle={billing_cycle}")
            return {
                "success": True,
                "plan": req.plan_slug,
                "billing_cycle": billing_cycle,
                "subscription_id": sub_id,
                "expires_at": expires_at,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify payment failed: {e}")
        raise HTTPException(500, detail="Payment verification failed. Contact support if amount was deducted.")


# ─── POST /check-usage/{user_id} ─────────────────────────
@router.post("/check-usage/{user_id}")
async def check_usage(user_id: str):
    user_id = _validate_user_id(user_id)
    try:
        supabase = get_supabase()
        result = supabase.rpc("increment_usage", {
            "p_user_id": user_id,
            "p_action": "test_generated",
        }).execute()

        if not result.data:
            raise HTTPException(500, detail="Usage check failed")

        return result.data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Usage check failed: {e}")
        raise HTTPException(500, detail="Usage check failed. Please try again.")