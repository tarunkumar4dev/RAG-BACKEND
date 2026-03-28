"""
Payment & Subscription endpoints for Razorpay integration.
v2: Added billing_cycle support (monthly/yearly with 20% yearly discount)
"""

import os
import hmac
import hashlib
import logging
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.core.database import get_supabase
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payment", tags=["payment"])

# ─── Constants ────────────────────────────────────────────
YEARLY_DISCOUNT = 0.20   # 20% off for yearly
YEARLY_MONTHS = 12

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
    logger.warning("razorpay package not installed — pip install razorpay")
except Exception as e:
    logger.error(f"Razorpay init failed: {e}")


# ─── Request Models ───────────────────────────────────────
class CreateOrderRequest(BaseModel):
    amount: int                          # paise
    payment_method: str = "upi"
    plan_slug: str                       # 'starter' | 'pro'
    billing_cycle: str = "monthly"       # 'monthly' | 'yearly'  ← NEW
    user_id: str
    vpa: Optional[str] = None

class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    plan_slug: str
    billing_cycle: str = "monthly"       # ← NEW
    user_id: str


# ─── Helpers ──────────────────────────────────────────────

def _calculate_expected_amount(monthly_paise: int, billing_cycle: str) -> int:
    """Calculate expected amount in paise for given billing cycle."""
    if billing_cycle == "yearly":
        return round(monthly_paise * YEARLY_MONTHS * (1 - YEARLY_DISCOUNT))
    return monthly_paise


def _get_subscription_duration_days(billing_cycle: str) -> int:
    """Return subscription duration in days."""
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
        raise HTTPException(500, detail="Razorpay not configured. Set RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET.")

    try:
        supabase = get_supabase()

        # Validate plan
        plan_result = supabase.table("plans").select("*").eq("slug", req.plan_slug).eq("is_active", True).execute()
        if not plan_result.data:
            raise HTTPException(400, detail="Invalid plan")

        plan = plan_result.data[0]

        # Validate amount based on billing cycle
        expected_amount = _calculate_expected_amount(plan["price_paise"], req.billing_cycle)

        if req.amount != expected_amount:
            logger.warning(
                f"Amount mismatch: got {req.amount}, expected {expected_amount} "
                f"(plan={req.plan_slug}, cycle={req.billing_cycle}, monthly={plan['price_paise']})"
            )
            raise HTTPException(
                400,
                detail=f"Amount mismatch: expected {expected_amount} paise for {req.billing_cycle} billing, got {req.amount}"
            )

        # Create Razorpay order
        cycle_label = "Yearly" if req.billing_cycle == "yearly" else "Monthly"
        order = razorpay_client.order.create({
            "amount": req.amount,
            "currency": "INR",
            "notes": {
                "plan_slug": req.plan_slug,
                "billing_cycle": req.billing_cycle,
                "user_id": req.user_id,
            },
        })

        # Record pending payment
        supabase.table("payments").insert({
            "user_id": req.user_id,
            "plan_id": plan["id"],
            "amount_paise": req.amount,
            "status": "created",
            "razorpay_order_id": order["id"],
            "metadata": {
                "payment_method": req.payment_method,
                "billing_cycle": req.billing_cycle,
            },
        }).execute()

        logger.info(f"Order created: {order['id']} | user={req.user_id} | plan={req.plan_slug} | cycle={req.billing_cycle} | amount={req.amount}")
        return order

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create order failed: {e}")
        raise HTTPException(500, detail=f"Order creation failed: {str(e)}")


# ─── POST /verify-payment ────────────────────────────────
@router.post("/verify-payment")
async def verify_payment(req: VerifyPaymentRequest):
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
            raise HTTPException(400, detail="Invalid payment signature")

        # Step 2: Try activate_subscription RPC (if it supports billing_cycle)
        try:
            result = supabase.rpc("activate_subscription", {
                "p_user_id": req.user_id,
                "p_plan_slug": req.plan_slug,
                "p_razorpay_order_id": req.razorpay_order_id,
                "p_razorpay_payment_id": req.razorpay_payment_id,
                "p_razorpay_signature": req.razorpay_signature,
            }).execute()

            if result.data and result.data.get("success"):
                # RPC worked — now update expiry for yearly if needed
                if req.billing_cycle == "yearly" and result.data.get("subscription_id"):
                    sub_id = result.data["subscription_id"]
                    yearly_expiry = (datetime.utcnow() + timedelta(days=365)).isoformat()
                    supabase.table("subscriptions").update({
                        "expires_at": yearly_expiry,
                        "metadata": {"billing_cycle": "yearly"},
                    }).eq("id", sub_id).execute()
                    result.data["expires_at"] = yearly_expiry
                    logger.info(f"Updated subscription {sub_id} to yearly expiry: {yearly_expiry}")

                logger.info(f"Payment verified: user={req.user_id} | plan={req.plan_slug} | cycle={req.billing_cycle}")
                return {
                    "success": True,
                    "plan": req.plan_slug,
                    "billing_cycle": req.billing_cycle,
                    "subscription_id": result.data.get("subscription_id"),
                    "expires_at": result.data.get("expires_at"),
                }
            else:
                raise Exception("RPC returned no success")

        except Exception as rpc_err:
            # Fallback: manual subscription activation if RPC doesn't exist or fails
            logger.warning(f"activate_subscription RPC failed ({rpc_err}), using manual activation")

            duration_days = _get_subscription_duration_days(req.billing_cycle)
            expires_at = (datetime.utcnow() + timedelta(days=duration_days)).isoformat()

            # Get plan
            plan_result = supabase.table("plans").select("id").eq("slug", req.plan_slug).execute()
            if not plan_result.data:
                raise HTTPException(500, detail="Plan not found during activation")
            plan_id = plan_result.data[0]["id"]

            # Update payment status
            supabase.table("payments").update({
                "status": "captured",
                "razorpay_payment_id": req.razorpay_payment_id,
            }).eq("razorpay_order_id", req.razorpay_order_id).execute()

            # Upsert subscription
            sub_data = {
                "user_id": req.user_id,
                "plan_id": plan_id,
                "status": "active",
                "started_at": datetime.utcnow().isoformat(),
                "expires_at": expires_at,
                "metadata": {"billing_cycle": req.billing_cycle},
            }

            # Try update existing, else insert
            existing = supabase.table("subscriptions").select("id").eq("user_id", req.user_id).execute()
            if existing.data:
                supabase.table("subscriptions").update(sub_data).eq("user_id", req.user_id).execute()
                sub_id = existing.data[0]["id"]
            else:
                insert_result = supabase.table("subscriptions").insert(sub_data).execute()
                sub_id = insert_result.data[0]["id"] if insert_result.data else None

            logger.info(f"Manual activation: user={req.user_id} | plan={req.plan_slug} | cycle={req.billing_cycle} | expires={expires_at}")
            return {
                "success": True,
                "plan": req.plan_slug,
                "billing_cycle": req.billing_cycle,
                "subscription_id": sub_id,
                "expires_at": expires_at,
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify payment failed: {e}")
        raise HTTPException(500, detail=f"Payment verification failed: {str(e)}")


# ─── POST /check-usage/{user_id} ─────────────────────────
@router.post("/check-usage/{user_id}")
async def check_usage(user_id: str):
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
        raise HTTPException(500, detail=f"Usage check failed: {str(e)}")