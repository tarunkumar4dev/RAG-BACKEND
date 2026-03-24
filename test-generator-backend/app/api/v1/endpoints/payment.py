"""
Payment & Subscription endpoints for Razorpay integration.
Place at: app/api/v1/endpoints/payment.py
"""

import os
import hmac
import hashlib
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.core.database import get_supabase
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payment", tags=["payment"])

# ─── Razorpay Client ──────────────────────────────────────
RAZORPAY_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RAZORPAY_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")

razorpay_client = None
try:
    import razorpay
    if RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET:
        razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))
        logger.info("✅ Razorpay client initialized")
    else:
        logger.warning("⚠️ RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set")
except ImportError:
    logger.warning("⚠️ razorpay package not installed — pip install razorpay")
except Exception as e:
    logger.error(f"Razorpay init failed: {e}")


# ─── Request Models ───────────────────────────────────────
class CreateOrderRequest(BaseModel):
    amount: int                     # paise (14900 = ₹149)
    payment_method: str = "upi"
    plan_slug: str                  # 'starter' | 'pro'
    user_id: str
    vpa: Optional[str] = None       # test UPI in sandbox mode

class VerifyPaymentRequest(BaseModel):
    razorpay_order_id: str
    razorpay_payment_id: str
    razorpay_signature: str
    plan_slug: str
    user_id: str


# ─── GET /plans ───────────────────────────────────────────
@router.get("/plans")
async def get_plans():
    """Return all active subscription plans."""
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
    """Return user's current plan, usage, and subscription info."""
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

        if plan["price_paise"] != req.amount:
            raise HTTPException(400, detail=f"Amount mismatch: expected {plan['price_paise']}, got {req.amount}")

        # Create Razorpay order
        order = razorpay_client.order.create({
            "amount": req.amount,
            "currency": "INR",
            "notes": {
                "plan_slug": req.plan_slug,
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
            "metadata": {"payment_method": req.payment_method},
        }).execute()

        logger.info(f"Order created: {order['id']} | user={req.user_id} | plan={req.plan_slug}")
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

        # Step 2: Activate subscription (atomic DB function)
        result = supabase.rpc("activate_subscription", {
            "p_user_id": req.user_id,
            "p_plan_slug": req.plan_slug,
            "p_razorpay_order_id": req.razorpay_order_id,
            "p_razorpay_payment_id": req.razorpay_payment_id,
            "p_razorpay_signature": req.razorpay_signature,
        }).execute()

        if not result.data or not result.data.get("success"):
            raise HTTPException(500, detail="Subscription activation failed")

        logger.info(f"✅ Payment verified: user={req.user_id} | plan={req.plan_slug}")

        return {
            "success": True,
            "plan": req.plan_slug,
            "subscription_id": result.data.get("subscription_id"),
            "expires_at": result.data.get("expires_at"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify payment failed: {e}")
        raise HTTPException(500, detail=f"Payment verification failed: {str(e)}")


# ─── POST /check-usage/{user_id} ─────────────────────────
@router.post("/check-usage/{user_id}")
async def check_usage(user_id: str):
    """Check + increment usage. Returns { allowed, used, limit, remaining }"""
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