"""
app/api/payment.py — Razorpay payments and subscriptions (v4)

Flow:
  1. POST /payment/create-order   (signed-in teacher) — server prices the plan
     from the plans table, creates the Razorpay order and records it in
     payments (status 'created'). That row is the binding record of what was
     bought: plan, billing cycle, amount, buyer.
  2. Razorpay checkout in the browser.
  3. POST /payment/verify-payment (same teacher) — checks Razorpay's signature,
     then calls the capture_payment() database function, which in ONE
     transaction marks the order paid, starts or extends the subscription and
     syncs teacher_profiles.plan_id. Calling it twice changes nothing.
  4. POST /payment/webhook — Razorpay's server-to-server payment.captured
     event runs the same capture_payment(), so a teacher who closes the tab
     after paying is still activated.

Changes over v3:
  * Activation never worked: the old activate_subscription() inserted a second
    payments row for the order (unique violation) and the fallback wrote
    columns that do not exist and the status 'captured', which the payments
    check constraint rejects. A payment could be marked captured with no plan.
  * The buyer comes from the Supabase JWT, not from user_id in the body.
  * Online checkout is limited to the teacher plans (Starter, Pro); institute
    and college plans go through sales.
  * plan-status and check-usage only answer for the signed-in user.
"""

import hmac
import hashlib
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.core.auth import AuthUser, require_user
from app.core.database import get_supabase, get_supabase_admin
from app.core.config import settings
from app.core.sanitize import sanitize_uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payment", tags=["payment"])

# ─── Constants ────────────────────────────────────────────
VALID_BILLING_CYCLES = {"monthly", "yearly"}
VALID_PAYMENT_METHODS = {"upi", "card", "netbanking", "wallet"}

# Plans that can be bought online. Institute and college plans are sold by the team.
ONLINE_PLAN_SLUGS = {"starter", "pro"}


# ─── Razorpay Client ──────────────────────────────────────
razorpay_client = None
try:
    import razorpay

    if settings.RAZORPAY_KEY_ID and settings.RAZORPAY_KEY_SECRET:
        razorpay_client = razorpay.Client(
            auth=(settings.RAZORPAY_KEY_ID, settings.RAZORPAY_KEY_SECRET)
        )
        logger.info("Razorpay client initialized")
    else:
        logger.warning("RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET not set")
except ImportError:
    logger.warning("razorpay package not installed")
except Exception:
    logger.exception("Razorpay init failed")


# ─── Request Models ───────────────────────────────────────
class CreateOrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Upper bound comes from settings so a pricing change does not require
    # editing this file. Lower bound stays at Rs 1.
    amount: int = Field(ge=100, le=settings.MAX_AMOUNT_PAISE)
    payment_method: str = Field(default="upi", max_length=20)
    plan_slug: str = Field(max_length=50)
    billing_cycle: str = Field(default="monthly", max_length=10)
    # Accepted for compatibility with older frontends; the buyer is the JWT user.
    user_id: Optional[str] = Field(default=None, max_length=50)
    vpa: Optional[str] = Field(default=None, max_length=100)


class VerifyPaymentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    razorpay_order_id: str = Field(max_length=100)
    razorpay_payment_id: str = Field(max_length=100)
    razorpay_signature: str = Field(max_length=200)
    # Accepted for compatibility, never trusted: plan, cycle and buyer come
    # from the payments row written at order creation.
    plan_slug: Optional[str] = Field(default=None, max_length=50)
    billing_cycle: Optional[str] = Field(default=None, max_length=10)
    user_id: Optional[str] = Field(default=None, max_length=50)


# ─── Helpers ──────────────────────────────────────────────
def _validate_billing_cycle(cycle: str) -> str:
    if cycle not in VALID_BILLING_CYCLES:
        raise HTTPException(400, detail="Invalid billing cycle. Use: monthly or yearly")
    return cycle


def _require_self(user: AuthUser, user_id: str) -> str:
    """Path-parameter endpoints answer only for the signed-in user."""
    try:
        user_id = sanitize_uuid(user_id)
    except ValueError:
        raise HTTPException(400, detail="Invalid user ID format")
    if user_id != user.id:
        raise HTTPException(403, detail="You can only access your own account.")
    return user_id


def _expected_amount_paise(plan: dict, billing_cycle: str) -> int:
    """Authoritative price for a plan + cycle, straight from the plans table."""
    column = "price_yearly_paise" if billing_cycle == "yearly" else "price_paise"
    amount = plan.get(column)
    if not amount:
        logger.error("Plan %s has no %s", plan.get("slug"), column)
        raise HTTPException(400, detail="This billing option is not available for this plan.")
    return int(amount)


def _verify_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """Razorpay checkout signature: HMAC-SHA256 over 'order_id|payment_id'."""
    msg = f"{order_id}|{payment_id}"
    expected = hmac.new(
        settings.RAZORPAY_KEY_SECRET.encode(), msg.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _fetch_online_plan(db, slug: str) -> dict:
    if slug not in ONLINE_PLAN_SLUGS:
        raise HTTPException(400, detail="This plan is sold through our team. Please contact sales.")
    res = db.table("plans").select("*").eq("slug", slug).eq("is_active", True).limit(1).execute()
    if not res.data:
        raise HTTPException(400, detail="Invalid plan")
    return res.data[0]


def _ensure_teacher_profile(db, user: AuthUser) -> None:
    """payments.user_id references teacher_profiles; some accounts never got a row."""
    existing = db.table("teacher_profiles").select("id").eq("id", user.id).limit(1).execute()
    if not existing.data:
        db.table("teacher_profiles").insert({"id": user.id, "email": user.email}).execute()


def _capture(db, order_id: str, payment_id: str, signature: Optional[str]) -> dict:
    """Runs capture_payment() (see supabase/migrations/20261001200000_payment_pipeline.sql)."""
    res = db.rpc(
        "capture_payment",
        {"p_order_id": order_id, "p_payment_id": payment_id, "p_signature": signature},
    ).execute()
    return res.data if isinstance(res.data, dict) else {}


# ─── GET /plans ───────────────────────────────────────────
@router.get("/plans")
async def get_plans():
    """Public: the pricing table."""
    try:
        db = get_supabase()
        result = (
            db.table("plans")
            .select("*")
            .eq("is_active", True)
            .order("sort_order")
            .execute()
        )
        return {"success": True, "plans": result.data or []}
    except Exception:
        logger.exception("Failed to fetch plans")
        raise HTTPException(500, detail="Failed to fetch plans")


# ─── GET /plan-status/{user_id} ──────────────────────────
@router.get("/plan-status/{user_id}")
async def get_plan_status(user_id: str, user: AuthUser = Depends(require_user)):
    user_id = _require_self(user, user_id)
    try:
        db = get_supabase_admin()
        result = db.rpc("get_user_plan_status", {"p_user_id": user_id}).execute()
        if not result.data:
            raise HTTPException(404, detail="User not found")
        return result.data
    except HTTPException:
        raise
    except Exception:
        logger.exception("Plan status error for user=%s", user_id)
        raise HTTPException(500, detail="Failed to fetch plan status")


# ─── POST /create-order ──────────────────────────────────
@router.post("/create-order")
async def create_order(req: CreateOrderRequest, user: AuthUser = Depends(require_user)):
    if not razorpay_client:
        raise HTTPException(503, detail="Payment service unavailable")

    billing_cycle = _validate_billing_cycle(req.billing_cycle)
    if req.payment_method not in VALID_PAYMENT_METHODS:
        raise HTTPException(400, detail="Invalid payment method")

    try:
        db = get_supabase_admin()
        plan = _fetch_online_plan(db, req.plan_slug)

        # Server decides the price. The client's amount is only checked so a
        # stale pricing page produces a clear error instead of a wrong charge.
        expected = _expected_amount_paise(plan, billing_cycle)
        if req.amount != expected:
            logger.warning(
                "Amount mismatch: got=%s expected=%s plan=%s cycle=%s user=%s",
                req.amount, expected, req.plan_slug, billing_cycle, user.id,
            )
            raise HTTPException(400, detail="Pricing has changed. Please refresh and try again.")

        _ensure_teacher_profile(db, user)

        order = razorpay_client.order.create(
            {
                "amount": expected,
                "currency": "INR",
                "notes": {"plan_slug": plan["slug"], "billing_cycle": billing_cycle, "user_id": user.id},
            }
        )

        # This row is the binding record of what was actually bought.
        db.table("payments").insert(
            {
                "user_id": user.id,
                "plan_id": plan["id"],
                "amount_paise": expected,
                "status": "created",
                "razorpay_order_id": order["id"],
                "metadata": {
                    "payment_method": req.payment_method,
                    "billing_cycle": billing_cycle,
                    "plan_slug": plan["slug"],
                },
            }
        ).execute()

        logger.info(
            "Order created: %s user=%s plan=%s cycle=%s amount=%s",
            order["id"], user.id, plan["slug"], billing_cycle, expected,
        )
        return order

    except HTTPException:
        raise
    except Exception:
        logger.exception("Create order failed for user=%s", user.id)
        raise HTTPException(500, detail="Order creation failed. Please try again.")


# ─── POST /verify-payment ────────────────────────────────
@router.post("/verify-payment")
async def verify_payment(req: VerifyPaymentRequest, user: AuthUser = Depends(require_user)):
    if not settings.RAZORPAY_KEY_SECRET:
        logger.error("RAZORPAY_KEY_SECRET not set: cannot verify payment")
        raise HTTPException(503, detail="Payment verification unavailable")

    db = get_supabase_admin()
    order_id = req.razorpay_order_id

    try:
        pay_res = (
            db.table("payments")
            .select("user_id, status")
            .eq("razorpay_order_id", order_id)
            .limit(1)
            .execute()
        )
        if not pay_res.data:
            logger.warning("Unknown order at verify: %s", order_id)
            raise HTTPException(404, detail="Order not found. Contact support.")
        if pay_res.data[0]["user_id"] != user.id:
            logger.warning("User mismatch at verify: order=%s user=%s", order_id, user.id)
            raise HTTPException(403, detail="This order belongs to another account.")

        if not _verify_signature(order_id, req.razorpay_payment_id, req.razorpay_signature):
            # Only an unpaid order may be marked failed; never touch a paid one.
            db.table("payments").update({"status": "failed"}).eq(
                "razorpay_order_id", order_id
            ).eq("status", "created").execute()
            logger.warning("Signature mismatch: order=%s user=%s", order_id, user.id)
            raise HTTPException(
                400, detail="Payment verification failed. Contact support if the amount was deducted."
            )

        result = _capture(db, order_id, req.razorpay_payment_id, req.razorpay_signature)
        if not result.get("success"):
            logger.error("capture_payment refused: order=%s reason=%s", order_id, result.get("error"))
            raise HTTPException(500, detail="Activation failed. Contact support with your payment ID.")

        logger.info(
            "Payment verified: user=%s plan=%s order=%s replay=%s",
            user.id, result.get("plan"), order_id, result.get("already_processed"),
        )
        return {
            "success": True,
            "already_processed": bool(result.get("already_processed")),
            "plan": result.get("plan"),
            "billing_cycle": result.get("billing_cycle"),
            "subscription_id": result.get("subscription_id"),
            "expires_at": result.get("expires_at"),
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Verify payment failed: order=%s", order_id)
        raise HTTPException(
            500, detail="Payment verification failed. Contact support if the amount was deducted."
        )


# ─── POST /webhook ───────────────────────────────────────
@router.post("/webhook")
async def razorpay_webhook(request: Request):
    """
    Razorpay server-to-server callback (payment.captured).

    Without this, a teacher who pays and closes the tab before the browser
    calls /verify-payment is charged but never activated. Razorpay retries on
    non-2xx, and capture_payment() is idempotent, so retries are safe.

    Razorpay dashboard -> Settings -> Webhooks:
      URL:    https://<api>/api/v1/payment/webhook
      Events: payment.captured
      Secret: same value as RAZORPAY_WEBHOOK_SECRET
    """
    if not settings.RAZORPAY_WEBHOOK_SECRET:
        raise HTTPException(503, detail="Webhook not configured")

    raw = await request.body()
    signature = request.headers.get("x-razorpay-signature", "")
    expected = hmac.new(settings.RAZORPAY_WEBHOOK_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        logger.warning("Webhook signature mismatch")
        raise HTTPException(400, detail="Invalid signature")

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(400, detail="Invalid payload")

    event = payload.get("event")
    if event != "payment.captured":
        # Acknowledge anything else so Razorpay stops retrying it.
        return {"status": "ignored", "event": event}

    entity = payload.get("payload", {}).get("payment", {}).get("entity", {})
    order_id = entity.get("order_id")
    payment_id = entity.get("id")
    if not order_id or not payment_id:
        raise HTTPException(400, detail="Malformed payload")

    try:
        result = _capture(get_supabase_admin(), order_id, payment_id, None)
        if result.get("error") == "order_not_found":
            logger.warning("Webhook for unknown order=%s", order_id)
            return {"status": "unknown_order"}
        if not result.get("success"):
            logger.error("Webhook capture refused: order=%s reason=%s", order_id, result.get("error"))
            return {"status": "refused", "reason": result.get("error")}
        logger.info(
            "Webhook capture: order=%s plan=%s replay=%s",
            order_id, result.get("plan"), result.get("already_processed"),
        )
        return {"status": "already_processed" if result.get("already_processed") else "ok"}
    except Exception:
        logger.exception("Webhook processing failed: order=%s", order_id)
        # 500 makes Razorpay retry, which is what we want for a transient fault.
        raise HTTPException(500, detail="Processing failed")


# ─── POST /check-usage/{user_id} ─────────────────────────
@router.post("/check-usage/{user_id}")
async def check_usage(user_id: str, user: AuthUser = Depends(require_user)):
    user_id = _require_self(user, user_id)
    try:
        db = get_supabase_admin()
        result = db.rpc(
            "increment_usage", {"p_user_id": user_id, "p_action": "test_generated"}
        ).execute()

        if not result.data:
            raise HTTPException(500, detail="Usage check failed")

        return result.data
    except HTTPException:
        raise
    except Exception:
        logger.exception("Usage check failed for user=%s", user_id)
        raise HTTPException(500, detail="Usage check failed")
