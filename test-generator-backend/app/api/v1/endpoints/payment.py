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
    coupon_code: Optional[str] = Field(default=None, max_length=50)
    # Accepted for compatibility with older frontends; the buyer is the JWT user.
    user_id: Optional[str] = Field(default=None, max_length=50)
    vpa: Optional[str] = Field(default=None, max_length=100)


class QuoteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_slug: str = Field(max_length=50)
    billing_cycle: str = Field(default="monthly", max_length=10)
    coupon_code: Optional[str] = Field(default=None, max_length=50)


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


HOLD_MESSAGES = {
    "coupon_invalid": "Invalid coupon code.",
    "coupon_used": "You have already redeemed this coupon.",
    "coupon_limit_reached": "This coupon has reached its usage limit.",
    "coupon_in_checkout": "This coupon is already open in another checkout. Finish that payment, or try again in 30 minutes.",
}


def _hold_coupon(db, user_id: str, coupon: str, plan_slug: str, cycle: str) -> dict:
    """a4_hold_coupon() (supabase/migrations/20261019150000_coupon_holds.sql): one live checkout per
    user + coupon, so two tabs cannot both pay the discounted price."""
    try:
        res = db.rpc("a4_hold_coupon", {"p_user_id": user_id, "p_coupon_code": coupon,
                                        "p_plan_slug": plan_slug, "p_cycle": cycle}).execute()
    except Exception:
        # Fail closed: without the hold the per-user limit can be bypassed. Orders without a coupon are unaffected.
        logger.exception("a4_hold_coupon failed (is migration 20261019150000 applied?)")
        raise HTTPException(503, detail="Coupons are temporarily unavailable. Please try again shortly.")
    hold = res.data if isinstance(res.data, dict) else {}
    if not hold.get("ok"):
        err = hold.get("error", "coupon_invalid")
        raise HTTPException(409 if err == "coupon_in_checkout" else 400, detail=HOLD_MESSAGES.get(err, f"Coupon error: {err}"))
    return hold


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


# ─── POST /quote ─────────────────────────────────────────
@router.post("/quote")
async def quote_price(req: QuoteRequest, user: AuthUser = Depends(require_user)):
    """
    Returns the authoritative quote for a plan, cycle and optional coupon code.
    Single source of truth for the checkout pricing breakdown.
    """
    billing_cycle = _validate_billing_cycle(req.billing_cycle)
    try:
        db = get_supabase_admin()
        plan = _fetch_online_plan(db, req.plan_slug)
        coupon = req.coupon_code.strip() if req.coupon_code and req.coupon_code.strip() else None

        quote_res = db.rpc(
            "a4_quote_price",
            {
                "p_user_id": user.id,
                "p_plan_slug": plan["slug"],
                "p_cycle": billing_cycle,
                "p_coupon_code": coupon,
            },
        ).execute()

        quote = quote_res.data if isinstance(quote_res.data, dict) else {}
        if not quote.get("ok"):
            err = quote.get("error", "invalid_quote")
            messages = {
                "coupon_invalid": "Invalid coupon code.",
                "coupon_expired": "This coupon code has expired.",
                "coupon_used": "You have already redeemed this coupon.",
                "coupon_limit_reached": "This coupon has reached its usage limit.",
                "coupon_not_applicable": "This coupon is not applicable to the selected plan.",
            }
            return {
                "success": False,
                "error": err,
                "message": messages.get(err, f"Coupon error: {err}"),
            }

        return {
            "success": True,
            "plan_slug": quote.get("plan_slug"),
            "billing_cycle": quote.get("billing_cycle"),
            "coupon_code": quote.get("coupon_code"),
            "base_paise": quote.get("base_paise"),
            "discount_paise": quote.get("discount_paise"),
            "final_paise": quote.get("final_paise"),
            "percent_off": quote.get("percent_off"),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception("Quote price error for user=%s", user.id)
        raise HTTPException(500, detail="Failed to calculate price quote")


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

        # Server decides the price using authoritative quote RPC.
        # This handles base plan pricing and validates/applies coupons.
        coupon = req.coupon_code.strip() if req.coupon_code and req.coupon_code.strip() else None
        quote_res = db.rpc(
            "a4_quote_price",
            {
                "p_user_id": user.id,
                "p_plan_slug": plan["slug"],
                "p_cycle": billing_cycle,
                "p_coupon_code": coupon,
            },
        ).execute()

        quote = quote_res.data if isinstance(quote_res.data, dict) else {}
        if not quote.get("ok"):
            err = quote.get("error", "invalid_quote")
            messages = {
                "coupon_invalid": "Invalid coupon code.",
                "coupon_expired": "This coupon code has expired.",
                "coupon_used": "You have already redeemed this coupon.",
                "coupon_limit_reached": "This coupon has reached its usage limit.",
                "coupon_not_applicable": "This coupon is not applicable to the selected plan.",
            }
            raise HTTPException(400, detail=messages.get(err, f"Coupon error: {err}"))

        expected = int(quote["final_paise"])
        if req.amount != expected:
            logger.warning(
                "Amount mismatch: got=%s expected=%s plan=%s cycle=%s coupon=%s user=%s",
                req.amount, expected, req.plan_slug, billing_cycle, coupon, user.id,
            )
            raise HTTPException(400, detail="Pricing has changed or coupon was updated. Please refresh and try again.")

        _ensure_teacher_profile(db, user)

        hold = _hold_coupon(db, user.id, coupon, plan["slug"], billing_cycle) if coupon else None
        if hold and hold.get("reuse_order_id"):
            # Same user, same coupon, same plan + cycle already has an order: hand back THAT order.
            # Razorpay accepts one successful payment per order, so the discount can be paid only once.
            existing = razorpay_client.order.fetch(hold["reuse_order_id"])
            if existing.get("status") == "paid":
                raise HTTPException(409, detail="This coupon order is already paid. Refresh the page to see your plan.")
            if int(existing.get("amount") or 0) != expected:
                raise HTTPException(409, detail="Pricing changed for an open coupon checkout. Please try again in 30 minutes.")
            logger.info("Order reused: %s user=%s coupon=%s", existing.get("id"), user.id, coupon)
            return existing

        order_notes = {
            "plan_slug": plan["slug"],
            "billing_cycle": billing_cycle,
            "user_id": user.id,
        }
        if coupon:
            order_notes["coupon_code"] = coupon

        order = razorpay_client.order.create(
            {
                "amount": expected,
                "currency": "INR",
                "notes": order_notes,
            }
        )

        if hold:
            try:
                db.table("coupon_holds").update({"razorpay_order_id": order["id"]})                     .eq("id", hold["hold_id"]).is_("razorpay_order_id", "null").execute()
            except Exception:
                logger.exception("Could not attach order %s to coupon hold %s", order["id"], hold.get("hold_id"))

        metadata = {
            "payment_method": req.payment_method,
            "billing_cycle": billing_cycle,
            "plan_slug": plan["slug"],
        }
        if coupon:
            metadata["coupon_code"] = coupon
            metadata["base_paise"] = quote.get("base_paise")
            metadata["discount_paise"] = quote.get("discount_paise")
            metadata["final_paise"] = quote.get("final_paise")
            metadata["percent_off"] = quote.get("percent_off")

        # This row is the binding record of what was actually bought.
        db.table("payments").insert(
            {
                "user_id": user.id,
                "plan_id": plan["id"],
                "amount_paise": expected,
                "status": "created",
                "razorpay_order_id": order["id"],
                "metadata": metadata,
            }
        ).execute()

        logger.info(
            "Order created: %s user=%s plan=%s cycle=%s amount=%s coupon=%s",
            order["id"], user.id, plan["slug"], billing_cycle, expected, coupon,
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
            .select("user_id, status, plan_id, metadata")
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

        # Ensure coupon redemption is recorded if coupon was used
        try:
            meta = pay_res.data[0].get("metadata") or {}
            c_code = meta.get("coupon_code")
            if c_code:
                chk = db.table("coupon_redemptions").select("id").eq("razorpay_order_id", order_id).limit(1).execute()
                if not chk.data:
                    db.table("coupon_redemptions").insert({
                        "coupon_code": c_code,
                        "user_id": user.id,
                        "razorpay_order_id": order_id,
                        "plan_id": pay_res.data[0].get("plan_id"),
                        "billing_cycle": meta.get("billing_cycle"),
                        "base_paise": meta.get("base_paise"),
                        "discount_paise": meta.get("discount_paise"),
                        "final_paise": meta.get("final_paise"),
                    }).execute()
        except Exception as cr_err:
            logger.warning("Coupon redemption logging failed: %s", cr_err)

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
        db = get_supabase_admin()
        result = _capture(db, order_id, payment_id, None)
        if result.get("error") == "order_not_found":
            logger.warning("Webhook for unknown order=%s", order_id)
            return {"status": "unknown_order"}
        if not result.get("success"):
            logger.error("Webhook capture refused: order=%s reason=%s", order_id, result.get("error"))
            return {"status": "refused", "reason": result.get("error")}

        # Ensure coupon redemption is recorded if coupon was used
        try:
            pay_res = db.table("payments").select("id, user_id, plan_id, metadata").eq("razorpay_order_id", order_id).limit(1).execute()
            if pay_res.data:
                meta = pay_res.data[0].get("metadata") or {}
                c_code = meta.get("coupon_code")
                if c_code:
                    chk = db.table("coupon_redemptions").select("id").eq("razorpay_order_id", order_id).limit(1).execute()
                    if not chk.data:
                        db.table("coupon_redemptions").insert({
                            "coupon_code": c_code,
                            "user_id": pay_res.data[0].get("user_id"),
                            "razorpay_order_id": order_id,
                            "plan_id": pay_res.data[0].get("plan_id"),
                            "billing_cycle": meta.get("billing_cycle"),
                            "base_paise": meta.get("base_paise"),
                            "discount_paise": meta.get("discount_paise"),
                            "final_paise": meta.get("final_paise"),
                        }).execute()
        except Exception as cr_err:
            logger.warning("Webhook coupon redemption logging failed: %s", cr_err)

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
