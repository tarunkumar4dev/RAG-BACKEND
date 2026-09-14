"""
app/api/payment.py — Razorpay payments and subscriptions (v3)

Fixes over v2, in order of how much money each one was costing:

1. YEARLY PRICE CAME FROM A FORMULA (v2: monthly * 12 * 0.80).
   Actual a4ai pricing is "2 months free" (~16.6%), and College Plus is
   ~33% off, so the formula matched no plan. Every yearly checkout died on
   "Amount mismatch". Yearly price is now a column: plans.price_yearly_paise.

2. AMOUNT CAP WAS Rs 50,000. College Pro yearly is Rs 1,99,999, so Pydantic
   rejected it before any handler ran. Cap now comes from settings.

3. PLAN-SWAP EXPLOIT. /verify-payment re-read plan_slug from the request and
   never checked it against the order. The Razorpay signature only covers
   order_id|payment_id, so a Rs 199 order could be verified as a Rs 2,999
   plan. Plan and billing cycle are now read from the stored payments row;
   the client's values are only used to detect and log tampering.

4. NO IDEMPOTENCY. The same order_id could be verified repeatedly, each time
   extending the subscription. Capture is now guarded by a status check plus
   a unique index on payments.razorpay_order_id.

5. RPC FALLBACK FIRED ON LEGITIMATE FAILURES. `raise Exception("RPC returned
   no success")` inside the try meant a deliberate rejection from
   activate_subscription (unknown plan, missing user) was caught and then
   manually activated anyway. The fallback now runs only when the RPC is
   genuinely unavailable, and a returned success=False is treated as a
   refusal.

6. ANON CLIENT FOR PRIVILEGED WRITES. payments/subscriptions writes now use
   the service-role client.

7. NO WEBHOOK. If the user closed the tab after paying, the subscription was
   never activated and the money was silently kept. POST /payment/webhook
   now captures those.

Also: timezone-aware UTC (datetime.utcnow() is deprecated), and error
responses still never echo internal exception text.
"""

import hmac
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.core.database import get_supabase, get_supabase_admin
from app.core.config import settings
from app.core.sanitize import sanitize_uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/payment", tags=["payment"])

# ─── Constants ────────────────────────────────────────────
YEARLY_DAYS = 365
MONTHLY_DAYS = 30

VALID_BILLING_CYCLES = {"monthly", "yearly"}
VALID_PAYMENT_METHODS = {"upi", "card", "netbanking", "wallet"}

# Terminal states — a payment in one of these must not be activated again.
CAPTURED_STATES = {"captured", "refunded"}


def _now() -> datetime:
    return datetime.now(timezone.utc)


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
    user_id: str = Field(max_length=50)
    vpa: Optional[str] = Field(default=None, max_length=100)


class VerifyPaymentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    razorpay_order_id: str = Field(max_length=100)
    razorpay_payment_id: str = Field(max_length=100)
    razorpay_signature: str = Field(max_length=200)
    # plan_slug and billing_cycle are accepted for backward compatibility
    # with the existing frontend, but they are NOT trusted. The values
    # recorded at order-creation time are authoritative.
    plan_slug: Optional[str] = Field(default=None, max_length=50)
    billing_cycle: Optional[str] = Field(default=None, max_length=10)
    user_id: str = Field(max_length=50)


# ─── Helpers ──────────────────────────────────────────────
def _validate_billing_cycle(cycle: str) -> str:
    if cycle not in VALID_BILLING_CYCLES:
        raise HTTPException(400, detail="Invalid billing cycle. Use: monthly or yearly")
    return cycle


def _validate_user_id(user_id: str) -> str:
    try:
        return sanitize_uuid(user_id)
    except ValueError:
        raise HTTPException(400, detail="Invalid user ID format")


def _expected_amount_paise(plan: dict, billing_cycle: str) -> int:
    """
    Authoritative price for a plan + cycle, straight from the plans table.

    Yearly is a stored column, not a computed discount, because a4ai's
    yearly prices are not a fixed percentage of monthly:
        Institute Starter  999/mo   ->  9,999/yr   (~16.6% off)
        College Plus     9,999/mo   -> 79,999/yr   (~33.3% off)
    A single multiplier cannot express both.
    """
    if billing_cycle == "yearly":
        yearly = plan.get("price_yearly_paise")
        if not yearly:
            logger.error(
                "Plan %s has no price_yearly_paise; yearly billing unavailable",
                plan.get("slug"),
            )
            raise HTTPException(
                400, detail="Yearly billing is not available for this plan."
            )
        return int(yearly)

    monthly = plan.get("price_paise")
    if not monthly:
        logger.error("Plan %s has no price_paise", plan.get("slug"))
        raise HTTPException(400, detail="This plan is not available for purchase.")
    return int(monthly)


def _duration_days(billing_cycle: str) -> int:
    return YEARLY_DAYS if billing_cycle == "yearly" else MONTHLY_DAYS


def _verify_signature(order_id: str, payment_id: str, signature: str) -> bool:
    """Razorpay checkout signature: HMAC-SHA256 over 'order_id|payment_id'."""
    msg = f"{order_id}|{payment_id}"
    expected = hmac.new(
        settings.RAZORPAY_KEY_SECRET.encode(), msg.encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature)


def _fetch_plan_by_slug(db, slug: str) -> dict:
    res = (
        db.table("plans")
        .select("*")
        .eq("slug", slug)
        .eq("is_active", True)
        .limit(1)
        .execute()
    )
    if not res.data:
        raise HTTPException(400, detail="Invalid plan")
    return res.data[0]


def _fetch_plan_by_id(db, plan_id: str) -> Optional[dict]:
    res = db.table("plans").select("*").eq("id", plan_id).limit(1).execute()
    return res.data[0] if res.data else None


def _activate(
    db,
    user_id: str,
    plan: dict,
    billing_cycle: str,
    order_id: str,
    payment_id: str,
) -> Tuple[str, str]:
    """
    Mark the payment captured and activate the subscription.

    Returns (subscription_id, expires_at_iso).
    Caller must have already verified the signature and confirmed this order
    has not been captured before.
    """
    expires_at = (_now() + timedelta(days=_duration_days(billing_cycle))).isoformat()

    # Capture the payment first. If this update matches no rows, another
    # request captured it concurrently and we must not activate again.
    captured = (
        db.table("payments")
        .update(
            {
                "status": "captured",
                "razorpay_payment_id": payment_id,
                "captured_at": _now().isoformat(),
            }
        )
        .eq("razorpay_order_id", order_id)
        .neq("status", "captured")
        .execute()
    )
    if not captured.data:
        raise HTTPException(409, detail="This payment has already been processed.")

    sub_data = {
        "user_id": user_id,
        "plan_id": plan["id"],
        "status": "active",
        "started_at": _now().isoformat(),
        "expires_at": expires_at,
        "metadata": {"billing_cycle": billing_cycle},
    }

    existing = (
        db.table("subscriptions").select("id").eq("user_id", user_id).limit(1).execute()
    )
    if existing.data:
        sub_id = existing.data[0]["id"]
        db.table("subscriptions").update(sub_data).eq("id", sub_id).execute()
    else:
        inserted = db.table("subscriptions").insert(sub_data).execute()
        if not inserted.data:
            logger.error("Subscription insert returned no row for user=%s", user_id)
            raise HTTPException(500, detail="Activation failed. Contact support.")
        sub_id = inserted.data[0]["id"]

    return sub_id, expires_at


# ─── GET /plans ───────────────────────────────────────────
@router.get("/plans")
async def get_plans():
    """Public: the pricing table. Anon client is correct here."""
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
async def get_plan_status(user_id: str):
    user_id = _validate_user_id(user_id)
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
async def create_order(req: CreateOrderRequest):
    if not razorpay_client:
        raise HTTPException(503, detail="Payment service unavailable")

    user_id = _validate_user_id(req.user_id)
    billing_cycle = _validate_billing_cycle(req.billing_cycle)

    if req.payment_method not in VALID_PAYMENT_METHODS:
        raise HTTPException(400, detail="Invalid payment method")

    try:
        db = get_supabase_admin()
        plan = _fetch_plan_by_slug(db, req.plan_slug)

        # Server decides the price. The client's amount is only checked so a
        # stale pricing page produces a clear error instead of a wrong charge.
        expected = _expected_amount_paise(plan, billing_cycle)
        if req.amount != expected:
            logger.warning(
                "Amount mismatch: got=%s expected=%s plan=%s cycle=%s user=%s",
                req.amount,
                expected,
                req.plan_slug,
                billing_cycle,
                user_id,
            )
            raise HTTPException(
                400, detail="Pricing has changed. Please refresh and try again."
            )

        order = razorpay_client.order.create(
            {
                "amount": expected,
                "currency": "INR",
                "notes": {
                    "plan_slug": req.plan_slug,
                    "billing_cycle": billing_cycle,
                    "user_id": user_id,
                },
            }
        )

        # This row is the binding record of what was actually bought.
        # /verify-payment reads plan and cycle from here, not from the client.
        db.table("payments").insert(
            {
                "user_id": user_id,
                "plan_id": plan["id"],
                "amount_paise": expected,
                "status": "created",
                "razorpay_order_id": order["id"],
                "metadata": {
                    "payment_method": req.payment_method,
                    "billing_cycle": billing_cycle,
                    "plan_slug": req.plan_slug,
                },
            }
        ).execute()

        logger.info(
            "Order created: %s user=%s plan=%s cycle=%s amount=%s",
            order["id"],
            user_id,
            req.plan_slug,
            billing_cycle,
            expected,
        )
        return order

    except HTTPException:
        raise
    except Exception:
        logger.exception("Create order failed for user=%s", user_id)
        raise HTTPException(500, detail="Order creation failed. Please try again.")


# ─── POST /verify-payment ────────────────────────────────
@router.post("/verify-payment")
async def verify_payment(req: VerifyPaymentRequest):
    user_id = _validate_user_id(req.user_id)

    if not settings.RAZORPAY_KEY_SECRET:
        logger.error("RAZORPAY_KEY_SECRET not set: cannot verify payment")
        raise HTTPException(503, detail="Payment verification unavailable")

    db = get_supabase_admin()

    try:
        # ── Step 1: signature ────────────────────────────────────────
        if not _verify_signature(
            req.razorpay_order_id, req.razorpay_payment_id, req.razorpay_signature
        ):
            db.table("payments").update({"status": "failed"}).eq(
                "razorpay_order_id", req.razorpay_order_id
            ).execute()
            logger.warning(
                "Signature mismatch: order=%s user=%s", req.razorpay_order_id, user_id
            )
            raise HTTPException(
                400,
                detail="Payment verification failed. Contact support if the amount was deducted.",
            )

        # ── Step 2: load the order we created ────────────────────────
        pay_res = (
            db.table("payments")
            .select("id, user_id, plan_id, amount_paise, status, metadata")
            .eq("razorpay_order_id", req.razorpay_order_id)
            .limit(1)
            .execute()
        )
        if not pay_res.data:
            logger.warning("Unknown order at verify: %s", req.razorpay_order_id)
            raise HTTPException(404, detail="Order not found. Contact support.")

        payment_row = pay_res.data[0]

        # The order belongs to whoever created it, not whoever is asking.
        if payment_row["user_id"] != user_id:
            logger.warning(
                "User mismatch at verify: order=%s owner=%s claimed=%s",
                req.razorpay_order_id,
                payment_row["user_id"],
                user_id,
            )
            raise HTTPException(403, detail="This order belongs to another account.")

        # ── Step 3: idempotency ──────────────────────────────────────
        if payment_row["status"] in CAPTURED_STATES:
            sub = (
                db.table("subscriptions")
                .select("id, expires_at")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            logger.info("Replayed verify for captured order=%s", req.razorpay_order_id)
            return {
                "success": True,
                "already_processed": True,
                "subscription_id": sub.data[0]["id"] if sub.data else None,
                "expires_at": sub.data[0]["expires_at"] if sub.data else None,
            }

        # ── Step 4: authoritative plan + cycle from OUR record ───────
        meta = payment_row.get("metadata") or {}
        billing_cycle = _validate_billing_cycle(meta.get("billing_cycle", "monthly"))

        plan = _fetch_plan_by_id(db, payment_row["plan_id"])
        if not plan:
            logger.error(
                "Plan %s missing for order %s",
                payment_row["plan_id"],
                req.razorpay_order_id,
            )
            raise HTTPException(500, detail="Activation failed. Contact support.")

        # If the client sent a different plan than it paid for, that is either
        # a stale tab or a tampering attempt. Either way we ignore it and
        # activate what was actually paid for, but we record the discrepancy.
        if req.plan_slug and req.plan_slug != plan.get("slug"):
            logger.warning(
                "Plan mismatch at verify: order=%s paid_for=%s claimed=%s user=%s",
                req.razorpay_order_id,
                plan.get("slug"),
                req.plan_slug,
                user_id,
            )
        if req.billing_cycle and req.billing_cycle != billing_cycle:
            logger.warning(
                "Cycle mismatch at verify: order=%s paid_for=%s claimed=%s",
                req.razorpay_order_id,
                billing_cycle,
                req.billing_cycle,
            )

        # ── Step 5: activate, preferring the DB routine ──────────────
        try:
            rpc = db.rpc(
                "activate_subscription",
                {
                    "p_user_id": user_id,
                    "p_plan_slug": plan["slug"],
                    "p_razorpay_order_id": req.razorpay_order_id,
                    "p_razorpay_payment_id": req.razorpay_payment_id,
                    "p_razorpay_signature": req.razorpay_signature,
                },
            ).execute()
            rpc_data = rpc.data
        except Exception:
            # The routine is unavailable (not deployed, transport error).
            # This is the only case where falling back is correct.
            logger.exception(
                "activate_subscription RPC unavailable; using direct activation "
                "(order=%s)",
                req.razorpay_order_id,
            )
            rpc_data = None

        if rpc_data is not None and not rpc_data.get("success"):
            # The routine ran and deliberately refused. Do NOT activate anyway.
            logger.error(
                "activate_subscription refused: order=%s reason=%s",
                req.razorpay_order_id,
                rpc_data.get("error") or rpc_data.get("message"),
            )
            raise HTTPException(
                500, detail="Activation failed. Contact support with your payment ID."
            )

        if rpc_data and rpc_data.get("success"):
            sub_id = rpc_data.get("subscription_id")
            expires_at = rpc_data.get("expires_at")

            # The routine assumes a monthly term; stretch it for yearly.
            if billing_cycle == "yearly" and sub_id:
                expires_at = (_now() + timedelta(days=YEARLY_DAYS)).isoformat()
                db.table("subscriptions").update(
                    {"expires_at": expires_at, "metadata": {"billing_cycle": "yearly"}}
                ).eq("id", sub_id).execute()

            db.table("payments").update(
                {
                    "status": "captured",
                    "razorpay_payment_id": req.razorpay_payment_id,
                    "captured_at": _now().isoformat(),
                }
            ).eq("razorpay_order_id", req.razorpay_order_id).execute()
        else:
            sub_id, expires_at = _activate(
                db,
                user_id,
                plan,
                billing_cycle,
                req.razorpay_order_id,
                req.razorpay_payment_id,
            )

        logger.info(
            "Payment verified: user=%s plan=%s cycle=%s order=%s",
            user_id,
            plan["slug"],
            billing_cycle,
            req.razorpay_order_id,
        )
        return {
            "success": True,
            "plan": plan["slug"],
            "billing_cycle": billing_cycle,
            "subscription_id": sub_id,
            "expires_at": expires_at,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Verify payment failed: order=%s", req.razorpay_order_id)
        raise HTTPException(
            500,
            detail="Payment verification failed. Contact support if the amount was deducted.",
        )


# ─── POST /webhook ───────────────────────────────────────
@router.post("/webhook")
async def razorpay_webhook(request: Request):
    """
    Razorpay server-to-server callback.

    Without this, a user who pays and then closes the tab before the browser
    calls /verify-payment is charged but never activated. Razorpay retries
    this endpoint, so it must be idempotent: the capture guard in _activate()
    provides that.

    Configure in the Razorpay dashboard:
      URL:    https://<your-api>/payment/webhook
      Events: payment.captured
      Secret: set RAZORPAY_WEBHOOK_SECRET to the same value
    """
    if not settings.RAZORPAY_WEBHOOK_SECRET:
        raise HTTPException(503, detail="Webhook not configured")

    raw = await request.body()
    signature = request.headers.get("x-razorpay-signature", "")

    expected = hmac.new(
        settings.RAZORPAY_WEBHOOK_SECRET.encode(), raw, hashlib.sha256
    ).hexdigest()
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

    entity = (
        payload.get("payload", {}).get("payment", {}).get("entity", {})
    )
    order_id = entity.get("order_id")
    payment_id = entity.get("id")
    if not order_id or not payment_id:
        raise HTTPException(400, detail="Malformed payload")

    db = get_supabase_admin()

    try:
        pay_res = (
            db.table("payments")
            .select("id, user_id, plan_id, status, metadata")
            .eq("razorpay_order_id", order_id)
            .limit(1)
            .execute()
        )
        if not pay_res.data:
            logger.warning("Webhook for unknown order=%s", order_id)
            return {"status": "unknown_order"}

        row = pay_res.data[0]
        if row["status"] in CAPTURED_STATES:
            return {"status": "already_processed"}

        plan = _fetch_plan_by_id(db, row["plan_id"])
        if not plan:
            logger.error("Webhook: plan %s missing for order %s", row["plan_id"], order_id)
            return {"status": "plan_missing"}

        meta = row.get("metadata") or {}
        billing_cycle = meta.get("billing_cycle", "monthly")
        if billing_cycle not in VALID_BILLING_CYCLES:
            billing_cycle = "monthly"

        try:
            _activate(db, row["user_id"], plan, billing_cycle, order_id, payment_id)
        except HTTPException as e:
            if e.status_code == 409:
                return {"status": "already_processed"}
            raise

        logger.info(
            "Webhook activated: user=%s plan=%s order=%s",
            row["user_id"],
            plan["slug"],
            order_id,
        )
        return {"status": "ok"}

    except Exception:
        logger.exception("Webhook processing failed: order=%s", order_id)
        # 500 makes Razorpay retry, which is what we want for a transient fault.
        raise HTTPException(500, detail="Processing failed")


# ─── POST /check-usage/{user_id} ─────────────────────────
@router.post("/check-usage/{user_id}")
async def check_usage(user_id: str):
    user_id = _validate_user_id(user_id)
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
        raise HTTPException(500, detail="Usage check failed. Please try again.")