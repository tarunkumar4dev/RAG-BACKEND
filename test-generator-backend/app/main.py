"""
A4AI Test Generator Backend — FastAPI (Security Hardened)
"""
import time
import logging
from collections import defaultdict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints.test_generator import router as test_router
from app.api.v1.endpoints.contest import router as contest_router
from app.api.v1.endpoints.payment import router as payment_router
from app.api.v1.endpoints.community_quiz import router as community_quiz_router
from app.api.v1.endpoints.modules import router as module_router
from app.api.v1.endpoints.whatsapp import router as whatsapp_router
from app.routers.test_checker_router import router as test_checker_router
from app.api.v1.endpoints.chat import router as chat_router
from app.core.config import settings
from app.core import api_errors, rate_limit
from app.routers.smart_assignment_router import router as smart_assignment_router
from app.routers.copy_checker_router import router as copy_checker_router
from app.routers.internal_jobs_router import router as internal_jobs_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from app.core import db_pool  # noqa: E402

if db_pool.config_error():
    logger.error(db_pool.config_error())


# ── FastAPI App ─────────────────────────────────────────────────────────

app = FastAPI(
    title="A4AI API",
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
)


# ── CORS — RESTRICTED ───────────────────────────────────────────────────

origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],  # PATCH: checker marks override
    # X-Attempt-Id: contest submit sends it from the browser; without it the CORS preflight is rejected (400)
    allow_headers=["Authorization", "Content-Type", "X-Requested-With", "X-Attempt-Id"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Rate Limiting Middleware ────────────────────────────────────────────

_rate_store: dict[str, list[float]] = defaultdict(list)
_last_rate_prune: float = 0.0
RATE_LIMIT_WINDOW = rate_limit.WINDOW_S
RATE_LIMIT_DEFAULT = settings.RATE_LIMIT_PER_MINUTE  # 300 req/min


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Per route template; per user for authenticated requests, per IP otherwise (app/core/rate_limit.py)."""
    global _last_rate_prune

    # CRITICAL: Always bypass CORS preflight OPTIONS requests immediately
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path

    # Bypass health / docs / provider webhooks / internal worker calls
    if rate_limit.is_exempt(path):
        return await call_next(request)

    client_ip = (
        request.headers.get("x-forwarded-for", "")
        .split(",")[0]
        .strip()
        or (request.client.host if request.client else "unknown")
    )

    now = time.time()

    # Periodic background pruning of expired keys (every 60s or when store grows large)
    if now - _last_rate_prune > 60 or len(_rate_store) > 5000:
        _last_rate_prune = now
        dead_keys = []
        for k, timestamps in list(_rate_store.items()):
            active = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
            if active:
                _rate_store[k] = active
            else:
                dead_keys.append(k)
        for k in dead_keys:
            _rate_store.pop(k, None)

    template = rate_limit.route_template(app, request.method, path)
    buckets = rate_limit.checks(request.method, template, request.headers.get("authorization"), client_ip,
                                request.headers.get("x-attempt-id"), RATE_LIMIT_DEFAULT)
    for key, limit in buckets:
        active_requests = [t for t in _rate_store[key] if now - t < RATE_LIMIT_WINDOW]
        _rate_store[key] = active_requests
        if len(active_requests) >= limit:
            logger.warning(
                f"Rate limit hit: {key.split('|')[0][:14]}… on {request.method} {template} "
                f"({len(active_requests)}/{limit})"
            )
            origin = request.headers.get("origin", "*")
            return JSONResponse(
                status_code=429,
                headers={
                    "Retry-After": "60",
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Credentials": "true",
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                },
                content={"detail": "Too many requests. Please wait a moment and try again."},
            )

    for key, _ in buckets:
        _rate_store[key].append(now)
    key, max_req = buckets[0]
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(max_req)
    response.headers["X-RateLimit-Remaining"] = str(max(0, max_req - len(_rate_store[key])))
    return response


# ── Security Headers Middleware ─────────────────────────────────────────

@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

    # FIX: MutableHeaders has no .pop() — use del with try/except
    try:
        del response.headers["server"]
    except (KeyError, AttributeError):
        pass

    return response


# ── Request Logging Middleware ──────────────────────────────────────────

@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Log every request with timing."""
    start = time.time()
    response = await call_next(request)
    elapsed = round((time.time() - start) * 1000, 1)

    if request.url.path not in ("/health", "/"):
        client_ip = (
            request.headers.get("x-forwarded-for", "")
            .split(",")[0]
            .strip()
            or "unknown"
        )
        logger.info(
            f"{request.method} {request.url.path} → {response.status_code} "
            f"({elapsed}ms) [IP: {client_ip}]"
        )

    response.headers["X-Process-Time-Ms"] = str(elapsed)
    return response


# ── Global Error Handler ────────────────────────────────────────────────

app.add_exception_handler(api_errors.ApiError, api_errors.api_error_handler)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch unhandled exceptions — NEVER leak internals."""
    logger.error(
        f"Unhandled error: {request.method} {request.url.path} — {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong. Please try again or contact support."},
    )


# ── Routers ─────────────────────────────────────────────────────────────

app.include_router(test_router, prefix="/api/v1")
app.include_router(contest_router, prefix="/api/v1")
app.include_router(payment_router, prefix="/api/v1")
app.include_router(community_quiz_router, prefix="/api/v1")
app.include_router(module_router, prefix="/api/v1")
app.include_router(module_router)  # Fallback for direct /modules/* and /worksheet/* requests
app.include_router(test_checker_router, prefix="/api/v1")
app.include_router(chat_router, prefix="/api/v1")
app.include_router(whatsapp_router, prefix="/api/v1")
app.include_router(smart_assignment_router, prefix="/api/v1")
app.include_router(copy_checker_router, prefix="/api/v1")
# no prefix: the Supabase vault calls exactly /internal/jobs/run
app.include_router(internal_jobs_router)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok"}