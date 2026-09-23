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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Rate Limiting Middleware ────────────────────────────────────────────

_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = settings.RATE_LIMIT_PER_MINUTE

HEAVY_ENDPOINTS = {
    "/api/v1/test-generator/generate-frontend": 5,
    "/api/v1/test-generator/export": 10,
    "/api/v1/modules/generate": 5,
}


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """IP-based rate limiting."""
    path = request.url.path

    if path in ("/health", "/", "/openapi.json") or path.startswith("/api/v1/whatsapp"):
        return await call_next(request)

    client_ip = (
        request.headers.get("x-forwarded-for", "")
        .split(",")[0]
        .strip()
        or (request.client.host if request.client else "unknown")
    )

    now = time.time()
    key = f"{client_ip}:{path}"
    max_req = HEAVY_ENDPOINTS.get(path, RATE_LIMIT_MAX)

    _rate_store[key] = [t for t in _rate_store[key] if now - t < RATE_LIMIT_WINDOW]

    if len(_rate_store[key]) >= max_req:
        logger.warning(
            f"Rate limit hit: {client_ip} on {path} ({len(_rate_store[key])}/{max_req})"
        )
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please wait a minute and try again."},
        )

    _rate_store[key].append(now)
    return await call_next(request)


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "ok"}