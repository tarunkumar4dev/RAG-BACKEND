"""
app/core/config.py — Application settings (production hardened)

Changes vs previous version:
  - SUPABASE_ANON_KEY resolution simplified (old ternary could never fail)
  - SUPABASE_SERVICE_KEY now required in production (admin writes depend on it)
  - CORS: no localhost defaults leak into production; fail loud instead
  - RAZORPAY keys moved here so payment.py has one source of truth
  - Startup validation raises in production instead of only warning
"""

import os
import logging
from typing import List

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _require(name: str) -> str:
    """Fail fast if a critical env var is missing."""
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _first_set(*names: str, default: str = "") -> str:
    """Return the first env var that is set among `names`."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


class Settings:
    # ── Environment (read first — other settings depend on it) ──────
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    IS_PRODUCTION: bool = ENVIRONMENT.lower() in ("production", "prod")
    DEBUG: bool = not IS_PRODUCTION
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # ── Supabase / Postgres ─────────────────────────────────────────
    SUPABASE_URL: str = _require("SUPABASE_URL")

    # Anon key: respects RLS. Used for reads on behalf of a user.
    SUPABASE_ANON_KEY: str = _first_set("SUPABASE_ANON_KEY", "SUPABASE_KEY")

    # Service key: bypasses RLS. Server-side trusted writes ONLY.
    # Never send this to the frontend.
    SUPABASE_SERVICE_KEY: str = _first_set("SUPABASE_SERVICE_KEY", "SUPABASE_KEY")

    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # ── Gemini ──────────────────────────────────────────────────────
    GEMINI_API_KEY: str = _require("GEMINI_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    GEMINI_GEN_MODEL: str = os.getenv("GEMINI_GEN_MODEL", "gemini-2.5-flash-lite")
    GEMINI_FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite")
    GEMINI_VAL_MODEL: str = os.getenv("GEMINI_VAL_MODEL", "gemini-2.5-flash-lite")
    GEMINI_THINKING_BUDGET: int = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))

    # ── Razorpay ────────────────────────────────────────────────────
    RAZORPAY_KEY_ID: str = os.getenv("RAZORPAY_KEY_ID", "")
    RAZORPAY_KEY_SECRET: str = os.getenv("RAZORPAY_KEY_SECRET", "")
    # Set this in the Razorpay dashboard when you enable webhooks.
    RAZORPAY_WEBHOOK_SECRET: str = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")

    # Highest single charge we will ever accept, in paise.
    # College Pro yearly is Rs 1,99,999 -> 1,99,99,900 paise. Cap set above that.
    MAX_AMOUNT_PAISE: int = int(os.getenv("MAX_AMOUNT_PAISE", str(3_00_00_000)))

    # ── Generation ──────────────────────────────────────────────────
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "5"))
    BATCH_DELAY: int = int(os.getenv("BATCH_DELAY", "2"))
    OVERSHOOT_PER_CHAPTER: int = int(os.getenv("OVERSHOOT_PER_CHAPTER", "1"))
    GENERATION_TEMPERATURE: float = 0.55
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "16384"))

    # ── Context Optimization ────────────────────────────────────────
    CONTEXT_CHARS_PER_CHUNK: int = int(os.getenv("CONTEXT_CHARS_PER_CHUNK", "750"))
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "8"))

    # ── RAG ─────────────────────────────────────────────────────────
    MAX_CHUNKS: int = int(os.getenv("MAX_CHUNKS", "10"))
    SIMILARITY_THRESHOLD: float = 0.65
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # ── Pipeline ────────────────────────────────────────────────────
    MAX_ITERATIONS: int = 1
    MAX_QUESTIONS_PER_REQUEST: int = 100
    DEDUP_THRESHOLD: float = 0.82

    # ── CBSE Pattern ────────────────────────────────────────────────
    CBSE_PATTERN_DEFAULT: bool = os.getenv("CBSE_PATTERN_DEFAULT", "true").lower() == "true"

    # ── Rate Limiting ───────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "50"))

    # ── CORS ────────────────────────────────────────────────────────
    # In development we default to local dev servers.
    # In production CORS_ORIGINS must be set explicitly (validated below).
    _DEV_CORS = "http://localhost:5173,http://localhost:3000,http://localhost:8080"
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "" if IS_PRODUCTION else _DEV_CORS)

    # ── App ─────────────────────────────────────────────────────────
    APP_NAME: str = "a4ai"
    APP_VERSION: str = "2.5.0"

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()


# ── Startup validation ──────────────────────────────────────────────
def _validate() -> None:
    problems: List[str] = []

    if not settings.SUPABASE_ANON_KEY:
        problems.append("SUPABASE_ANON_KEY (or SUPABASE_KEY) is not set")

    if settings.IS_PRODUCTION:
        if not settings.SUPABASE_SERVICE_KEY:
            problems.append("SUPABASE_SERVICE_KEY is required in production")
        if not settings.CORS_ORIGINS:
            problems.append("CORS_ORIGINS must be set explicitly in production")
        if "*" in settings.CORS_ORIGINS:
            problems.append("CORS_ORIGINS must not contain '*' in production")
        if any(o.startswith("http://localhost") for o in settings.cors_origin_list):
            problems.append("CORS_ORIGINS contains localhost in production")
        if not settings.RAZORPAY_KEY_SECRET:
            logger.warning("RAZORPAY_KEY_SECRET not set: payments will be disabled")

    if problems:
        joined = "\n  - ".join(problems)
        if settings.IS_PRODUCTION:
            raise RuntimeError(f"Invalid production configuration:\n  - {joined}")
        logger.warning("Configuration warnings:\n  - %s", joined)

    if settings.DEBUG:
        logger.info("Running in DEVELOPMENT mode")
        logger.info("  CORS: %s", settings.CORS_ORIGINS)
    else:
        logger.info("Running in PRODUCTION mode")
        logger.info("  CORS origins: %d configured", len(settings.cors_origin_list))


_validate()