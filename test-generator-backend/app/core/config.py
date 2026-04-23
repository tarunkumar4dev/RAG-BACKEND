import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── Supabase / Postgres ─────────────────────────────────────────
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # ── Gemini — COST OPTIMIZED ─────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    GEMINI_GEN_MODEL: str = os.getenv("GEMINI_GEN_MODEL", "gemini-2.5-flash-lite")
    GEMINI_FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
    GEMINI_VAL_MODEL: str = os.getenv("GEMINI_VAL_MODEL", "gemini-2.5-flash-lite")

    # Thinking budget — 0 = no thinking (cheapest)
    GEMINI_THINKING_BUDGET: int = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))

    # ── Generation (FIXED for truncation) ───────────────────────────
    # Previous: BATCH_SIZE=10, MAX_OUTPUT_TOKENS=8192 → response truncated
    # New: BATCH_SIZE=4, MAX_OUTPUT_TOKENS=16384 → clean responses
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "4"))
    BATCH_DELAY: int = int(os.getenv("BATCH_DELAY", "2"))
    OVERSHOOT_PER_CHAPTER: int = int(os.getenv("OVERSHOOT_PER_CHAPTER", "2"))
    GENERATION_TEMPERATURE: float = 0.55
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "16384"))
    CONTEXT_CHARS_PER_CHUNK: int = int(os.getenv("CONTEXT_CHARS_PER_CHUNK", "400"))
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))

    # ── RAG ─────────────────────────────────────────────────────────
    MAX_CHUNKS: int = int(os.getenv("MAX_CHUNKS", "12"))
    SIMILARITY_THRESHOLD: float = 0.65
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # ── Pipeline ────────────────────────────────────────────────────
    MAX_ITERATIONS: int = 2
    MAX_QUESTIONS_PER_REQUEST: int = 100
    DEDUP_THRESHOLD: float = 0.82

    # ── CBSE Pattern ────────────────────────────────────────────────
    CBSE_PATTERN_DEFAULT: bool = os.getenv("CBSE_PATTERN_DEFAULT", "true").lower() == "true"

    # ── Rate Limiting ───────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "50"))

    # ── CORS ────────────────────────────────────────────────────────
    CORS_ORIGINS: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://localhost:5174,http://localhost:8080"
    )

    # ── App ─────────────────────────────────────────────────────────
    APP_NAME: str = "A4AI Test Generator"
    APP_VERSION: str = "2.3.0"  # bumped for truncation fix
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()