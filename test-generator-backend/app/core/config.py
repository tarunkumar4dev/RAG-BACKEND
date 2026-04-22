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
    # Switched from 2.5-flash → 2.5-flash-lite (6x cheaper output)
    # Fallback changed from 2.5-pro → 2.5-flash (never Pro — 25x cheaper output)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    GEMINI_GEN_MODEL: str = os.getenv("GEMINI_GEN_MODEL", "gemini-2.5-flash-lite")
    GEMINI_FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")
    GEMINI_VAL_MODEL: str = os.getenv("GEMINI_VAL_MODEL", "gemini-2.5-flash-lite")

    # Thinking budget — CRITICAL for cost control
    # 0 = no thinking (cheapest), -1 = unlimited (expensive)
    GEMINI_THINKING_BUDGET: int = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))

    # ── Generation (cost optimized) ─────────────────────────────────
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "10"))  # 6 → 10: fewer API calls
    BATCH_DELAY: int = int(os.getenv("BATCH_DELAY", "2"))
    OVERSHOOT_PER_CHAPTER: int = int(os.getenv("OVERSHOOT_PER_CHAPTER", "2"))
    GENERATION_TEMPERATURE: float = 0.55
    MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))  # 12288 → 4096: cap runaway output
    CONTEXT_CHARS_PER_CHUNK: int = int(os.getenv("CONTEXT_CHARS_PER_CHUNK", "400"))  # 500 → 400: less input
    MAX_CONTEXT_CHUNKS: int = int(os.getenv("MAX_CONTEXT_CHUNKS", "5"))  # 8 → 5: 30% less input

    # ── RAG ─────────────────────────────────────────────────────────
    MAX_CHUNKS: int = int(os.getenv("MAX_CHUNKS", "12"))  # 15 → 12
    SIMILARITY_THRESHOLD: float = 0.65
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # ── Pipeline ────────────────────────────────────────────────────
    MAX_ITERATIONS: int = 2  # 3 → 2: one less retry saves tokens
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
    APP_VERSION: str = "2.2.0"  # bumped for cost optimization
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()