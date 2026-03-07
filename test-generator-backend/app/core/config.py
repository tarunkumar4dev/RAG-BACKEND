import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # ── Supabase / Postgres ─────────────────────────────────────────
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")          # service_role key
    SUPABASE_ANON_KEY: str = os.getenv("SUPABASE_ANON_KEY", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # ── Gemini (March 2026) ─────────────────────────────────────────
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

    # Primary: 3 Flash — fast + cheap, good for most tasks
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    GEMINI_GEN_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

    # Fallback: 2.5 Flash — alternative if primary is unavailable
    GEMINI_FALLBACK_MODEL: str = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash")

    # Validation: Flash is fine for checking answers (cheap)
    GEMINI_VAL_MODEL: str = os.getenv("GEMINI_VAL_MODEL", "gemini-3-flash-preview")

    # ── Generation v5 Tuning ────────────────────────────────────────
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "12"))
    BATCH_DELAY: int = int(os.getenv("BATCH_DELAY", "20"))
    OVERSHOOT_FACTOR: float = float(os.getenv("OVERSHOOT_FACTOR", "1.30"))
    GENERATION_TEMPERATURE: float = 0.5

    # ── RAG ─────────────────────────────────────────────────────────
    MAX_CHUNKS: int = int(os.getenv("MAX_CHUNKS", "15"))
    SIMILARITY_THRESHOLD: float = 0.65
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # ── Pipeline Limits ─────────────────────────────────────────────
    MAX_ITERATIONS: int = 3
    MAX_QUESTIONS_PER_REQUEST: int = 50
    DEDUP_THRESHOLD: float = 0.82

    # ── Rate Limiting ───────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "50"))

    # ── CORS ────────────────────────────────────────────────────────
    CORS_ORIGINS: str = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://localhost:3000,http://localhost:5174"
    )

    # ── App ─────────────────────────────────────────────────────────
    APP_NAME: str = "A4AI Test Generator"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()