from supabase import create_client, Client
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

_client: Client | None = None

def get_supabase() -> Client:
    """
    Returns Supabase client. Creates once, reuses.
    NO lru_cache — safe for serverless & Cloud Run.
    """
    global _client
    if _client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")
        _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
        logger.info("✅ Supabase client created")
    return _client