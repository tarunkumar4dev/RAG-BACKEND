from supabase import create_client, Client
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_supabase() -> Client:
    """
    Returns Supabase client using ANON key (respects RLS).
    Creates once, reuses. Safe for serverless & Cloud Run.
    """
    global _client
    if _client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment"
            )
        _client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
        logger.info("✅ Supabase client created (anon)")
    return _client


# Admin client — bypasses RLS. Use ONLY for trusted server-side ops.
_admin_client: Client | None = None


def get_supabase_admin() -> Client:
    """
    Returns Supabase admin client using SERVICE key (bypasses RLS).
    NEVER expose this client to the frontend.
    """
    global _admin_client
    if _admin_client is None:
        if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment"
            )
        _admin_client = create_client(
            settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY
        )
        logger.info("✅ Supabase admin client created")
    return _admin_client