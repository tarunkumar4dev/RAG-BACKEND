"""
app/core/database.py — Supabase client factory (production hardened)

Three clients, three purposes. Pick deliberately:

  get_supabase()              anon key, no user JWT.
                              Only for genuinely public reads
                              (NCERT questions, chapters, plans list).
                              Under correct RLS this CANNOT write to
                              user-owned tables, and it should not.

  get_supabase_for_user(jwt)  anon key + the caller's access token.
                              RLS evaluates as that user. This is what
                              user-scoped reads/writes should use.

  get_supabase_admin()        service key, bypasses RLS entirely.
                              Trusted server-side writes ONLY:
                              payment capture, subscription activation,
                              usage counters. Never returned to a client.

Changes vs previous version:
  - Thread-safe lazy init (two concurrent requests could previously
    each build a client on cold start)
  - Added get_supabase_for_user() so RLS actually applies to user writes
  - Admin client fails loudly if the service key is missing, instead of
    silently falling back to the anon key
"""

import logging
import threading
from typing import Optional

from supabase import create_client, Client

from app.core.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_client: Optional[Client] = None
_admin_client: Optional[Client] = None


def get_supabase() -> Client:
    """
    Anon client. Respects RLS, but carries no user identity, so every
    policy sees it as anonymous. Use for public data only.
    """
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                if not settings.SUPABASE_URL or not settings.SUPABASE_ANON_KEY:
                    raise RuntimeError(
                        "SUPABASE_URL and SUPABASE_ANON_KEY must be set"
                    )
                _client = create_client(
                    settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY
                )
                logger.info("Supabase client created (anon)")
    return _client


def get_supabase_for_user(access_token: str) -> Client:
    """
    Per-request client bound to the caller's JWT, so RLS policies
    evaluate against auth.uid(). Not cached: each user needs their own.

    Pass the raw token from the Authorization: Bearer <token> header.
    """
    if not access_token:
        raise ValueError("access_token is required")

    client = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
    client.postgrest.auth(access_token)
    return client


def get_supabase_admin() -> Client:
    """
    Service-key client. Bypasses RLS. Server-side trusted operations only.
    NEVER expose this client, its key, or its results unfiltered to a client.
    """
    global _admin_client
    if _admin_client is None:
        with _lock:
            if _admin_client is None:
                if not settings.SUPABASE_SERVICE_KEY:
                    raise RuntimeError(
                        "SUPABASE_SERVICE_KEY is not set. Admin operations "
                        "(payment capture, subscription activation) require it. "
                        "Refusing to fall back to the anon key."
                    )
                if settings.SUPABASE_SERVICE_KEY == settings.SUPABASE_ANON_KEY:
                    raise RuntimeError(
                        "SUPABASE_SERVICE_KEY is identical to the anon key. "
                        "Set a real service role key."
                    )
                _admin_client = create_client(
                    settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY
                )
                logger.info("Supabase admin client created (service role)")
    return _admin_client