"""
app/core/auth.py — Supabase JWT verification for FastAPI endpoints.

verify_token() validates an access token against Supabase Auth (auth.get_user), so expired,
revoked and signed-out sessions are rejected. Verified users are cached for up to 60 seconds
(never beyond the token's own expiry) to keep per-request latency low.

FastAPI dependencies:
    require_user   -> AuthUser; raises 401 when there is no valid bearer token
    optional_user  -> AuthUser | None
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request
from gotrue.errors import AuthApiError
from starlette.concurrency import run_in_threadpool

from app.core.database import get_supabase

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 60
_CACHE_MAX_ENTRIES = 10_000
_MAX_TOKEN_LENGTH = 8192


@dataclass(frozen=True)
class AuthUser:
    id: str
    email: Optional[str]
    is_anonymous: bool
    role: str  # Supabase role claim, e.g. "authenticated"


_cache: Dict[str, Tuple[AuthUser, float]] = {}
_cache_lock = threading.Lock()


def _cache_key(token: str) -> str:
    # Never keep raw tokens as dictionary keys.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _token_expiry(token: str) -> Optional[float]:
    """Reads `exp` from the JWT payload WITHOUT verifying it — only used to bound the cache."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except Exception:
        return None


def _cache_get(key: str) -> Optional[AuthUser]:
    with _cache_lock:
        item = _cache.get(key)
        if item is None:
            return None
        user, expires_at = item
        if time.time() >= expires_at:
            del _cache[key]
            return None
        return user


def _cache_put(key: str, user: AuthUser, token_exp: Optional[float]) -> None:
    expires_at = time.time() + CACHE_TTL_SECONDS
    if token_exp is not None:
        expires_at = min(expires_at, token_exp)
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX_ENTRIES:
            now = time.time()
            for k in [k for k, (_, exp) in _cache.items() if exp <= now]:
                del _cache[k]
            if len(_cache) >= _CACHE_MAX_ENTRIES:
                _cache.clear()
        _cache[key] = (user, expires_at)


async def verify_token(token: Optional[str]) -> Optional[AuthUser]:
    """Returns the user for a valid Supabase access token, or None if the token is invalid.

    Raises HTTP 503 only when Supabase Auth itself is unreachable, so an outage is not
    mistaken for "logged out".
    """
    if not token or len(token) > _MAX_TOKEN_LENGTH or token.count(".") != 2:
        return None
    token_exp = _token_expiry(token)
    if token_exp is not None and token_exp <= time.time():
        return None

    key = _cache_key(token)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        response = await run_in_threadpool(get_supabase().auth.get_user, token)
    except AuthApiError:
        return None  # invalid, expired or revoked token
    except Exception as e:
        logger.error("Supabase Auth unreachable during token verification: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail="Authentication service unavailable")

    u = getattr(response, "user", None)
    if u is None or not getattr(u, "id", None):
        return None
    user = AuthUser(
        id=str(u.id),
        email=getattr(u, "email", None),
        is_anonymous=bool(getattr(u, "is_anonymous", False)),
        role=getattr(u, "role", None) or "authenticated",
    )
    _cache_put(key, user, token_exp)
    return user


def _bearer_token(request: Request) -> Optional[str]:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer":
        return None
    return token.strip() or None


async def optional_user(request: Request) -> Optional[AuthUser]:
    token = _bearer_token(request)
    return await verify_token(token) if token else None


async def require_user(request: Request) -> AuthUser:
    user = await optional_user(request)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user
