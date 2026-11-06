"""Verify Supabase access tokens locally.

JWKS (asymmetric signing keys, RS256/ES256) from SUPABASE_URL/auth/v1/.well-known/jwks.json, or the legacy shared
secret (HS256) from SUPABASE_JWT_SECRET. Checks signature, exp, iss (= SUPABASE_URL/auth/v1), aud = "authenticated",
role = "authenticated" and a UUID sub. 'none' and algorithm confusion (HS256 without a secret) are rejected.
Identity (role, institute, department) is derived from DB rows for `sub` — never from token claims or the client.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any, Optional
from uuid import UUID

import jwt
from jwt import PyJWKClient

ASYMMETRIC = ["RS256", "ES256"]
_jwks_lock = threading.Lock()
_jwks: dict[str, tuple[PyJWKClient, float]] = {}


class AuthError(Exception):
    pass


def issuer(env=os.environ) -> str:
    base = (env.get("SUPABASE_URL") or "").rstrip("/")
    if not base:
        raise AuthError("SUPABASE_URL not configured")
    return f"{base}/auth/v1"


def _jwks_client(iss: str) -> PyJWKClient:
    with _jwks_lock:
        hit = _jwks.get(iss)
        if hit and time.time() - hit[1] < 3600:
            return hit[0]
        client = PyJWKClient(f"{iss}/.well-known/jwks.json", cache_keys=True, timeout=5)
        _jwks[iss] = (client, time.time())
        return client


def verify_supabase_jwt(token: str, env=os.environ, jwks_client: Optional[Any] = None) -> dict:
    if not token or token.count(".") != 2:
        raise AuthError("malformed token")
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as e:
        raise AuthError("malformed token") from e
    alg = header.get("alg")
    iss = issuer(env)
    opts = {"require": ["exp", "iat", "sub", "iss", "aud"]}
    try:
        if alg == "HS256":
            secret = env.get("SUPABASE_JWT_SECRET")
            if not secret:
                raise AuthError("HS256 token but SUPABASE_JWT_SECRET is not configured")
            claims = jwt.decode(token, secret, algorithms=["HS256"], audience="authenticated", issuer=iss,
                                options=opts, leeway=10)
        elif alg in ASYMMETRIC:
            key = (jwks_client or _jwks_client(iss)).get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=ASYMMETRIC, audience="authenticated", issuer=iss,
                                options=opts, leeway=10)
        else:
            raise AuthError("unsupported token algorithm")
    except jwt.ExpiredSignatureError as e:
        raise AuthError("token expired") from e
    except jwt.InvalidIssuerError as e:
        raise AuthError("wrong issuer") from e
    except jwt.InvalidAudienceError as e:
        raise AuthError("wrong audience") from e
    except jwt.PyJWTError as e:
        raise AuthError("invalid token") from e
    if claims.get("role") != "authenticated":
        raise AuthError("not an authenticated user token")
    try:
        UUID(str(claims["sub"]))
    except ValueError as e:
        raise AuthError("invalid subject") from e
    return claims


def first_name_from_claims(claims: dict) -> Optional[str]:
    meta = claims.get("user_metadata") or {}
    name = (meta.get("full_name") or meta.get("name") or "").strip()
    return name.split()[0][:40] if name else None
