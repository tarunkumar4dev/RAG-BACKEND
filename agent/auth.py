"""get_session(): JWT -> frozen SessionCtx. Role/institute come from the DB, never from the request body."""
from __future__ import annotations

import os
import sys
from uuid import UUID

from fastapi import HTTPException, Request

from .db import connect
from .schemas import SessionCtx

_ADMIN = {"owner", "admin"}

# ---- dev mode (AGENT_DEV_MODE=1): fake staff identity from the seeded local DB ------------------
DEV_ROLE_USERS = {  # fixed ids from agent/seed/seed.sql (fake data)
    "admin": "aaaaaaaa-0000-0000-0000-000000000001",    # Demo Admin
    "teacher": "aaaaaaaa-0000-0000-0000-000000000002",  # Priya Sharma
}
DB_URL_VARS = ("AGENT_REAL_DB_URL", "AGENT_DATABASE_URL", "AGENT_DATABASE_RO_URL", "DATABASE_URL", "SUPABASE_DB_URL",
               "SUPABASE_URL", "POSTGRES_URL", "POSTGRES_PRISMA_URL")
_LOCAL_HOSTS = {"", "localhost", "127.0.0.1", "::1"}


def _is_local_url(url: str) -> bool:
    from urllib.parse import parse_qs, urlparse
    u = urlparse(url)
    hosts = [u.hostname or ""] + parse_qs(u.query).get("host", [])
    return all(h.startswith("/") or h.lower() in _LOCAL_HOSTS for h in hosts)


def dev_mode_enabled(env=os.environ) -> bool:
    return env.get("AGENT_DEV_MODE") == "1"


def check_dev_mode_safe(env=os.environ) -> None:
    """Refuse to start dev mode next to anything that looks like production."""
    if not dev_mode_enabled(env):
        return
    if env.get("VERCEL"):
        raise RuntimeError("AGENT_DEV_MODE=1 is not allowed on Vercel")
    bad = [k for k in DB_URL_VARS if env.get(k) and not _is_local_url(env[k])]
    if bad:  # names only, never values
        raise RuntimeError(f"AGENT_DEV_MODE=1 refused: non-local database configured in {', '.join(bad)}")
_TEACHER = {"teacher", "hod", "proctor"}  # HOD/proctor get own-classes scope here (conservative)


def ctx_for_user(user_id: str) -> SessionCtx:
    with connect(readonly=True) as c:
        rows = c.execute("select institute_id, role, user_name from institute_members where user_id=%s and "
                         "coalesce(status,'active')='active'", (user_id,)).fetchall()
    insts = {r["institute_id"] for r in rows}
    if len(insts) != 1:  # none, or member of several institutes -> refuse rather than guess
        raise HTTPException(status_code=403, detail="no single active institute membership")
    roles = {r["role"] for r in rows}
    role = "admin" if roles & _ADMIN else "teacher" if roles & _TEACHER else None
    if role is None:
        raise HTTPException(status_code=403, detail="staff only")
    names = [r.get("user_name") for r in rows if r.get("user_name")] if rows and "user_name" in rows[0] else []
    return SessionCtx(user_id=UUID(user_id), role=role, institute_id=next(iter(insts)),
                      first_name=names[0].split()[0] if names else None,
                      teacher_id=UUID(user_id) if role == "teacher" else None)


async def get_session(request: Request) -> SessionCtx:
    from .mode import real_mode
    if real_mode():  # live data: only a verified Supabase JWT; no dev headers of any kind
        from starlette.concurrency import run_in_threadpool
        from .jwt_auth import AuthError, first_name_from_claims, verify_supabase_jwt
        from .real_identity import ctx_for_real_user
        auth = request.headers.get("authorization", "")
        if not auth.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Authentication required")
        try:
            claims = verify_supabase_jwt(auth.split(" ", 1)[1].strip())
        except AuthError as e:
            raise HTTPException(status_code=401, detail=str(e))
        return await run_in_threadpool(ctx_for_real_user, claims["sub"], first_name_from_claims(claims))
    dev_role = request.headers.get("x-dev-role")
    if dev_role is not None and dev_mode_enabled():
        check_dev_mode_safe()
        uid = DEV_ROLE_USERS.get(dev_role.strip().lower())
        if uid is None:
            raise HTTPException(status_code=400, detail="X-Dev-Role must be admin or teacher")
        return ctx_for_user(uid)
    # Local/dev only: header identity. Refused on Vercel and unless AGENT_DEV_AUTH=1.
    dev = request.headers.get("x-agent-dev-user")
    if dev and os.getenv("AGENT_DEV_AUTH") == "1" and not os.getenv("VERCEL"):
        return ctx_for_user(str(UUID(dev)))
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test-generator-backend"))
    from app.core.auth import verify_token  # type: ignore  (existing Supabase JWT check)
    user = await verify_token(auth.split(" ", 1)[1])
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return ctx_for_user(str(getattr(user, "id", None) or getattr(user, "user_id")))
