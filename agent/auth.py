"""get_session(): JWT -> frozen SessionCtx. Role/institute come from the DB, never from the request body."""
from __future__ import annotations

import os
import sys
from uuid import UUID

from fastapi import HTTPException, Request

from .db import connect
from .schemas import SessionCtx

_ADMIN = {"owner", "admin"}
_TEACHER = {"teacher", "hod", "proctor"}  # HOD/proctor get own-classes scope here (conservative)


def ctx_for_user(user_id: str) -> SessionCtx:
    with connect(readonly=True) as c:
        rows = c.execute("select institute_id, role from institute_members where user_id=%s and "
                         "coalesce(status,'active')='active'", (user_id,)).fetchall()
    insts = {r["institute_id"] for r in rows}
    if len(insts) != 1:  # none, or member of several institutes -> refuse rather than guess
        raise HTTPException(status_code=403, detail="no single active institute membership")
    roles = {r["role"] for r in rows}
    role = "admin" if roles & _ADMIN else "teacher" if roles & _TEACHER else None
    if role is None:
        raise HTTPException(status_code=403, detail="staff only")
    return SessionCtx(user_id=UUID(user_id), role=role, institute_id=next(iter(insts)),
                      teacher_id=UUID(user_id) if role == "teacher" else None)


async def get_session(request: Request) -> SessionCtx:
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
