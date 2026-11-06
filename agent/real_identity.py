"""SessionCtx for a verified Supabase user, from REAL DB rows only (mirrors a4ai/supabase/migrations/01_access_helpers.sql).

admin  = institutes.owner_id = uid, or active institute_members row with role admin/institute
hod    = active member row role 'hod' with department_id
teacher= any other active staff row (teacher / proctor / hod without department)
students / unknown users / several institutes -> refused (403), never guessed.
"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import HTTPException

from .db_real import connect_real
from .schemas import SessionCtx

STAFF_ROLES = ("admin", "institute", "hod", "proctor", "teacher")


def ctx_for_real_user(user_id: str, first_name: Optional[str] = None) -> SessionCtx:
    uid = str(UUID(user_id))
    with connect_real() as c:
        owned = [r["id"] for r in c.execute("select id from institutes where owner_id = %s", (uid,)).fetchall()]
        rows = c.execute(
            "select institute_id, role, department_id, user_name from institute_members "
            "where user_id = %s and status = 'active' and role = any(%s)", (uid, list(STAFF_ROLES))).fetchall()
    insts = set(owned) | {r["institute_id"] for r in rows}
    if not insts:
        raise HTTPException(status_code=403, detail="a4ai Agent is available to institute staff only")
    if len(insts) > 1:
        raise HTTPException(status_code=403, detail="member of several institutes; not supported yet")
    inst = next(iter(insts))
    roles = {r["role"] for r in rows}
    dept = next((r["department_id"] for r in rows if r["role"] == "hod" and r["department_id"]), None)
    if owned or roles & {"admin", "institute"}:
        role = "admin"
    elif dept is not None:
        role = "hod"
    else:
        role = "teacher"
    name = first_name or next((r["user_name"].split()[0] for r in rows if r.get("user_name")), None)
    return SessionCtx(user_id=UUID(uid), role=role, institute_id=inst,
                      teacher_id=None if role == "admin" else UUID(uid),
                      department_id=dept if role == "hod" else None, data_source="live",
                      first_name=(name or None) and name[:40])
