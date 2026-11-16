"""Approval gate: single-use, bound to session + user + sha256(args), 15 min TTL. Replay -> stored result."""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable

from . import config
from .schemas import SessionCtx


def args_hash(args: dict) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def create_approval(conn, ctx: SessionCtx, session_id: str, action: str, args: dict) -> dict:
    aid = str(uuid.uuid4())
    exp = datetime.now(timezone.utc) + timedelta(seconds=config.APPROVAL_TTL_S)
    h = args_hash(args)
    conn.execute("insert into agent_approvals(id, session_id, user_id, action, args_sha256, args, expires_at) "
                 "values (%s,%s,%s,%s,%s,%s::jsonb,%s)",
                 (aid, session_id, str(ctx.user_id), action, h, json.dumps(args, default=str), exp))
    return {"approval_id": aid, "action": action, "args_sha256": h, "expires_at": exp.isoformat()}


class ApprovalError(Exception):
    def __init__(self, code: str, msg: str):
        super().__init__(msg)
        self.code = code


def execute_approval(conn, ctx: SessionCtx, session_id: str, approval_id: str,
                     executor: Callable[[str, dict], dict]) -> dict:
    with conn.transaction():
        row = conn.execute("select * from agent_approvals where id=%s and session_id=%s and user_id=%s for update",
                           (approval_id, session_id, str(ctx.user_id))).fetchone()
        if not row:
            raise ApprovalError("not_found", "approval not found")
        if row["used_at"] is not None:
            return {"replayed": True, **(row["result"] or {})}
        if row["expires_at"] < datetime.now(timezone.utc):
            raise ApprovalError("expired", "approval expired")
        if args_hash(row["args"]) != row["args_sha256"]:
            raise ApprovalError("forbidden", "args changed after approval was issued")
        result = executor(row["action"], row["args"])
        conn.execute("update agent_approvals set used_at=now(), result=%s::jsonb where id=%s",
                     (json.dumps(result, default=str), approval_id))
    return {"replayed": False, **result}
