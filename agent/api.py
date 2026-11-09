"""Endpoints: POST /agent/chat (SSE), POST /agent/approve, GET /agent/sessions/{id}, GET /agent/drafts/{id}/pdf."""
from __future__ import annotations

import json
import uuid
from typing import Iterator

import os

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .approval import ApprovalError, execute_approval
from .auth import check_dev_mode_safe, dev_mode_enabled, get_session
from .mode import READ_ONLY_MESSAGE, check_modes, real_mode
from .store import MEMORY
from .core import Agent
from .db import connect
from .llm import get_provider
from .schemas import ApproveRequest, ChatRequest, SessionCtx

router = APIRouter(prefix="/agent", tags=["agent"])


def _own_session(conn, ctx: SessionCtx, session_id: str, allow_admin: bool = False) -> dict:
    try:
        uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(404, "not found")
    row = conn.execute("select * from agent_sessions where id=%s", (session_id,)).fetchone()
    if row is None or str(row["institute_id"]) != str(ctx.institute_id):
        raise HTTPException(404, "not found")
    if str(row["user_id"]) != str(ctx.user_id) and not (allow_admin and ctx.role == "admin"):
        raise HTTPException(404, "not found")  # 404, never 403: don't leak existence
    return row


def _history(conn, session_id: str) -> list[dict]:
    rows = conn.execute("select content from agent_messages where session_id=%s and role='llm' order by seq",
                        (session_id,)).fetchall()
    msgs = [r["content"] for r in rows][-30:]
    while msgs and msgs[0].get("role") != "user":  # never start mid tool-exchange
        msgs.pop(0)
    return msgs


def _mem_session(ctx: SessionCtx, session_id: str, allow_admin: bool = False) -> dict:
    row = MEMORY.get(session_id)
    if row is None or row["institute_id"] != str(ctx.institute_id):
        raise HTTPException(404, "not found")
    if row["user_id"] != str(ctx.user_id) and not (allow_admin and ctx.role == "admin"):
        raise HTTPException(404, "not found")
    return row


def _sse(ev: dict) -> str:
    return f"data: {json.dumps(ev, default=str)}\n\n"


def _chat_live(body: ChatRequest, ctx: SessionCtx):
    """Real data: nothing is written to Supabase. Session, history and drafts stay in this process."""
    if body.session_id:
        _mem_session(ctx, body.session_id)
        sid = body.session_id
    else:
        sid = str(uuid.uuid4())
        MEMORY.create_session(sid, str(ctx.user_id), str(ctx.institute_id), ctx.role, body.message)
    hist = list(MEMORY.history.get(sid, []))[-30:]
    while hist and hist[0].get("role") != "user":
        hist.pop(0)
    agent = Agent(get_provider(), ctx, sid, history=hist, memory_store=MEMORY)

    def stream() -> Iterator[str]:
        events = []
        try:
            for ev in agent.run(body.message):
                events.append(ev)
                yield _sse(ev)
        finally:
            MEMORY.history[sid] = agent.history[-60:]
            MEMORY.events.setdefault(sid, []).append(events)
            MEMORY.touch(sid)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/meta")
def meta(ctx: SessionCtx = Depends(get_session)):
    inst = None
    try:
        if ctx.data_source == "live":
            from .db_real import connect_real
            with connect_real() as c:
                r = c.execute("select name from institutes where id = %s", (str(ctx.institute_id),)).fetchone()
        else:
            with connect(readonly=True) as c:
                r = c.execute("select name from institutes where id = %s", (str(ctx.institute_id),)).fetchone()
        inst = r["name"] if r else None
    except Exception:
        inst = None
    return {"data_source": ctx.data_source, "read_only": ctx.data_source == "live", "role": ctx.role,
            "first_name": ctx.first_name, "institute_name": inst}


@router.get("/sessions")
def list_sessions(ctx: SessionCtx = Depends(get_session)):
    if ctx.data_source == "live":
        rows = [r for r in MEMORY.list_for(str(ctx.user_id))]
        return {"sessions": [{**r, "created_at": _iso(r["created_at"]), "updated_at": _iso(r["updated_at"])}
                             for r in rows[:50]]}
    with connect(readonly=False) as conn:
        rows = conn.execute(
            "select s.id, s.created_at, coalesce(max(m.created_at), s.created_at) as updated_at, "
            "(select m2.content->>'text' from agent_messages m2 where m2.session_id = s.id and m2.role = 'llm' "
            " and m2.content->>'role' = 'user' order by m2.seq limit 1) as title "
            "from agent_sessions s left join agent_messages m on m.session_id = s.id "
            "where s.user_id = %s and s.institute_id = %s group by s.id order by updated_at desc limit 50",
            (str(ctx.user_id), str(ctx.institute_id))).fetchall()
    return {"sessions": [{"session_id": str(r["id"]), "title": (r["title"] or "New chat")[:60],
                          "created_at": str(r["created_at"]), "updated_at": str(r["updated_at"])} for r in rows]}


def _iso(ts: float) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


@router.post("/chat")
def chat(body: ChatRequest, ctx: SessionCtx = Depends(get_session)):
    if ctx.data_source == "live":
        return _chat_live(body, ctx)
    conn = connect(readonly=False)
    if body.session_id:
        _own_session(conn, ctx, body.session_id)
        sid = body.session_id
    else:
        sid = str(uuid.uuid4())
        conn.execute("insert into agent_sessions(id, user_id, institute_id, role) values (%s,%s,%s,%s)",
                     (sid, str(ctx.user_id), str(ctx.institute_id), ctx.role))
    agent = Agent(get_provider(), ctx, sid, conn_rw=conn, history=_history(conn, sid))

    def stream() -> Iterator[str]:
        events = []
        try:
            for ev in agent.run(body.message):
                events.append(ev)
                yield f"data: {json.dumps(ev, default=str)}\n\n"
        finally:
            base = conn.execute("select coalesce(max(seq),0) m from agent_messages where session_id=%s",
                                (sid,)).fetchone()["m"]
            new = agent.history[len(_history(conn, sid)):] if agent.history else []
            for i, m in enumerate(new):
                conn.execute("insert into agent_messages(session_id, seq, role, content) values (%s,%s,'llm',%s::jsonb)",
                             (sid, base + 1 + i, json.dumps(m, default=str)))
            conn.execute("insert into agent_messages(session_id, seq, role, content) values (%s,%s,'events',%s::jsonb)",
                         (sid, base + 1 + len(new), json.dumps(events, default=str)))
            conn.close()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def make_executor(conn, ctx: SessionCtx, session_id: str):
    def run(action: str, args: dict) -> dict:
        d = conn.execute("select id, status from agent_drafts where id=%s and session_id=%s and user_id=%s",
                         (args["draft_id"], session_id, str(ctx.user_id))).fetchone()
        if d is None:
            raise ApprovalError("not_found", "draft not found")
        if action == "finalize_test":
            conn.execute("update agent_drafts set status='final' where id=%s", (args["draft_id"],))
            return {"status": "finalized", "draft_id": args["draft_id"],
                    "download_path": f"/agent/drafts/{args['draft_id']}/pdf"}
        if action == "assign_to_section":
            with connect(readonly=True) as ro:
                ok = ro.execute("select 1 from batches b where b.id=%s and b.institute_id=%s and (%s='admin' or "
                                "exists(select 1 from teaching_assignments t where t.batch_id=b.id and t.teacher_id=%s "
                                "and coalesce(t.is_active,true)))",
                                (args["section_id"], str(ctx.institute_id), ctx.role, str(ctx.teacher_id))).fetchone()
            if not ok:
                raise ApprovalError("forbidden", "section not in your scope")
            conn.execute("update agent_drafts set status='assigned', paper = paper || jsonb_build_object("
                         "'assigned_section', %s::text) where id=%s", (args["section_id"], args["draft_id"]))
            return {"status": "assigned", "draft_id": args["draft_id"], "section_id": args["section_id"]}
        raise ApprovalError("invalid_input", "unknown action")
    return run


@router.post("/approve")
def approve(body: ApproveRequest, ctx: SessionCtx = Depends(get_session)):
    if ctx.data_source == "live":
        raise HTTPException(403, READ_ONLY_MESSAGE)
    with connect(readonly=False) as conn:
        _own_session(conn, ctx, body.session_id)
        try:
            return execute_approval(conn, ctx, body.session_id, body.approval_id,
                                    make_executor(conn, ctx, body.session_id))
        except ApprovalError as e:
            raise HTTPException({"not_found": 404, "expired": 410, "forbidden": 403}.get(e.code, 400), str(e))


@router.get("/sessions/{session_id}")
def get_session_view(session_id: str, ctx: SessionCtx = Depends(get_session)):
    if ctx.data_source == "live":
        row = _mem_session(ctx, session_id, allow_admin=True)
        return {"session_id": session_id, "role": row["role"], "created_at": _iso(row["created_at"]),
                "turns": MEMORY.events.get(session_id, [])}
    with connect(readonly=False) as conn:
        row = _own_session(conn, ctx, session_id, allow_admin=True)
        ev = conn.execute("select content from agent_messages where session_id=%s and role='events' order by seq",
                          (session_id,)).fetchall()
    return {"session_id": session_id, "role": row["role"], "created_at": str(row["created_at"]),
            "turns": [r["content"] for r in ev]}


@router.get("/drafts/{draft_id}/pdf")
def draft_pdf(draft_id: str, ctx: SessionCtx = Depends(get_session)):
    if ctx.data_source == "live":
        raise HTTPException(403, READ_ONLY_MESSAGE)
    from .export import export_pdf
    with connect(readonly=False) as conn:
        try:
            uuid.UUID(draft_id)
        except ValueError:
            raise HTTPException(404, "not found")
        d = conn.execute("select paper, status from agent_drafts where id=%s and user_id=%s and institute_id=%s",
                         (draft_id, str(ctx.user_id), str(ctx.institute_id))).fetchone()
    if d is None:
        raise HTTPException(404, "not found")
    if d["status"] not in ("final", "assigned"):
        raise HTTPException(409, "approve the test first")  # download only after approval
    return Response(export_pdf(d["paper"]), media_type="application/pdf")


DEFAULT_CORS_ORIGINS = "http://localhost:5173"


def cors_origins(env=os.environ) -> list[str]:
    raw = env.get("AGENT_CORS_ORIGINS") or DEFAULT_CORS_ORIGINS
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    if any("*" in o for o in origins):
        raise RuntimeError("AGENT_CORS_ORIGINS must list exact origins (no wildcard)")
    return origins


def create_app(env=os.environ, check_real_db: bool = True) -> FastAPI:
    check_modes(env)  # dev + real together -> refuse
    check_dev_mode_safe(env)  # refuse to start: dev mode + non-local DB
    if real_mode(env) and check_real_db:
        from .db_real import startup_check
        startup_check()  # refuses to start if the DB role can write or read auth/storage
    a = FastAPI(title="a4ai Agent (standalone)")
    headers = ["Authorization", "Content-Type", "Last-Event-ID"]
    if dev_mode_enabled(env):
        headers.append("X-Dev-Role")  # only in dev mode, so the browser preflight allows the dev header
    a.add_middleware(CORSMiddleware, allow_origins=cors_origins(env),
                     allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"], allow_headers=headers,
                     expose_headers=["X-Agent-Request-Id"], allow_credentials=False)
    from . import api_v2
    a.include_router(router)
    a.include_router(api_v2.router)
    return a


app = create_app()
