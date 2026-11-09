"""Endpoints: POST /agent/chat (SSE), POST /agent/approve, GET /agent/sessions/{id}, GET /agent/drafts/{id}/pdf.
API v2 (sessions list/messages/patch/delete, history, suggestions, feedback, cancel): agent/api_v2.py.

Conversation state (sessions, messages, query history, feedback) lives in the agent store (agent/store.py) in BOTH
modes. Seed/dev mode additionally keeps a row in the local demo Postgres agent_sessions table, because paper drafts
and approvals reference it. Real mode never writes to Supabase."""
from __future__ import annotations

import json
import logging
import uuid
from typing import Iterator, Optional

import os

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .approval import ApprovalError, execute_approval
from .api_v2 import own_session
from .auth import check_dev_mode_safe, dev_mode_enabled, get_session
from .mode import READ_ONLY_MESSAGE, check_modes, real_mode
from .core import Agent
from .db import connect
from .llm import get_provider
from .runs import RUNS
from .schemas import ApproveRequest, ChatRequest, SessionCtx
from .store import DRAFTS, Owner, auto_title, get_store, iso, valid_id

router = APIRouter(prefix="/agent", tags=["agent"])
HISTORY_MESSAGES = 30


def _own_session(conn, ctx: SessionCtx, session_id: str, allow_admin: bool = False) -> dict:
    """Seed/dev Postgres agent_sessions row (drafts / approvals)."""
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


def build_history(messages: list[dict]) -> list[dict]:
    """Stored messages -> provider-neutral history: user text + each assistant turn's own llm messages."""
    out: list[dict] = []
    for m in messages:
        if m["role"] == "user":
            out.append({"role": "user", "text": m["text"] or ""})
        else:
            out.extend(m.get("llm") or [{"role": "assistant", "text": m.get("text") or "", "tool_calls": []}])
    out = out[-HISTORY_MESSAGES:]
    while out and out[0].get("role") != "user":  # never start mid tool-exchange
        out.pop(0)
    return out


def _sse(ev: dict) -> str:
    return f"data: {json.dumps(ev, default=str)}\n\n"


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


def _prepare_turn(body: ChatRequest, ctx: SessionCtx) -> tuple[str, str, str, list[dict], bool]:
    """-> (session_id, title, user_text, history, store_user_message). Raises 404/422 before streaming."""
    store, owner = get_store(), Owner.of(ctx)
    if not body.session_id:
        text = body.message or ""
        sid = str(uuid.uuid4())
        sess = store.create_session(owner, sid, ctx.role, auto_title(text))
        return sid, sess["title"], text, [], True
    sess = own_session(ctx, body.session_id)
    sid = body.session_id
    if body.regenerate_from_seq is not None:
        m = store.get_message(owner, sid, body.regenerate_from_seq)
        if m is None:
            raise HTTPException(404, "not found")
        msgs = store.all_messages(owner, sid)
        users = [x for x in msgs if x["role"] == "user" and x["seq"] <= m["seq"]]
        if not users:
            raise HTTPException(404, "not found")
        user = users[-1]
        store.truncate(owner, sid, user["seq"] + 1)
        prior = [x for x in msgs if x["seq"] < user["seq"]]
        return sid, sess["title"], user["text"] or "", build_history(prior), False
    if body.edit_of_seq is not None:
        m = store.get_message(owner, sid, body.edit_of_seq)
        if m is None:
            raise HTTPException(404, "not found")
        if m["role"] != "user":
            raise HTTPException(422, "edit_of_seq must point at a user message")
        msgs = store.all_messages(owner, sid)
        first_user = next((x["seq"] for x in msgs if x["role"] == "user"), None)
        store.truncate(owner, sid, m["seq"])
        title = sess["title"]
        if m["seq"] == first_user and sess["title_source"] == "auto":
            title = auto_title(body.message or "")
            store.set_auto_title(owner, sid, title)
        return sid, title, body.message or "", build_history([x for x in msgs if x["seq"] < m["seq"]]), True
    return sid, sess["title"], body.message or "", build_history(store.all_messages(owner, sid)), True


@router.post("/chat")
def chat(body: ChatRequest, ctx: SessionCtx = Depends(get_session)):
    live = ctx.data_source == "live"
    store, owner = get_store(), Owner.of(ctx)
    try:  # before anything is written: no key -> clean 503, no half-created session
        provider = get_provider()
    except KeyError:
        raise HTTPException(503, "LLM provider not configured (API key env var not set)")
    sid, title, text, hist, store_user = _prepare_turn(body, ctx)
    if store_user:
        store.add_message(owner, sid, "user", text)
        store.record_query(owner, text)
    conn = None
    request_id = uuid.uuid4().hex[:16]
    cancel_event = RUNS.register(request_id, owner)
    try:
        if not live:  # seed/dev: drafts + approvals hang off the local demo agent_sessions row
            conn = connect(readonly=False)
            conn.execute("insert into agent_sessions(id, user_id, institute_id, role) values (%s,%s,%s,%s) "
                         "on conflict (id) do nothing", (sid, str(ctx.user_id), str(ctx.institute_id), ctx.role))
        agent = Agent(provider, ctx, sid, conn_rw=conn, history=hist, memory_store=DRAFTS if live else None,
                      request_id=request_id, session_title=title, cancel_event=cancel_event)
    except Exception:
        RUNS.unregister(request_id)
        if conn is not None:
            conn.close()
        raise

    def stream() -> Iterator[str]:
        events: list[dict] = []
        try:
            for ev in agent.run(text):
                events.append(ev)
                yield _sse(ev)
        finally:
            RUNS.unregister(request_id)
            turn = agent.history[len(hist) + 1:] if len(agent.history) > len(hist) else []
            texts = [e["text"] for e in events if e.get("type") == "text_delta" and e.get("text")]
            store.add_message(owner, sid, "assistant", "\n\n".join(texts) or None, events=events, llm=turn)
            if conn is not None:
                conn.close()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                                      "X-Agent-Request-Id": request_id})


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
    """1.1 view (turns = event lists). Agent store sessions: owner only. Seed/dev fallback: the local demo
    Postgres row (owner or same-institute admin, as in 1.1)."""
    if not valid_id(session_id):
        raise HTTPException(404, "not found")
    store, owner = get_store(), Owner.of(ctx)
    row = store.get_session(owner, session_id)
    if row is not None:
        turns = [m["events"] or [] for m in store.all_messages(owner, session_id) if m["role"] == "assistant"]
        return {"session_id": session_id, "role": row["role"], "created_at": row["created_at"], "turns": turns}
    if ctx.data_source == "live":
        raise HTTPException(404, "not found")
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


class RedactAgentQueryStrings(logging.Filter):
    """uvicorn access log lines carry the full URL; /agent/* query strings (suggestions ?q=, sessions ?q=) are user
    text, so they are cut before the line is written."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str) and args[2].startswith("/agent"):
            record.args = args[:2] + (args[2].split("?", 1)[0],) + args[3:]
        return True


def install_log_redaction() -> None:
    lg = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, RedactAgentQueryStrings) for f in lg.filters):
        lg.addFilter(RedactAgentQueryStrings())


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
    install_log_redaction()
    try:
        get_store().purge()  # retention: sessions / queries unused for 90 days
    except Exception as e:  # noqa: BLE001  never block startup on the purge; message has no user content
        logging.getLogger(__name__).warning(f"agent store purge failed: {type(e).__name__}")
    return a


app = create_app()
