"""Endpoints: POST /agent/chat (SSE), POST /agent/approve, GET /agent/sessions/{id}, GET /agent/drafts/{id}/pdf.
API v2 (sessions list/messages/patch/delete, history, suggestions, feedback, cancel): agent/api_v2.py.

Conversation state (sessions, messages, query history, feedback) lives in the agent store (agent/store.py) in BOTH
modes. Seed/dev mode additionally keeps a row in the local demo Postgres agent_sessions table, because paper drafts
and approvals reference it. Real mode never writes to Supabase."""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Iterator, Optional

import os

from fastapi import APIRouter, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from .approval import ApprovalError, execute_approval
from .api_v2 import own_session
from .api_models import MetaResponse, RateLimitDetail, RateLimitResponse
from .auth import check_dev_mode_safe, dev_mode_enabled, get_identity, get_session
from .mode import READ_ONLY_MESSAGE, check_modes, real_mode
from .core import Agent
from .db import connect
from .llm import fast_provider_for, get_provider
from .rate_limit import check_institute_budget, check_user_rate, record_tokens
from .runs import RUNS
from .schemas import CONTRACT_VERSION, ApproveRequest, ChatRequest, SessionCtx
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


@router.get("/meta", response_model=MetaResponse)
def meta(ctx: SessionCtx = Depends(get_identity)):
    """Never 403 for a not-yet-enabled institute: answers enabled=false so the UI can hide the agent."""
    from .meta import build_meta
    return build_meta(ctx)


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


CHAT_RESPONSES = {
    200: {"description": "SSE stream: `data: <AgentEvent JSON>` frames; first `session`, last `final`.",
          "headers": {"X-Agent-Request-Id": {"description": "request_id of every event; use it to cancel",
                                             "schema": {"type": "string"}}}},
    404: {"description": "unknown / foreign session_id or seq"},
    429: {"model": RateLimitResponse, "description": "rate limit or institute token budget hit; nothing stored",
          "headers": {"Retry-After": {"schema": {"type": "integer"}}}},
    503: {"description": "LLM provider not configured (nothing stored)"},
}


@router.post("/chat", responses=CHAT_RESPONSES)
def chat(body: ChatRequest, ctx: SessionCtx = Depends(get_session)):
    return run_chat(body, ctx)


def run_chat(body: ChatRequest, ctx: SessionCtx, voice: bool = False, on_done=None, canned=None, speech_stream=None):
    """One chat turn (POST /agent/chat, and POST /agent/voice/chat with voice=True). `on_done(events, agent)` runs
    when the stream ends; `canned` (agent/voice/guard.Canned) answers without the model."""
    live = ctx.data_source == "live"
    store, owner = get_store(), Owner.of(ctx)
    if getattr(body, "cancel_current", False):
        RUNS.cancel_owner(owner)  # barge-in: the previous answer stops generating (and speaking) now
    # Rate limits: before anything is stored
    for rl_check in (lambda: check_user_rate(str(ctx.user_id)),
                     lambda: check_institute_budget(str(ctx.institute_id))):
        hit = rl_check()
        if hit is not None:
            detail = RateLimitDetail(scope=hit.scope, message=hit.message, retry_after_s=hit.retry_after_s)
            return Response(
                content=RateLimitResponse(detail=detail).model_dump_json(),
                status_code=429, media_type="application/json",
                headers={"Retry-After": str(hit.retry_after_s)})
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
                      request_id=request_id, session_title=title, cancel_event=cancel_event,
                      fast_provider=fast_provider_for(provider), voice=voice,
                      language=getattr(body, "language", "hinglish"))
    except Exception:
        RUNS.unregister(request_id)
        if conn is not None:
            conn.close()
        raise
    agent.speech_stream = speech_stream

    def stream() -> Iterator[str]:
        events: list[dict] = []
        run = agent.run_canned(text, canned.text, canned.speech, canned.action) if canned is not None else agent.run(text)
        try:
            for ev in run:
                events.append(ev)
                yield _sse(ev)
        finally:
            RUNS.unregister(request_id)
            if on_done is not None:
                try:
                    on_done(events, agent)
                except Exception:  # noqa: BLE001  a trace hook must never break the turn
                    pass
            turn = agent.history[len(hist) + 1:] if len(agent.history) > len(hist) else []
            texts = [e["text"] for e in events if e.get("type") == "text_delta" and e.get("text")]
            stored = [e for e in events if e.get("type") != "audio"]  # audio is played live, never stored
            store.add_message(owner, sid, "assistant", "\n\n".join(texts) or None, events=stored, llm=turn)
            # Record token usage for per-institute daily budget
            total_tokens = agent.tokens[0] + agent.tokens[1] + agent.cache_read
            if total_tokens > 0:
                record_tokens(str(ctx.institute_id), total_tokens)
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
    store, owner = get_store(), Owner.of(ctx)
    if body.decision == "cancel":
        # Cancel = decline: persist the outcome without executing anything
        store.update_approval_in_events(owner, body.session_id, body.approval_id, "cancelled")
        return {"approval_id": body.approval_id, "status": "cancelled"}
    with connect(readonly=False) as conn:
        _own_session(conn, ctx, body.session_id)
        try:
            result = execute_approval(conn, ctx, body.session_id, body.approval_id,
                                      make_executor(conn, ctx, body.session_id))
        except ApprovalError as e:
            raise HTTPException({"not_found": 404, "expired": 410, "forbidden": 403}.get(e.code, 400), str(e))
    # Persist the approval outcome in the stored events so replay shows "approved" not "pending"
    result_ref = result.get("download_path")
    store.update_approval_in_events(owner, body.session_id, body.approval_id, "approved", result_ref=result_ref)
    return result


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


@router.get("/files/{file_id}")
def download_file(file_id: str, ctx: SessionCtx = Depends(get_session)):
    from .files import serve_file
    return serve_file(ctx, file_id)


# Vite dev runs the a4ai frontend on 8080 (vite.config.ts); 5173 is Vite's stock port. Production sets AGENT_CORS_ORIGINS.
DEFAULT_CORS_ORIGINS = "http://localhost:8080,http://localhost:5173"


# localhost, loopback (v4 / v6) and RFC 1918 LAN ranges, any port, http or https
DEV_ORIGIN_REGEX = (r"^https?://(localhost|127\.0\.0\.1|\[::1\]|10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|"
                    r"172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?$")


class LogRefusedPreflight:
    """ASGI middleware: a CORS preflight answered 400 is logged with its Origin (the CORS middleware itself says
    nothing), so a "Connecting…" panel can be traced to the origin the browser used."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http" or scope.get("method") != "OPTIONS":
            return await self.app(scope, receive, send)
        origin = next((v.decode("latin-1") for k, v in scope.get("headers", []) if k == b"origin"), "")

        async def send_wrapped(message):
            if message.get("type") == "http.response.start" and message.get("status") == 400 and origin:
                logging.getLogger(__name__).warning(
                    "CORS preflight refused for origin %s on %s (allowed: AGENT_CORS_ORIGINS=%s; dev mode allows any "
                    "local / LAN origin)", origin, scope.get("path"), os.getenv("AGENT_CORS_ORIGINS") or DEFAULT_CORS_ORIGINS)
            await send(message)
        return await self.app(scope, receive, send_wrapped)


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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Sync endpoints and SSE generators run in anyio's worker threads (default limit 40): each running stream holds
    one while it waits for the model, so 50 concurrent users need more. AGENT_THREADPOOL (default 200)."""
    import anyio.to_thread
    try:
        anyio.to_thread.current_default_thread_limiter().total_tokens = int(os.getenv("AGENT_THREADPOOL", "200"))
    except (ValueError, RuntimeError):
        pass
    # voice: open the Sarvam connections and check the key now, off the request path (agent/voice/providers.warmup)
    from .voice.providers import close_http_clients, warmup
    import threading
    threading.Thread(target=warmup, name="voice-warmup", daemon=True).start()
    yield
    close_http_clients()


def create_app(env=os.environ, check_real_db: bool = True) -> FastAPI:
    check_modes(env)  # dev + real together -> refuse
    check_dev_mode_safe(env)  # refuse to start: dev mode + non-local DB
    if real_mode(env) and check_real_db:
        from .db_real import startup_check
        startup_check()  # refuses to start if the DB role can write or read auth/storage
    a = FastAPI(title="a4ai Agent", version=CONTRACT_VERSION, lifespan=_lifespan)
    headers = ["Authorization", "Content-Type", "Last-Event-ID", "Idempotency-Key"]  # Idempotency-Key: voice uploads
    if dev_mode_enabled(env):
        headers.append("X-Dev-Role")  # only in dev mode, so the browser preflight allows the dev header
    # Dev mode: the frontend may be open as localhost / 127.0.0.1 / [::1] / a LAN address on any port (Vite listens on
    # every interface, phones test over Wi-Fi), so any private origin is allowed. Real / production: the exact list.
    cors_kwargs: dict = {"allow_origins": cors_origins(env)}
    if dev_mode_enabled(env):
        cors_kwargs["allow_origin_regex"] = DEV_ORIGIN_REGEX
    a.add_middleware(CORSMiddleware, allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"], allow_headers=headers,
                     expose_headers=["X-Agent-Request-Id"], allow_credentials=False, **cors_kwargs)
    a.add_middleware(LogRefusedPreflight)   # outermost: says WHICH origin a 400 preflight came from
    from . import api_v2, ops
    from .openapi import install as install_openapi
    a.include_router(router)
    a.include_router(api_v2.router)
    a.include_router(ops.router)
    from .voice.api import router as voice_router  # voice layer: new routes only (VOICE_ENABLED=1 to switch on)
    a.include_router(voice_router)
    install_openapi(a)
    install_log_redaction()
    from .logging_config import configure as configure_logging
    configure_logging()
    try:
        get_store().purge()  # retention: sessions / queries unused for 90 days
    except Exception as e:  # noqa: BLE001  never block startup on the purge; message has no user content
        logging.getLogger(__name__).warning(f"agent store purge failed: {type(e).__name__}")
    return a


app = create_app()
