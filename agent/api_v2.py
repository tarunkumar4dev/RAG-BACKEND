"""API v2 routes (docs/AGENT_API_V2.md): sessions, messages, query history, suggestions, feedback, cancel.

All state lives in the agent store (agent/store.py), keyed by user_id + institute_id from the verified SessionCtx.
Anything that is not the caller's -> 404. Nothing here logs request content.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from .api_models import (CancelResponse, FeedbackRequest, FeedbackResponse, MessageList, QueryItem, SessionItem,
                         SessionList, SessionPatch, Suggestion)
from .auth import get_session
from .runs import RUNS
from .schemas import SessionCtx
from .store import DRAFTS, Owner, get_store, valid_id
from .suggest import suggest

router = APIRouter(prefix="/agent", tags=["agent-v2"])

NOT_FOUND = "not found"


def own_session(ctx: SessionCtx, session_id: str) -> dict:
    if not valid_id(session_id):
        raise HTTPException(404, NOT_FOUND)
    row = get_store().get_session(Owner.of(ctx), session_id)
    if row is None:
        raise HTTPException(404, NOT_FOUND)  # someone else's session looks exactly like a missing one
    return row


def session_item(r: dict) -> dict:
    return {k: r[k] for k in ("id", "title", "created_at", "last_message_at", "message_count", "pinned", "preview")}


def _bad_cursor():
    raise HTTPException(422, "invalid cursor")


@router.get("/sessions", response_model=SessionList)
def list_sessions(cursor: Optional[str] = Query(None, max_length=200), limit: int = Query(20, ge=1, le=50),
                  q: Optional[str] = Query(None, max_length=80), pinned: Optional[bool] = None,
                  ctx: SessionCtx = Depends(get_session)):
    try:
        rows, nxt = get_store().list_sessions(Owner.of(ctx), cursor=cursor, limit=limit, q=q, pinned=pinned)
    except ValueError:
        _bad_cursor()
    items = [session_item(r) for r in rows]
    return {"items": items, "next_cursor": nxt,
            "sessions": [{"session_id": r["id"], "title": r["title"], "created_at": r["created_at"],
                          "updated_at": r["last_message_at"]} for r in items]}


@router.get("/sessions/{session_id}/messages", response_model=MessageList)
def list_messages(session_id: str, cursor: Optional[str] = Query(None, max_length=200),
                  limit: int = Query(100, ge=1, le=200), ctx: SessionCtx = Depends(get_session)):
    own_session(ctx, session_id)
    try:
        items, nxt = get_store().list_messages(Owner.of(ctx), session_id, cursor=cursor, limit=limit)
    except ValueError:
        _bad_cursor()
    return {"items": items, "next_cursor": nxt}


@router.patch("/sessions/{session_id}", response_model=SessionItem)
def patch_session(session_id: str, body: SessionPatch, ctx: SessionCtx = Depends(get_session)):
    own_session(ctx, session_id)
    r = get_store().patch_session(Owner.of(ctx), session_id, title=body.title, pinned=body.pinned)
    if r is None:
        raise HTTPException(404, NOT_FOUND)
    return session_item(r)


@router.delete("/sessions/{session_id}", status_code=204, response_class=Response)
def delete_session(session_id: str, ctx: SessionCtx = Depends(get_session)):
    own_session(ctx, session_id)
    get_store().delete_session(Owner.of(ctx), session_id)
    DRAFTS.drop(session_id)
    if ctx.data_source != "live":  # seed/dev: the local demo Postgres row (drafts/approvals hang off it)
        try:
            from .db import connect
            with connect(readonly=False) as conn:
                conn.execute("delete from agent_sessions where id = %s and user_id = %s and institute_id = %s",
                             (session_id, str(ctx.user_id), str(ctx.institute_id)))
        except Exception:  # noqa: BLE001  (local demo DB not running: the store delete already happened)
            pass
    return Response(status_code=204)


@router.get("/history/queries", response_model=list[QueryItem])
def list_queries(q: Optional[str] = Query(None, max_length=80), limit: int = Query(10, ge=1, le=50),
                 ctx: SessionCtx = Depends(get_session)):
    return [{k: r[k] for k in ("id", "text", "last_used_at", "use_count")}
            for r in get_store().list_queries(Owner.of(ctx), q=q, limit=limit)]


@router.delete("/history/queries/{query_id}", status_code=204, response_class=Response)
def delete_query(query_id: str, ctx: SessionCtx = Depends(get_session)):
    if len(query_id) > 64 or not get_store().delete_query(Owner.of(ctx), query_id):
        raise HTTPException(404, NOT_FOUND)
    return Response(status_code=204)


@router.delete("/history/queries", status_code=204, response_class=Response)
def clear_queries(ctx: SessionCtx = Depends(get_session)):
    get_store().clear_queries(Owner.of(ctx))
    return Response(status_code=204)


@router.get("/suggestions", response_model=list[Suggestion])
def suggestions(q: Optional[str] = Query(None, max_length=80), limit: int = Query(8, ge=1, le=20),
                ctx: SessionCtx = Depends(get_session)):
    return suggest(get_store(), ctx, q, limit)


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(body: FeedbackRequest, ctx: SessionCtx = Depends(get_session)):
    own_session(ctx, body.session_id)
    if not get_store().set_feedback(Owner.of(ctx), body.session_id, body.message_seq, body.rating, body.note):
        raise HTTPException(404, NOT_FOUND)
    return {"ok": True}


XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@router.get("/files/{file_id}", response_class=Response,
            responses={200: {"content": {XLSX: {"schema": {"type": "string", "format": "binary"}}},
                             "description": "the generated file (Content-Disposition: attachment)"},
                       404: {"description": "unknown / foreign file"},
                       410: {"description": "your file, but expired (1 hour); ask the agent again"}})
def get_file(file_id: str, ctx: SessionCtx = Depends(get_session)):
    from .files import serve_file
    return serve_file(ctx, file_id)


@router.post("/chat/{request_id}/cancel", status_code=202, response_model=CancelResponse)
def cancel(request_id: str, ctx: SessionCtx = Depends(get_session)):
    if len(request_id) > 64 or not RUNS.cancel(request_id, Owner.of(ctx)):
        raise HTTPException(404, NOT_FOUND)
    return {"request_id": request_id, "cancelled": True}
