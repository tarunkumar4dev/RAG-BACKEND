"""API v2 routes (docs/AGENT_API_V2.md): sessions, messages, query history, suggestions, feedback, cancel."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from .api_models import (CancelResponse, FeedbackRequest, FeedbackResponse, MessageList, QueryItem, SessionItem,
                         SessionList, SessionPatch, Suggestion)
from .auth import get_session
from .schemas import SessionCtx

router = APIRouter(prefix="/agent", tags=["agent-v2"])


def _todo():
    raise HTTPException(501, "not implemented yet")


@router.get("/sessions", response_model=SessionList)
def list_sessions(cursor: Optional[str] = Query(None, max_length=200), limit: int = Query(20, ge=1, le=50),
                  q: Optional[str] = Query(None, max_length=80), pinned: Optional[bool] = None,
                  ctx: SessionCtx = Depends(get_session)):
    _todo()


@router.get("/sessions/{session_id}/messages", response_model=MessageList)
def list_messages(session_id: str, cursor: Optional[str] = Query(None, max_length=200),
                  limit: int = Query(100, ge=1, le=200), ctx: SessionCtx = Depends(get_session)):
    _todo()


@router.patch("/sessions/{session_id}", response_model=SessionItem)
def patch_session(session_id: str, body: SessionPatch, ctx: SessionCtx = Depends(get_session)):
    _todo()


@router.delete("/sessions/{session_id}", status_code=204, response_class=Response)
def delete_session(session_id: str, ctx: SessionCtx = Depends(get_session)):
    _todo()


@router.get("/history/queries", response_model=list[QueryItem])
def list_queries(q: Optional[str] = Query(None, max_length=80), limit: int = Query(10, ge=1, le=50),
                 ctx: SessionCtx = Depends(get_session)):
    _todo()


@router.delete("/history/queries/{query_id}", status_code=204, response_class=Response)
def delete_query(query_id: str, ctx: SessionCtx = Depends(get_session)):
    _todo()


@router.delete("/history/queries", status_code=204, response_class=Response)
def clear_queries(ctx: SessionCtx = Depends(get_session)):
    _todo()


@router.get("/suggestions", response_model=list[Suggestion])
def suggestions(q: Optional[str] = Query(None, max_length=80), limit: int = Query(8, ge=1, le=20),
                ctx: SessionCtx = Depends(get_session)):
    _todo()


@router.post("/feedback", response_model=FeedbackResponse)
def feedback(body: FeedbackRequest, ctx: SessionCtx = Depends(get_session)):
    _todo()


@router.post("/chat/{request_id}/cancel", status_code=202, response_model=CancelResponse)
def cancel(request_id: str, ctx: SessionCtx = Depends(get_session)):
    _todo()
