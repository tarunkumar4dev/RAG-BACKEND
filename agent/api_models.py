"""API v2 request/response models (docs/AGENT_API_V2.md). STABLE: a separate frontend session builds against these.

Every endpoint is scoped to the SessionCtx built from the verified JWT. Ids of another user / institute -> 404.
Timestamps are ISO-8601 UTC strings. Cursors are opaque strings: send back exactly what the server returned.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schemas import AgentEvent


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionItem(_Out):
    id: str
    title: str
    created_at: str
    last_message_at: str
    message_count: int
    pinned: bool
    preview: Optional[str] = None  # last assistant text (or last user text), max 120 chars


class LegacySessionItem(_Out):
    """Contract 1.1 shape, kept so the current UI keeps working. Deprecated: use `items`."""
    session_id: str
    title: str
    created_at: str
    updated_at: str


class SessionList(_Out):
    items: list[SessionItem]
    next_cursor: Optional[str] = None
    sessions: list[LegacySessionItem] = Field(default_factory=list, description="deprecated alias of items (1.1)")


class MessageItem(_Out):
    seq: int
    role: Literal["user", "assistant"]
    text: Optional[str] = None
    events: Optional[list[AgentEvent]] = None  # assistant only: the exact streamed events, for replay
    created_at: str


class MessageList(_Out):
    items: list[MessageItem]
    next_cursor: Optional[str] = None


class SessionPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: Optional[str] = Field(default=None, min_length=1, max_length=80)
    pinned: Optional[bool] = None

    @model_validator(mode="after")
    def _one(self):
        if self.title is None and self.pinned is None:
            raise ValueError("send title and/or pinned")
        if self.title is not None and not self.title.strip():
            raise ValueError("title must not be blank")
        return self


class QueryItem(_Out):
    id: str
    text: str
    last_used_at: str
    use_count: int


class Suggestion(_Out):
    type: Literal["recent", "template", "entity"]
    text: str  # ready to send as the chat message
    hint: Optional[str] = None  # e.g. "Batch · 22 students", "Recent · 3×"


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(max_length=64)
    message_seq: int = Field(ge=1)
    rating: Literal["up", "down"]
    note: Optional[str] = Field(default=None, max_length=1000)


class FeedbackResponse(_Out):
    ok: bool = True


class CancelResponse(_Out):
    request_id: str
    cancelled: bool = True
