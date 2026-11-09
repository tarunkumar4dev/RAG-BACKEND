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


# ---- 1.3 --------------------------------------------------------------------------------------------------
class MetaScope(_Out):
    """What the caller can see. level: institute (admin) > department (HOD) > section (proctor) > classes
    (teacher's own batch/subject pairs) > none (nothing assigned yet)."""
    level: Literal["institute", "department", "section", "classes", "none"]
    batch_count: int
    student_count: int
    batches: list[str] = Field(default_factory=list, description="batch names, max 50, sorted")


class MetaFeatures(_Out):
    paper_generation: bool  # false in live read-only mode
    defaulters_export: bool
    proctor_month_report: bool
    cancel: bool = True


class MetaLimits(_Out):
    chat_turns_per_10min: int
    institute_daily_tokens: Optional[int] = None  # null = no institute budget configured


class MetaResponse(_Out):
    contract_version: str
    enabled: bool  # false: the agent is not switched on for this institute yet (AGENT_ENABLED_INSTITUTES)
    data_source: Literal["demo", "live"]
    read_only: bool
    role: Literal["admin", "hod", "teacher"]
    first_name: Optional[str] = None
    institute_name: Optional[str] = None
    scope: Optional[MetaScope] = None  # null when enabled is false or the scope could not be loaded
    features: MetaFeatures
    limits: MetaLimits


class RateLimitDetail(_Out):
    """`detail` of a 429 from POST /agent/chat (also sent as the Retry-After header). Nothing was stored."""
    code: Literal["rate_limited"] = "rate_limited"
    scope: Literal["user", "institute"]  # user: too many turns in 10 min; institute: daily token budget used up
    message: str  # friendly Hinglish, ready to show
    retry_after_s: int


class RateLimitResponse(_Out):
    detail: RateLimitDetail


class HealthResponse(_Out):
    status: Literal["ok"] = "ok"


class ReadyResponse(_Out):
    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]  # e.g. {"store": true, "db": true}
