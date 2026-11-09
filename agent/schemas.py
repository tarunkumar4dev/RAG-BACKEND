"""FROZEN CONTRACT (5 Nov 2026). AgentEvent = discriminated union streamed over SSE.

Every event carries `seq` (monotonic per request) and `request_id`.
stat_card / chart / table carry `source` ("tool:<name>") and `as_of` (ISO, Asia/Kolkata).
Do not rename fields without bumping CONTRACT_VERSION (frontend mirrors this file).
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

CONTRACT_VERSION = "1.2"  # 1.1: stat_card.period/delta, thinking_status tool.<name> steps
# 1.2 (additive): `session` event first in every stream, final.cancelled, clarify chip.hint (docs/AGENT_API_V2.md)

QuestionType = Literal["mcq", "numeric", "assertion_reason", "short_answer", "long_answer", "case_study", "very_short"]
VerifyStatus = Literal["verified", "failed", "manual_review", "rubric_ok"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seq: int = Field(ge=0)
    request_id: str = Field(min_length=1, max_length=64)


class _Sourced(_Base):
    source: str = Field(pattern=r"^tool:[a-z_]+$")
    as_of: str  # ISO-8601 with +05:30


class TextDelta(_Base):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingStatus(_Base):
    type: Literal["thinking_status"] = "thinking_status"
    key: str  # i18n key, e.g. "status.retrieving_ncert"
    params: dict[str, Any] = Field(default_factory=dict)


class StatCard(_Sourced):
    type: Literal["stat_card"] = "stat_card"
    title: str
    value: Optional[str]  # None => render "—"
    unit: Optional[str] = None
    subtitle: Optional[str] = None
    period: Optional[str] = None  # e.g. "1–5 Nov 2026"
    delta: Optional[str] = None   # e.g. "+2.1 pts vs Oct" — only when the tool computed it


class ChartSeries(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    points: list[tuple[str, Optional[float]]]


class Chart(_Sourced):
    type: Literal["chart"] = "chart"
    kind: Literal["line", "bar"]
    title: str
    series: list[ChartSeries]
    y_unit: Optional[str] = None


class Table(_Sourced):
    type: Literal["table"] = "table"
    title: str
    columns: list[str]
    rows: list[list[Optional[str]]]  # None cells render "—"


class ClarifyChip(BaseModel):
    model_config = ConfigDict(extra="forbid")
    label: str
    value: str  # text sent back as the user's next message when the chip is tapped
    hint: Optional[str] = None  # disambiguation, e.g. "Electronics · 22 students" / "0 students"


class Clarify(_Base):
    type: Literal["clarify"] = "clarify"
    question: str
    chips: list[ClarifyChip] = Field(max_length=5)


class BlueprintSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section: str  # "A".."E"
    qtype: QuestionType
    count: int = Field(ge=0)
    marks_each: float = Field(gt=0)


class Blueprint(_Base):
    type: Literal["blueprint"] = "blueprint"
    class_grade: int = Field(ge=1, le=12)
    subject: str
    chapters: list[str]
    total_marks: float
    duration_min: int
    sections: list[BlueprintSection]


class PreviewQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    section: str
    qtype: QuestionType
    marks: float
    text: str  # may contain $LaTeX$
    options: Optional[list[str]] = None
    answer: Optional[str] = None
    marking_scheme: Optional[list[str]] = None
    sub_questions: Optional[list[dict[str, Any]]] = None
    status: VerifyStatus


class PaperPreview(_Base):
    type: Literal["paper_preview"] = "paper_preview"
    draft_id: str
    title: str
    total_marks: float
    duration_min: int
    questions: list[PreviewQuestion]


class VerificationSummary(_Base):
    type: Literal["verification_summary"] = "verification_summary"
    objective_verified: int
    objective_total: int
    rubric_checked: int
    rubric_total: int
    manual_review: list[str]
    marks_sum_ok: bool
    duplicates: list[str] = Field(default_factory=list)
    rounds: int = 0


class PendingApproval(_Base):
    type: Literal["pending_approval"] = "pending_approval"
    approval_id: str
    action: Literal["finalize_test", "assign_to_section"]
    summary: str
    args_sha256: str
    expires_at: str


class SessionStarted(_Base):
    """First event of every /agent/chat stream (seq 0): which session this turn belongs to."""
    type: Literal["session"] = "session"
    session_id: str
    title: str


class Final(_Base):
    type: Literal["final"] = "final"
    session_id: str
    usage: dict[str, Any] = Field(default_factory=dict)
    cancelled: bool = False  # true when POST /agent/chat/{request_id}/cancel stopped the run


class Error(_Base):
    type: Literal["error"] = "error"
    code: Literal["budget_exceeded", "forbidden", "not_found", "provider_error", "invalid_input", "internal"]
    message: str
    partial: dict[str, Any] = Field(default_factory=dict)


AgentEvent = Annotated[
    Union[SessionStarted, TextDelta, ThinkingStatus, StatCard, Chart, Table, Clarify, Blueprint, PaperPreview,
          VerificationSummary, PendingApproval, Final, Error],
    Field(discriminator="type"),
]
AgentEventAdapter: TypeAdapter = TypeAdapter(AgentEvent)


# ---- request bodies -------------------------------------------------------
class ChatRequest(BaseModel):
    """message is required, except with regenerate_from_seq (the stored user message is re-run).
    regenerate_from_seq / edit_of_seq need session_id and are mutually exclusive (docs/AGENT_API_V2.md)."""
    model_config = ConfigDict(extra="forbid")
    session_id: Optional[str] = None
    message: Optional[str] = Field(default=None, min_length=1, max_length=4000)
    regenerate_from_seq: Optional[int] = Field(default=None, ge=1)
    edit_of_seq: Optional[int] = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check(self):
        if self.regenerate_from_seq is not None and self.edit_of_seq is not None:
            raise ValueError("regenerate_from_seq and edit_of_seq are mutually exclusive")
        if (self.regenerate_from_seq is not None or self.edit_of_seq is not None) and not self.session_id:
            raise ValueError("regenerate_from_seq / edit_of_seq need session_id")
        if self.regenerate_from_seq is None and not (self.message or "").strip():
            raise ValueError("message is required")
        return self


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    approval_id: str


class SessionCtx(BaseModel):
    """Frozen server-side identity. NEVER part of any tool schema. Built only from DB rows for the verified user."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    user_id: UUID
    role: Literal["admin", "hod", "teacher"]
    institute_id: UUID
    teacher_id: Optional[UUID] = None  # auth user id of the teacher (== user_id for teachers / HODs)
    department_id: Optional[UUID] = None  # HOD's department
    data_source: Literal["demo", "live"] = "demo"
    first_name: Optional[str] = None  # display only (greeting); never used for scope


def now_ist() -> str:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds")
