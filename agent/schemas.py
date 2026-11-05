"""FROZEN CONTRACT (5 Nov 2026). AgentEvent = discriminated union streamed over SSE.

Every event carries `seq` (monotonic per request) and `request_id`.
stat_card / chart / table carry `source` ("tool:<name>") and `as_of` (ISO, Asia/Kolkata).
Do not rename fields without bumping CONTRACT_VERSION (frontend mirrors this file).
"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

CONTRACT_VERSION = "1.0"

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
    value: str  # entity id or canonical value sent back as the user's choice


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


class Final(_Base):
    type: Literal["final"] = "final"
    session_id: str
    usage: dict[str, Any] = Field(default_factory=dict)


class Error(_Base):
    type: Literal["error"] = "error"
    code: Literal["budget_exceeded", "forbidden", "not_found", "provider_error", "invalid_input", "internal"]
    message: str
    partial: dict[str, Any] = Field(default_factory=dict)


AgentEvent = Annotated[
    Union[TextDelta, ThinkingStatus, StatCard, Chart, Table, Clarify, Blueprint, PaperPreview,
          VerificationSummary, PendingApproval, Final, Error],
    Field(discriminator="type"),
]
AgentEventAdapter: TypeAdapter = TypeAdapter(AgentEvent)


# ---- request bodies -------------------------------------------------------
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: Optional[str] = None
    message: str = Field(min_length=1, max_length=4000)


class ApproveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    approval_id: str


class SessionCtx(BaseModel):
    """Frozen server-side identity. NEVER part of any tool schema."""
    model_config = ConfigDict(frozen=True, extra="forbid")
    user_id: UUID
    role: Literal["admin", "teacher"]
    institute_id: UUID
    teacher_id: Optional[UUID] = None  # auth user id of the teacher (== user_id for teachers)


def now_ist() -> str:
    from zoneinfo import ZoneInfo
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds")
