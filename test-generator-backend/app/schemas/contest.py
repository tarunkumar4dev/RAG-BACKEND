# app/schemas/contest.py
# ──────────────────────────────────────────────────────────
# Pydantic models for Contest API
#
# FIXED:
#   [422 on create] ContestQuestionIn.question_type: was Literal["MCQ","Short","Long"]
#                   → any other type from the generator (SHORT_ANSWER, etc.) caused a
#                   422. Relaxed to plain str so all generated types are accepted.
#   [safety] options relaxed to Optional[List[Any]] so both ["4","8"] and
#            [{"label":"A","text":"4"}] shapes are accepted.
# ──────────────────────────────────────────────────────────

from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any
from datetime import datetime
from uuid import UUID


# ═══════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ═══════════════════════════════════════════════════════════

class ContestQuestionIn(BaseModel):
    """Single question to include in contest"""
    question_text: str
    question_type: str = "MCQ"                     # was Literal[...] — relaxed to accept any generated type
    options: Optional[List[Any]] = None            # ["4","8"] OR [{"label":"A","text":"..."}]
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    marks: int = 1
    difficulty: str = "Medium"
    chapter: Optional[str] = None


class CreateContestRequest(BaseModel):
    """Teacher creates a contest from generated questions"""
    title: str = "Test Paper"
    subject: str
    class_grade: str
    board: str = "CBSE"
    logo_base64: Optional[str] = None

    # Settings
    duration_minutes: int = Field(default=30, ge=5, le=180)
    answer_mode: Literal["instant", "after_test", "none"] = "after_test"
    show_explanation: bool = True
    enable_camera: bool = True
    enable_tab_detection: bool = True
    allow_back_navigation: bool = True
    max_warnings: int = Field(default=3, ge=1, le=10)
    max_attempts: int = Field(default=1, ge=1, le=5)

    # Scheduling
    scheduled_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    # Questions
    questions: List[ContestQuestionIn]


class SubmitContestRequest(BaseModel):
    """Student submits their answers"""
    student_name: Optional[str] = None
    student_email: Optional[str] = None
    answers: List[dict]             # [{"questionId": "...", "selected": "A", "timeSpent": 12}]
    warning_count: int = 0
    warning_log: List[dict] = []    # [{"reason": "tab switch", "at": "..."}]
    time_taken_seconds: int = 0


class StartAttemptRequest(BaseModel):
    """Student starts a contest attempt"""
    student_name: Optional[str] = None
    student_email: Optional[str] = None


# ═══════════════════════════════════════════════════════════
# RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════

class ContestQuestionOut(BaseModel):
    """Question as sent to student (no correct_answer or explanation)"""
    id: str
    question_number: int
    question_text: str
    question_type: str
    options: Optional[List[Any]] = None
    marks: int
    difficulty: str
    chapter: Optional[str] = None


class ContestQuestionWithAnswer(BaseModel):
    """Question with answer — only sent after submission if answer_mode allows"""
    id: str
    question_number: int
    question_text: str
    question_type: str
    options: Optional[List[Any]] = None
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    marks: int
    difficulty: str
    chapter: Optional[str] = None


class CreateContestResponse(BaseModel):
    """Response after creating a contest"""
    contest_id: str
    short_code: str
    share_link: str
    total_questions: int
    total_marks: int


class ContestInfoResponse(BaseModel):
    """Public contest info for student landing page"""
    contest_id: str
    title: str
    subject: str
    class_grade: str
    board: str
    logo_base64: Optional[str] = None
    duration_minutes: int
    total_questions: int
    total_marks: int
    enable_camera: bool
    enable_tab_detection: bool
    allow_back_navigation: bool
    max_warnings: int
    status: str
    teacher_name: Optional[str] = None
    institute_name: Optional[str] = None


class ContestDataResponse(BaseModel):
    """Full contest data for student taking the test"""
    contest_id: str
    attempt_id: str
    title: str
    subject: str
    class_grade: str
    duration_minutes: int
    enable_camera: bool
    enable_tab_detection: bool
    allow_back_navigation: bool
    max_warnings: int
    answer_mode: str
    show_explanation: bool
    questions: List[ContestQuestionOut]


class SubmitContestResponse(BaseModel):
    """Response after submitting answers"""
    attempt_id: str
    status: str
    score: Optional[int] = None
    total_marks: int
    percentage: Optional[float] = None
    answered_count: int
    total_questions: int
    # Only included if answer_mode is "after_test"
    questions_with_answers: Optional[List[ContestQuestionWithAnswer]] = None


class AttemptSummary(BaseModel):
    """Summary of a student's attempt — for teacher dashboard"""
    attempt_id: str
    student_name: Optional[str] = None
    student_email: Optional[str] = None
    status: str
    score: Optional[int] = None
    total_marks: int
    percentage: Optional[float] = None
    time_taken_seconds: Optional[int] = None
    warning_count: int
    submitted_at: Optional[datetime] = None


class ContestLeaderboardResponse(BaseModel):
    """Leaderboard for a contest"""
    contest_id: str
    title: str
    total_attempts: int
    attempts: List[AttemptSummary]