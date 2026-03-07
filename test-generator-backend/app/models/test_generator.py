"""
Test Generator Models — Pydantic schemas for the full pipeline.

Covers: request, generation, response, feedback, save, quiz.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"          # ← fixed: underscore, not space


class BloomLevel(str, Enum):
    REMEMBER = "remember"
    UNDERSTAND = "understand"
    APPLY = "apply"
    ANALYZE = "analyze"
    EVALUATE = "evaluate"
    CREATE = "create"


class QuestionFormat(str, Enum):
    MCQ = "mcq"
    SHORT_ANSWER = "short_answer"
    LONG_ANSWER = "long_answer"
    ASSERTION_REASON = "assertion_reason"
    CASE_BASED = "case_based"


class TestPattern(str, Enum):
    SIMPLE = "simple"
    BLUEPRINT = "blueprint"
    MATRIX = "matrix"
    BUCKETS = "buckets"


# ── Chapter section (one row in the teacher's form) ──────────────────────────

class ChapterSection(BaseModel):
    chapter: str
    topic: Optional[str] = None
    subtopics: List[str] = []
    quantity: int = Field(default=5, ge=1, le=100)
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    format: QuestionFormat = QuestionFormat.MCQ
    bloom_levels: List[BloomLevel] = []
    marks_per_question: int = Field(default=1, ge=1)


# ── Main request ──────────────────────────────────────────────────────────────

class TestGenerationRequest(BaseModel):
    exam_title: str
    board: str = "CBSE"
    class_grade: str = "10"
    subject: str

    chapters: List[ChapterSection] = Field(..., min_length=1)
    pattern: TestPattern = TestPattern.SIMPLE

    bloom_enabled: bool = False

    reference_file_url: Optional[str] = None

    teacher_id: str

    iteration: int = Field(default=0, ge=0)
    previous_test_id: Optional[str] = None
    teacher_feedback: Optional[str] = None


# ── Generated question ────────────────────────────────────────────────────────

class GeneratedQuestion(BaseModel):
    id: str
    text: str
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: str
    marks: int
    difficulty: DifficultyLevel
    bloom_level: Optional[BloomLevel] = None
    chapter: str
    topic: Optional[str] = None
    format: QuestionFormat
    validation_status: Literal["verified", "needs_review"] = "verified"
    validation_notes: Optional[str] = None


# ── Response ──────────────────────────────────────────────────────────────────

class TestGenerationResponse(BaseModel):
    test_id: str
    exam_title: str
    questions: List[GeneratedQuestion]
    total_marks: int
    total_questions: int
    iteration: int
    generation_time_seconds: float
    status: Literal["preview", "saved", "exported"] = "preview"


# ── Feedback ──────────────────────────────────────────────────────────────────

class QuestionFeedback(BaseModel):
    question_id: str
    action: Literal["approve", "reject", "edit"]
    comment: Optional[str] = None
    edited_text: Optional[str] = None


class TestFeedbackRequest(BaseModel):
    test_id: str
    teacher_id: str
    feedbacks: List[QuestionFeedback]
    global_comment: Optional[str] = None


# ── Save / Export ─────────────────────────────────────────────────────────────

class SaveTestRequest(BaseModel):
    test_id: str
    teacher_id: str
    export_format: Literal["pdf", "docx"] = "pdf"


# ── Quiz ──────────────────────────────────────────────────────────────────────

class QuizSettings(BaseModel):
    test_id: str
    teacher_id: str
    duration_minutes: int = Field(default=60, ge=5)
    max_marks: int
    passing_marks: int
    shuffle_questions: bool = True
    shuffle_options: bool = True
    camera_required: bool = False
    tab_switch_limit: int = Field(default=3, ge=0)