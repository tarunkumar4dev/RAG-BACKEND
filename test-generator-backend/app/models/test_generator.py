"""
Test Generator Models — Pydantic schemas for the full pipeline.

Covers: request, generation, response, feedback, save, quiz, manual questions.

v3 Changes:
  - Added ManualQuestionPayload model for teacher-added questions
  - Added is_manual flag to GeneratedQuestion
  - Added "image" and "manual" QuestionFormat values
  - SaveTestRequest now accepts full questions list

v2 Changes:
  - Added AnswerTable model for Accountancy tabular answers
  - Added answer_table field to GeneratedQuestion
  - Added JOURNAL_ENTRY, LEDGER, TRIAL_BALANCE to QuestionFormat
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


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
    # Commerce / Accountancy formats
    JOURNAL_ENTRY = "journal_entry"
    LEDGER = "ledger"
    TRIAL_BALANCE = "trial_balance"
    # v3: Manual-only formats
    IMAGE = "image"
    MANUAL = "manual"


class TestPattern(str, Enum):
    SIMPLE = "simple"
    BLUEPRINT = "blueprint"
    MATRIX = "matrix"
    BUCKETS = "buckets"


# ── Chapter section (one row in the teacher's form) ──────────────────────────

FORMAT_MAP = {
    "MCQ": QuestionFormat.MCQ,
    "Short Answer": QuestionFormat.SHORT_ANSWER,
    "Long Answer": QuestionFormat.LONG_ANSWER,
    "Assertion-Reason": QuestionFormat.ASSERTION_REASON,
    "Journal Entry": QuestionFormat.JOURNAL_ENTRY,
    "Ledger": QuestionFormat.LEDGER,
    "Trial Balance": QuestionFormat.TRIAL_BALANCE,
    "PDF": QuestionFormat.MCQ,
    "DOC": QuestionFormat.MCQ,
}

MARKS_MAP = {
    QuestionFormat.MCQ: 1,
    QuestionFormat.SHORT_ANSWER: 2,
    QuestionFormat.LONG_ANSWER: 5,
    QuestionFormat.ASSERTION_REASON: 1,
    QuestionFormat.JOURNAL_ENTRY: 4,
    QuestionFormat.LEDGER: 4,
    QuestionFormat.TRIAL_BALANCE: 5,
    QuestionFormat.IMAGE: 3,
    QuestionFormat.MANUAL: 2,
}


# ── Tabular answer structure ─────────────────────────────────────────────

class AnswerTable(BaseModel):
    """Structured table answer for Accountancy questions."""
    type: str
    headers: List[str]
    rows: List[List[str]]
    total_row: Optional[List[str]] = None


# ── Main request ──────────────────────────────────────────────────────────────

class ChapterSection(BaseModel):
    chapter: str
    topic: Optional[str] = None
    subtopics: List[str] = []
    quantity: int
    difficulty: DifficultyLevel
    format: QuestionFormat
    marks_per_question: int


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
    validation_status: Literal["verified", "needs_review", "manual"] = "verified"
    validation_notes: Optional[str] = None
    # Tabular answer for Accountancy
    answer_table: Optional[AnswerTable] = None
    # v3: Manual question fields
    is_manual: bool = False
    image_url: Optional[str] = None  # for image-based manual questions
    section: Optional[str] = None  # CBSE section (A/B/C/D/E)


# ── v3: Manual Question Payload (from frontend) ──────────────────────────────

class ManualQuestionPayload(BaseModel):
    """
    Represents a question added manually by the teacher.
    Accepts both camelCase (frontend) and snake_case keys.
    """
    id: str
    text: str
    options: Optional[List[str]] = None
    correctAnswer: Optional[str] = None      # frontend camelCase
    correct_answer: Optional[str] = None     # backend snake_case
    explanation: Optional[str] = ""
    marks: int = 1
    difficulty: str = "medium"
    chapter: Optional[str] = "Manual Addition"
    topic: Optional[str] = None
    format: str = "manual"  # mcq, short_answer, long_answer, image, manual
    type: Optional[str] = None  # alternative name from frontend
    imageUrl: Optional[str] = None
    image_url: Optional[str] = None
    section: Optional[str] = None
    isManual: bool = True
    is_manual: bool = True

    def to_generated_question(self) -> GeneratedQuestion:
        """Convert to the standard GeneratedQuestion model for DB save/export."""
        correct = self.correctAnswer or self.correct_answer or ""
        img_url = self.imageUrl or self.image_url

        # Map format/type → QuestionFormat
        fmt_raw = (self.type or self.format or "manual").lower()
        fmt_map = {
            "mcq": QuestionFormat.MCQ,
            "short": QuestionFormat.SHORT_ANSWER,
            "short_answer": QuestionFormat.SHORT_ANSWER,
            "long": QuestionFormat.LONG_ANSWER,
            "long_answer": QuestionFormat.LONG_ANSWER,
            "image": QuestionFormat.IMAGE,
            "manual": QuestionFormat.MANUAL,
        }
        fmt = fmt_map.get(fmt_raw, QuestionFormat.MANUAL)

        # Map difficulty
        diff_raw = self.difficulty.lower()
        diff_map = {
            "easy": DifficultyLevel.EASY,
            "medium": DifficultyLevel.MEDIUM,
            "hard": DifficultyLevel.HARD,
        }
        diff = diff_map.get(diff_raw, DifficultyLevel.MEDIUM)

        return GeneratedQuestion(
            id=self.id,
            text=self.text,
            options=self.options if fmt == QuestionFormat.MCQ else None,
            correct_answer=correct,
            explanation=self.explanation or "",
            marks=self.marks,
            difficulty=diff,
            bloom_level=None,
            chapter=self.chapter or "Manual Addition",
            topic=self.topic,
            format=fmt,
            validation_status="manual",
            is_manual=True,
            image_url=img_url,
            section=self.section,
        )


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
    # v3: allow passing the final question list (including manual additions)
    questions: Optional[List[dict]] = None


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