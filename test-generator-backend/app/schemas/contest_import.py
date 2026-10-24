# app/schemas/contest_import.py
# ──────────────────────────────────────────────────────────
# "PDF → contest questions" import (POST/GET /api/v1/contests/import-pdf)
#
# ImportResult.questions is directly usable as CreateContestRequest.questions:
#   - correct_answer is always the bare uppercase option label ("B") for an MCQ — the contest grader accepts
#     label or text, but the after-test review screen compares labels only.
#   - source_number / answer_source / usable are for the teacher's review screen; POST /contests drops them.
# ──────────────────────────────────────────────────────────

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ImportRequest(BaseModel):
    path: str = Field(description="<uid>/contest-imports/<uuid>.pdf in the private 'answer-keys' bucket")


class ImportOption(BaseModel):
    label: str               # "A".."F"
    text: str                # without the "(a)" prefix


class ImportQuestion(BaseModel):
    question_text: str
    question_type: Literal["MCQ", "Short", "Long"]
    options: List[ImportOption] = []
    correct_answer: Optional[str] = None
    explanation: Optional[str] = None
    marks: int = 1
    difficulty: Literal["Easy", "Medium", "Hard"] = "Medium"
    chapter: Optional[str] = None
    source_number: int
    answer_source: Literal["key", "ai", "missing"]
    usable: bool = True       # False: the teacher must fix or drop it before publishing (see warnings)


class ImportWarning(BaseModel):
    question: Optional[int] = None    # source_number, None = whole paper
    code: str
    message: str


class ImportResult(BaseModel):
    title: str
    subject: Optional[str] = None
    class_grade: Optional[str] = None
    institute: Optional[str] = None
    board: Optional[str] = None               # only when printed on the paper
    duration_minutes: Optional[int] = None    # already clamped to 5-180 (POST /contests limits)
    total_marks: int
    answer_key_found: bool
    answer_key_count: int
    questions: List[ImportQuestion]
    warnings: List[ImportWarning] = []


class ImportJobOut(BaseModel):
    status: Literal["queued", "processing", "done", "failed"]
    progress: int = 0
    error: Optional[str] = None
    retryable: bool = False
    result: Optional[ImportResult] = None
