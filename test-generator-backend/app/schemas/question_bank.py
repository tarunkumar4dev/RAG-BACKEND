# app/schemas/question_bank.py
# ──────────────────────────────────────────────────────────
# "Question Bank → Paper Builder" (/api/v1/question-banks)
#
# Every request model forbids unknown fields and has NO user-id field: the owner is always the JWT user.
# BankQuestionIn is the subset of the contest import's ImportQuestion (app/schemas/contest_import.py) that a bank
# keeps, after the teacher's review; "teacher" is added to answer_source for an answer the teacher set / confirmed.
# ──────────────────────────────────────────────────────────

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

QType = Literal["mcq", "true_false", "short", "long"]

MAX_QUESTIONS_PER_BANK = 300
MAX_OPTIONS = 6
MAX_SECTIONS = 6
MAX_PAPER_QUESTIONS = 200          # the exporter's own cap (ExportRequest.questions)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BankOptionIn(_Strict):
    label: str = Field(default="", max_length=8)
    text: str = Field(max_length=1000)


class BankQuestionIn(_Strict):
    question_text: str = Field(max_length=6000)
    question_type: str = Field(default="", max_length=20)        # MCQ / Short / Long as the parser sends it
    options: List[BankOptionIn] = Field(default=[], max_length=MAX_OPTIONS)
    correct_answer: Optional[str] = Field(default=None, max_length=4000)
    answer_source: Optional[Literal["key", "ai", "missing", "teacher"]] = None
    marks: Optional[float] = Field(default=None, gt=0, le=100)
    usable: bool = True
    source_number: Optional[int] = Field(default=None, ge=0, le=100000)


class CreateBankRequest(_Strict):
    name: str = Field(min_length=1, max_length=120)
    source_filename: Optional[str] = Field(default=None, max_length=255)
    source_tag: Optional[str] = Field(default=None, max_length=200)   # the paper's title line, e.g. "CBSE 2023 …"
    subject: Optional[str] = Field(default=None, max_length=50)
    class_grade: Optional[str] = Field(default=None, max_length=20)
    board: Optional[str] = Field(default=None, max_length=30)
    questions: List[BankQuestionIn] = Field(min_length=1, max_length=MAX_QUESTIONS_PER_BANK)


class PatchQuestionRequest(_Strict):
    """Only the fields that are sent change. answer: null clears it; a value makes it a teacher answer."""
    question_text: Optional[str] = Field(default=None, min_length=1, max_length=6000)
    options: Optional[List[BankOptionIn]] = Field(default=None, max_length=MAX_OPTIONS)
    answer: Optional[str] = Field(default=None, max_length=4000)
    marks: Optional[float] = Field(default=None, gt=0, le=100)
    excluded: Optional[bool] = None
    reviewed: Optional[Literal[True]] = None      # teacher looked at a flagged question and accepts it as it is


class PaperSection(_Strict):
    qtype: QType
    count: int = Field(ge=1, le=100)
    marks_each: int = Field(ge=1, le=100)


class BuildPaperRequest(_Strict):
    title: str = Field(min_length=1, max_length=200)
    institute_name: Optional[str] = Field(default=None, max_length=200)
    sections: List[PaperSection] = Field(min_length=1, max_length=MAX_SECTIONS)
    include_answer_key: bool = False
    template: str = Field(default="modern", max_length=30)
    shuffle: bool = False
    year_filter: Optional[int] = Field(default=None, ge=1950, le=2100)
    # optional extras (defaults come from the bank)
    format: Literal["pdf", "docx"] = "pdf"
    allow_needs_review: bool = False
    subject: Optional[str] = Field(default=None, max_length=50)
    class_grade: Optional[str] = Field(default=None, max_length=20)
    board: Optional[str] = Field(default=None, max_length=30)
    duration: Optional[str] = Field(default=None, max_length=20)
    teacher_name: Optional[str] = Field(default=None, max_length=100)
