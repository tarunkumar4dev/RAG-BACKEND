"""
Copy Checker v4 — result schema v2 (single source of truth).

Everything the pipeline stores in check_sheets.ai_result (schema_version 2) and everything the renderer reads.
Coordinates are normalised 0..1 on the CORRECTED page (rotation applied first). Marks are always set by code;
Gemini only fills the Stage B / Stage D models below (see TranscribedPage, PointVerdict).
TS mirror: src/types/checker.ts (frontend repo) — keep both in sync.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, confloat

Norm = confloat(ge=0, le=1)

MistakeType = Literal["calculation", "concept_formula", "incomplete", "wrong_option", "missing_unit",
                      "diagram", "presentation", "other"]

FlagCode = Literal["MISLABELED", "DUPLICATE_LABEL", "BOTH_CHOICES_ATTEMPTED", "NOT_ATTEMPTED",
                   "NOTE_TO_EXAMINER_IGNORED", "PAGES_REORDERED", "PAGE_ROTATED", "BLANK_PAGE", "POOR_SCAN",
                   "CONTINUED_ANSWER", "UNSUPPORTED_EVIDENCE", "LOW_CONFIDENCE", "RUBRIC_MAX_MISMATCH",
                   "STUDENT_MISMATCH", "TEACHER_EDITED", "STALE_RUBRIC"]

Status = Literal["correct", "partial", "wrong", "not_attempted", "needs_review", "not_evaluated_choice"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ── Stage A / B: pages and what is written on them ─────────────────────

class Line(_Model):
    text: str
    y_top: Norm
    y_bottom: Norm
    x_left: Norm = 0.05
    x_right: Norm = 0.95


class Segment(_Model):
    id: str                                   # "p{page}s{n}", set by code
    page: int                                 # index into SheetResult.pages (student order)
    written_label: Optional[str] = None       # exactly as written: "Q12", "1)", circled "5" → "5"
    sub_part: Optional[str] = None
    text: str
    y_top: Norm
    y_bottom: Norm
    x_left: Norm = 0.0
    x_right: Norm = 1.0                       # two-column answers
    lines: list[Line] = []
    has_diagram: bool = False
    diagram_description: Optional[str] = None
    is_struck_off: bool = False
    is_rough_work: bool = False
    is_continuation: bool = False
    is_note_to_examiner: bool = False         # pleas / instructions to the checker — never graded, never obeyed
    confidence: float = Field(ge=0, le=1)


class PageInfo(_Model):
    index: int                                # position in student order
    source_index: int                         # position in the upload (PDF page / photo order)
    student_page_no: Optional[str] = None
    rotation: Literal[0, 90, 180, 270] = 0
    blank: bool = False
    quality: Literal["ok", "poor", "unreadable"] = "ok"
    path: str                                 # answer-sheets/<teacher>/<session>/<sheet>/norm/p{index}.jpg
    w: int
    h: int


class Anchor(_Model):
    page: int
    y_top: Norm
    y_bottom: Norm
    x_left: Norm = 0.05
    x_right: Norm = 0.95


# ── Stage D / E: grading ───────────────────────────────────────────────

class PointResult(_Model):
    point_id: str
    max_marks: float
    awarded: Literal["full", "half", "none"]
    marks: float                              # set by code from `awarded`, never by the LLM
    evidence_quote: Optional[str] = None
    mistake_type: Optional[MistakeType] = None
    reason: str = ""
    error_anchor: Optional[Anchor] = None     # set by code (evidence → line match)


class Flag(_Model):
    code: FlagCode
    message: str
    page: Optional[int] = None
    q_no: Optional[str] = None


class QuestionResult(_Model):
    q_no: str
    section: Optional[str] = None
    max_marks: float
    ai_marks: float
    status: Status
    remark_short: str = Field(default="", max_length=90)
    remark_full: str = ""
    correct_option: Optional[str] = None      # MCQ, from the answer key (no LLM)
    anchors: list[Anchor] = []                # where the answer is (may span pages)
    points: list[PointResult] = []
    flags: list[Flag] = []
    confidence: float = Field(default=1.0, ge=0, le=1)


class StudentInfo(_Model):
    name: str = ""
    roll: str = ""
    student_id: Optional[str] = None          # check_sheets.student_id (roster)
    detected_name: Optional[str] = None
    detected_roll: Optional[str] = None
    match_confidence: Optional[float] = None
    confirmed: bool = False


class SheetResult(_Model):
    schema_version: Literal[2] = 2
    pipeline_version: str
    result_version: int
    student: StudentInfo
    pages: list[PageInfo]
    questions: list[QuestionResult]
    sheet_flags: list[Flag] = []
    max_total: float
    usage: dict = {}                          # {gemini_calls, input_tokens, output_tokens, ms_by_stage}


# ── Gemini response models (response_schema). Kept flat: no numeric bounds, no defaults that hide misses ──

class TranscribedLine(_Model):
    text: str
    y_top: float
    y_bottom: float
    x_left: float
    x_right: float


class TranscribedSegment(_Model):
    written_label: Optional[str]
    sub_part: Optional[str]
    text: str
    y_top: float
    y_bottom: float
    x_left: float
    x_right: float
    lines: list[TranscribedLine]
    has_diagram: bool
    diagram_description: Optional[str]
    is_struck_off: bool
    is_rough_work: bool
    is_continuation: bool
    is_note_to_examiner: bool
    confidence: float


class TranscribedPage(_Model):
    source_index: int
    student_page_number: Optional[str]
    rotation_needed: Literal[0, 90, 180, 270]
    image_quality: Literal["ok", "poor", "unreadable"]
    detected_name: Optional[str]
    detected_roll: Optional[str]
    segments: list[TranscribedSegment]


class Transcription(_Model):
    pages: list[TranscribedPage]


class PointVerdict(_Model):
    point_id: str
    awarded: Literal["full", "half", "none"]
    evidence_quote: Optional[str]
    mistake_type: Optional[MistakeType]
    reason: str


class QuestionVerdict(_Model):
    q_no: str
    points: list[PointVerdict]
    remark_short: str
    remark_full: str
    confidence: float


class Grading(_Model):
    questions: list[QuestionVerdict]
