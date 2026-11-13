"""Benchmark fixture + paper schema (Phase 0).

A fixture is a generation config plus expectations the runner can check.
A paper is the normalised view of a generated test-paper the metrics score.

Both are plain dataclasses to keep I/O decoupled from the service's pydantic models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Paper (what metrics consume) ─────────────────────────────────────────

@dataclass
class PaperQuestion:
    qid: str                          # stable id within a paper (fixture.qid or generated uuid)
    text: str
    format: str                       # "mcq" | "assertion_reason" | "short_answer" | "long_answer" | ...
    options: list[str] | None         # None for non-MCQ
    correct_answer: str               # for MCQ: full option text ("B) ...") or option text; for others: free-form
    marks: int
    difficulty: str | None = None
    topic: str | None = None
    family: str | None = None
    validation_status: str | None = None


@dataclass
class Paper:
    questions: list[PaperQuestion]
    total_marks: int
    subject: str
    class_grade: str
    chapters: list[str]

    @staticmethod
    def from_backend_response(resp: dict[str, Any]) -> "Paper":
        """Normalise a FrontendGenerateResponse dict into a Paper."""
        qs = []
        for q in resp.get("questions", []):
            qs.append(PaperQuestion(
                qid=str(q.get("id") or q.get("qid") or f"q{len(qs)+1}"),
                text=q.get("text") or "",
                format=(q.get("format") or "mcq"),
                options=q.get("options"),
                correct_answer=q.get("correct_answer") or "",
                marks=int(q.get("marks") or 1),
                difficulty=q.get("difficulty"),
                topic=q.get("topic"),
                family=q.get("family"),
                validation_status=q.get("validation_status"),
            ))
        return Paper(
            questions=qs,
            total_marks=int(resp.get("totalMarks") or sum(q.marks for q in qs)),
            subject=resp.get("subject", ""),
            class_grade=str(resp.get("classGrade", "")),
            chapters=list(resp.get("chapters", []) or []),
        )


# ── Fixture (what the harness loads) ─────────────────────────────────────

@dataclass
class ExpectedNumericItem:
    """One numeric MCQ item with an INDEPENDENTLY verified answer.

    `expected_answer` is a float (currency amounts in rupees, ratios as decimals,
    percentages as the % number itself e.g. 12 means 12%). `tolerance` is absolute.

    `qid` matches a PaperQuestion.qid when post-hoc scoring a saved paper; when
    `qid=None`, the runner matches by nearest text similarity (not implemented in
    Phase 0 — qid required for now).
    """
    qid: str
    description: str
    expected_answer: float
    tolerance: float = 1.0
    unit: str = "INR"              # "INR" | "%" | "ratio" | "months"
    key_source: str = "author-computed"   # "author-computed" | "sme-verified" | "pending_sme"


@dataclass
class BenchmarkConfig:
    """Fixture describing a generation request + expectations."""
    id: str
    description: str
    request: dict[str, Any]                 # payload for /generate-frontend (shape = FrontendGenerateRequest)
    expected_total_questions: int
    expected_total_marks: int
    expected_numeric_items: list[ExpectedNumericItem] = field(default_factory=list)
    golden_paper_file: str | None = None    # relative path under fixtures/, optional
    pdf_file: str | None = None             # relative path to a pre-rendered PDF, optional
    key_source: str = "author-computed"     # fixture-level source of expectations
    notes: str = ""

    def resolve_path(self, rel: str) -> str:
        from pathlib import Path
        base = Path(__file__).parent / "fixtures"
        return str((base / rel).resolve())
