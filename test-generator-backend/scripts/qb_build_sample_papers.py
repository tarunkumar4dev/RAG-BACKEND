"""
Builds real papers from both fixture banks through the existing export system, for a human to look at.
Offline (recorded parser replies, no database): fixture → bank rows → sections → PDF and DOCX.

    python scripts/qb_build_sample_papers.py [output dir]      (default: ./qb_sample_papers, git-ignored name)
"""

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import qb_contest_flow_snapshot as snap  # noqa: E402  (sets the offline environment)
from app.services import question_bank_service as qb  # noqa: E402

PAPERS = [
    # (fixture, file stem, template, sections, institute)
    ("business_studies", "business_studies_modern", "modern",
     [("mcq", 4, 1), ("true_false", 4, 1), ("short", 3, 3), ("long", 2, 5)], "Sunrise Commerce Classes"),
    ("business_studies", "business_studies_colorful", "colorful",
     [("mcq", 5, 1), ("true_false", 4, 1), ("short", 4, 3), ("long", 2, 5)], "Sunrise Commerce Classes"),
    ("economics", "economics_classic", "classic", [("mcq", 20, 1), ("mcq", 5, 2)], None),
    ("economics", "economics_modern", "modern", [("mcq", 25, 1)], "Education Beast"),
]


def bank_rows(name: str) -> tuple:
    result = snap.run_case(name)
    rows, _ = qb.normalise_questions(qb.payload_from_import(result), result.get("title"))
    for r in rows:
        r["id"] = str(uuid.uuid4())
    return result, rows


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("qb_sample_papers")
    out.mkdir(parents=True, exist_ok=True)
    for name, stem, template, spec, institute in PAPERS:
        result, rows = bank_rows(name)
        sections = [{"qtype": t, "count": c, "marks_each": m} for t, c, m in spec]
        questions = qb.paper_questions(qb.select_for_paper(rows, sections), sections)
        for fmt in ("pdf", "docx"):
            data = qb.render_paper(questions, title="Unit Test 1", fmt=fmt, template=template,
                                   include_answer_key=True, subject=result.get("subject") or "General",
                                   class_grade=result.get("class_grade") or "-", board=result.get("board") or "CBSE",
                                   institute_name=institute, duration="1 Hour", paper_date="2026-11-14")
            path = out / f"{stem}.{fmt}"
            path.write_bytes(data)
            print(f"{path}  {len(data)} bytes  {len(questions)} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
