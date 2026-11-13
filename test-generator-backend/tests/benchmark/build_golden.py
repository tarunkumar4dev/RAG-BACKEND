"""Generate one golden paper + fixture per numeric family (Phase 1B).

Each golden paper is a seeded SOLVE-FIRST output saved for regression of
structure (exactly-one-correct, dedup, count/marks, PDF). Numeric-answer
verification for these is left to the SME (key_source=pending_sme) per the
project ground rules — the solver's own correctness is covered independently by
tests/unit/test_accountancy_solver.py.

Run:  python tests/benchmark/build_golden.py
It writes tests/benchmark/fixtures/<NN>_<family>.json and
tests/benchmark/fixtures/<family>_golden.json. Deterministic.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("GEMINI_API_KEY", "mock")
os.environ.setdefault("ENVIRONMENT", "development")

from app.services import accountancy_scenarios as sc   # noqa: E402
from app.services import blueprint as bp               # noqa: E402

FIX = Path(__file__).parent / "fixtures"
TOPIC_LABEL = {
    "interest_on_capital": "Interest on Capital",
    "interest_on_drawings": "Interest on Drawings",
    "interest_on_drawings_flat": "Interest on Drawings",
    "interest_on_drawings_irregular": "Interest on Drawings",
    "pl_appropriation": "P&L Appropriation Account",
    "profit_share": "Distribution of Profit",
    "commission_before": "Manager's Commission",
    "commission_after": "Manager's Commission",
    "guarantee_to_partner": "Guarantee of Profit",
    "guarantee_to_firm": "Guarantee of Profit",
    "opening_capital_plain": "Fundamentals of Partnership",
    "opening_capital_ioc_on_opening": "Interest on Capital",
    "ioc_when_inadequate": "Interest on Capital",
    "partners_loan_interest": "Partner's Loan",
    "past_adjustment": "Past Adjustments",
}

N = 10
START_INDEX = 10   # family fixtures numbered 10.. (01-04 are the author-computed minis)


def build():
    for pos, family in enumerate(sc.ALL_FAMILIES):
        qs = sc.generate_numeric_mcqs(N, seed=1000 + pos, difficulty="hard",
                                      families=[family])
        for i, q in enumerate(qs, 1):
            q["id"] = f"{family}_{i}"
        paper = {
            "subject": "Accountancy",
            "classGrade": "12",
            "chapters": ["Fundamentals of Partnership"],
            "totalMarks": sum(q["marks"] for q in qs),
            "questions": qs,
        }
        golden_name = f"{family}_golden.json"
        (FIX / golden_name).write_text(json.dumps(paper, indent=2, default=str),
                                       encoding="utf-8")

        fixture = {
            "id": f"acc12_{family}",
            "description": f"Solve-first golden paper: {N} numeric MCQs, family={family}.",
            "request": {
                "examTitle": f"{TOPIC_LABEL[family]} — {N} MCQ",
                "board": "CBSE", "classGrade": "12", "subject": "Accountancy",
                "cbsePattern": False,
                "simpleData": [{
                    "topic": "Fundamentals of Partnership",
                    "subtopic": TOPIC_LABEL[family],
                    "quantity": N, "difficulty": "hard", "format": "mcq", "marks": 1,
                }],
            },
            "expected_total_questions": N,
            "expected_total_marks": N,
            "expected_numeric_items": [{
                "qid": f"{family}_1",
                "description": f"{family} numeric answer — SME to hand-verify.",
                "expected_answer": 0.0, "tolerance": 1.0, "unit": "INR",
                "key_source": "pending_sme",
            }],
            "golden_paper_file": golden_name,
            "key_source": "solver-structural",
            "notes": "Numeric answers follow the solver (unit-tested independently); "
                     "SME sign-off converts expected_numeric_items to sme-verified.",
        }
        (FIX / f"{START_INDEX + pos:02d}_acc12_{family}.json").write_text(
            json.dumps(fixture, indent=2), encoding="utf-8")
        print(f"wrote {golden_name} + fixture ({family})")


def build_rotation():
    """The realistic 50-MCQ Fundamentals paper, built from the weighted blueprint."""
    seq = bp.plan_sequence(bp.get_blueprint("Fundamentals of Partnership"), 50)
    qs = sc.generate_numeric_mcqs(50, seed=2025, difficulty="hard", family_sequence=seq)
    for i, q in enumerate(qs, 1):
        q["id"] = f"rot_{i}"
    paper = {"subject": "Accountancy", "classGrade": "12",
             "chapters": ["Fundamentals of Partnership"],
             "totalMarks": sum(q["marks"] for q in qs), "questions": qs}
    (FIX / "fundamentals_rotation_golden.json").write_text(
        json.dumps(paper, indent=2, default=str), encoding="utf-8")
    print("wrote fundamentals_rotation_golden.json (blueprint-planned)")


if __name__ == "__main__":
    build()
    build_rotation()
