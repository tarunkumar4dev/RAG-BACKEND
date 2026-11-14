"""The existing "Contest from PDF" flow must not change because of the Question Bank feature.

Baseline = the parser's full result for both fixtures, written by scripts/qb_contest_flow_snapshot.py --write on
untouched origin/main (7cba4eb) before any Question Bank code existed. Any difference fails."""

import subprocess
import sys

import pytest

from tests.conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import qb_contest_flow_snapshot as snap  # noqa: E402

CONTEST_FILES = [
    "app/routers/contest_import_router.py",
    "app/services/contest_import_service.py",
    "app/services/contest_import_gemini.py",
    "app/schemas/contest_import.py",
    "app/services/contest_service.py",
    "app/api/v1/endpoints/contest.py",
    "app/schemas/contest.py",
]


@pytest.mark.parametrize("name", sorted(snap.CASES))
def test_contest_import_result_is_identical_to_the_baseline(name):
    pdf, recordings = snap.CASES[name]
    assert pdf.exists() and recordings.exists()
    baseline = (snap.BASELINE / f"{name}.json").read_text(encoding="utf-8")
    assert snap.canonical(snap.run_case(name)) == baseline


def test_contest_code_has_no_diff_against_main():
    """The feature adds files next to the contest import; it does not edit it."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if git("rev-parse", "--verify", "origin/main").returncode != 0:
        pytest.skip("origin/main is not available in this checkout")
    r = git("diff", "--stat", "origin/main", "--", *CONTEST_FILES)
    assert r.returncode == 0 and r.stdout.strip() == "", r.stdout


def test_contest_import_schema_is_unchanged():
    """The shape the frontend's review screen and POST /contests rely on."""
    from app.schemas.contest_import import ImportQuestion, ImportResult
    assert list(ImportQuestion.model_fields) == [
        "question_text", "question_type", "options", "correct_answer", "explanation", "marks", "difficulty",
        "chapter", "source_number", "answer_source", "usable"]
    assert list(ImportResult.model_fields) == [
        "title", "subject", "class_grade", "institute", "board", "duration_minutes", "total_marks",
        "answer_key_found", "answer_key_count", "questions", "warnings"]
