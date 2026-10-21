"""G1 on the messy fixture with recorded (mock) Gemini replies: the code around the model — scoring, page order,
rotation, positions, call count. Not a test of the model (that is G3, with live recordings)."""

import json

import pytest

from app.services.answer_checker.copy_check import grade_copy
from app.services.smart_assignment_service import build_rubric, usage_meter
from tests.conftest import FIXTURES

EXPECTED = json.loads((FIXTURES / "expected_evaluation.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def checked():
    with usage_meter() as rubric_meter:
        rubric = build_rubric(str(FIXTURES / "question_paper.pdf"), str(FIXTURES / "answer_key.pdf"))
    result, ordered, usage = grade_copy([str(FIXTURES / "answer_sheet_messy.pdf")], rubric)
    return rubric, rubric_meter, result, ordered, usage


def test_marks_match_expected(checked):
    _, _, result, _, _ = checked
    got = {q["q_no"]: q["marks_awarded"] for q in result["graded_questions"] if q["counted"]}
    for q in EXPECTED["questions"]:
        assert abs(got[str(q["q"])] - q["expected"]) <= EXPECTED["tolerance_per_question"], q["q"]
        assert got[str(q["q"])] <= q["max"]
    assert abs(result["total_marks_awarded"] - EXPECTED["expected_total"]) <= EXPECTED["tolerance_total"]
    assert result["total_marks_awarded"] <= EXPECTED["max_marks"]


def test_only_one_internal_choice_counts(checked):
    _, _, result, _, _ = checked
    q15 = [q for q in result["graded_questions"] if q.get("or_group") == "15"]
    assert sum(q["counted"] for q in q15) == 1


def test_pages_upright_and_in_student_order(checked):
    _, _, result, ordered, _ = checked
    assert [p.student_page_no for p in ordered] == ["1", "2", "3", "4", "5", "6", "7"]
    assert [p.source_index for p in ordered] == [0, 1, 3, 2, 4, 5, 6]
    assert [p.rotation for p in ordered] == [0, 0, 0, 180, 0, 0, 0]
    codes = {f["code"] for f in result["sheet_flags"]}
    assert {"PAGES_REORDERED", "PAGE_ROTATED"} <= codes


def test_positions_are_on_final_pages(checked):
    _, _, result, _, _ = checked
    pos = {q["q_no"]: (q["page_index"], q["y_start"]) for q in result["graded_questions"]}
    student_page = {str(q["q"]): q["found_on_student_pages"] for q in EXPECTED["questions"]}
    for q_no, pages in student_page.items():
        if pages:
            assert pos[q_no][0] + 1 == pages[0], q_no          # page index = student page - 1
    assert pos["6"] == (None, None)                             # not attempted, no invented position
    assert pos["7"] == (0, 493) and pos["14"] == (2, 187)       # sub-parts answered in reverse order


def test_gemini_calls_counted(checked):
    _, rubric_meter, _, _, usage = checked
    assert rubric_meter.calls == 1
    assert usage["gemini_calls"] == 1
    assert (usage["pages"], usage["pages_sent"]) == (7, 7)
