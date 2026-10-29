"""Read-time dedupe of unmerged NCERT copies. Run: cd test-generator-backend && python -m pytest tests/unit"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.ncert_dedupe import dedupe_ncert_rows  # noqa: E402


def row(id, number, text, chapter="Accounting Ratios", section="5.6.1", **extra):
    return {"id": id, "chapter": chapter, "section": section, "question_number": number,
            "question_text": text, **extra}


GOOD = "Calculate Current Ratio from the following information: | Inventories | 50,000 |"
GOOD_NO_COLON = "Calculate Current Ratio from the following information | Inventories | 50,000 |"
BAD = "Calculate Current Ratio from the following information: | Inventories | 50 | Total | 000 |"


def ids(rows):
    return [r["id"] for r in rows]


def test_illustration_copies_keep_the_majority_extraction():
    rows = [row(1, "Illustration 1", BAD), row(2, "Illustration 1", BAD), row(3, "Illustration 1", GOOD),
            row(4, "Illustration 1", GOOD), row(5, "Illustration 1", GOOD_NO_COLON),
            row(6, "Illustration 1", GOOD_NO_COLON)]
    assert ids(dedupe_ncert_rows(rows)) == [3]


def test_copies_in_different_sections_are_one_question():
    rows = [row(10, "Illustration 5", "Total assets", section="5.6.1"),
            row(11, "Illustration 5", "Total assets", section="5.6.2")]
    assert ids(dedupe_ncert_rows(rows)) == [10]


def test_same_number_other_chapter_is_kept():
    rows = [row(1, "Illustration 1", GOOD), row(2, "Illustration 1", GOOD, chapter="Cash Flow Statement")]
    assert ids(dedupe_ncert_rows(rows)) == [1, 2]


def test_different_illustrations_are_kept():
    rows = [row(1, "Illustration 1", GOOD), row(2, "Illustration 2", GOOD), row(3, "Example 1", GOOD)]
    assert ids(dedupe_ncert_rows(rows)) == [1, 2, 3]


def test_unnumbered_rows_dedupe_on_text_and_options():
    rows = [row(1, "1", "Find x.", options=["(A) 1", "(B) 2"]), row(2, "1", "find  X", options=["A) 1", "B) 2"]),
            row(3, "1", "Find x.", options=["(A) 3", "(B) 4"]), row(4, "2", "Find y.")]
    assert ids(dedupe_ncert_rows(rows)) == [1, 3, 4]


def test_quality_rank_wins_over_votes():
    rows = [row(1, "Example 1", BAD, quality_rank=1), row(2, "Example 1", BAD, quality_rank=1),
            row(3, "Example 1", GOOD, quality_rank=0)]
    assert ids(dedupe_ncert_rows(rows)) == [3]


def test_order_is_kept():
    rows = [row(9, "Illustration 2", "b"), row(3, "Illustration 1", "a"), row(5, "Illustration 2", "b")]
    assert ids(dedupe_ncert_rows(rows)) == [3, 5]  # input order; tie → lowest id


def test_garbled_copy_loses_to_a_clean_one():
    garbled = "Prove that sin cos 1 1, sin cos 1 sec tanθ－ θ＋＝ θ＋"
    rows = [row(1, "Example 12", garbled), row(2, "Example 12", garbled), row(3, "Example 12", "Prove that (sin θ – cos θ + 1)"),
            row(4, "Example 11", "(1 sin ) (1 sin )"), row(5, "Example 11", "(1 + sin θ)(1 - sin θ)")]
    assert ids(dedupe_ncert_rows(rows)) == [3, 5]


def test_example_sub_parts():
    rows = [row(1, "Example 1 (i)", "a"), row(2, "Example 1(i)", "a."), row(3, "Example 1 (ii)", "b")]
    assert ids(dedupe_ncert_rows(rows)) == [1, 3]
