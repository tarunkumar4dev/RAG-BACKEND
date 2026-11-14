"""question_bank_service: normalisation (never an invented answer), duplicates, true/false and year rules, paper
selection and its "bank has too few" messages. Pure functions — no Gemini, no database."""

import random
import sys

import pytest

from app.services import question_bank_service as qb
from tests.conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import qb_contest_flow_snapshot as snap  # noqa: E402


def q(text="What is money?", qtype="MCQ", options=("Cash", "A medium of exchange", "Gold", "Debt"), answer="B",
      source="key", marks=1, usable=True):
    return {"question_text": text, "question_type": qtype,
            "options": [{"label": "abcd"[i], "text": o} for i, o in enumerate(options)],
            "correct_answer": answer, "answer_source": source, "marks": marks, "usable": usable}


# ── fixtures end to end (recorded parser replies) ────────────────────────────

def test_economics_fixture_into_a_bank():
    result = snap.run_case("economics")
    rows, s = qb.normalise_questions(qb.payload_from_import(result), result["title"])
    assert (s["saved"], s["duplicates_removed"], s["needs_review"], s["answers_matched"]) == (50, 0, 0, 50)
    assert s["per_type"] == {"mcq": 50, "true_false": 0, "short": 0, "long": 0}
    assert [r["position"] for r in rows] == list(range(1, 51))
    # text, options and answers are the parser's, untouched
    for row, src in zip(rows, result["questions"]):
        assert row["question_text"] == src["question_text"]
        assert [o["text"] for o in row["options"]] == [o["text"] for o in src["options"]]
        assert (row["answer"], row["answer_source"]) == (src["correct_answer"], "key")
    assert "₹" in rows[42]["question_text"] and "$" in rows[41]["options"][0]["text"]


def test_business_studies_fixture_into_a_bank():
    result = snap.run_case("business_studies")
    rows, s = qb.normalise_questions(qb.payload_from_import(result), result["title"])
    assert s["per_type"] == {"mcq": 6, "true_false": 4, "short": 4, "long": 2}
    assert (s["saved"], s["needs_review"], s["answers_matched"], s["ai_answers_dropped"]) == (16, 1, 11, 0)
    by_pos = {r["position"]: r for r in rows}
    assert by_pos[6]["answer"] is None and by_pos[6]["answer_source"] is None and by_pos[6]["needs_review"]
    assert [by_pos[n]["answer"] for n in (7, 8, 9, 10)] == ["False", "False", "True", "True"]
    assert by_pos[7]["qtype"] == "true_false" and by_pos[7]["options"] == []
    # written questions without a model answer: answer stays null, and they are still usable in a paper
    assert all(by_pos[n]["answer"] is None and not by_pos[n]["needs_review"] for n in (13, 14, 15, 16))
    assert (by_pos[1]["exam_year"], by_pos[1]["source_tag"]) == (2023, "CBSE 2023")
    assert (by_pos[3]["exam_year"], by_pos[12]["exam_year"], by_pos[2]["exam_year"]) == (2022, 2021, None)
    assert by_pos[14]["question_text"] == "व्यवसाय और पेशे में कोई दो अंतर लिखिए।"       # Hindi passes through


# ── answers are never invented ───────────────────────────────────────────────

def test_ai_guess_is_dropped_and_flagged():
    rows, s = qb.normalise_questions([q(source="ai")])
    assert rows[0]["answer"] is None and rows[0]["answer_source"] is None and rows[0]["needs_review"]
    assert s["ai_answers_dropped"] == 1 and s["answers_matched"] == 0
    assert any("guessed by AI" in w for w in qb.stats_warnings(s))


def test_teacher_confirmed_answer_is_kept_as_teacher():
    rows, _ = qb.normalise_questions([q(source="teacher", answer="(c)")])
    assert (rows[0]["answer"], rows[0]["answer_source"], rows[0]["needs_review"]) == ("C", "teacher", False)


def test_missing_answer_stays_null():
    rows, s = qb.normalise_questions([q(answer=None, source="missing"), q(text="Explain money.", qtype="Long",
                                                                          options=(), answer="", source="missing")])
    assert [r["answer"] for r in rows] == [None, None]
    assert [r["needs_review"] for r in rows] == [True, False]      # MCQ without answer: review; long answer: fine
    assert s["answers_matched"] == 0


@pytest.mark.parametrize("answer", ["E", "Z", "Silver", "b and c"])
def test_mcq_answer_that_is_not_an_option_is_dropped(answer):
    rows, s = qb.normalise_questions([q(answer=answer)])
    assert rows[0]["answer"] is None and rows[0]["needs_review"] and s["unresolved_answers"] == 1


@pytest.mark.parametrize("answer,label", [("B", "B"), ("(b)", "B"), ("b.", "B"), ("B) A medium of exchange", "B"),
                                          ("a medium  of EXCHANGE", "B"), ("Cash", "A")])
def test_mcq_answer_forms(answer, label):
    rows, _ = qb.normalise_questions([q(answer=answer)])
    assert rows[0]["answer"] == label


def test_option_text_starting_with_a_letter_is_not_read_as_a_label():
    rows, _ = qb.normalise_questions([q(options=("A bank", "B and C only", "Cash", "Debt"), answer="A bank")])
    assert rows[0]["answer"] == "A"


def test_unusable_question_is_flagged_even_with_an_answer():
    rows, _ = qb.normalise_questions([q(usable=False)])
    assert rows[0]["needs_review"] and rows[0]["answer"] == "B"


def test_mcq_with_one_option_is_flagged():
    rows, _ = qb.normalise_questions([q(options=("Only one",), answer="A")])
    assert rows[0]["qtype"] == "mcq" and rows[0]["needs_review"]


# ── normalise: whitespace, duplicates, types, years ──────────────────────────

def test_whitespace_is_collapsed_and_duplicates_dropped():
    rows, s = qb.normalise_questions([
        q(text="  What   is\tmoney?  "),
        q(text="what is MONEY?"),                                             # same stem, same options
        q(text="What is money?", options=("Coins", "Notes", "Both", "None")),  # same stem, other options: different
        q(text="   ", answer="A"),
    ])
    assert [r["question_text"] for r in rows] == ["What is money?", "What is money?"]
    assert (s["received"], s["saved"], s["duplicates_removed"], s["skipped_empty"]) == (4, 2, 1, 1)
    assert [r["position"] for r in rows] == [1, 2] and len({r["content_hash"] for r in rows}) == 2


def test_line_breaks_inside_a_question_are_kept():
    text = "Assertion (A): Demand rises.\nReason (R):   Price falls."
    rows, _ = qb.normalise_questions([q(text=text)])
    assert rows[0]["question_text"] == "Assertion (A): Demand rises.\nReason (R): Price falls."


@pytest.mark.parametrize("item,expected", [
    (q(options=("True", "False"), answer="A"), "true_false"),
    (q(qtype="Short", options=(), answer="False"), "true_false"),
    (q(text="State true or false: money is a store of value.", qtype="Short", options=(), answer=None,
       source="missing"), "true_false"),
    (q(qtype="Short", options=(), answer="A medium of exchange"), "short"),
    (q(qtype="Long", options=(), answer=None, source="missing"), "long"),
    (q(qtype="Short"), "mcq"),                                  # it has options, whatever the parser called it
    (q(qtype="", options=(), answer=None, source="missing"), "short"),
])
def test_qtype_rules(item, expected):
    rows, _ = qb.normalise_questions([item])
    assert rows[0]["qtype"] == expected


def test_true_false_answers():
    rows, _ = qb.normalise_questions([
        q(text="S1", options=("True", "False"), answer="False"),
        q(text="S2", qtype="Short", options=(), answer="true."),
        q(text="S3 true or false?", qtype="Short", options=(), answer="maybe"),
    ])
    assert [(r["answer"], r["needs_review"]) for r in rows] == [("B", False), ("True", False), (None, True)]


@pytest.mark.parametrize("text,year,tag", [
    ("Define GDP. [CBSE 2023]", 2023, "CBSE 2023"),
    ("Define GDP. (CBSE 2019, Delhi)", 2019, "CBSE 2019, Delhi"),
    ("Define GDP. (2021)", 2021, "2021"),
    ("In 2023 exports rose. Why?", None, None),                 # a year in the sentence is not a tag
    ("Sales were (₹ 2024 crore). Explain.", None, None),         # an amount, not a year
    ("Compare the data (Table 2020). Explain.", 2020, "Table 2020"),   # known limit: any short bracket with a year
    ("Define GDP. [CBSE 3023]", None, None),
])
def test_year_tag(text, year, tag):
    assert qb.year_tag(text) == (year, tag)


def test_paper_title_gives_the_year_when_the_question_has_none():
    rows, _ = qb.normalise_questions([q(), q(text="Other [ICSE 2020]")], "CBSE 2023 Sample Paper — Economics")
    assert [(r["exam_year"], r["source_tag"]) for r in rows] == [
        (2023, "CBSE 2023 Sample Paper — Economics"), (2020, "ICSE 2020")]


def test_safe_filename_and_download_name():
    assert qb.safe_filename("..\\..\\secret/paper<1>.pdf") == "paper1.pdf"
    assert qb.safe_filename("\x00\n") == "question-paper"
    assert qb.download_name("../Unit Test: 1 (Eco)") == "Unit_Test__1__Eco"
    assert qb.download_name("अर्थशास्त्र") == "question_paper"


# ── selection ────────────────────────────────────────────────────────────────

def bank(n_mcq=8, n_long=3):
    rows = [{"id": f"m{i}", "position": i, "qtype": "mcq", "question_text": f"MCQ {i}",
             "options": [{"label": "A", "text": "x"}, {"label": "B", "text": "y"}], "answer": "A",
             "needs_review": False, "excluded": False, "exam_year": 2023 if i <= 3 else None}
            for i in range(1, n_mcq + 1)]
    rows += [{"id": f"l{i}", "position": 100 + i, "qtype": "long", "question_text": f"Long {i}", "options": [],
              "answer": None, "needs_review": False, "excluded": False, "exam_year": None}
             for i in range(1, n_long + 1)]
    return rows


def sec(qtype, count, marks=1):
    return {"qtype": qtype, "count": count, "marks_each": marks}


def test_select_in_bank_order():
    picked = qb.select_for_paper(bank(), [sec("mcq", 3), sec("long", 2, 5)])
    assert [[r["id"] for r in s] for s in picked] == [["m1", "m2", "m3"], ["l1", "l2"]]


def test_too_few_questions_message():
    with pytest.raises(qb.PaperError) as e:
        qb.select_for_paper(bank(n_mcq=6), [sec("mcq", 10), sec("long", 2, 5), sec("true_false", 1)])
    assert e.value.shortfalls == ["Section A needs 10 MCQs, bank has 6.",
                                  "Section C needs 1 True/False question, bank has 0."]


def test_needs_review_and_excluded_are_skipped():
    rows = bank(n_mcq=5)
    rows[0]["needs_review"] = True
    rows[1]["excluded"] = True
    with pytest.raises(qb.PaperError) as e:
        qb.select_for_paper(rows, [sec("mcq", 4)])
    assert str(e.value) == "Section A needs 4 MCQs, bank has 3 (1 more needs review)."
    picked = qb.select_for_paper(rows, [sec("mcq", 4)], allow_needs_review=True)
    assert [r["id"] for r in picked[0]] == ["m1", "m3", "m4", "m5"]          # never the excluded one


def test_year_filter():
    picked = qb.select_for_paper(bank(), [sec("mcq", 3)], year_filter=2023)
    assert [r["id"] for r in picked[0]] == ["m1", "m2", "m3"]
    with pytest.raises(qb.PaperError) as e:
        qb.select_for_paper(bank(), [sec("mcq", 4)], year_filter=2023)
    assert str(e.value) == "Section A needs 4 MCQs, bank has 3 from 2023."


def test_two_sections_of_one_type_never_share_a_question():
    picked = qb.select_for_paper(bank(n_mcq=8), [sec("mcq", 5), sec("mcq", 3, 2)], shuffle=True,
                                 rng=random.Random(1))
    ids = [r["id"] for s in picked for r in s]
    assert len(ids) == len(set(ids)) == 8
    with pytest.raises(qb.PaperError) as e:
        qb.select_for_paper(bank(n_mcq=8), [sec("mcq", 5), sec("mcq", 4)])
    assert str(e.value) == "Section B needs 4 MCQs, bank has 3 left after the earlier sections."


def test_shuffle_only_picks_from_the_bank():
    rows = bank(n_mcq=30)
    for seed in range(20):
        picked = qb.select_for_paper(rows, [sec("mcq", 10)], shuffle=True, rng=random.Random(seed))
        assert len({r["id"] for r in picked[0]}) == 10 and all(r in rows for r in picked[0])


def test_paper_questions_copy_the_bank_and_never_add_an_answer():
    rows = bank(n_mcq=2, n_long=1) + [
        {"id": "t1", "position": 200, "qtype": "true_false", "question_text": "Money is an asset.", "options": [],
         "answer": "False", "needs_review": False, "excluded": False},
        {"id": "t2", "position": 201, "qtype": "true_false", "question_text": "No key.", "options": [],
         "answer": None, "needs_review": True, "excluded": False}]
    sections = [sec("mcq", 2, 2), sec("true_false", 2, 1), sec("long", 1, 5)]
    out = qb.paper_questions(qb.select_for_paper(rows, sections, allow_needs_review=True), sections)
    assert [(p["bank_question_id"], p["section"], p["marks"], p["format"]) for p in out] == [
        ("m1", "A", 2, "mcq"), ("m2", "A", 2, "mcq"), ("t1", "B", 1, "mcq"), ("t2", "B", 1, "mcq"),
        ("l1", "C", 5, "long_answer")]
    assert out[0]["options"] == ["A) x", "B) y"] and out[0]["correctAnswer"] == "A"
    assert out[2]["options"] == ["A) True", "B) False"] and out[2]["correctAnswer"] == "B"
    assert (out[3]["correctAnswer"], out[3]["answerFound"]) == ("", False)
    assert (out[4]["correctAnswer"], out[4]["answerFound"]) == ("", False)
    assert out[0]["section_meta"]["subtitle"] == "(Multiple Choice Questions — 2 marks each)"
    assert {p["text"] for p in out} <= {r["question_text"] for r in rows}
