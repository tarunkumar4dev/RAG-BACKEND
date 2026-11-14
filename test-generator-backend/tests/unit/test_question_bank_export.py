"""Real files through the existing export system, from both fixture banks: the paper holds only bank questions,
the answer key holds only answers the bank has, and papers that are not from a bank render as before."""

import io
import re
import sys
import uuid

import pymupdf
import pytest
from docx import Document

from app.services import export_service as ex
from app.services import question_bank_service as qb
from tests.conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import qb_contest_flow_snapshot as snap  # noqa: E402

MIXED = [("mcq", 5, 1), ("true_false", 4, 1), ("short", 4, 3), ("long", 2, 5)]


def ws(s) -> str:
    return " ".join(str(s or "").split())


def printed(s) -> str:
    """How the exporter prints bank text: ₹ becomes "Rs." (existing behaviour); ligatures come back from the PDF."""
    return ws(s).replace("₹", "Rs.")


def bank(name):
    result = snap.run_case(name)
    rows, _ = qb.normalise_questions(qb.payload_from_import(result), result["title"])
    for r in rows:
        r["id"] = str(uuid.uuid4())
    return result, rows


def build(name, spec, fmt="pdf", template="modern", key=True, **kw):
    result, rows = bank(name)
    sections = [{"qtype": t, "count": c, "marks_each": m} for t, c, m in spec]
    picked = qb.select_for_paper(rows, sections, **kw)
    questions = qb.paper_questions(picked, sections)
    data = qb.render_paper(questions, title="Unit Test 1", fmt=fmt, template=template, include_answer_key=key,
                           subject=result["subject"], class_grade=result["class_grade"], board="CBSE",
                           institute_name="Sunrise Commerce Classes", paper_date="2026-11-14")
    return rows, [r for sec in picked for r in sec], questions, data


def pdf_pages(data) -> list:
    return [ws(p.get_text()) for p in pymupdf.open(stream=data, filetype="pdf")]


def docx_text(data) -> str:
    d = Document(io.BytesIO(data))
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:
        for row in t.rows:
            parts.extend(c.text for c in row.cells)
    return ws("\n".join(parts))


def split_at_key(pages):
    i = next(n for n, t in enumerate(pages) if "ANSWER KEY" in t)
    return " ".join(pages[:i]), " ".join(pages[i:])


# ── the paper is made only of bank questions ─────────────────────────────────

@pytest.mark.parametrize("name,spec", [("economics", [("mcq", 20, 1), ("mcq", 5, 2)]),
                                       ("business_studies", [("mcq", 5, 1), ("true_false", 4, 1), ("short", 3, 3),
                                                             ("long", 2, 5)])])
def test_pdf_holds_exactly_the_selected_bank_questions(name, spec):
    rows, chosen, questions, data = build(name, spec)
    assert data[:5] == b"%PDF-"
    paper, key = split_at_key(pdf_pages(data))
    for r in chosen:
        assert printed(r["question_text"]) in paper, r["question_text"][:50]
        for o in r["options"]:
            assert printed(o["text"]) in paper
    left_out = [r for r in rows if r not in chosen]
    assert left_out
    for r in left_out:
        assert printed(r["question_text"]) not in paper
    assert f"Q{len(chosen)}." in paper and f"Q{len(chosen) + 1}." not in paper
    for marker in ("Answer:", "Ans:", "Explanation", qb.NO_ANSWER_TEXT):   # the student paper shows no answers
        assert marker not in paper


def test_dollar_amounts_are_printed_as_in_the_bank():
    """The exporter reads "$…$" as maths and drops the signs; a bank paper must print the text literally."""
    rows, chosen, _, data = build("economics", [("mcq", 50, 1)], key=False)
    text = " ".join(pdf_pages(data))
    assert "$25 to $30" in text and "$30 to $36" in text and "Rs.80 per $ to Rs.84 per $" in text
    assert ex._process_latex("from $25 to $30") == "from 25 to 30"        # everyone else: as before
    with ex.literal_text():
        assert ex._process_latex("from $25 to $30 and H2O") == "from $25 to $30 and H2O"
    assert ex._process_latex("H2O", use_tags=True) == "H<sub>2</sub>O"    # switched off again after the block


def test_awkward_characters_are_printed_as_typed_and_never_break_the_file():
    _, rows = bank("business_studies")
    odd = "Is P&L < Balance Sheet > 5% & <b>bold</b> </para> ok? x_1 a^2 {curly}"
    rows[0]["question_text"] = odd
    rows[0]["options"][0]["text"] = "R&D <i>cost</i>"
    sections = [{"qtype": "mcq", "count": 3, "marks_each": 1}]
    questions = qb.paper_questions(qb.select_for_paper(rows, sections), sections)
    kw = dict(title="Tom & Jerry <Test> 100%", include_answer_key=True, subject="A&B", class_grade="12",
              board="CBSE", institute_name="Sharma&Sons <Classes>", teacher_name="A & B")
    for template in ("modern", "classic", "compact", "colorful"):
        text = " ".join(pdf_pages(qb.render_paper(questions, fmt="pdf", template=template, **kw)))
        assert odd in text and "R&D <i>cost</i>" in text, template
        assert "‹Test›" in text and ("Sharma & Sons ‹Classes›" in text or "SHARMA & SONS ‹CLASSES›" in text
                                     or template in ("classic", "compact")), template
        assert "&amp;" not in text and "&B;" not in text
        word = docx_text(qb.render_paper(questions, fmt="docx", template=template, **kw))
        assert odd in word and "Tom & Jerry <Test> 100%" in word, template
    # everyone else: the exporter's maths handling is as before
    assert ex._process_latex("x_1 a^2 {c}", use_tags=True) == "x<sub>1</sub> a<super>2</super> c"
    assert ex._latex_to_paragraph("<b>bold</b> a < b") == "<b>bold</b> a &lt; b"


# ── the answer key holds only answers that were found ────────────────────────

def test_answer_key_matches_the_bank_and_marks_missing_answers():
    rows, chosen, questions, data = build("business_studies", MIXED)
    paper, key = split_at_key(pdf_pages(data))
    by_pos = {r["position"]: r for r in rows}
    assert [by_pos[n]["answer"] for n in (1, 2, 3, 4, 5)] == ["B", "C", "C", "C", "C"]
    for n, (letter, text) in enumerate([("b", "A teacher teaching in a school for a salary"), ("c", "Employment"),
                                        ("c", "Wages or salaries"), ("c", "Tertiary industries"), ("c", "Business")], 1):
        assert f"Q{n}. ({letter}) {text}" in key
    assert "Q6. (b) False" in key and "Q8. (a) True" in key
    assert "Q10. An economic activity; production or procurement" in key
    # Q12, Q13 (short, no model answer in the PDF) and Q14, Q15 (long): said plainly, nothing made up
    assert key.count(qb.NO_ANSWER_TEXT) == 4 == sum(1 for q in questions if not q["answerFound"])
    for n in (12, 13, 14, 15):
        assert f"Q{n}. {qb.NO_ANSWER_TEXT}" in key


def test_mcq_without_a_key_entry_gets_no_answer_in_the_key():
    rows, chosen, questions, data = build("business_studies", [("mcq", 6, 1)], allow_needs_review=True)
    _, key = split_at_key(pdf_pages(data))
    assert questions[5]["correctAnswer"] == "" and f"Q6. {qb.NO_ANSWER_TEXT}" in key
    assert "Q6. (" not in key


def test_economics_answer_key_equals_the_pdf_key():
    result, _ = bank("economics")
    rows, chosen, questions, data = build("economics", [("mcq", 50, 1)])
    _, key = split_at_key(pdf_pages(data))
    for n, src in enumerate(result["questions"], 1):
        label = src["correct_answer"]
        option = next(o["text"] for o in src["options"] if o["label"] == label)
        assert f"Q{n}. ({label.lower()}) {printed(option)}" in key, n


def test_no_answer_key_unless_asked():
    _, _, _, data = build("business_studies", MIXED, key=False)
    assert "ANSWER KEY" not in " ".join(pdf_pages(data))
    _, _, _, data = build("business_studies", MIXED, fmt="docx", key=False)
    assert "ANSWER KEY" not in docx_text(data).upper()


# ── DOCX ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("template", ["modern", "colorful", "classic"])
def test_docx_holds_the_paper_then_the_answer_key(template):
    rows, chosen, questions, data = build("business_studies", MIXED, fmt="docx", template=template)
    assert data[:2] == b"PK"
    text = docx_text(data)
    for r in chosen:
        assert ws(r["question_text"]) in text
    assert "व्यवसाय और पेशे में कोई दो अंतर लिखिए।" in text            # Hindi is kept in Word files
    assert text.upper().count("ANSWER KEY") >= 1
    assert text.count(qb.NO_ANSWER_TEXT) == 4
    assert text.index(ws(chosen[0]["question_text"])) < text.upper().index("ANSWER KEY")


def test_hindi_is_detected_for_pdf():
    _, _, questions, _ = build("business_studies", MIXED, fmt="docx")
    assert qb.pdf_cannot_print(questions)
    assert not qb.pdf_cannot_print(questions[:5], "Unit Test — ₹ $ ½ é")
    assert qb.pdf_cannot_print(questions[:5], "इकाई परीक्षा")


# ── every template, both formats ─────────────────────────────────────────────

def test_every_template_renders():
    assert {"modern", "classic", "compact", "colorful"} <= qb.allowed_templates()
    for template in sorted(qb.allowed_templates()):
        for fmt, magic in (("pdf", b"%PDF-"), ("docx", b"PK")):
            spec = [("mcq", 5, 1), ("true_false", 4, 1), ("short", 3, 3), ("long", 2, 5)]
            _, chosen, _, data = build("business_studies", spec, fmt=fmt, template=template)
            assert data.startswith(magic), (template, fmt)
            text = " ".join(pdf_pages(data)) if fmt == "pdf" else docx_text(data)
            assert printed(chosen[0]["question_text"]) in text, (template, fmt)
            # the exporter's fixed CBSE wording must not leak into a bank paper
            for claim in ("20 MCQs", "5 Sections", "Assertion-Reason", "Very Short Answer", "calculator",
                          "CBSE Pattern", "Partnership Firms"):
                assert claim not in text, (template, fmt, claim)


def test_accountancy_bank_paper_keeps_the_chosen_template():
    result, rows = bank("business_studies")
    sections = [{"qtype": "mcq", "count": 4, "marks_each": 1}]
    questions = qb.paper_questions(qb.select_for_paper(rows, sections), sections)
    kw = dict(title="Unit Test", template="modern", include_answer_key=False, subject="Accountancy",
              class_grade="12", board="CBSE")
    for fmt in ("pdf", "docx"):
        data = qb.render_paper(questions, fmt=fmt, **kw)
        text = " ".join(pdf_pages(data)) if fmt == "pdf" else docx_text(data)
        assert "Part A" not in text and "SECTION A — Multiple Choice Questions — 1 mark each" in text


# ── papers that are not from a bank are untouched ────────────────────────────

def plain_questions(sectioned):
    return [{"id": f"q{i}", "text": f"Question {i}?", "options": ["A) one", "B) two"], "correctAnswer": "A",
             "explanation": "", "marks": 1, "format": "mcq", **({"section": "A"} if sectioned else {})}
            for i in range(1, 4)]


def test_section_order_for_other_papers_is_unchanged():
    assert ex._get_section_order(plain_questions(False)) == (None, None)
    order, meta = ex._get_section_order(plain_questions(True))
    assert order is ex.SECTION_ORDER and meta is ex.CBSE_SECTIONS_META
    assert ex._custom_section_meta(plain_questions(True)) is None
    acc = [{**q, "section": "A_1m"} for q in plain_questions(False)]
    assert ex._get_section_order(acc) == (ex.ACCOUNTANCY_SECTION_ORDER, ex.ACCOUNTANCY_SECTIONS_META)


def test_other_papers_keep_their_fixed_wording():
    kw = dict(exam_title="T", board="CBSE", class_grade="10", paper_date="2026-11-14")
    modern = " ".join(pdf_pages(ex.generate_pdf(questions=plain_questions(True), subject="Science",
                                                template="modern", **kw)))
    assert "divided into 5 Sections." in modern and "Use of calculator is not permitted." in modern
    assert "CBSE Pattern" in modern and "MCQs / Assertion-Reason of 1 mark each" in modern
    classic = " ".join(pdf_pages(ex.generate_pdf(questions=plain_questions(True), subject="Science",
                                                 template="classic", **kw)))
    assert "5 Sections" in classic and "20 MCQs" in classic
    acc = " ".join(pdf_pages(ex.generate_pdf(questions=plain_questions(False), subject="Accountancy",
                                             template="modern", **kw)))
    assert "Part A" in acc
