"""Paper-wide duplicate gate (fix/p0-quality, task 6): the same question in two sections is shipped once."""
import json
from types import SimpleNamespace

from app.models.test_generator import ChapterSection, GeneratedQuestion, TestGenerationRequest
from app.services import test_generator_service as g

from audit_support import FakeGenModels, force_gemini, no_network  # noqa: F401

_n = iter(range(10_000))


def _q(text, section=None, fmt="short_answer", options=None, is_or=False, or_of=None, chapter="Light"):
    q = GeneratedQuestion(id=f"id{next(_n)}", text=text, options=options, correct_answer="an answer long enough",
                          explanation="an explanation long enough", marks=2, difficulty="medium", chapter=chapter,
                          format=fmt)
    q._section, q._is_or, q._or_of = section, is_or, or_of
    return q


MIRROR = "A concave mirror of focal length 15 cm forms an image of an object placed 30 cm from it. Find the image distance."


def test_same_question_in_section_b_and_c_is_shipped_once():
    b, c, other = _q(MIRROR, "B"), _q(MIRROR + " ", "C"), _q("State the laws of reflection of light.", "C")
    assert g._dedupe_paper([b, other, c]) == [b, other]


def test_same_sum_with_other_names_is_a_duplicate():
    a = _q("Ram and Shyam are partners sharing profits in the ratio 3:2. Calculate interest on capital at 6% p.a. on Rs 50,000.")
    b = _q("Amit and Vijay are partners sharing profits in the ratio 3:2. Calculate interest on capital at 6% p.a. on Rs 50,000.")
    assert g._dedupe_paper([a, b]) == [a]


def test_different_questions_survive():
    qs = [_q("Define Photosynthesis."), _q("Define Respiration."),                       # same exact_key "n n"
          _q(MIRROR), _q(MIRROR.replace("15", "10").replace("30", "25")),                # same sum, other numbers
          _q("Find the roots of the quadratic equation x^2 - 5x + 6 = 0."),
          _q("Find the roots of the quadratic equation x^2 - 7x + 12 = 0."),
          _q("Which of the following is a property of acids?"), _q("Which of the following is a property of bases?")]
    assert g._dedupe_paper(qs) == qs


def test_mcq_with_same_stem_but_other_options_is_kept():
    stem = "Which of the following statements is correct?"
    a = _q(stem, fmt="mcq", options=["A) Light is a wave", "B) Sound is light", "C) Heat is cold", "D) None"])
    b = _q(stem, fmt="mcq", options=["A) Current flows in a circuit", "B) Ohm is a unit of power", "C) Volt is energy", "D) All"])
    c = _q(stem, fmt="mcq", options=["A) Light is a wave", "B) Sound is light", "C) Heat is cold", "D) None"])
    assert g._dedupe_paper([a, b, c]) == [a, b]


def test_or_alternative_goes_with_its_dropped_main_question():
    first = _q(MIRROR, "B")
    dup_main = _q(MIRROR, "D")
    its_or = _q("Draw a ray diagram for a convex lens forming a real image.", "D", is_or=True, or_of=dup_main.id)
    free_or = _q("Explain the dispersion of white light by a glass prism.", "D", is_or=True, or_of=first.id)
    assert g._dedupe_paper([first, dup_main, its_or, free_or]) == [first, free_or]


def test_cbse_paper_drops_the_repeat_across_sections(monkeypatch, no_network, force_gemini):  # noqa: F811
    """End to end through generate_cbse_paper: sections B and C both come back with the same question."""
    from app.core.config import settings
    monkeypatch.setattr(settings, "GEMINI_GEN_MODEL", "primary")
    monkeypatch.setattr(settings, "GEMINI_FALLBACK_MODEL", "")

    def parse(raw, chapter, request, section_key=None):
        fmt = chapter.format.value
        opts = list(g.ASSERTION_REASON_OPTIONS) if fmt == "assertion_reason" else (
            [f"A) {section_key} one", "B) two", "C) three", "D) four"] if fmt == "mcq" else None)
        shared = [_q(MIRROR, section_key)] if section_key in ("B", "C") else []
        own = [_q(f"Section {section_key} {fmt} question {i}: explain idea {i * 7 + 1} of the chapter in your own words.",
                  section_key, fmt=fmt, options=opts) for i in range(chapter.quantity)]
        return shared + own

    monkeypatch.setattr(g, "_parse_batch", parse)
    monkeypatch.setattr(g, "_get_gemini_client",
                        lambda: SimpleNamespace(models=FakeGenModels(lambda n: json.dumps({"questions": []}))))
    ch = ChapterSection(chapter="Light", quantity=5, difficulty="medium", format="mcq", marks_per_question=1)
    req = TestGenerationRequest(exam_title="T", subject="Science", class_grade="10", teacher_id="u", chapters=[ch])
    paper = g.generate_cbse_paper(req, [{"chapter": "Light", "content": "Light reflects from a plane mirror. " * 20}])
    assert [q._section for q in paper if q.text == MIRROR] == ["B"]
    assert len({(q.text, q._section) for q in paper}) == len(paper)
