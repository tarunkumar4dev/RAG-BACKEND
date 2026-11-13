"""
Phase 1B tests for the SOLVE-FIRST numeric MCQ engine.

Key property under test: for EVERY family and MANY seeds, the built MCQ has
exactly one option equal to the solver's answer, four distinct options, and the
stem repeats every number the options/working rely on. The solver is the oracle;
these tests do not re-derive the arithmetic (that is test_accountancy_solver).
"""
import re
from fractions import Fraction

import pytest

from app.services import accountancy_scenarios as sc
from app.services import accountancy_solver as S

SEEDS = list(range(25))


def _amount_from_option(opt: str) -> Fraction:
    # "C) Rs. 2,70,000" -> Fraction(270000)
    m = re.search(r"Rs\.\s*([\d,]+)", opt)
    assert m, f"no amount in option {opt!r}"
    return Fraction(int(m.group(1).replace(",", "")))


@pytest.mark.parametrize("family", sc.ALL_FAMILIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_every_family_seed_builds_clean_mcq(family, seed):
    mcq = sc.build_numeric_mcq(family, seed=seed, difficulty="medium")

    # 4 options, all distinct
    assert len(mcq.options) == 4
    vals = [_amount_from_option(o) for o in mcq.options]
    assert len(set(vals)) == 4, (family, seed, mcq.options)

    # the correct_answer text is one of the options, and its letter is consistent
    assert mcq.correct_answer in mcq.options
    assert mcq.correct_answer.startswith(mcq.correct_letter + ")")

    # exactly one option equals the correct value
    correct_val = _amount_from_option(mcq.correct_answer)
    assert sum(1 for v in vals if v == correct_val) == 1


@pytest.mark.parametrize("family", sc.ALL_FAMILIES)
def test_correct_option_equals_solver_answer(family):
    """Rebuild the scenario deterministically and confirm the option marked
    correct equals the solver's own computed value (not a text match)."""
    import random
    # Reproduce build_numeric_mcq's first successful draft for seed=7.
    seed, variant = 7, 0
    gen = sc.FAMILY_GENERATORS[family]
    chosen = None
    for attempt in range(60):
        rng = random.Random(f"{family}:{seed}:{variant}:{attempt}")
        draft = gen(rng, "medium", variant)
        try:
            S.make_options(draft.correct, [v for _, v in draft.distractors])
        except ValueError:
            continue
        chosen = draft
        break
    assert chosen is not None, f"no clean draft for {family}"

    mcq = sc.build_numeric_mcq(family, seed=seed, difficulty="medium", variant=variant)
    assert _amount_from_option(mcq.correct_answer) == S.rupees(chosen.correct)


@pytest.mark.parametrize("family", sc.ALL_FAMILIES)
def test_stem_contains_its_numbers(family):
    """The deterministic stem must carry the data; a downstream LLM polish is
    only allowed to keep those numbers (see polish_stem_with_llm)."""
    mcq = sc.build_numeric_mcq(family, seed=3, difficulty="medium")
    # The stem should contain at least two grouped numbers (amounts/rates).
    nums = re.findall(r"\d[\d,]*", mcq.text)
    assert len(nums) >= 2, (family, mcq.text)


def test_generate_numeric_mcqs_rotates_and_is_deterministic():
    a = sc.generate_numeric_mcqs(15, seed=42, difficulty="hard")
    b = sc.generate_numeric_mcqs(15, seed=42, difficulty="hard")
    assert len(a) == 15
    assert [q["text"] for q in a] == [q["text"] for q in b]   # deterministic
    # Each is a well-formed MCQ dict.
    for q in a:
        assert q["format"] == "mcq"
        assert len(q["options"]) == 4
        assert q["correct_answer"] in q["options"]
        assert q["validation_status"] == "verified"


def test_no_answer_position_monoculture():
    """Across a 20-question paper the correct letter should not be a single value
    (the D4 defect). Not a strict balance yet — that is Phase 1E — but prove it
    varies."""
    qs = sc.generate_numeric_mcqs(20, seed=1, difficulty="medium")
    letters = []
    for q in qs:
        letters.append(q["correct_answer"].split(")")[0])
    assert len(set(letters)) >= 2, letters


def test_detect_numeric_family():
    assert sc.detect_numeric_family("Fundamentals of Partnership", "Interest on Capital") == "interest_on_capital"
    assert sc.detect_numeric_family("Fundamentals of Partnership", "Interest on Drawings") == "interest_on_drawings"
    assert sc.detect_numeric_family("Accounting Ratios", None) is None
    assert sc.is_fundamentals_chapter("Fundamentals of Partnership") is True
    assert sc.is_fundamentals_chapter("Dissolution of a Firm") is False


# ── LLM prose polish keeps numbers code-owned ────────────────────────────

def test_polish_keeps_template_when_numbers_drift():
    stem = "Arun introduced Rs. 1,00,000 on 1 October at 10% p.a."
    # LLM that mangles a number -> must be rejected, template kept.
    bad = lambda _p: "Arun introduced Rs. 2,00,000 on 1 October at 10% p.a."
    assert sc.polish_stem_with_llm(stem, bad) == stem


def test_polish_accepts_rewrite_that_preserves_numbers():
    stem = "Arun introduced Rs. 1,00,000 on 1 October at 10% p.a."
    good = lambda _p: "On 1 October, Arun brought in additional capital of Rs. 1,00,000 at 10% p.a."
    out = sc.polish_stem_with_llm(stem, good)
    assert out != stem
    assert "1,00,000" in out


def test_polish_falls_back_on_llm_error():
    stem = "Bela's drawings were Rs. 40,000 at 5%."
    boom = lambda _p: (_ for _ in ()).throw(RuntimeError("llm down"))
    assert sc.polish_stem_with_llm(stem, boom) == stem
