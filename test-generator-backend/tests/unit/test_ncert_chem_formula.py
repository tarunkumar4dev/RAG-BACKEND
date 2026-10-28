"""needs_formula_render: periodic-table-aware, not [A-Z][a-z]?\\d+. Run: cd test-generator-backend && python -m pytest tests/unit"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.ncert_chem_formula import has_formula, is_chemistry, needs_formula_render  # noqa: E402

CHEM = ("Chemistry", "Equilibrium")


@pytest.mark.parametrize("text, expected", [
    # the brief's cases
    ("H2SO4", True),
    ("CO2", True),
    ("A4 size paper", False),
    ("Q2. What is the pH of the solution?", False),
    ("Class XII", False),
    ("C6H6", True),
    ("Fe2O3", True),
    ("2-methylpropane", False),
    ("NH4+", True),
    ("CaCO3", True),
    # more of the same rule
    ("Balance: Fe + H2O → Fe3O4 + H2", True),
    ("2H+ + 2e- → H2", True),
    ("CuSO4·5H2O is blue", True),
    ("(SO4)2- ion", True),
    ("Ca(OH)2 is slaked lime", True),
    ("H2O(l) and NaCl(aq)", True),
    ("Fe^3+ ions", True),
    ("Na+ and Cl- ions", True),
    ("HCl and NaCl", False),          # nothing to subscript
    ("He said NO, IS IT SO?", False),
    ("Fig. 2 shows", False),
    ("Example 2 :", False),
    ("Refer to Chapter 4", False),
    ("#H2 tag", False),
    ("B5 and C10 sizes", False),
    ("P1 and S1 are points", False),
    ("NCERT CBSE Exercise 3.1", False),
    ("", False),
])
def test_has_formula(text, expected):
    assert has_formula(text) is expected


@pytest.mark.parametrize("subject, chapter, expected", [
    ("Chemistry", "Equilibrium", True),
    ("chemistry", None, True),
    ("Science", "Acids, Bases and Salts", True),
    ("Science", "Atoms and Molecules", True),
    ("Science", "Electricity", False),
    ("Mathematics", "Polynomials", False),
    ("Physics", "Units", False),
])
def test_is_chemistry(subject, chapter, expected):
    assert is_chemistry(subject, chapter) is expected


def test_needs_formula_render_gated_by_subject():
    assert needs_formula_render("H2SO4", *CHEM) is True
    assert needs_formula_render("H2SO4", "Science", "Electricity") is False
    assert needs_formula_render("x2 + y2", "Mathematics", "Polynomials") is False
