"""
app/services/ncert_chem_formula.py — does a chemistry question contain a formula that needs rendering
(H2SO4 → H₂SO₄, NH4+ → NH₄⁺)? Sets quality_flags.needs_formula_render; rendering itself is Slice B
(docs/chemistry-formula-edge-cases.md).

A word is a formula only if ALL of it parses as element symbols (the list below), "(…)" groups and counts —
"A4", "Q2", "Class XII", "NCERT" do not — and it has something to render: a count ≥ 2 or a trailing charge.
"HCl" / "NaCl" parse but have nothing to render, so they are not flagged. A count of exactly 1 is never written
in a formula ("P1", "S1" are labels). Polyatomic groups (SO4, NO3, CO3, PO4, OH, NH4, CH3, C2H5) are element
sequences, so they parse without a separate list. Only Chemistry, and the chemistry chapters of Science.
"""
from __future__ import annotations

import re

ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Br Ag I Ba Pb Hg Au Pt Zr"
    .split()
)
# parse as formulas but are paper sizes / labels
DENYLIST = frozenset({"A4", "B4", "B5", "C10"})
# the word before a label: "Q 2", "Fig 3", "Example 2", "Class XII", "Chapter 4", "No. 5"
_LABEL_WORDS = frozenset({"q", "fig", "figure", "example", "class", "chapter", "no"})
SCIENCE_CHEMISTRY_CHAPTERS = frozenset(c.lower() for c in (
    # Class 10
    "Acids, Bases and Salts", "Chemical Reactions and Equations", "Metals and Non-metals",
    "Carbon and its Compounds", "Periodic Classification of Elements",
    # Class 9 (old and rationalised books)
    "Matter In Our Surroundings", "Is Matter Around Us Pure", "Atoms and Molecules", "Structure Of The Atom",
    "Atomic Foundations of Matter", "Journey Inside the Atom", "Exploring Mixtures and their Separation",
))

_SPLIT = re.compile(r"[\s,;:/=→⇌⟶]+")
_EDGE = "\"'“”‘’[]{}"
_TRAILING_PUNCT = re.compile(r"[.!?]+$")
_COEFFICIENT = re.compile(r"^[0-9]+")
_CHARGE = re.compile(r"(?:\^\{?[0-9]*[+\-−]+\}?|[+\-−]+)$")  # NH4+, Fe^3+, SO4^{2-}
_STATE = re.compile(r"\((?:aq|s|l|g)\)$")  # H2O(l), NaCl(aq)
_COUNT = re.compile(r"[0-9]+")


def _parse(body: str) -> tuple[int, bool] | None:
    """(units, has_count) if all of body is elements / (groups) with counts, else None."""
    i, units, has_count = 0, 0, False
    while i < len(body):
        if body[i] == "(":
            j = body.find(")", i)
            inner = _parse(body[i + 1:j]) if j > i + 1 else None
            if inner is None:
                return None
            units, has_count, i = units + inner[0], has_count or inner[1], j + 1
        elif body[i:i + 2] in ELEMENTS and body[i + 1:i + 2].islower():
            units, i = units + 1, i + 2
        elif body[i] in ELEMENTS:
            units, i = units + 1, i + 1
        else:
            return None
        m = _COUNT.match(body, i)
        if m:
            if m.group() == "1" or m.group().startswith("0"):
                return None
            has_count, i = True, m.end()
    return (units, has_count) if units else None


def _is_formula(word: str) -> bool:
    if word.count("(") != word.count(")"):
        word = word.strip("()")
    if word in DENYLIST:
        return False
    parts = word.split("·")  # hydrate: CuSO4·5H2O
    renderable = False
    for part in parts:
        part = _STATE.sub("", _COEFFICIENT.sub("", part))
        charge = _CHARGE.search(part)
        if charge:
            part = part[:charge.start()]
        parsed = _parse(part)
        if parsed is None:
            return False
        renderable = renderable or parsed[1] or bool(charge)
    return renderable


def has_formula(text: str | None) -> bool:
    """True when the text contains at least one chemical formula with a subscript or charge to render."""
    prev = ""
    for raw in _SPLIT.split(text or ""):
        word = _TRAILING_PUNCT.sub("", raw.strip(_EDGE))
        if word and not raw.startswith("#") and prev not in _LABEL_WORDS and _is_formula(word):
            return True
        prev = raw.strip(_EDGE + ".").lower()
    return False


def is_chemistry(subject: str | None, chapter: str | None) -> bool:
    s = (subject or "").strip().lower()
    return s == "chemistry" or (s == "science" and (chapter or "").strip().lower() in SCIENCE_CHEMISTRY_CHAPTERS)


def needs_formula_render(text: str | None, subject: str | None, chapter: str | None) -> bool:
    return is_chemistry(subject, chapter) and has_formula(text)
