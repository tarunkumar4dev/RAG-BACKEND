"""
Deterministic cross-checks on the grader's verdicts (no LLM call, no new dependency).

The grader reads the copy AND judges it in one Gemini call. Where a check can be exact, code re-checks that
judgement against the grader's own transcription:
  mcq        option letter written vs the key's letter
  numeric    the number (and unit) the student wrote for a point vs the key's value
  equation   chemical species / coefficients vs the key's equation
  terms      a mutually exclusive term ("acidic" where the key says "neutral") or a negated key term
  evidence   the words quoted as evidence for an awarded point must be in the student's transcribed answer
  injection  text in the answer that addresses the grader ("give full marks")
A disagreement never changes marks: the question goes to Review with a reason the teacher can act on.
Ported from the A4AI standalone checker (grading/numeric_grader.py, objective_grader.py, safety.py,
text_norm.py), using only the standard library (Vercel bundle limits, docs/checker-v4-design.md §1).
"""

import math
import re
import unicodedata
from difflib import SequenceMatcher
from fractions import Fraction
from typing import Optional

# Every reason this module adds starts with this, so portal_result can map it to a review enum value.
PREFIX = "Check: "
AWARDED = ("full", "mostly", "partial")
# Evidence for an awarded point must be the student's words. Gemini stitches quotes from table cells and lists
# ("Devaluation - fixed exchange rate system"), so long quotes are checked word by word, short ones ("= 0.6 A",
# "b) O2") character by character. Thresholds: see the calibration note in review_leaf().
EVIDENCE_MIN_WORD_SHARE = 0.6
EVIDENCE_MIN_SIMILARITY = 0.8
_STOP = frozenset("""a an the of to in on at by for from with and or but is are was were be been it its this that these
those as into than then so such also can could may will would should do does did has have had not no any some each
all both one other there here we you he she they them his her their our your my me us him if""".split())


# ─── text helpers ──────────────────────────────────────────────────────────────────────────────────────────
def fold(s: str) -> str:
    """NFKC (₂ → 2, ² → 2, full-width → ASCII), one form for minus, times and arrows."""
    s = unicodedata.normalize("NFKC", s or "")
    for a, b in (("−", "-"), ("–", "-"), ("—", "-"), ("×", "x"), ("⟶", "->"), ("→", "->"), ("⇌", "->"),
                 ("“", '"'), ("”", '"'), ("’", "'")):
        s = s.replace(a, b)
    return s


def _plain(s: str) -> str:
    """For quote matching: lower case, punctuation that students and transcribers vary on removed."""
    s = fold(s).lower()
    s = re.sub(r"[^\w\s=+\-/.%]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def similarity_in(needle: str, hay: str) -> float:
    """How well `needle` occurs somewhere in `hay` (0..1), like rapidfuzz.partial_ratio, stdlib only."""
    needle, hay = _plain(needle), _plain(hay)
    if not needle:
        return 1.0
    if needle in hay:
        return 1.0
    if not hay:
        return 0.0
    if len(needle) > len(hay):
        return SequenceMatcher(None, needle, hay, autojunk=False).ratio()
    best = 0.0
    for a, b, size in SequenceMatcher(None, hay, needle, autojunk=False).get_matching_blocks():
        if not size:
            continue
        start = max(0, min(a - b, len(hay) - len(needle)))
        best = max(best, SequenceMatcher(None, hay[start:start + len(needle)], needle, autojunk=False).ratio())
    return best


def _words(s: str) -> list:
    return re.findall(r"[a-z][a-z']*", fold(s).lower())


def _stem(w: str) -> str:
    """'increases' / 'increased' / 'increasing' / 'increase' → 'increas' (enough to tell inflections apart
    from opposites)."""
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[:-len(suf)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 4 else w


def evidence_grounded(evidence: str, answer: str) -> bool:
    """Are the quoted words the student's? Long quotes: >= EVIDENCE_MIN_WORD_SHARE of their content words occur
    in the answer (prefix match for abbreviations like 'Govt'). Short quotes: close character match."""
    # every number in the quote must be one the student wrote: a quote of the key's 68,500 for a written 68,700
    # passes any text similarity, so numbers are checked exactly (commas ignored: 54,000 == 54000)
    flat_digits = fold(answer).replace(",", "")
    if any(n.replace(",", "") not in flat_digits for n in re.findall(r"\d[\d,]*(?:\.\d+)?", fold(evidence))):
        return False
    ev = [w for w in re.findall(r"[a-z0-9]+", fold(evidence).lower()) if len(w) >= 3 and w not in _STOP]
    if len(ev) < 3:
        pieces = [x for x in re.split(r"\.{3}|…|\|", evidence) if len(_plain(x)) >= 2]
        if not pieces or min(similarity_in(x, answer) for x in pieces) >= EVIDENCE_MIN_SIMILARITY:
            return True
        # quotes joined from two places ("1,87,000 1,87,000" = both totals of a trial balance): every number /
        # word of the quote must still be in the answer, so an invented "0.6" for a written "0.5" fails
        toks = re.findall(r"[a-z0-9][a-z0-9,./]*[a-z0-9]|[a-z0-9]", fold(evidence).lower())
        flat = fold(answer).lower()
        return bool(toks) and all(t in flat for t in toks)
    have = set(re.findall(r"[a-z0-9]+", fold(answer).lower()))
    stems = {_stem(w) for w in have}

    def found(w: str) -> bool:
        return w in have or _stem(w) in stems or (len(w) >= 4 and any(h.startswith(w[:4]) and
                                                                      abs(len(h) - len(w)) <= 3 for h in have))
    return sum(found(w) for w in ev) / len(ev) >= EVIDENCE_MIN_WORD_SHARE


# ─── prompt injection (safety.py) ──────────────────────────────────────────────────────────────────────────
_INJECTION = re.compile(
    r"\b(ignore|disregard|forget)\b.{0,40}\b(instructions?|rubric|marking scheme|above|previous|rules?|key)\b"
    r"|\b(award|give|assign|grant)\b.{0,25}\b(full|maximum|max|all|100\s*%|total)\s*(marks?|points?|score)\b"
    r"|\byou are (?:an?|the)\s+(?:ai|grader|examiner|assistant|model|language model)\b"
    r"|\b(system prompt|new instructions?|as an ai)\b"
    r"|\bthis answer (?:is|should be) (?:marked|graded) (?:as )?(?:correct|full)\b",
    re.I | re.S,
)


def injection_phrase(text: str) -> Optional[str]:
    m = _INJECTION.search(fold(text))
    return m.group(0)[:60] if m else None


# ─── numbers and units (numeric_grader.py) ─────────────────────────────────────────────────────────────────
UNITS = {
    "m": ("L", 1), "cm": ("L", 0.01), "mm": ("L", 0.001), "km": ("L", 1000),
    "s": ("T", 1), "sec": ("T", 1), "ms": ("T", 0.001), "min": ("T", 60), "h": ("T", 3600), "hr": ("T", 3600),
    "kg": ("M", 1), "g": ("M", 0.001), "mg": ("M", 1e-6),
    "n": ("F", 1), "kn": ("F", 1000), "j": ("E", 1), "kj": ("E", 1000), "cal": ("E", 4.184), "kwh": ("E", 3.6e6),
    "w": ("P", 1), "kw": ("P", 1000), "v": ("V", 1), "a": ("I", 1), "ma": ("I", 0.001),
    "ohm": ("R", 1), "ω": ("R", 1), "Ω": ("R", 1), "pa": ("Pr", 1), "hz": ("Hz", 1), "l": ("Vol", 0.001),
    "ml": ("Vol", 1e-6), "m/s": ("LT", 1), "km/h": ("LT", 1 / 3.6), "kmph": ("LT", 1 / 3.6), "cm/s": ("LT", 0.01),
    "m/s2": ("LT2", 1), "m/s^2": ("LT2", 1), "°c": ("C", 1), "k": ("K", 1), "%": ("pct", 1), "d": ("D", 1),
}
# Indian grouping (1,00,000) as well as 100,000; a fraction; a power of ten
_NUM = r"[-+]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?(?:\s*/\s*\d+(?:\.\d+)?)?(?:\s*x\s*10\s*\^?\s*[-+]?\d+)?"
_QTY = re.compile(rf"(?<![\w.])({_NUM})(?:\s*([a-zA-ZΩω°/%^0-9]{{1,6}}))?")
_CURRENCY = re.compile(r"(?i)\b(?:rs|inr)\b\.?|₹")
# "Step 2:", "(ii)", "3." / "3)" before the content — never the "0." of "0.6 A"
_STEP_LABEL = re.compile(r"(?i)^\s*(?:step\s*\d+\s*[:.)\-]|\(\s*[a-z]{1,4}\s*\)|\d{1,2}[.)](?=\s))\s*")


class Qty:
    __slots__ = ("value", "unit", "text")

    def __init__(self, value: float, unit: Optional[str], text: str):
        self.value, self.unit, self.text = value, unit, text


def _number(s: str) -> Optional[float]:
    s = s.replace(",", "").replace(" ", "")
    m = re.fullmatch(r"([-+]?[\d.]+)x10\^?([-+]?\d+)", s)
    try:
        if m:
            return float(m.group(1)) * 10 ** int(m.group(2))
        return float(Fraction(s)) if "/" in s else float(s)
    except (ValueError, ZeroDivisionError):
        return None


def quantities(text: str) -> list:
    """Every number in the text, with the unit written right after it (if it is a known unit)."""
    s = _CURRENCY.sub(" ", fold(text))
    out = []
    for m in _QTY.finditer(s):
        v = _number(m.group(1))
        if v is None:
            continue
        unit = (m.group(2) or "").lower().rstrip(".") or None
        if unit and unit not in UNITS:
            unit = None          # a following word ("20 cheques"), not a unit
        out.append(Qty(v, unit, m.group(0).strip()))
    return out


def _decimals(text: str) -> int:
    m = re.search(r"\.(\d+)", text)
    return len(m.group(1)) if m else 0


def same_quantity(expected: Qty, got: Qty) -> Optional[bool]:
    """True / False, or None when the units can't be compared. Precision = that of the less precise value."""
    ev, gv = expected.value, got.value
    if expected.unit and got.unit and expected.unit != got.unit:
        ed, gd = UNITS.get(expected.unit), UNITS.get(got.unit)
        if not (ed and gd):
            return None
        if ed[0] != gd[0]:
            return False
        ev, gv = ev * ed[1], gv * gd[1]
    d = min(_decimals(expected.text), _decimals(got.text))
    return math.isclose(ev, gv, rel_tol=1e-9, abs_tol=1e-12) or (d > 0 and abs(ev - gv) <= 0.5 * 10 ** -d + 1e-12)


def _without_notes(text: str) -> str:
    """'v = +20 cm (or 20 cm on the other side)' → 'v = +20 cm': drop worded parentheticals, keep '(-20)'."""
    return re.sub(r"\((?=[^)]*[A-Za-z]{2})[^)]*\)", " ", fold(text))


def expected_value(point: dict) -> Optional[Qty]:
    """The key's value for this point: rubric final_value when the marking scheme has it, else the expected text
    when it holds exactly one number."""
    fv = point.get("final_value")
    if fv:
        qs = quantities(str(fv))
        if len(qs) == 1:
            return qs[0]
    qs = quantities(_STEP_LABEL.sub("", _without_notes(point.get("expected") or "")))
    return qs[0] if len(qs) == 1 else None


def final_answer(answer: str) -> Optional[Qty]:
    """The student's final answer: an 'Ans / ∴ / Therefore' line, else the right side of the last '=',
    else the last line, when that is a single number."""
    lines = [l.strip() for l in fold(answer).splitlines() if l.strip()]
    if not lines:
        return None
    pick = lines[-1]
    for l in reversed(lines):
        m = re.match(r"(?i)^(?:ans(?:wer)?|∴|therefore|hence|final answer)\s*[:.\-]?\s*(.+)$", l)
        if m:
            pick = m.group(1)
            break
    pick = pick.split("=")[-1]
    qs = quantities(pick)
    return qs[-1] if qs else None


# ─── chemical equations (numeric_grader.py) ────────────────────────────────────────────────────────────────
_ARROW = r"\s*(?:->|=>|<=>|-+>|=)\s*"
_SPECIES = re.compile(r"^\s*(\d*)\s*([A-Z][A-Za-z0-9()\[\]·.]*)\s*$")


def equation(text: str) -> Optional[tuple]:
    """First line shaped like 'A + B -> C + D' → (reactants, products) as {species: coefficient}."""
    for line in fold(text).splitlines():
        line = re.sub(r"\((?:s|l|g|aq)\)|[↑↓]", "", line)
        line = re.sub(r"(?i)\b(?:heat|catalyst|sunlight|electricity|conc\.?)\b|[Δ∆]", " ", line)
        m = re.search(r"([A-Z0-9][A-Za-z0-9() +]*?)" + _ARROW + r"([A-Z0-9][A-Za-z0-9() +]*)", line)
        if not m:
            continue
        sides = []
        for side in (m.group(1), m.group(2)):
            d = {}
            for part in side.strip(" .").split("+"):
                pm = _SPECIES.match(part)
                if not pm:
                    return None
                key = pm.group(2).replace(" ", "").lower().replace("i", "l").replace("0", "o")  # ZnCI2 == ZnCl2
                d[key] = d.get(key, 0) + int(pm.group(1) or 1)
            sides.append(d)
        return sides[0], sides[1]
    return None


def compare_equations(expected: str, got: str) -> Optional[str]:
    """'correct' | 'unbalanced' | 'different', or None when either side has no parsable equation."""
    e, g = equation(_without_notes(expected)), equation(got)
    if e is None or g is None:
        return None
    if set(e[0]) != set(g[0]) or set(e[1]) != set(g[1]):
        return "different"
    ratios = {Fraction(g[0][k], e[0][k]) for k in e[0]} | {Fraction(g[1][k], e[1][k]) for k in e[1]}
    return "correct" if len(ratios) == 1 else "unbalanced"


# ─── mutually exclusive terms and negation (text_norm.py) ──────────────────────────────────────────────────
# Kept to sets where naming another member really contradicts the key (no "more/less", "left/right").
ANTONYM_SETS = [
    {"acidic", "basic", "neutral", "alkaline"}, {"increase", "decrease"},
    {"positive", "negative"}, {"convex", "concave"}, {"real", "virtual"}, {"erect", "inverted"},
    {"exothermic", "endothermic"}, {"oxidation", "reduction"}, {"series", "parallel"},
    {"debit", "credit"}, {"surplus", "deficit"}, {"profit", "loss"},
    {"appreciation", "depreciation"}, {"favourable", "unfavourable"}, {"favorable", "unfavorable"},
]
NEGATIONS = {"not", "no", "never", "cannot", "isn't", "doesn't", "don't", "aren't", "neither", "nor", "without"}


def contradiction(expected: str, evidence: str) -> Optional[str]:
    """A key term replaced by its opposite, or negated, in the words quoted for this point."""
    e_words, g_words = _words(_without_notes(expected)), _words(evidence)
    if not g_words:
        return None
    e_set, g_set = set(e_words), set(g_words)
    e_stems, g_stems = {_stem(w) for w in e_words}, {_stem(w): w for w in g_words}
    for group in ANTONYM_SETS:
        stems = {_stem(w): w for w in group}
        need = set(stems) & e_stems
        if len(need) == 1 and not (need & set(g_stems)):
            other = sorted((set(stems) - need) & set(g_stems))
            if other:
                return f"the student wrote '{g_stems[other[0]]}' where the key says '{stems[next(iter(need))]}'"
    # negation: only for short keys that state one thing ("is neutral"), never for comparisons
    # ("devaluation is official, whereas depreciation is automatic" → "devaluation is not automatic" is right)
    if not (NEGATIONS & e_set) and len(e_words) <= 8 and not e_set & {"whereas", "while", "but", "unlike"}:
        for i, w in enumerate(g_words):
            if len(w) >= 4 and w in e_set and NEGATIONS & set(g_words[max(0, i - 3):i]):
                return f"the student wrote '{' '.join(g_words[max(0, i - 3):i + 1])}'"
    return None


# ─── MCQ (objective_grader.py) ─────────────────────────────────────────────────────────────────────────────
_KEY_OPTION = re.compile(r"(?i)^\s*\(?\s*([a-d])\s*[).:]")
# '(b)', 'b)', 'b.', 'Ans: b', 'option B', or the answer being just the letter. Never the article in
# 'a displacement reaction' (the standalone checker's bug #8).
_STUDENT_OPTION = re.compile(
    r"(?i)^\s*(?:(?:ans(?:wer)?|option|opt)\s*[:.\-]?\s*)?(?:\(\s*([a-d])\s*\)|([a-d])\s*[).:](?![\w])|([a-d])\s*$)")


def key_option(leaf: dict, points: list) -> tuple:
    """(letter, [accepted option texts]) from the rubric's correct_option, else from '(c) 7 (or seven)'."""
    letter = str(leaf.get("correct_option") or "").strip().lower()[:1] or None
    src = (points[0].get("expected") if points else "") or leaf.get("model_answer") or ""
    m = _KEY_OPTION.match(fold(src))
    if not letter and m:
        letter = m.group(1).lower()
    text = fold(src)[m.end():] if m else fold(src)
    texts = [_without_notes(text).strip(" .")]
    for note in re.findall(r"\(\s*(?:or|accept)\s+([^)]*)\)", text, re.I):
        texts += [t.strip() for t in re.split(r"/|,| or ", note) if t.strip()]
    for p in points[:1]:
        texts += [str(a) for a in p.get("accept") or []]
    return letter if letter in ("a", "b", "c", "d") else None, [t for t in texts if t]


def student_options(answer: str) -> list:
    s = fold(answer).strip()
    letters = sorted({x.lower() for x in re.findall(r"\(\s*([a-dA-D])\s*\)", s)})
    if len(letters) > 1:
        return letters
    m = _STUDENT_OPTION.match(s)
    return [next(g for g in m.groups() if g).lower()] if m else []


# ─── the checks for one graded leaf ────────────────────────────────────────────────────────────────────────
def _is_diagram(answer_type: str, point: dict) -> bool:
    if answer_type == "diagram":
        return True
    text = f"{point.get('text', '')} {point.get('expected', '')}".lower()
    return bool(re.search(r"\b(diagram|draw|drawing|sketch|label|graph|figure|ray)", text))


def review_leaf(leaf: dict, answer_type: str, points: list, answer: str, attempted: bool) -> list:
    """Review reasons for one graded leaf (a question without sub-parts, or one sub-part).
    leaf: the rubric entry; points: _score_leaf's points_out (verdict, evidence, expected, …); answer: the
    grader's transcription of the student's answer."""
    reasons = []
    answer = answer or ""
    if not attempted:
        return reasons
    inj = injection_phrase(answer)
    if inj:
        reasons.append(f"{PREFIX}the answer contains text addressed to the examiner (\"{inj}\"); "
                       f"it was ignored for marking")

    if answer_type == "mcq" and points:
        letter, texts = key_option(leaf, points)
        chosen = student_options(answer)
        verdict = points[0]["verdict"]
        if len(chosen) > 1:
            if verdict in AWARDED:
                reasons.append(f"{PREFIX}more than one option written ({', '.join(chosen)}) but marks were given")
        elif chosen and letter:
            if chosen[0] != letter and verdict in AWARDED:
                reasons.append(f"{PREFIX}marks given, but the student chose ({chosen[0]}); the key says ({letter})")
            elif chosen[0] == letter and verdict not in AWARDED:
                reasons.append(f"{PREFIX}the student chose ({letter}), which matches the key, but got no marks")
        elif not chosen and verdict not in AWARDED and any(_plain(answer) == _plain(t) for t in texts):
            reasons.append(f"{PREFIX}the answer matches the key's option, but got no marks")
        return reasons          # one point; the rest below is for written answers

    for p in points:
        verdict, evidence = p["verdict"], p.get("evidence") or ""
        label = f"point {p['id']}"
        diagram = _is_diagram(answer_type, p)

        # evidence grounding: an awarded point must quote words that are really in the answer
        if verdict in AWARDED and not diagram and not evidence.lower().startswith(("[diagram", "[image")):
            if not evidence.strip():
                reasons.append(f"{PREFIX}{label} got marks without any of the student's words quoted")
                continue
            if not evidence_grounded(evidence, answer):
                reasons.append(f"{PREFIX}{label} got marks for words that are not in the student's answer")
                continue

        expected = p.get("expected") or ""
        if not expected or diagram:
            continue

        # chemical equation in the key
        if "->" in fold(expected) or "→" in expected:
            got = compare_equations(expected, evidence or answer)
            if got == "different" and verdict == "full":
                reasons.append(f"{PREFIX}{label}: full marks, but the equation differs from the key")
                continue
            if got == "unbalanced" and verdict == "full":
                reasons.append(f"{PREFIX}{label}: full marks, but the equation is not balanced")
                continue
            if got == "correct" and verdict == "none":
                reasons.append(f"{PREFIX}{label}: the equation matches the key, but got no marks")
                continue

        # a number in the key
        if answer_type in ("numerical", "short", "one_word"):
            exp = expected_value(p)
            if exp is not None:
                got = quantities(evidence)
                if verdict == "full" and got:
                    same = [same_quantity(exp, g) for g in got]
                    if not any(same) and None not in same:
                        reasons.append(f"{PREFIX}{label}: full marks, but the student wrote {got[-1].text} "
                                       f"and the key says {exp.text}")
                        continue

        # an opposite or negated key term in the quoted words
        if verdict in ("full", "mostly"):
            why = contradiction(expected, evidence)
            if why:
                reasons.append(f"{PREFIX}{label}: marks given, but {why}")

    # under-credit on a numerical's final answer: the student's final value equals the key's
    if answer_type == "numerical" and points and points[-1]["verdict"] in ("none", "partial"):
        exp, fin = expected_value(points[-1]), final_answer(answer)
        if exp is not None and fin is not None and same_quantity(exp, fin) and (not exp.unit or fin.unit):
            reasons.append(f"{PREFIX}the final answer {fin.text} matches the key, but point {points[-1]['id']} "
                           f"got {'no' if points[-1]['verdict'] == 'none' else 'only part'} marks")
    return reasons
