"""VERIFICATION GATE.

deterministic (mcq, numeric, assertion_reason, very_short with check):
  independent solver pass WITHOUT the key + sympy for numeric/trig + rules
  (exactly one correct MCQ option, valid A-R option set).
rubric (short/long answer, case study): marking scheme present, scheme marks == question marks,
  no duplicate, in syllabus.
unparseable -> manual_review (never silently passes).
Paper: marks sum EXACTLY to total, no duplicates.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

import sympy as sp
from sympy.parsing.sympy_parser import (implicit_multiplication_application, parse_expr,
                                        standard_transformations, convert_xor)

DETERMINISTIC = {"mcq", "numeric", "assertion_reason", "very_short"}
RUBRIC = {"short_answer", "long_answer", "case_study"}
AR_OPTIONS = [
    "both a and r are true and r is the correct explanation of a",
    "both a and r are true but r is not the correct explanation of a",
    "a is true but r is false",
    "a is false but r is true",
]
LETTERS = "ABCD"
_TF = standard_transformations + (implicit_multiplication_application, convert_xor)
_LOCALS = {"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
           "cosec": sp.csc, "sqrt": sp.sqrt, "pi": sp.pi, "deg": sp.pi / 180,
           "Rational": sp.Rational, "Integer": sp.Integer}


def _latex_to_plain(t: str) -> str:
    t = t.strip().strip("$").strip()
    t = re.sub(r"\\(left|right|displaystyle|,|;|!)", "", t)
    t = re.sub(r"\^\s*\{?\\circ\}?", "*deg", t)
    t = t.replace("\\circ", "*deg").replace("\\cdot", "*").replace("\\times", "*").replace("\\div", "/")
    t = t.replace("\\theta", "theta").replace("\\pi", "pi").replace("\\alpha", "alpha").replace("\\beta", "beta")
    t = re.sub(r"\\[dt]?frac\s*(\d)\s*(\d)", r"((\1)/(\2))", t)  # \frac32 shorthand
    for _ in range(10):  # innermost-first: sqrt{..}, (d|t)frac{..}{..}, ^{..}
        prev = t
        t = re.sub(r"\\?sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", t)
        t = re.sub(r"\\?[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", t)
        t = re.sub(r"\^\s*\{([^{}]*)\}", r"**(\1)", t)
        if t == prev:
            break
    t = t.replace("\\", "")
    t = (t.replace("√", "sqrt").replace("π", "pi").replace("°", "*deg").replace("×", "*").replace("÷", "/")
         .replace("−", "-").replace("θ", "theta"))
    t = re.sub(r"sqrt\s*(\d+)", r"sqrt(\1)", t)
    t = t.replace("{", "(").replace("}", ")")
    fn = r"(sin|cos|tan|cot|sec|cosec|csc)"
    arg = r"(theta|alpha|beta|[A-Z]|\d+(?:\.\d+)?(?:\*deg)?|\([^()]*\))"
    t = re.sub(fn + r"\s*\^\s*\(?(\d+)\)?\s*" + arg, r"(\1(\3))**\2", t)  # sin^2 30° -> (sin(30*deg))**2
    t = re.sub(fn + r"\s+" + arg, r"\1(\2)", t)  # sin 30° -> sin(30*deg)
    return t


def to_sympy(s: Any) -> Optional[sp.Expr]:
    """Parse '√3/2', 'sin(30°)^2', '$\\dfrac{\\sqrt{3}}{2}$'. Returns None when unparseable."""
    if s is None:
        return None
    raw = str(s)
    if not raw.strip() or len(raw) > 200 or "=" in raw:
        return None
    t = _latex_to_plain(raw)
    if "," in re.sub(r"\([^()]*\)", "", t):  # top-level comma = several values, not one expression
        return None
    if re.search(r"[a-zA-Z]{2,}", re.sub(r"Rational|Integer|sqrt|sin|cos|tan|cot|sec|csc|cosec|pi|deg|theta|alpha|beta",
                                         "", t)):
        return None  # leftover words (e.g. 'dfrac', prose) -> unparseable, never a silent symbol
    try:
        e = parse_expr(t, local_dict=_LOCALS, transformations=_TF, evaluate=True)
        return e if isinstance(e, sp.Expr) else None
    except Exception:
        return None


def parse_pairs(s: Any) -> Optional[dict]:
    """'A = 45°, B = 15°' -> {'A': '45', 'B': '15'} (degrees dropped on both sides when compared)."""
    if s is None:
        return None
    parts = [p.strip() for p in re.split(r"[,;]|\band\b", str(s).strip().strip("$")) if p.strip()]
    out = {}
    for p in parts:
        m = re.match(r"^\\?([A-Za-z]{1,6})\s*=\s*(.+)$", p.strip().strip("$"))
        if not m:
            return None
        out[m.group(1).upper()] = re.sub(r"\^?\s*\{?\\circ\}?|°", "", m.group(2)).strip()
    return out if len(out) >= 2 else None


def sym_equal(a: Any, b: Any) -> Optional[bool]:
    pa, pb = parse_pairs(a), parse_pairs(b)
    if pa is not None or pb is not None:
        if pa is None or pb is None or set(pa) != set(pb):
            return None if (pa is None or pb is None) else False
        res = [sym_equal(pa[k], pb[k]) for k in pa]
        return None if any(r is None for r in res) else all(res)
    ea, eb = to_sympy(a), to_sympy(b)
    if ea is None or eb is None:
        return None
    try:
        d = sp.simplify(ea - eb)
        if d == 0:
            return True
        if d.free_symbols:  # identity in theta etc: numeric spot-check at 3 angles
            vals = [abs(complex(d.subs({s: v for s in d.free_symbols}).evalf())) for v in (0.3, 0.7, 1.1)]
            return all(v < 1e-9 for v in vals)
        return abs(complex(d.evalf())) < 1e-9
    except Exception:
        return None


def _distinct_options(opts: list) -> bool:
    vals = [to_sympy(o) for o in opts]
    if all(v is not None for v in vals):
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                if sym_equal(opts[i], opts[j]):
                    return False
        return True
    keys = [re.sub(r"\s+", "", _latex_to_plain(str(o)).lower()) for o in opts]
    return len(set(keys)) == len(keys)


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def _letter(ans: Any) -> Optional[str]:
    m = re.match(r"^\s*\(?([A-Da-d])\)?(?:[\s.):]|$)", str(ans or ""))
    return m.group(1).upper() if m else None


def classify(q: dict) -> str:
    return "deterministic" if q.get("qtype") in DETERMINISTIC else "rubric"


Solver = Callable[[list[dict]], dict]  # questions WITHOUT keys -> {id: answer}


def strip_key(q: dict) -> dict:
    return {k: q[k] for k in ("id", "qtype", "text", "options") if k in q}


def _check_deterministic(q: dict, solved: Optional[str]) -> tuple[str, list[str], dict]:
    reasons: list[str] = []
    detail: dict = {"solver_answer": solved, "sympy": None, "key_matches": None}
    qt = q["qtype"]
    if qt in ("mcq", "assertion_reason"):
        opts = q.get("options") or []
        key = _letter(q.get("answer"))
        if len(opts) != 4 or not _distinct_options(opts):
            reasons.append("needs 4 distinct options")
        if key is None:
            return "manual_review", reasons + ["answer key not a letter A-D"], detail
        if qt == "assertion_reason" and [_norm(o) for o in opts] != [_norm(o) for o in AR_OPTIONS]:
            reasons.append("A-R options are not the standard CBSE set")
        if qt == "assertion_reason" and not re.search(r"assertion", q.get("text", ""), re.I):
            reasons.append("A-R question has no Assertion/Reason statements")
        # sympy: for MCQ with a check expression, exactly one option must equal it.
        # A-R options are prose (standard set) -> sympy n/a; solver + option-set rules decide.
        if qt == "assertion_reason":
            detail["sympy"] = "n/a (assertion-reason)"
        elif q.get("check_expr"):
            hits = [sym_equal(q["check_expr"], o) for o in opts]
            if any(h is None for h in hits) and not any(hits):
                detail["sympy"] = "unparseable"
                return "manual_review", reasons + ["sympy could not parse options/check"], detail
            n = sum(1 for h in hits if h)
            detail["sympy"] = f"{n} option(s) equal check_expr"
            if n != 1:
                reasons.append(f"exactly one correct option required, sympy found {n}")
            elif LETTERS[hits.index(True)] != key:
                reasons.append(f"sympy says {LETTERS[hits.index(True)]}, key says {key}")
        sl = _letter(solved)
        detail["key_matches"] = (sl == key) if sl else None
        if sl is None:
            return "manual_review", reasons + ["independent solver gave no answer"], detail
        if sl != key:
            reasons.append(f"independent solver chose {sl}, key is {key}")
    else:  # numeric / very_short
        key = q.get("answer")
        if q.get("check_expr") and parse_pairs(key) is None:
            eq = sym_equal(q["check_expr"], key)
            detail["sympy"] = {True: "key == check_expr", False: "key != check_expr", None: "unparseable"}[eq]
            if eq is None:
                return "manual_review", reasons + ["sympy could not parse"], detail
            if not eq:
                reasons.append("sympy: key does not equal check expression")
        eq2 = sym_equal(solved, key)
        detail["key_matches"] = eq2
        if eq2 is None:
            if _norm(str(solved)) and _norm(str(solved)) == _norm(str(key)):
                detail["key_matches"] = True
            else:
                return "manual_review", reasons + ["solver answer not comparable to key"], detail
        elif not eq2:
            reasons.append("independent solver answer differs from key")
    return ("failed" if reasons else "verified"), reasons, detail


def _check_rubric(q: dict, syllabus: Optional[set[str]]) -> tuple[str, list[str], dict]:
    reasons: list[str] = []
    parts = q.get("sub_questions") or []
    scheme = q.get("marking_scheme") or []
    if q["qtype"] == "case_study":
        if not parts:
            reasons.append("case study has no sub-questions")
        psum = sum(float(p.get("marks", 0)) for p in parts)
        if parts and abs(psum - float(q["marks"])) > 1e-9:
            reasons.append(f"sub-question marks {psum:g} != {q['marks']:g}")
        if any(not (p.get("answer") or p.get("marking_scheme")) for p in parts):
            reasons.append("a sub-question has no answer/marking scheme")
    else:
        if not scheme:
            reasons.append("marking scheme missing")
        smarks = [float(s.get("marks", 0)) for s in scheme if isinstance(s, dict)]
        if smarks and abs(sum(smarks) - float(q["marks"])) > 1e-9:
            reasons.append(f"marking scheme sums to {sum(smarks):g}, question carries {q['marks']:g}")
    if syllabus and _norm(q.get("chapter", "")) not in syllabus:
        reasons.append(f"chapter '{q.get('chapter')}' not in requested syllabus")
    return ("failed" if reasons else "rubric_ok"), reasons, {}


def verify_paper(paper: dict, solver: Optional[Solver], syllabus: Optional[list[str]] = None) -> dict:
    qs = paper["questions"]
    syl = {_norm(s) for s in syllabus} if syllabus else None
    det = [q for q in qs if classify(q) == "deterministic"]
    solved: dict = {}
    solver_error = None
    if solver and det:
        try:
            solved = solver([strip_key(q) for q in det]) or {}
        except Exception as e:  # solver down -> manual_review, never pass
            solver_error = type(e).__name__
    seen: dict[str, str] = {}
    dups: list[str] = []
    results = {}
    for q in qs:
        n = _norm(q.get("text", ""))
        if n in seen:
            dups.append(q["id"])
        seen.setdefault(n, q["id"])
        if classify(q) == "deterministic":
            st, rs, d = _check_deterministic(q, solved.get(q["id"]))
        else:
            st, rs, d = _check_rubric(q, syl)
        if q["id"] in dups:
            st, rs = "failed", rs + ["duplicate of " + seen[n]]
        if not (isinstance(q.get("marks"), (int, float)) and q["marks"] > 0):
            st, rs = "failed", rs + ["marks must be > 0"]
        results[q["id"]] = {"status": st, "reasons": rs, **d}
        q["status"] = st
    total = sum(float(q.get("marks", 0)) for q in qs)
    marks_ok = abs(total - float(paper["total_marks"])) < 1e-9
    obj = [q for q in qs if classify(q) == "deterministic"]
    rub = [q for q in qs if classify(q) == "rubric"]
    return {
        "objective_verified": sum(1 for q in obj if results[q["id"]]["status"] == "verified"),
        "objective_total": len(obj),
        "rubric_checked": sum(1 for q in rub if results[q["id"]]["status"] == "rubric_ok"),
        "rubric_total": len(rub),
        "manual_review": [q["id"] for q in qs if results[q["id"]]["status"] == "manual_review"],
        "failed": [q["id"] for q in qs if results[q["id"]]["status"] == "failed"],
        "marks_sum": total,
        "marks_sum_ok": marks_ok,
        "duplicates": dups,
        "solver_error": solver_error,
        "per_question": results,
    }


def can_finalize(summary: dict) -> bool:
    return summary["marks_sum_ok"] and not summary["failed"] and not summary["duplicates"]
