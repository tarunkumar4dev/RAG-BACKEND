"""
"Question Bank → Paper Builder": everything between the reviewed contest-import questions and a printed paper.
No Gemini call, no database call — pure functions (the router does the I/O).

  normalise_questions  reviewed questions (the contest import's shape) → bank_questions rows + stats.
                       Trim, collapse whitespace, hash, drop duplicates, derive qtype / year tag / needs_review.
                       ANSWERS ARE NEVER INVENTED: an answer is kept only when it came from the PDF's key ("key")
                       or from the teacher ("teacher"). An AI guess ("ai") is dropped and the question is flagged.
  select_for_paper     sections [{qtype, count, marks_each}] → the chosen bank rows, or PaperError
                       ("Section A needs 10 MCQs, bank has 6").
  paper_questions      chosen rows → the dicts export_service renders (and /test-generator/save stores).
  render_paper         existing export_service → PDF / DOCX bytes; the answer key (existing standalone answer-key
                       export) is appended as its own pages when asked for.

The extraction itself is the existing contest import (contest_import_service) — untouched.
"""

import hashlib
import io
import random
import re
import unicodedata
import uuid
from datetime import date
from typing import List, Optional

QTYPES = ("mcq", "true_false", "short", "long")
# (singular, plural) for messages; heading for the paper
QTYPE_NAMES = {
    "mcq": ("MCQ", "MCQs", "Multiple Choice Questions"),
    "true_false": ("True/False question", "True/False questions", "True / False"),
    "short": ("short-answer question", "short-answer questions", "Short Answer Questions"),
    "long": ("long-answer question", "long-answer questions", "Long Answer Questions"),
}
EXPORT_FORMAT = {"mcq": "mcq", "true_false": "mcq", "short": "short_answer", "long": "long_answer"}
SECTION_LETTERS = "ABCDEF"
NO_ANSWER_TEXT = "— (no answer in the source PDF)"
MAX_TEXT = 6000
MAX_ANSWER = 4000

_TRUE = {"true", "t", "correct", "right", "yes", "सही", "सत्य"}
_FALSE = {"false", "f", "incorrect", "wrong", "no", "गलत", "ग़लत", "असत्य"}
_TF_STEM = re.compile(r"\btrue\s*(?:/|or)\s*false\b|सही\s*(?:/|या)\s*(?:गलत|ग़लत)|सत्य\s*(?:/|या)\s*असत्य", re.I)
# "[CBSE 2023]", "(CBSE 2022)", "[Delhi 2019 Set 1]", "(2021)" — a short bracketed tag with a 4-digit year
_YEAR_TAG = re.compile(r"[\[(]\s*([^\[\]()]{0,40}?\b((?:19|20)\d{2})\b[^\[\]()]{0,20}?)\s*[\])]")
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")


# ═══════════════════════════════════════════════════════════════════════
# TEXT
# ═══════════════════════════════════════════════════════════════════════

def clean_text(value, limit: int = MAX_TEXT) -> str:
    """Trim; collapse spaces inside each line; keep line breaks (Assertion / Reason, statements I / II); drop
    control characters. The words themselves — Hindi or English — are not changed."""
    s = unicodedata.normalize("NFC", str(value or "")).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in s.split("\n")]
    s = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return s[:limit]


def _canon(value) -> str:
    """For comparing only: NFKC, case-folded, all whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def content_hash(question_text: str, options: list) -> str:
    """Same stem + same options = the same question ("Which of the following is correct?" with other options is
    a different one)."""
    parts = [_canon(question_text)] + [_canon(o.get("text")) for o in options or []]
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def safe_filename(name, fallback: str = "question-paper") -> str:
    """A client file name → display-only base name: no path parts, no control characters."""
    s = str(name or "").replace("\\", "/").split("/")[-1]
    s = re.sub(r"[\x00-\x1f\x7f<>:\"|?*]", "", s).strip(" .")
    return s[:200] or fallback


def download_name(title: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", title or "")[:50].strip("_") or "question_paper"


def year_tag(text: str) -> tuple:
    """(exam_year, source_tag) from a bracketed tag in the question, e.g. "… [CBSE 2023]" → (2023, "CBSE 2023")."""
    last = None
    for m in _YEAR_TAG.finditer(text or ""):
        last = m
    if not last:
        return None, None
    year = int(last.group(2))
    if not 1950 <= year <= date.today().year + 1:
        return None, None
    if re.search(r"[₹$%]|\brs\b|\bcrore|\blakh", last.group(1), re.I):      # "(₹ 2024 crore)" is an amount
        return None, None
    return year, " ".join(last.group(1).split())[:60]


def paper_tag(*texts) -> tuple:
    """(exam_year, source_tag) for a whole paper from its title / board line ("CBSE 2023 Sample Paper")."""
    for t in texts:
        t = " ".join(str(t or "").split())
        m = _YEAR.search(t)
        if m and 1950 <= int(m.group(1)) <= date.today().year + 1:
            return int(m.group(1)), t[:60]
    return None, None


# ═══════════════════════════════════════════════════════════════════════
# NORMALISE (reviewed import questions → bank rows)
# ═══════════════════════════════════════════════════════════════════════

def _options(raw: list) -> list:
    """[{label, text}] with labels A, B, C… in order; empty options dropped."""
    out = []
    for o in raw or []:
        text = clean_text(o.get("text") if isinstance(o, dict) else o, 1000)
        if text:
            out.append({"label": SECTION_LETTERS[len(out)] if len(out) < 6 else str(len(out) + 1), "text": text})
    return out[:6]


def _is_true_false_options(options: list) -> bool:
    return len(options) == 2 and {_canon(o["text"]) for o in options} == {"true", "false"}


def _tf_value(answer) -> Optional[str]:
    c = _canon(answer).strip(" .()")
    if c in _TRUE:
        return "True"
    if c in _FALSE:
        return "False"
    return None


def derive_qtype(declared, options: list, answer, text: str) -> str:
    """The parser only knows MCQ / Short / Long; true/false is recognised here by rule."""
    t = _canon(declared).replace(" ", "_").replace("/", "_").replace("-", "_")
    if t in ("true_false", "truefalse", "tf", "t_f"):
        return "true_false"
    if _is_true_false_options(options):
        return "true_false"
    if len(options) >= 2 or t in ("mcq", "multiple_choice", "objective"):
        return "mcq"
    if _tf_value(answer) or _TF_STEM.search(text or ""):
        return "true_false"
    return "long" if t in ("long", "long_answer", "essay") else "short"


def option_label(answer, options: list) -> Optional[str]:
    """The option an MCQ answer points to: "B", "(b)", "B) text", or the option's own text. None if none."""
    a = str(answer or "").strip()
    if not a or not options:
        return None
    labels = [o["label"] for o in options]
    m = re.fullmatch(r"\(?\s*([A-Fa-f])\s*\)?\.?", a) or re.match(r"\(?([A-Fa-f])[).:\-]\s", a)
    if m:
        label = m.group(1).upper()
        return label if label in labels else None
    hits = [o["label"] for o in options if _canon(o["text"]) == _canon(a)]
    return hits[0] if len(hits) == 1 else None


def normalise_answer(qtype: str, options: list, raw, source) -> tuple:
    """(answer, answer_source, problem) — problem is None, "ai" (an AI guess, dropped) or "unresolved"
    (an answer that matches none of the options / is not True or False, dropped)."""
    text = clean_text(raw, MAX_ANSWER)
    if not text:
        return None, None, None
    if source == "ai":
        return None, None, "ai"
    kept = "key" if source == "key" else "teacher"     # "missing" / None with an answer = typed in by the teacher
    if qtype == "mcq":
        label = option_label(text, options)
        return (label, kept, None) if label else (None, None, "unresolved")
    if qtype == "true_false":
        if options:
            label = option_label(text, options)
            if not label and _tf_value(text):
                label = next((o["label"] for o in options if _canon(o["text"]) == _tf_value(text).lower()), None)
            return (label, kept, None) if label else (None, None, "unresolved")
        value = _tf_value(text)
        return (value, kept, None) if value else (None, None, "unresolved")
    return text, kept, None


def needs_review(qtype: str, question_text: str, options: list, answer, usable: bool = True) -> bool:
    """True when a paper should not use the question until the teacher has looked at it: the parser could not read
    it completely, an MCQ has fewer than 2 options, or an objective question (MCQ / true-false) has no answer.
    A written question without a model answer is NOT flagged: it is a complete question, its answer-key line just
    says that the source PDF had no answer."""
    if not usable or not (question_text or "").strip():
        return True
    if qtype == "mcq" and len(options or []) < 2:
        return True
    return qtype in ("mcq", "true_false") and not answer


def _marks(value):
    if value is None:
        return None
    try:
        m = float(value)
    except (TypeError, ValueError):
        return None
    if not 0 < m <= 100:
        return None
    return int(m) if m == int(m) else m


def normalise_questions(questions: list, source_tag: Optional[str] = None) -> tuple:
    """Reviewed questions (dicts in the BankQuestionIn shape) → (rows for bank_questions, stats).
    source_tag = the paper's title line; used for the year when a question has no tag of its own."""
    paper_year, paper_label = paper_tag(source_tag)
    rows, seen = [], set()
    stats = {"received": len(questions), "saved": 0, "duplicates_removed": 0, "skipped_empty": 0,
             "needs_review": 0, "answers_matched": 0, "ai_answers_dropped": 0, "unresolved_answers": 0,
             "per_type": {t: 0 for t in QTYPES}}
    for q in questions:
        text = clean_text(q.get("question_text"))
        if not text:
            stats["skipped_empty"] += 1
            continue
        options = _options(q.get("options"))
        qtype = derive_qtype(q.get("question_type"), options, q.get("correct_answer"), text)
        if qtype in ("short", "long"):
            options = []
        digest = content_hash(text, options)
        if digest in seen:
            stats["duplicates_removed"] += 1
            continue
        seen.add(digest)
        answer, answer_source, problem = normalise_answer(qtype, options, q.get("correct_answer"),
                                                          q.get("answer_source"))
        if problem == "ai":
            stats["ai_answers_dropped"] += 1
        elif problem == "unresolved":
            stats["unresolved_answers"] += 1
        year, tag = year_tag(text)
        if year is None:
            year, tag = paper_year, paper_label
        flagged = needs_review(qtype, text, options, answer, bool(q.get("usable", True))) or problem is not None
        rows.append({
            "position": len(rows) + 1, "qtype": qtype, "question_text": text, "options": options,
            "answer": answer, "answer_source": answer_source, "marks": _marks(q.get("marks")),
            "exam_year": year, "source_tag": tag, "needs_review": flagged, "excluded": False,
            "content_hash": digest,
        })
        stats["per_type"][qtype] += 1
        stats["needs_review"] += flagged
        stats["answers_matched"] += answer is not None
    stats["saved"] = len(rows)
    return rows, stats


def stats_warnings(stats: dict) -> list:
    """Plain sentences for the teacher about what was changed on the way into the bank."""
    out = []
    if stats["duplicates_removed"]:
        n = stats["duplicates_removed"]
        out.append(f"{n} duplicate question{'s were' if n > 1 else ' was'} left out.")
    if stats["skipped_empty"]:
        n = stats["skipped_empty"]
        out.append(f"{n} question{'s' if n > 1 else ''} with no text {'were' if n > 1 else 'was'} left out.")
    if stats["ai_answers_dropped"]:
        n = stats["ai_answers_dropped"]
        out.append(f"{n} answer{'s' if n > 1 else ''} guessed by AI {'were' if n > 1 else 'was'} not saved "
                   f"(not from the PDF's answer key, not confirmed by you). Set them by hand in the bank.")
    if stats["unresolved_answers"]:
        n = stats["unresolved_answers"]
        out.append(f"{n} answer{'s do' if n > 1 else ' does'} not match the question's options and "
                   f"{'were' if n > 1 else 'was'} not saved.")
    if stats["needs_review"]:
        n = stats["needs_review"]
        out.append(f"{n} question{'s need' if n > 1 else ' needs'} review before a paper can use "
                   f"{'them' if n > 1 else 'it'}.")
    return out


def payload_from_import(result: dict) -> list:
    """The contest import's result → the questions list of POST /question-banks, with no teacher edits.
    The frontend does this mapping after review; scripts/qb_extract_check.py and the tests use it too."""
    return [{"question_text": q.get("question_text") or "", "question_type": q.get("question_type") or "",
             "options": [{"label": o.get("label") or "", "text": o.get("text") or ""} for o in q.get("options") or []],
             "correct_answer": q.get("correct_answer"), "answer_source": q.get("answer_source"),
             "marks": q.get("marks"), "usable": bool(q.get("usable", True)),
             "source_number": q.get("source_number")} for q in result.get("questions") or []]


# ═══════════════════════════════════════════════════════════════════════
# SELECT (bank rows → the questions of one paper)
# ═══════════════════════════════════════════════════════════════════════

class PaperError(ValueError):
    """The bank cannot fill the requested paper; str(e) is the message for the teacher."""

    def __init__(self, shortfalls: list):
        super().__init__(" ".join(shortfalls))
        self.shortfalls = shortfalls


def _section_value(section, key):
    return section[key] if isinstance(section, dict) else getattr(section, key)


def select_for_paper(rows: list, sections: list, *, year_filter: Optional[int] = None,
                     allow_needs_review: bool = False, shuffle: bool = False,
                     rng: Optional[random.Random] = None) -> list:
    """[[bank rows of section A], [section B], …]. Only rows of this bank, each at most once, never an excluded
    one, never a needs_review one unless allowed. PaperError lists every section that cannot be filled."""
    rng = rng or random.Random()
    used: set = set()
    picked, shortfalls = [], []
    for i, section in enumerate(sections):
        qtype, count = _section_value(section, "qtype"), int(_section_value(section, "count"))
        letter = SECTION_LETTERS[i]
        of_type = [r for r in rows if r.get("qtype") == qtype and not r.get("excluded")
                   and (year_filter is None or r.get("exam_year") == year_filter)]
        usable = [r for r in of_type if allow_needs_review or not r.get("needs_review")]
        pool = [r for r in usable if r["id"] not in used]
        if len(pool) < count:
            singular, plural, _ = QTYPE_NAMES[qtype]
            need = f"{count} {singular if count == 1 else plural}"
            where = f" from {year_filter}" if year_filter is not None else ""
            have = f"has {len(pool)}{where}"
            if len(usable) != len(pool):
                have += " left after the earlier sections"
            held = len(of_type) - len(usable)
            hint = f" ({held} more need{'s' if held == 1 else ''} review)" if held else ""
            shortfalls.append(f"Section {letter} needs {need}, bank {have}{hint}.")
            picked.append([])
            continue
        ordered = sorted(pool, key=lambda r: (r.get("position") or 0, str(r["id"])))
        chosen = rng.sample(ordered, count) if shuffle else ordered[:count]
        used.update(r["id"] for r in chosen)
        picked.append(chosen)
    if shortfalls:
        raise PaperError(shortfalls)
    return picked


def paper_questions(picked: list, sections: list) -> list:
    """Chosen rows → export_service question dicts, in paper order. Text and options are the bank's, unchanged;
    marks are the section's. correctAnswer is "" when the bank has no answer (never filled in from anywhere)."""
    out = []
    for i, (rows, section) in enumerate(zip(picked, sections)):
        qtype, marks = _section_value(section, "qtype"), int(_section_value(section, "marks_each"))
        letter = SECTION_LETTERS[i]
        meta = {"title": f"Section {letter}",
                "subtitle": f"({QTYPE_NAMES[qtype][2]} — {marks} mark{'' if marks == 1 else 's'} each)",
                "instruction": f"All questions are compulsory. Each carries {marks} mark{'' if marks == 1 else 's'}.",
                "marks": marks}
        for r in rows:
            options = list(r.get("options") or [])
            answer = r.get("answer") or ""
            if qtype == "true_false" and not options:
                options = [{"label": "A", "text": "True"}, {"label": "B", "text": "False"}]
                answer = {"True": "A", "False": "B"}.get(answer, "")
            out.append({
                "id": str(uuid.uuid4()),
                "bank_question_id": str(r["id"]),
                "text": r.get("question_text") or "",
                # "A) text": the exporter strips a leading "A) " itself; a bare "A mother …" would lose its "A "
                "options": [f"{o['label']}) {o['text']}" for o in options],
                "correctAnswer": answer,
                "answerFound": bool(answer),
                "explanation": "",
                "marks": marks,
                "difficulty": "medium",
                "chapter": "",
                "format": EXPORT_FORMAT[qtype],
                "section": letter,
                "section_meta": meta,
            })
    return out


# ═══════════════════════════════════════════════════════════════════════
# RENDER (existing export system)
# ═══════════════════════════════════════════════════════════════════════

_INDIC = re.compile(r"[ऀ-෿]")       # Devanagari … Sinhala


def pdf_cannot_print(questions: List[dict], *more_texts) -> bool:
    """The exporter's PDF fonts (Helvetica / Times) have no Indian-script glyphs: Hindi text comes out as black
    boxes. DOCX is not affected (Word picks a font that has them)."""
    texts = list(more_texts)
    for q in questions:
        texts.append(q.get("text"))
        texts.extend(q.get("options") or [])
        texts.append(q.get("correctAnswer"))
    return any(_INDIC.search(str(t or "")) for t in texts)


def allowed_templates() -> set:
    from app.services.export_service import TEMPLATE_PRESETS
    return set(TEMPLATE_PRESETS)


def _answer_key_questions(questions: list) -> list:
    """For the answer key only: a question without an answer says so instead of showing a blank."""
    return [q if q.get("correctAnswer") else {**q, "correctAnswer": NO_ANSWER_TEXT} for q in questions]


def _append_pdf(paper: bytes, extra: bytes) -> bytes:
    import pymupdf
    out = pymupdf.open(stream=paper, filetype="pdf")
    more = pymupdf.open(stream=extra, filetype="pdf")
    try:
        out.insert_pdf(more)
        return out.tobytes(garbage=3, deflate=True)
    finally:
        more.close()
        out.close()


def _append_docx(paper: bytes, extra: bytes) -> bytes:
    """The answer-key document's body after a page break in the paper document. Both come from export_service,
    whose formatting is set on the runs / tables themselves, so the copied content keeps its look."""
    from copy import deepcopy
    from docx import Document
    from docx.enum.text import WD_BREAK
    doc, more = Document(io.BytesIO(paper)), Document(io.BytesIO(extra))
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    body = doc.element.body
    sect = body.sectPr
    for el in more.element.body:
        if el.tag.endswith("}sectPr"):
            continue
        if sect is not None:
            sect.addprevious(deepcopy(el))
        else:
            body.append(deepcopy(el))
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _pdf_header_text(value):
    """Title / institute / subject go into the PDF header as markup, unescaped (and some layouts upper-case them,
    so entities cannot be used): "<Test>" would vanish and "C&B" would print as "C&B;". Angle brackets become
    ‹ ›, and an "&" gets spaces around it."""
    if not value:
        return value
    s = str(value).replace("<", "‹").replace(">", "›")
    return " ".join(re.sub(r"\s*&\s*", " & ", s).split())


def render_paper(questions: List[dict], *, title: str, fmt: str, template: str, include_answer_key: bool,
                 subject: str, class_grade: str, board: str, institute_name: Optional[str] = None,
                 duration: Optional[str] = None, teacher_name: Optional[str] = None,
                 paper_date: Optional[str] = None) -> bytes:
    """The question paper through the existing exporter (never with answers marked inside it), plus — when asked —
    the existing standalone answer key appended as its own page(s)."""
    from app.services import export_service as ex
    head = _pdf_header_text if fmt == "pdf" else (lambda v: v)
    common = dict(exam_title=head(title), board=head(board), class_grade=head(class_grade), subject=head(subject),
                  paper_date=paper_date, template=template, teacher_name=head(teacher_name),
                  institute_name=head(institute_name), duration=head(duration))
    with ex.literal_text():
        if fmt == "docx":
            paper = ex.generate_docx(questions=questions, include_answers=False, include_explanations=False, **common)
            if not include_answer_key:
                return paper
            key = ex.generate_answer_key_docx(questions=_answer_key_questions(questions),
                                              include_explanations=False, **common)
            return _append_docx(paper, key)
        paper = ex.generate_pdf(questions=questions, include_answers=False, include_explanations=False, **common)
        if not include_answer_key:
            return paper
        key = ex.generate_answer_key_pdf(questions=_answer_key_questions(questions),
                                         include_explanations=False, **common)
        return _append_pdf(paper, key)
