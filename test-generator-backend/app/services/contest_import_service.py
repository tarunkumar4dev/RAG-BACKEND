"""
"PDF → contest questions": read a question paper PDF (and the answer key inside it) into the POST /contests shape.

  check_pdf      size / type / password / page count (before any Gemini call)
  import_paper   1. PARSE  the PDF goes to Gemini as a document (no text layer needed, scanned papers work), structured
                           JSON with a schema; > CHUNK_PAGES pages → page chunks in parallel, merged in page order.
                           Gemini only transcribes: header, questions, options, and the raw answer-key entries.
                 2. build_result (pure, no Gemini): option labels, key letter → label, section level → difficulty,
                           duration, and every validation warning. Nothing is silently fixed.
                 3. KEY CHECK (optional, CONTEST_IMPORT_KEY_CHECK=0 turns it off): a cheap text-only call solves each
                           keyed MCQ on its own; a disagreement is a KEY_MISMATCH_SUSPECT warning, the key is never
                           changed. Runs only when more than KEY_CHECK_MIN_LEFT_S are left; skipped with
                           KEY_CHECK_SKIPPED otherwise or when it fails.

Gemini calls go through contest_import_gemini (plain REST, thinking capped — the production SDK can't), with the
checker's rules: 404/503 → next model, 429/500/504 → GeminiBusy, one retry. Budget: the job's deadline (270 s);
the parse ends by deadline - PARSE_RESERVE_S, the key check by deadline - 30 s. Tests: GEMINI_MOCK recordings
IMPORT-PARSE / IMPORT-SOLVE.
"""

import contextvars
import json
import logging
import math
import os
import re
import string
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import pymupdf

from app.services import contest_import_gemini as gemini
from app.services import smart_assignment_service as svc
from app.services.answer_checker.upload_validation import sniff
from app.services.contest_service import _canon

logger = logging.getLogger(__name__)

MAX_PDF_BYTES = 15 * 1024 * 1024
MAX_PAGES = 30
CHUNK_PAGES = 20
PARSE_TIMEOUT_S = float(os.getenv("CONTEST_IMPORT_PARSE_TIMEOUT_S", "210"))
SOLVE_TIMEOUT_S = float(os.getenv("CONTEST_IMPORT_SOLVE_TIMEOUT_S", "60"))
KEY_CHECK_MIN_LEFT_S = 90
CALL_CAP = 24          # 2 chunks × (2 models × 2 tries) + key-check batches × 2 models, with room
DURATION_MIN, DURATION_MAX = 5, 180   # CreateContestRequest.duration_minutes
# Model order for the import only (404/503 → next). 3.8-flash wrote the 50-MCQ paper's JSON in 192 s,
# gemini-3-flash-preview in 41 s (thinking capped, 24 Oct 2026).
IMPORT_MODELS = [m.strip() for m in os.getenv("CONTEST_IMPORT_MODELS", "gemini-3-flash-preview,gemini-3.8-flash")
                 .split(",") if m.strip()]
PARSE_THINKING_BUDGET = 2048
SOLVE_THINKING_BUDGET = 1024
PARSE_MAX_OUTPUT_TOKENS = 65536   # thinking counts here too; ~11k tokens of JSON for 50 MCQs
SOLVE_MAX_OUTPUT_TOKENS = 32768


def key_check_enabled() -> bool:
    return os.getenv("CONTEST_IMPORT_KEY_CHECK", "1") != "0"


class ImportRejected(ValueError):
    """The file itself is the problem (too big, not a PDF, no questions…): the message is for the teacher."""


# ═══════════════════════════════════════════════════════════════════════
# FILE CHECKS
# ═══════════════════════════════════════════════════════════════════════

def check_pdf(path: str) -> int:
    """Page count, or ImportRejected with a teacher-readable reason."""
    if os.path.getsize(path) > MAX_PDF_BYTES:
        raise ImportRejected(f"The PDF is too big (max {MAX_PDF_BYTES // (1024 * 1024)} MB).")
    with open(path, "rb") as f:
        if sniff(f.read(1024)) != "pdf":
            raise ImportRejected("This file is not a PDF.")
    try:
        doc = pymupdf.open(path, filetype="pdf")
    except Exception:
        raise ImportRejected("The PDF could not be opened (damaged file). Save it again and re-upload.")
    try:
        if doc.needs_pass:
            raise ImportRejected("The PDF is password-protected. Upload it without a password.")
        if doc.page_count == 0:
            raise ImportRejected("The PDF has no pages.")
        if doc.page_count > MAX_PAGES:
            raise ImportRejected(f"The PDF has {doc.page_count} pages; the limit is {MAX_PAGES}.")
        return doc.page_count
    finally:
        doc.close()


def _split(path: str, pages: int) -> list:
    """[(pdf path, first page, last page, is_temp)] — one entry when the paper fits in one call."""
    if pages <= CHUNK_PAGES:
        return [(path, 1, pages, False)]
    size = math.ceil(pages / math.ceil(pages / CHUNK_PAGES))
    src = pymupdf.open(path, filetype="pdf")
    out = []
    try:
        for a in range(0, pages, size):
            b = min(pages, a + size)
            part = pymupdf.open()
            part.insert_pdf(src, from_page=a, to_page=b - 1)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix="cimport_part_")
            tmp.close()
            part.save(tmp.name)
            part.close()
            out.append((tmp.name, a + 1, b, True))
    finally:
        src.close()
    return out


# ═══════════════════════════════════════════════════════════════════════
# PASS 1 — PARSE (Gemini transcribes; it does not map, fix or invent)
# ═══════════════════════════════════════════════════════════════════════

# google-genai 0.8.0 (production) rejects nullable / minimum / any_of / property_ordering in a schema, so missing
# values are "" or 0 and the code below turns them into None.
_S = {"type": "STRING"}
_I = {"type": "INTEGER"}
_OPTION = {"type": "OBJECT", "properties": {"label": _S, "text": _S}, "required": ["label", "text"]}
_KEY_ROW = {"type": "OBJECT", "properties": {"number": _I, "letter": _S, "text": _S},
            "required": ["number", "letter", "text"]}
PARSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "header": {"type": "OBJECT", "properties": {
            "institute": _S, "board": _S, "title": _S, "class_grade": _S, "subject": _S, "topic": _S,
            "duration_text": _S, "max_marks": _I},
            "required": ["institute", "board", "title", "class_grade", "subject", "topic", "duration_text",
                         "max_marks"]},
        "questions": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "number": _I,
            "section_level": {"type": "STRING", "enum": ["Easy", "Medium", "Hard", "Unknown"]},
            "type": {"type": "STRING", "enum": ["MCQ", "Short", "Long"]},
            "text": _S,
            "options": {"type": "ARRAY", "items": _OPTION},
            "marks": _I,
            "explanation": _S,
            "ai_answer": _S},
            "required": ["number", "section_level", "type", "text", "options", "marks", "explanation", "ai_answer"]}},
        "answer_key": {"type": "OBJECT", "properties": {
            "found": {"type": "BOOLEAN"},
            "table": {"type": "ARRAY", "items": _KEY_ROW},
            "list": {"type": "ARRAY", "items": _KEY_ROW}},
            "required": ["found", "table", "list"]},
    },
    "required": ["header", "questions", "answer_key"],
}


PARSE_PROMPT = """You get a school exam QUESTION PAPER as a PDF (it may be a scan). Transcribe it into the JSON schema.
You are a careful transcriber: copy, never rewrite, summarise, translate or correct.

HEADER (top of the first page): institute = the school / coaching name; board = the exam board ONLY if printed
(e.g. "CBSE", "ICSE", "Maharashtra State Board"), else ""; title = the paper's main heading line exactly
as printed (e.g. "Class XII — Economics"), not the institute name; class_grade = the class as printed ("XII", "10th");
subject; topic = chapter / topic line if printed; duration_text = the time allowed exactly as printed ("1 Hour",
"90 minutes"); max_marks = maximum marks as a number. Use "" / 0 for anything not printed.

QUESTIONS — every numbered question, in paper order:
- number = the printed question number (1, 2, 3 …).
- section_level = the level of the section the question is in ("SECTION A — Level: Easy" → Easy; Moderate →
  Medium; Difficult / Tough → Hard). Unknown when the paper has no levels.
- type = MCQ when it has options to choose from (including Assertion–Reason and "Consider the statements I / II"
  questions); Short or Long for written-answer questions (Long when the marks or the section say long answer).
- text = the full question stem without its number: keep every line, Assertion (A) / Reason (R) lines, statements
  I / II / III, "Which of the statements is/are correct?", tables written as text. Copy numbers, ₹, $, %, fractions,
  sub/superscripts (write H₂O, x², ½) exactly.
- options = every option with label = its printed letter ("a", "b", …) and text = the option text WITHOUT the
  "(a)" prefix. Options may be one per line, two per line in columns, or all four in one paragraph
  ("(a) … (b) … (c) … (d) …") — split them correctly in every layout. Short / Long questions: [].
- marks = marks printed for the question or for its section ("Q.1–10 · 1 mark each"); 0 if not printed.
- explanation = the explanation / solution printed in the PDF for this question, else "". Never write your own.
- ai_answer: ONLY if the PDF has NO answer key at all, the label of the option you believe is correct for each
  MCQ; otherwise "".

ANSWER KEY — usually at the END (sometimes a separate page). found = true if the PDF prints answers anywhere.
- table: rows of an answer table such as "Q | Ans | Correct option": number, letter as printed ("(b)"), text = the
  correct option's text if the table prints it, else "".
- list: every other form of printed answers — a grid "Q1 b  Q2 b …", "1-b", "1. (b)", "Ans: B" lines under the
  questions: number, letter, text "" (or the printed answer text for written-answer questions).
If the paper has both a grid and a table, give BOTH (the table in table, the grid in list). Copy letters as
printed; do not correct the key. Do not put answers you worked out yourself in the key.
"""

_CHUNK_NOTE = """
These are pages {a}–{b} of a {n}-page paper. Return only what is on these pages. If these pages start in the middle
of a question whose number is on an earlier page, return that part as a question with number = -1 (its text = the
rest of the stem, if any, and its options). header: "" / 0 when these pages have no header.
"""


PARSE_RESERVE_S = 40   # left after the parse for mapping, the key check decision and saving


def _parse_chunk(path: str, a: int, b: int, n: int, deadline: float) -> dict:
    """One chunk, the checker's retry rule (one retry: 429 → Retry-After capped at CHECKER_MAX_RETRY_WAIT_S, else
    2 s; then GeminiBusy / ValueError). Every call ends by deadline - PARSE_RESERVE_S: a slow model gets one long
    attempt instead of two that both time out."""
    prompt = PARSE_PROMPT + (_CHUNK_NOTE.format(a=a, b=b, n=n) if (a, b) != (1, n) else "")
    call_deadline = deadline - PARSE_RESERVE_S
    part, uploaded = gemini.pdf_part(path, call_deadline)
    try:
        last: Optional[Exception] = None
        for attempt in (1, 2):
            if call_deadline - time.monotonic() < 30:
                break
            try:
                with svc.call_settings(log_tag=f"import p{a}-{b}"):
                    return gemini.generate_json(
                        [part, {"text": prompt}], "IMPORT-PARSE", PARSE_SCHEMA, models=IMPORT_MODELS,
                        call_deadline=call_deadline, max_timeout=PARSE_TIMEOUT_S,
                        max_output_tokens=PARSE_MAX_OUTPUT_TOKENS, thinking_budget=PARSE_THINKING_BUDGET)
            except svc.GeminiBusy as e:
                last = e
                wait = 2.0
                if e.code == 429:
                    wait = min(e.retry_after if e.retry_after is not None else 5.0, svc.CHECKER_MAX_RETRY_WAIT_S)
            except ValueError as e:
                last, wait = e, 2.0
            if attempt == 1:
                time.sleep(wait)
        raise last or TimeoutError("no time left to read the paper")
    finally:
        gemini.delete_file(uploaded)


def _parse(pdf_path: str, pages: int, deadline: float, progress: Callable[[int], None]) -> dict:
    chunks = _split(pdf_path, pages)
    try:
        if len(chunks) == 1:
            return _parse_chunk(pdf_path, 1, pages, pages, deadline)
        parts: list = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
            futures = {pool.submit(contextvars.copy_context().run, _parse_chunk, p, a, b, pages, deadline): i
                       for i, (p, a, b, _) in enumerate(chunks)}
            done = 0
            for f, i in futures.items():
                parts[i] = f.result()
                done += 1
                progress(20 + 50 * done // len(chunks))
        return merge_parts(parts)
    finally:
        for p, _, _, is_temp in chunks:
            if is_temp:
                try:
                    os.unlink(p)
                except OSError:
                    pass


def merge_parts(parts: list) -> dict:
    """Chunk replies in page order → one reply. A question cut by a chunk boundary (number -1, or the same number at
    the end of one chunk and the start of the next) is joined to the question before it."""
    header: dict = {}
    questions: list = []
    key = {"found": False, "table": [], "list": []}
    for part in parts:
        for k, v in (part.get("header") or {}).items():
            if v and not header.get(k):
                header[k] = v
        for i, q in enumerate(part.get("questions") or []):
            num = _int(q.get("number"))
            continues = i == 0 and questions and (num == -1 or num == _int(questions[-1].get("number")))
            if continues:
                prev = questions[-1]
                extra = (q.get("text") or "").strip()
                if extra and extra not in (prev.get("text") or ""):
                    prev["text"] = f"{(prev.get('text') or '').rstrip()}\n{extra}"
                seen = {str(o.get("label")).strip().lower() for o in prev.get("options") or []}
                prev["options"] = list(prev.get("options") or []) + [
                    o for o in q.get("options") or [] if str(o.get("label")).strip().lower() not in seen]
                continue
            questions.append(dict(q))
        ak = part.get("answer_key") or {}
        key["found"] = key["found"] or bool(ak.get("found"))
        key["table"] += ak.get("table") or []
        key["list"] += ak.get("list") or []
    return {"header": header, "questions": questions, "answer_key": key}


# ═══════════════════════════════════════════════════════════════════════
# NORMALISATION HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _int(v) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"-?\d+", str(v or ""))
    return int(m.group()) if m else None


_LABEL_PREFIX = re.compile(r"^(?:ans(?:wer)?|option|opt|correct(?:\s+option)?)\b\s*[:.\-]?\s*", re.I)
_NUMBER_PREFIX = re.compile(r"^(?:Q\.?\s*)?\d+\s*[-.):]\s*(?=\S)", re.I)     # "1-b", "Q1. (b)"


def norm_letter(v) -> Optional[str]:
    """'b' / '(b)' / 'B)' / 'b.' / 'Ans: B' / 'Option (c)' / '1-b' / '1. (b)' / '2' → 'B' / 'C' … ;
    None when it is not a label."""
    s = _NUMBER_PREFIX.sub("", _LABEL_PREFIX.sub("", str(v or "").strip()))
    m = re.fullmatch(r"[\(\[]?\s*([A-Za-z]|[1-9])\s*[\)\]]?\.?", s)
    if not m:
        return None
    c = m.group(1)
    if c.isdigit():
        return string.ascii_uppercase[int(c) - 1]
    return c.upper()


_OPTION_PREFIX = re.compile(r"^\s*(?:\(\s*[A-Fa-f]\s*\)\s*|[A-Fa-f]\s*[.)]\s+)")


def _option_text(text) -> str:
    return _OPTION_PREFIX.sub("", str(text or "")).strip()


def _stem(text, num: Optional[int]) -> str:
    s = str(text or "").strip()
    if num is not None and num > 0:
        s = re.sub(rf"^(?:Q(?:uestion)?\s*\.?\s*)?{num}\s*[.):]\s*", "", s, flags=re.I).strip()
    return s


_LEVELS = {"easy": "Easy", "medium": "Medium", "moderate": "Medium", "average": "Medium",
           "hard": "Hard", "difficult": "Hard", "tough": "Hard", "challenging": "Hard"}


def _level(v) -> str:
    return _LEVELS.get(str(v or "").strip().lower(), "Medium")


def _qtype(v, n_options: int) -> str:
    t = str(v or "").strip().lower()
    if t in ("mcq", "multiple choice", "objective"):
        return "MCQ"
    if t in ("short", "long"):
        return t.capitalize()
    return "MCQ" if n_options >= 2 else "Short"


_ROMAN = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11,
          "XII": 12}


def norm_class(v) -> Optional[str]:
    """'Class XII' / 'XII' / '12th' / '12' → '12'; anything else as printed; '' → None."""
    s = str(v or "").strip()
    if not s:
        return None
    m = re.search(r"\b(XII|XI|IX|X|VIII|VII|VI|IV|V|III|II|I)\b", s.upper())
    if m:
        return str(_ROMAN[m.group(1)])
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", s, re.I)
    return m.group(1) if m else s


def duration_minutes(text) -> Optional[int]:
    """'1 Hour' → 60, '1½ hours' / '1.5 hrs' → 90, '2 hours 30 minutes' → 150, '90 min' → 90, '1:30' → 90."""
    t = str(text or "").lower()
    t = re.sub(r"\s*½", ".5", t).replace(" 1/2", ".5")
    t = re.sub(r"\b(one|two|three)\b", lambda m: {"one": "1", "two": "2", "three": "3"}[m.group(1)], t)
    m = re.search(r"\b(\d{1,2}):(\d{2})\b", t)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    h = re.search(r"(\d+(?:\.\d+)?)\s*(?:h\b|hrs?\b|hours?\b)", t)
    mins = re.search(r"(\d+)\s*(?:m\b|mins?\b|minutes?\b)", t)
    if not h and not mins:
        return None
    return round(float(h.group(1)) * 60 if h else 0) + (int(mins.group(1)) if mins else 0)


def _same_text(a, b) -> bool:
    ca, cb = _canon(a), _canon(b)
    return bool(ca) and bool(cb) and (ca == cb or ca in cb or cb in ca)


# ═══════════════════════════════════════════════════════════════════════
# BUILD + VALIDATE (pure — unit tested without Gemini)
# ═══════════════════════════════════════════════════════════════════════

def _options(raw: list, num: int, warn) -> list:
    opts = [{"label": norm_letter(o.get("label")), "text": _option_text(o.get("text"))}
            for o in raw or [] if isinstance(o, dict)]
    labels = [o["label"] for o in opts]
    if opts and (None in labels or len(set(labels)) != len(labels)):
        printed = ", ".join(str(o.get("label")) for o in raw if isinstance(o, dict))
        for i, o in enumerate(opts):
            o["label"] = string.ascii_uppercase[i]
        warn(num, "OPTION_LABELS_RENUMBERED",
             f"Q{num}: option labels were unclear ({printed}); they were numbered A, B, C… in order. Check them.")
    return opts


def _key_entries(answer_key: dict, warn) -> dict:
    """number → {"letter", "text"}. The table is the source of truth; the grid / list fills gaps and cross-checks."""
    key: dict = {}
    for row in answer_key.get("table") or []:
        n = _int(row.get("number"))
        if n is None or n <= 0:
            continue
        if n in key:
            warn(n, "KEY_DUPLICATE_ENTRY", f"The answer key lists Q{n} more than once; the first entry is used.")
            continue
        key[n] = {"letter": norm_letter(row.get("letter")), "text": str(row.get("text") or "").strip(),
                  "printed": str(row.get("letter") or "").strip()}
    in_table = set(key)
    for row in answer_key.get("list") or []:
        n = _int(row.get("number"))
        if n is None or n <= 0:
            continue
        letter = norm_letter(row.get("letter"))
        if n in in_table:
            if letter and key[n]["letter"] and letter != key[n]["letter"]:
                warn(n, "KEY_GRID_MISMATCH", f"Q{n}: the answer table says {key[n]['letter']} but the answer grid "
                                             f"says {letter}. The table was used — check which is right.")
            continue
        if n in key:
            if letter and key[n]["letter"] and letter != key[n]["letter"]:
                warn(n, "KEY_DUPLICATE_ENTRY", f"The answer key lists Q{n} more than once with different answers "
                                               f"({key[n]['letter']}, {letter}); the first entry is used.")
            continue
        key[n] = {"letter": letter, "text": str(row.get("text") or "").strip(),
                  "printed": str(row.get("letter") or "").strip()}
    return key


def build_result(raw: dict) -> dict:
    """Gemini's transcription → ImportResult dict (answers mapped, every problem a warning)."""
    warnings: list = []

    def warn(q, code, message):
        warnings.append({"question": q, "code": code, "message": message})

    header = raw.get("header") or {}
    raw_qs = [q for q in raw.get("questions") or [] if isinstance(q, dict)]
    if not raw_qs:
        raise ImportRejected("No questions were found in this PDF. Check that it is a question paper.")
    answer_key = raw.get("answer_key") or {}
    key = _key_entries(answer_key, warn)
    key_found = bool(answer_key.get("found")) or bool(key)
    chapter = str(header.get("topic") or "").strip() or None

    questions = []
    prev = 0
    for q in raw_qs:
        num = _int(q.get("number"))
        if num is None or num <= 0:
            num = prev + 1
            warn(num, "NUMBER_UNREADABLE", f"A question after Q{prev} has no readable number; it was "
                                           f"numbered {num}. Check its position.")
        prev = num
        opts = _options(q.get("options"), num, warn)
        qtype = _qtype(q.get("type"), len(opts))
        if qtype != "MCQ":
            opts = []
        text = _stem(q.get("text"), num)
        usable = True
        if not text:
            usable = False
            warn(num, "EMPTY_QUESTION_TEXT", f"Q{num}: the question text could not be read.")
        if qtype == "MCQ":
            if len(opts) < 2:
                usable = False
                warn(num, "TOO_FEW_OPTIONS", f"Q{num}: only {len(opts)} option(s) found. Add the options or remove "
                                             f"the question.")
            elif len(opts) > 6:
                usable = False
                warn(num, "TOO_MANY_OPTIONS", f"Q{num}: {len(opts)} options found (max 6). Check the options.")
            elif len(opts) > 4:
                warn(num, "TOO_MANY_OPTIONS", f"Q{num}: {len(opts)} options. The contest screen is made for "
                                              f"4 (A–D); check that options E/F show correctly.")

        labels = [o["label"] for o in opts]
        correct, source = None, "missing"
        entry = key.get(num)
        if qtype == "MCQ":
            if key_found:
                if entry is None:
                    warn(num, "ANSWER_MISSING", f"Q{num}: the answer key has no entry for this question.")
                else:
                    letter = entry["letter"]
                    if not letter and entry["text"]:
                        letter = next((o["label"] for o in opts if _same_text(o["text"], entry["text"])), None)
                    if not letter or letter not in labels:
                        warn(num, "KEY_LETTER_NOT_IN_OPTIONS",
                             f"Q{num}: the answer key says '{entry['printed'] or entry['text']}', which is not one "
                             f"of this question's options ({', '.join(labels) or 'none'}). Set the answer by hand.")
                    else:
                        correct, source = letter, "key"
                        if entry["text"]:
                            mine = next(o["text"] for o in opts if o["label"] == letter)
                            other = next((o["label"] for o in opts if o["label"] != letter
                                          and _canon(o["text"]) == _canon(entry["text"])), None)
                            if other and not _same_text(mine, entry["text"]):
                                warn(num, "KEY_TEXT_MISMATCH",
                                     f"Q{num}: the key says ({letter.lower()}) but its answer text matches option "
                                     f"({other.lower()}). The letter was used — check which is right.")
            else:
                ai = norm_letter(q.get("ai_answer"))
                if ai in labels:
                    correct, source = ai, "ai"
                    warn(num, "AI_ANSWER", f"Q{num}: the PDF has no answer key, so the answer ({ai}) was worked out "
                                           f"by AI. Check it before publishing.")
                else:
                    warn(num, "ANSWER_MISSING", f"Q{num}: the PDF has no answer key and AI could not answer it.")
        else:
            text_answer = (entry or {}).get("text") or None
            if text_answer:
                correct, source = text_answer, "key"
            else:
                warn(num, "ANSWER_MISSING", f"Q{num}: no model answer for this written question in the PDF.")

        questions.append({
            "question_text": text,
            "question_type": qtype,
            "options": opts,
            "correct_answer": correct,
            "explanation": str(q.get("explanation") or "").strip() or None,
            "marks": max(1, _int(q.get("marks")) or 1),
            "difficulty": _level(q.get("section_level")),
            "chapter": chapter,
            "source_number": num,
            "answer_source": source,
            "usable": usable,
        })

    numbers = [q["source_number"] for q in questions]
    for n in sorted({n for n in numbers if numbers.count(n) > 1}):
        warn(n, "DUPLICATE_QUESTION_NUMBER", f"Q{n} appears {numbers.count(n)} times in the paper.")
    for n in sorted(set(range(1, max(numbers) + 1)) - set(numbers)):
        warn(n, "MISSING_QUESTION_NUMBER", f"Q{n} was not found in the paper.")

    if key_found and len(key) != len(questions):
        warn(None, "KEY_COUNT_MISMATCH", f"The answer key has {len(key)} answers but {len(questions)} questions "
                                         f"were found.")
    for n in sorted(set(key) - set(numbers)):
        warn(n, "KEY_WITHOUT_QUESTION", f"The answer key has an answer for Q{n}, but there is no Q{n} in the paper.")

    total = sum(q["marks"] for q in questions)
    stated = _int(header.get("max_marks")) or 0
    if stated > 0 and stated != total:
        warn(None, "MARKS_TOTAL_MISMATCH", f"The questions add up to {total} marks, the paper says {stated}.")

    duration = duration_minutes(header.get("duration_text"))
    if duration is not None and not DURATION_MIN <= duration <= DURATION_MAX:
        clamped = min(DURATION_MAX, max(DURATION_MIN, duration))
        warn(None, "DURATION_CLAMPED", f"The paper's duration ({duration} min) is outside what a contest allows "
                                       f"({DURATION_MIN}–{DURATION_MAX} min); it was set to {clamped} min.")
        duration = clamped

    subject = str(header.get("subject") or "").strip() or None
    title = (str(header.get("title") or "").strip()
             or " — ".join(x for x in (subject, chapter) if x) or "Imported paper")
    return {
        "title": title,
        "subject": subject,
        "class_grade": norm_class(header.get("class_grade")),
        "institute": str(header.get("institute") or "").strip() or None,
        "board": str(header.get("board") or "").strip() or None,
        "duration_minutes": duration,
        "total_marks": total,
        "answer_key_found": key_found,
        "answer_key_count": len(key),
        "questions": questions,
        "warnings": warnings,
    }


# ═══════════════════════════════════════════════════════════════════════
# PASS 2 — KEY CHECK (cheap, text only, never changes the key)
# ═══════════════════════════════════════════════════════════════════════

SOLVE_SCHEMA = {"type": "OBJECT", "properties": {"answers": {"type": "ARRAY", "items": {
    "type": "OBJECT", "properties": {"number": _I, "label": _S}, "required": ["number", "label"]}}},
    "required": ["answers"]}
SOLVE_BATCH = 20       # one 50-MCQ call took > 60 s on an overloaded model; batches run in parallel

SOLVE_PROMPT = """Solve these multiple-choice questions from a school exam ({subject}). For each question return its
number and the label of the single best option. If a question cannot be answered without a figure or table that is
not given here, return label "". Questions (JSON):
{questions}
"""


def _solve_batch(batch: list, subject: str, call_deadline: float) -> dict:
    payload = [{"number": q["source_number"], "question": q["question_text"],
                "options": {o["label"]: o["text"] for o in q["options"]}} for q in batch]
    prompt = SOLVE_PROMPT.format(subject=subject, questions=json.dumps(payload, ensure_ascii=False))
    with svc.call_settings(log_tag=f"import key-check Q{batch[0]['source_number']}+"):
        data = gemini.generate_json([{"text": prompt}], "IMPORT-SOLVE", SOLVE_SCHEMA, models=IMPORT_MODELS,
                                    call_deadline=call_deadline, max_timeout=SOLVE_TIMEOUT_S,
                                    max_output_tokens=SOLVE_MAX_OUTPUT_TOKENS, thinking_budget=SOLVE_THINKING_BUDGET)
    ai = {}
    for a in data.get("answers") or []:
        n = _int(a.get("number")) if isinstance(a, dict) else None
        if n is not None:
            ai.setdefault(n, norm_letter(a.get("label")))
    return ai


def check_key(result: dict, deadline: float) -> None:
    """Adds KEY_MISMATCH_SUSPECT / KEY_CHECK_SKIPPED warnings to result; never raises, never changes an answer."""
    mcqs = [q for q in result["questions"]
            if q["question_type"] == "MCQ" and q["answer_source"] == "key" and q["usable"]]
    if not mcqs:
        return

    def skipped(why: str) -> None:
        result["warnings"].append({"question": None, "code": "KEY_CHECK_SKIPPED",
                                   "message": f"The answer key was not double-checked by AI ({why})."})

    left = deadline - time.monotonic()
    if left <= KEY_CHECK_MIN_LEFT_S:
        skipped("not enough time left")
        return
    call_deadline = deadline - 30
    subject = result.get("subject") or "school subject"
    batches = [mcqs[i:i + SOLVE_BATCH] for i in range(0, len(mcqs), SOLVE_BATCH)]
    ai: dict = {}
    failed: list = []
    with ThreadPoolExecutor(max_workers=len(batches)) as pool:
        futures = [pool.submit(contextvars.copy_context().run, _solve_batch, b, subject, call_deadline) for b in batches]
        for batch, f in zip(batches, futures):
            try:
                ai.update(f.result())
            except Exception as e:
                logger.warning(f"[import] key check batch Q{batch[0]['source_number']}+ skipped: {type(e).__name__}")
                failed.append(batch)
    if len(failed) == len(batches):
        skipped("AI busy or too slow")
        return
    if failed:
        nums = ", ".join(f"Q{b[0]['source_number']}–Q{b[-1]['source_number']}" for b in failed)
        skipped(f"AI busy or too slow for {nums}")
    for q in mcqs:
        mine = ai.get(q["source_number"])
        if mine and mine != q["correct_answer"] and mine in {o["label"] for o in q["options"]}:
            result["warnings"].append({
                "question": q["source_number"], "code": "KEY_MISMATCH_SUSPECT",
                "message": f"Q{q['source_number']}: the answer key says {q['correct_answer']}, an independent AI "
                           f"solve says {mine}. The key was kept — check this question."})


# ═══════════════════════════════════════════════════════════════════════
# ENTRY
# ═══════════════════════════════════════════════════════════════════════

def import_paper(pdf_path: str, pages: int, deadline: float, progress: Callable[[int], None] = lambda p: None) -> dict:
    """The whole import for one checked PDF. Raises ImportRejected / GeminiBusy / ValueError (unreadable reply)."""
    start = time.monotonic()
    with svc.usage_meter(cap=CALL_CAP) as meter:
        progress(20)
        raw = _parse(pdf_path, pages, deadline, progress)
        progress(70)
        result = build_result(raw)
        progress(80)
        if key_check_enabled():
            check_key(result, deadline)
        else:
            result["warnings"].append({"question": None, "code": "KEY_CHECK_SKIPPED",
                                       "message": "The answer key was not double-checked by AI (turned off)."})
        progress(95)
    logger.info(f"[import] {pages} pages → {len(result['questions'])} questions in {time.monotonic() - start:.1f}s; "
                f"usage {meter.as_dict()}")
    return result
