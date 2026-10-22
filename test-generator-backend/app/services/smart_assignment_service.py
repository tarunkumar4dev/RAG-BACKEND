"""
Smart Assignment Checker Service — v2.0

2-pass pipeline:
  Pass 1 — RUBRIC (question paper + answer key, one call):
            Every question / sub-part with its marks, and the answer key's MARKING POINTS
            ("definition [2] · law 1 [1] · law 2 [1]"). Points always add up to the question's marks.
            The rubric does not depend on the student, so it can be built once and reused for a
            whole class (build_rubric → grade_sheet).

  Pass 2 — GRADE (student sheet + rubric):
            a) Identity check: name, class/section, roll number must be on the sheet
               (skipped when the caller already knows the student, e.g. a portal submission).
            b) For every marking point the model gives a verdict — full / mostly / partial / none —
               and quotes what the student wrote as evidence.
            c) The SERVER turns verdicts into marks. The model never does arithmetic.

Key rules enforced SERVER-SIDE (not trusted from the LLM):
  - points of a question always sum to the question's marks (rescaled if the key disagrees)
  - marks per point come from verdict × point marks, on a 0.5 grid
  - OR-alternatives: only the best-scoring alternative of a group counts
  - total <= paper total; everything >= 0
  - low confidence / illegible / estimated scheme / answer not found → needs_review
"""

import json
import logging
import math
import re
import time
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from app.core.config import settings

logger = logging.getLogger(__name__)

CHECKER_API_KEY = os.getenv("CHECKER_GEMINI_API_KEY")
# gemini-3.6-flash retires 19 Nov 2026. Production also sets CHECKER_GEMINI_MODEL explicitly.
CHECKER_MODEL = os.getenv("CHECKER_GEMINI_MODEL", "gemini-3.8-flash")

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        key = CHECKER_API_KEY or settings.GEMINI_API_KEY
        if not key:
            raise ValueError("No Gemini API key. Set CHECKER_GEMINI_API_KEY in .env")
        _client = genai.Client(api_key=key)
    return _client


# Tried in order when the primary model is overloaded (503) or out of quota (429)
CHECKER_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("CHECKER_GEMINI_FALLBACK_MODELS", "gemini-2.5-flash,gemini-3-flash-preview").split(",")
    if m.strip() and m.strip() != CHECKER_MODEL
]
_OVERLOAD_CODES = {500, 503, 504}
# A fallback that no longer exists for this key (retired / access restricted) must not end the chain —
# gemini-2.5-flash is limited to earlier users since Sept 2026 and has returned 404 for some.
_MODEL_GONE_CODES = {404}
# Give up on a single generate call after this long (Gemini can take 20-50s just to return a 503)
_GENERATE_DEADLINE_S = float(os.getenv("CHECKER_GEMINI_DEADLINE_S", "120"))

# Last model that answered successfully (only used to log when a fallback kicks in)
_preferred_model: Optional[str] = None


def _log_usage(tag: str, model: str, response, seconds: float) -> None:
    """One line per Gemini call: "[checker] grade tokens: input=… output=… thinking=…".
    Production installs google-genai 0.8.0 (api/requirements.txt), whose usage_metadata has no
    thoughts_token_count — thinking is then total - input - output."""
    um = getattr(response, "usage_metadata", None)
    if um is None:
        return
    inp = getattr(um, "prompt_token_count", 0) or 0
    out = getattr(um, "candidates_token_count", 0) or 0
    total = getattr(um, "total_token_count", 0) or 0
    thinking = getattr(um, "thoughts_token_count", None)
    if thinking is None:
        thinking = max(total - inp - out, 0)
    candidates = getattr(response, "candidates", None) or []
    finish = getattr(candidates[0], "finish_reason", None) if candidates else None
    logger.info(f"[checker] {tag.lower()} tokens: input={inp} output={out} thinking={thinking} "
                f"model={model} time={seconds:.1f}s finish={finish}")


def _generate(client: genai.Client, contents: list, tag: str):
    """
    generate_content with resilience against transient Gemini outages.
    Each round tries every model once (primary model first):
      5xx / 429 → move straight to the next model.
      404 (model not available) → next model, and never retried in later rounds.
    If a whole round fails with 5xx, wait briefly and do another round until the deadline.
    Any other error is raised as-is.
    """
    global _preferred_model
    # Always start with the primary model; fallbacks only when it is down
    models = [CHECKER_MODEL, *CHECKER_FALLBACK_MODELS]

    deadline = time.time() + _GENERATE_DEADLINE_S
    last_error: Optional[Exception] = None
    round_no = 0
    gone: set = set()
    while True:
        round_no += 1
        all_quota = True
        for model in [m for m in models if m not in gone]:
            if time.time() > deadline:
                break
            try:
                t_call = time.time()
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    # temperature 0 + fixed seed: the same sheet should get the same marks every time
                    config={"response_mime_type": "application/json", "temperature": 0, "seed": 7},
                )
                _log_usage(tag, model, response, time.time() - t_call)
                if model != _preferred_model:
                    logger.info(f"[{tag}] Using {model}")
                _preferred_model = model
                return response
            except genai_errors.APIError as e:
                last_error = e
                if e.code in _MODEL_GONE_CODES:
                    logger.warning(f"[{tag}] {model} not available ({e.code}), skipping it")
                    gone.add(model)
                    continue
                if e.code not in _OVERLOAD_CODES and e.code != 429:
                    raise
                all_quota = all_quota and e.code == 429
                logger.warning(f"[{tag}] {model} unavailable ({e.code}), trying next model")

        # Every model is out of quota / gone, or we've run out of time → stop
        if all_quota or len(gone) == len(models) or time.time() + 5 > deadline:
            break
        logger.warning(f"[{tag}] All models busy (round {round_no}), retrying in 5s")
        time.sleep(5)

    raise last_error


def _generate_json(client: genai.Client, contents: list, tag: str, max_retries: int) -> dict:
    """_generate + json.loads, retrying on unreadable JSON or transient errors."""
    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = _generate(client, contents, tag)
            data = json.loads(response.text)
            if not isinstance(data, dict):
                raise ValueError("model returned JSON that is not an object")
            return data
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[{tag}] Attempt {attempt} unreadable reply: {e}")
            last_error = e
        except genai_errors.APIError as e:
            logger.warning(f"[{tag}] Attempt {attempt} API error: {e}")
            last_error = e
        if attempt < max_retries:
            time.sleep(2 * attempt)
    raise ValueError(f"{tag} failed: {last_error}")


# ═══════════════════════════════════════════════════════════════════════
# GEMINI FILES — upload in parallel, always delete afterwards
# (student sheets are personal data; Gemini would otherwise keep them for 48 h)
# ═══════════════════════════════════════════════════════════════════════

def upload_files(paths: list) -> list:
    client = _get_client()
    with ThreadPoolExecutor(max_workers=max(1, len(paths))) as pool:
        futures = [pool.submit(client.files.upload, file=p) for p in paths]
        uploaded, errors = [], []
        for f in futures:
            try:
                uploaded.append(f.result())
            except Exception as e:
                errors.append(e)
    if errors:
        delete_files(uploaded)
        raise errors[0]
    return uploaded


def delete_files(files: list) -> None:
    client = _get_client()
    for f in files:
        try:
            if f is not None:
                client.files.delete(name=f.name)
        except Exception as e:
            logger.warning(f"Could not delete Gemini file {getattr(f, 'name', '?')}: {e}")


# ═══════════════════════════════════════════════════════════════════════
# PASS 1 — RUBRIC (question paper + answer key)
# ═══════════════════════════════════════════════════════════════════════

RUBRIC_PROMPT = """You get two documents: FIRST the exam QUESTION PAPER, SECOND its ANSWER KEY / MARKING SCHEME.
Build the marking rubric a strict, fair teacher would use.

STEP A — QUESTION PAPER STRUCTURE
1. Extract EVERY question (scan every page and section). Keep the paper's numbering in q_no ("1", "2", "14").
2. Sub-parts (a, b, c / i, ii) are listed separately in "subparts" with sub_id = q_no + label ("5a", "3ii").
3. Marks: read [3], (3), 3M, "(1 mark each)", "Q6–Q10 carry 3 marks each", "5 × 2 = 10". Apply section-wise
   marks to every question in the range. A sub-part without its own marks: split the parent's marks equally.
4. Internal choice: for "Q12 ... OR ..." list each alternative as its own entry with the SAME or_group
   (e.g. "12"), q_no "12" and "12-OR". For "attempt any 4 of 6" give all 6 the same or_group and
   set "attempt_count": 4 on each. Otherwise or_group = null.
5. stated_max_marks = the maximum marks printed on the paper ("Max. Marks: 80", "M.M. 50"), else null.
6. section = the section the question is printed under, as its letter or short name ("A", "B", "C"),
   null when the paper has no sections. Students may write "C-4" for the 4th question of Section C.

STEP B — MARKING POINTS from the ANSWER KEY
For every question WITHOUT sub-parts, and for every SUB-PART, list "points": the separate things the
key gives marks for. Each point has its own marks, and the points MUST add up exactly to that
question's / sub-part's marks.
- If the key splits marks ("definition [2], law 1 [1]", "½ mark each", "formula 1 + substitution 1 +
  answer 1"), copy that split exactly and set scheme_source = "key".
- If the key gives only an answer without a split, divide the marks into the natural parts a CBSE
  examiner would check (each fact, step, term, label), and set scheme_source = "estimated".
- MCQ / one-word / fill-in-the-blank: one point = the correct option/word.
- Numericals: method / formula, substitution, final answer with unit — as separate points.
- Diagrams: drawing + each required label as points.
- "expected": what the student must write for this point, in the key's words.
- Do NOT invent answers that contradict the key. If the key is missing an answer, write the standard
  textbook answer and set scheme_source = "estimated".

Respond ONLY with JSON (no markdown):
{
  "questions": [
    {
      "q_no": "4",
      "section": "A",
      "question_text": "What is refraction of light? State the two laws of refraction.",
      "marks": 4,
      "answer_type": "descriptive",
      "or_group": null,
      "attempt_count": null,
      "scheme_source": "key",
      "model_answer": "Bending of light ... ; laws ...",
      "points": [
        {"id": "4.1", "text": "Definition of refraction", "expected": "bending of light when it passes obliquely from one medium to another", "marks": 2},
        {"id": "4.2", "text": "First law", "expected": "incident ray, refracted ray and normal lie in the same plane", "marks": 1},
        {"id": "4.3", "text": "Second law (Snell's law)", "expected": "sin i / sin r = constant for a given pair of media", "marks": 1}
      ],
      "subparts": []
    },
    {
      "q_no": "5", "section": "A", "question_text": "Write the function of ...", "marks": 2, "answer_type": "short",
      "or_group": null, "attempt_count": null, "scheme_source": "key", "model_answer": "See sub-parts", "points": [],
      "subparts": [
        {"sub_id": "5a", "text": "haemoglobin", "marks": 1, "answer_type": "short", "scheme_source": "key",
         "model_answer": "carries oxygen", "points": [{"id": "5a.1", "text": "Function", "expected": "carries oxygen in blood", "marks": 1}]},
        {"sub_id": "5b", "text": "xylem", "marks": 1, "answer_type": "short", "scheme_source": "key",
         "model_answer": "transports water", "points": [{"id": "5b.1", "text": "Function", "expected": "transports water and minerals", "marks": 1}]}
      ]
    }
  ],
  "stated_max_marks": 15,
  "paper_notes": "anything unusual (missing key answers, unclear marks)"
}
answer_type: "mcq" | "one_word" | "definition" | "short" | "descriptive" | "numerical" | "diagram"
Point ids: "<q_no or sub_id>.<n>". Marks are numbers (0.5 allowed), never null.
"""


def _num(v, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) and x >= 0 else default
    except (TypeError, ValueError):
        return default


def _half_floor(x: float) -> float:
    """Round DOWN to the 0.5 grid (Indian board marking)."""
    return math.floor(x * 2 + 1e-9) / 2


def _normalize_points(item: dict, marks: float, prefix: str) -> None:
    """Make item['points'] a clean list that sums exactly to `marks`."""
    points = [p for p in (item.get("points") or []) if isinstance(p, dict)]
    if not points:
        item["points"] = [{
            "id": f"{prefix}.1", "text": "Answer",
            "expected": item.get("model_answer") or item.get("question_text") or item.get("text") or "",
            "marks": marks,
        }]
        item["scheme_source"] = "estimated"
        return

    for i, p in enumerate(points, 1):
        p["id"] = str(p.get("id") or f"{prefix}.{i}")
        p["marks"] = _num(p.get("marks"))
        p["text"] = str(p.get("text") or "")
        p["expected"] = str(p.get("expected") or "")

    total = sum(p["marks"] for p in points)
    if abs(total - marks) > 0.01:
        # Key and paper disagree → scale the points to the paper's marks (paper is the truth)
        logger.info(f"[RUBRIC] {prefix}: points sum {total:g} != marks {marks:g}, rescaling")
        if total <= 0:
            for p in points:
                p["marks"] = marks / len(points)
        else:
            for p in points:
                p["marks"] = p["marks"] * marks / total
        # Keep a 0.5 grid where possible; put the rounding remainder on the largest point
        for p in points:
            p["marks"] = _half_floor(p["marks"])
        rest = round(marks - sum(p["marks"] for p in points), 2)
        max(points, key=lambda p: p["marks"])["marks"] += rest
        item["scheme_source"] = "adjusted"
    item["points"] = points


def _counted_total(questions: list) -> float:
    """Paper total: every question once, OR-groups count only the marks a student can actually earn."""
    total = 0.0
    groups: dict = {}
    for q in questions:
        g = q.get("or_group")
        if g:
            groups.setdefault(str(g), []).append(q)
        else:
            total += q["marks"]
    for members in groups.values():
        need = int(_num(members[0].get("attempt_count"), 0)) or 1
        best = sorted((m["marks"] for m in members), reverse=True)[:need]
        total += sum(best)
    return total


def _validate_rubric(rubric: dict) -> dict:
    questions = [q for q in (rubric.get("questions") or []) if isinstance(q, dict)]
    for q in questions:
        q["q_no"] = str(q.get("q_no") or "").strip() or "?"
        q["marks"] = _num(q.get("marks"), 1.0)
        sec = str(q.get("section") or "").strip()
        q["section"] = re.sub(r"(?i)^section\s*", "", sec).upper() or None
        q["or_group"] = str(q["or_group"]) if q.get("or_group") not in (None, "", "null") else None
        subs = [s for s in (q.get("subparts") or []) if isinstance(s, dict)]
        if subs:
            for i, s in enumerate(subs, 1):
                s["sub_id"] = str(s.get("sub_id") or f"{q['q_no']}{chr(96 + i)}")
                s["marks"] = _num(s.get("marks"))
            if sum(s["marks"] for s in subs) <= 0:
                for s in subs:
                    s["marks"] = q["marks"] / len(subs)
            # Sub-part marks are what students are graded on → parent = their sum
            sub_sum = sum(s["marks"] for s in subs)
            if abs(sub_sum - q["marks"]) > 0.01:
                logger.info(f"[RUBRIC] Q{q['q_no']}: sub-parts sum {sub_sum:g} != {q['marks']:g}, using sub-parts")
                q["marks"] = sub_sum
            for s in subs:
                _normalize_points(s, s["marks"], s["sub_id"])
            q["points"] = []
        else:
            _normalize_points(q, q["marks"], q["q_no"])
        q["subparts"] = subs
        q["scheme_source"] = q.get("scheme_source") or "key"

    rubric["questions"] = questions
    rubric["stated_max_marks"] = _num(rubric.get("stated_max_marks")) or None
    rubric["_computed_total"] = _counted_total(questions)
    rubric["total_marks"] = rubric["stated_max_marks"] or rubric["_computed_total"]
    rubric["total_questions"] = len({q["q_no"].replace("-OR", "") for q in questions})
    return rubric


_NO_KEY_NOTE = """

NO ANSWER KEY WAS PROVIDED — only the question paper. Write the standard CBSE/NCERT answer for every
question yourself, split it into marking points as an examiner would, and set scheme_source = "estimated"
on EVERY question and sub-part."""

_MULTI_PAGE_NOTE = """
The question paper is given as {qp} file(s) and the answer key as {key} file(s), in page order:
first all question-paper pages, then all answer-key pages."""


def _build_rubric_from_files(qp_files, key_files, max_retries: int = 2) -> dict:
    """qp_files / key_files: one Gemini file or a list (multi-page photos). key_files may be empty."""
    qp_files = qp_files if isinstance(qp_files, list) else [qp_files]
    key_files = key_files if isinstance(key_files, list) else ([key_files] if key_files else [])
    client = _get_client()
    base = RUBRIC_PROMPT + _MULTI_PAGE_NOTE.format(qp=len(qp_files), key=len(key_files))
    if not key_files:
        base += _NO_KEY_NOTE
    prompt = base
    rubric: dict = {}
    for attempt in range(1, max_retries + 1):
        rubric = _validate_rubric(_generate_json(client, [*qp_files, *key_files, prompt], "RUBRIC", max_retries))
        if not key_files:
            for q in rubric["questions"]:
                q["scheme_source"] = "estimated"
                for s in q["subparts"]:
                    s["scheme_source"] = "estimated"
        stated, computed = rubric["stated_max_marks"], rubric["_computed_total"]
        if not stated or abs(stated - computed) <= 0.01 or attempt == max_retries:
            break
        # Printed max marks is the truth — questions or section-wise marks were missed. Retry once.
        found = ", ".join(f"Q{q['q_no']}={q['marks']:g}" for q in rubric["questions"])
        logger.warning(f"[RUBRIC] marks sum {computed:g} != stated {stated:g} ({found}), retrying")
        prompt = (base +
                  f"\n\nCORRECTION: a previous reading found only {found} (sum {computed:g}), but the paper "
                  f"states Maximum Marks = {stated:g}. Re-read every page and section and return the "
                  f"complete structure.")

    stated, computed = rubric["stated_max_marks"], rubric["_computed_total"]
    if stated and abs(stated - computed) > 0.01:
        rubric["marks_mismatch"] = f"Questions add up to {computed:g}, paper says {stated:g}"
        rubric["total_marks"] = computed  # grade against what we can actually see
    return rubric


def build_rubric(question_paper_path, answer_key_path=None, max_retries: int = 2) -> dict:
    """Pass 1 on its own: rubric for a paper + key (reusable for every student of the class).
    Each argument is a path or a list of paths (multi-page photos); the key may be None / []."""
    qp = question_paper_path if isinstance(question_paper_path, list) else [question_paper_path]
    key = answer_key_path if isinstance(answer_key_path, list) else ([answer_key_path] if answer_key_path else [])
    start = time.time()
    files = upload_files([*qp, *key])
    try:
        rubric = _build_rubric_from_files(files[:len(qp)], files[len(qp):], max_retries)
    finally:
        delete_files(files)
    rubric["_rubric_time_s"] = round(time.time() - start, 2)
    return rubric


import re as _re

_STEP_SPLIT = _re.compile(r"(?:^|\s)(?:Step\s*\d+\s*[:.)-]|\(\s*[a-h]\s*\)|\(\s*(?:i|ii|iii|iv|v|vi)\s*\))\s*", _re.I)


def _split_answer(answer: str) -> list:
    """'Step 1: … Step 2: …' / '(a) … (b) …' → parts; otherwise sentences."""
    text = (answer or "").strip()
    parts = [p.strip(" ;.") for p in _STEP_SPLIT.split(text) if p and p.strip(" ;.")]
    if len(parts) >= 2:
        return parts, True
    sentences = [s.strip() for s in _re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3]
    return (sentences or [text]), False


def rubric_from_test_fast(test: dict, questions: list) -> dict:
    """Rubric for a saved a4ai test WITHOUT any AI call (instant, free, deterministic).
    Generated answers are already structured ("Step 1: … Step 2: …", "(a) … (b) …"); each question's
    DB marks are spread over those parts on a 0.5 grid. MCQs are one point."""
    start = time.time()
    qs = sorted(questions, key=lambda q: (q.get("position") or 0))
    out, prev_q = [], None
    for i, q in enumerate(qs, 1):
        q_no = str(q.get("position") or i)
        marks = _num(q.get("marks"), 1.0)
        fmt = (q.get("format") or "").lower()
        answer = (q.get("correct_answer") or q.get("model_answer") or "").strip()
        if "mcq" in fmt or "true_false" in fmt or "fill" in fmt:
            atype = "mcq" if "mcq" in fmt or "true" in fmt else "one_word"
            parts, structured = [answer], True
        else:
            atype = "descriptive" if marks >= 4 else "short"
            parts, structured = _split_answer(answer)
        # Not more points than half-marks available; merge the tail into the last point
        max_pts = max(1, int(marks * 2))
        if len(parts) > max_pts:
            parts = parts[:max_pts - 1] + [" ".join(parts[max_pts - 1:])]
        each = _half_floor(marks / len(parts)) or 0.5
        pts = [{"id": f"{q_no}.{n}", "text": f"Point {n}" if len(parts) > 1 else "Answer",
                "expected": p[:600], "marks": each} for n, p in enumerate(parts, 1)]
        pts[-1]["marks"] = round(marks - each * (len(parts) - 1), 2)   # remainder → last point
        if pts[-1]["marks"] <= 0:                                        # tiny marks, many parts
            pts = [{"id": f"{q_no}.1", "text": "Answer", "expected": answer[:1200], "marks": marks}]
        item = {"q_no": q_no, "question_text": (q.get("text") or "")[:1500], "marks": marks,
                "answer_type": atype, "or_group": None, "attempt_count": None,
                "scheme_source": "key" if structured else "estimated",
                "model_answer": answer[:3000], "points": pts, "subparts": []}
        if q.get("is_or") and prev_q is not None:     # OR-alternative of the previous question
            group = prev_q["or_group"] or prev_q["q_no"]
            prev_q["or_group"] = item["or_group"] = group
        out.append(item)
        prev_q = item
    rubric = _validate_rubric({"questions": out,
                               "stated_max_marks": test.get("total_marks") or None,
                               "paper_notes": "Built from the saved test (no AI)."})
    if rubric["stated_max_marks"] and abs(rubric["stated_max_marks"] - rubric["_computed_total"]) > 0.01:
        rubric["marks_mismatch"] = (f"Questions add up to {rubric['_computed_total']:g}, "
                                    f"test says {rubric['stated_max_marks']:g}")
        rubric["total_marks"] = rubric["_computed_total"]
    rubric["_rubric_time_s"] = round(time.time() - start, 3)
    return rubric


def build_rubric_from_test(test: dict, questions: list, max_retries: int = 2) -> dict:
    """AI version of rubric_from_test_fast (better splits for unstructured answers, but one slow call
    — can exceed serverless time limits). Kept for offline use / comparison."""
    start = time.time()
    qs = sorted(questions, key=lambda q: (q.get("position") or 0))
    paper = []
    for i, q in enumerate(qs, 1):
        item = {
            "q_no": str(q.get("position") or i),
            "question_text": (q.get("text") or "")[:1500],
            "marks": q.get("marks") or 1,
            "format": q.get("format") or "",
            "options": q.get("options"),
            "answer": (q.get("correct_answer") or q.get("model_answer") or "")[:3000],
        }
        if q.get("marking_scheme"):
            item["marking_scheme"] = q["marking_scheme"]
        if q.get("sub_parts"):
            item["sub_parts"] = q["sub_parts"]
        if q.get("is_or"):
            item["is_or_alternative"] = True
        paper.append(item)

    total = sum(_num(q.get("marks"), 1.0) for q in qs)
    doc = json.dumps({"exam_title": test.get("exam_title"), "subject": test.get("subject"),
                      "class": test.get("class_grade"), "max_marks": test.get("total_marks") or total,
                      "questions": paper}, ensure_ascii=False)
    prompt = (RUBRIC_PROMPT.replace(
        "You get two documents: FIRST the exam QUESTION PAPER, SECOND its ANSWER KEY / MARKING SCHEME.",
        "You get the teacher's saved test as JSON: every question with its marks and its answer "
        "(\"answer\", sometimes with a \"marking_scheme\"). Treat \"answer\" as the ANSWER KEY.")
        + "\n\nKeep q_no and marks EXACTLY as given. Use the paper's max_marks as stated_max_marks. "
          "Questions marked is_or_alternative are the OR-choice of the question before them (same or_group).")
    client = _get_client()
    rubric = _validate_rubric(_generate_json(client, [doc, prompt], "RUBRIC-TEST", max_retries))

    # The saved marks are the truth: force each question back to its DB marks
    db_marks = {str(q.get("position") or i): _num(q.get("marks"), 1.0) for i, q in enumerate(qs, 1)}
    for q in rubric["questions"]:
        m = db_marks.get(q["q_no"].replace("-OR", ""))
        if m and abs(m - q["marks"]) > 0.01:
            q["marks"] = m
            if q["subparts"]:
                for s in q["subparts"]:
                    s["marks"] = m / len(q["subparts"])
                    _normalize_points(s, s["marks"], s["sub_id"])
            else:
                _normalize_points(q, m, q["q_no"])
    rubric = _validate_rubric(rubric)
    rubric["_rubric_time_s"] = round(time.time() - start, 2)
    return rubric


# ═══════════════════════════════════════════════════════════════════════
# PASS 2 — GRADE STUDENT SHEET
# ═══════════════════════════════════════════════════════════════════════

# One definition per level, used by both the prompt and the UI copy. No percentage tables —
# the verdict is about how well each marking point is met.
STRICTNESS_RULES = {
    "easy":    "Lenient. A point is FULL when the student shows the right idea in any words, even loosely. "
               "Use MOSTLY/PARTIAL generously when there is a related correct idea.",
    "medium":  "Standard board marking. FULL when the essential idea of the point is clearly and correctly "
               "stated (synonyms and own words are fine). MOSTLY when a small qualifier or detail is missing. "
               "PARTIAL when only half the idea is there.",
    "hard":    "Strict. FULL only when the key technical terms of the point are present and correct. "
               "Missing terms → MOSTLY or PARTIAL. Vague answers → NONE.",
    "extreme": "Very strict. FULL only for a complete, precise statement with exact terms, symbols and units. "
               "Anything less is at most PARTIAL.",
}

VERDICT_FACTOR = {"full": 1.0, "mostly": 0.75, "partial": 0.5, "none": 0.0}
_BINARY_TYPES = {"mcq", "one_word"}


def _rubric_for_prompt(rubric: dict) -> list:
    """Only what the grader needs (no timings, no notes)."""
    out = []
    for q in rubric.get("questions", []):
        item = {"q_no": q["q_no"], "question_text": q.get("question_text", ""), "marks": q["marks"],
                "answer_type": q.get("answer_type", ""), "or_group": q.get("or_group")}
        if q["subparts"]:
            item["subparts"] = [{"sub_id": s["sub_id"], "text": s.get("text", ""), "marks": s["marks"],
                                 "answer_type": s.get("answer_type", q.get("answer_type", "")),
                                 "points": [{k: p[k] for k in ("id", "expected", "marks")} for p in s["points"]]}
                                for s in q["subparts"]]
        else:
            item["points"] = [{k: p[k] for k in ("id", "expected", "marks")} for p in q["points"]]
        out.append(item)
    return out


def _build_grading_prompt(rubric: dict, strictness: str, require_identity: bool) -> str:
    rubric_json = json.dumps(_rubric_for_prompt(rubric), indent=1, ensure_ascii=False)
    rule = STRICTNESS_RULES.get(strictness, STRICTNESS_RULES["medium"])

    if require_identity:
        identity = """STEP 1 — STUDENT IDENTITY
Find the Student Name, Class / Section and Roll Number (top of the sheet, cover page or header).
If ANY of the three is missing or unreadable → is_valid = false, list it in missing_fields, and return
an empty "answers" list. If all three are there → is_valid = true."""
    else:
        identity = """STEP 1 — STUDENT IDENTITY
The student is already known. Set is_valid = true and copy any name / class / roll number you see
(empty strings if none)."""

    return f"""You are an experienced examiner checking a handwritten answer sheet against a marking rubric.

{identity}

STEP 2 — READ THE ANSWERS
Students may answer in any order and may number answers differently ("Ans 4", "4)", "Q.4").
Match each answer to the rubric question by its number AND its content.
Read handwriting carefully. Transcribe what the student wrote — do not correct it.

STEP 3 — JUDGE EVERY MARKING POINT
STRICTNESS: {strictness.upper()} — {rule}
For each point give a verdict:
  "full"    — the point is met
  "mostly"  — met except a small detail / qualifier
  "partial" — about half of the point is there, or it is there with a significant error
  "none"    — missing or wrong
Rules:
- Judge ONLY against the point's "expected" content. Do not reward extra unrelated writing.
- A WRONG statement on the point (e.g. "acidic" when the answer is "neutral") is "none" for that point,
  even if nearby words are right.
- Chemical equations: unbalanced but correct reactants/products → "partial" on the equation point.
- mcq / one_word: only "full" or "none".
- "evidence": quote the student's own words for this point (max 20 words), "" if nothing written.
- "reason": one short sentence why (for the teacher).
- "borderline": true when a careful teacher could reasonably give a different verdict (e.g. a one-word
  answer that names the right thing but not the action asked for). The teacher will review it.
- OR-questions (same or_group): grade whichever alternative(s) the student actually attempted; mark the
  others attempted = false.
- If you cannot read part of an answer, set legibility_issue = true and confidence = "low" for it.
- Do NOT calculate marks. The server does that from your verdicts.
- "page_index": 0-based index of the page where this answer starts (0 = first page).
- "y_start": where on that page the answer starts, 0 (top of page) to 1000 (bottom of page).
  Estimate visually. If you cannot tell, set null.

RUBRIC:
{rubric_json}

Respond ONLY with JSON (no markdown):
{{
  "is_valid": true,
  "student_info": {{"name": "Rahul Verma", "class": "10-A", "roll_no": "23", "missing_fields": []}},
  "invalid_reason": "",
  "answers": [
    {{
      "q_no": "4", "sub_id": null, "attempted": true,
      "page_index": 0, "y_start": 320,
      "student_answer": "Refraction is the bending of light when it goes from one medium to another. Law: sin i / sin r = constant.",
      "points": [
        {{"id": "4.1", "verdict": "mostly", "evidence": "bending of light when it goes from one medium to another", "reason": "Correct idea, 'obliquely' missing.", "borderline": false}},
        {{"id": "4.2", "verdict": "none", "evidence": "", "reason": "First law not stated.", "borderline": false}},
        {{"id": "4.3", "verdict": "full", "evidence": "sin i / sin r = constant", "reason": "Snell's law correct.", "borderline": false}}
      ],
      "feedback": "Good definition; state both laws of refraction.",
      "confidence": "high",
      "legibility_issue": false
    }},
    {{
      "q_no": "5", "sub_id": "5a", "attempted": true,
      "page_index": 1, "y_start": 150,
      "student_answer": "haemoglobin carries oxygen",
      "points": [{{"id": "5a.1", "verdict": "full", "evidence": "carries oxygen", "reason": "Correct.", "borderline": false}}],
      "feedback": "Correct.", "confidence": "high", "legibility_issue": false
    }}
  ],
  "overall_remarks": "2-3 sentences for the student: strengths, what to fix.",
  "readability_score": "good"
}}
One entry in "answers" per question without sub-parts, and one per SUB-PART (with sub_id) — never one
for a parent question that has sub-parts. Include unattempted ones with attempted = false.
missing_fields values: "name" | "class" | "roll_no". confidence: "high" | "medium" | "low".
readability_score: "good" | "average" | "poor".
"""


def _score_leaf(leaf: dict, ans: Optional[dict], answer_type: str, scheme_source: str) -> dict:
    """Marks for one gradable unit (question without sub-parts, or one sub-part)."""
    verdicts = {}
    if ans:
        for p in ans.get("points") or []:
            if isinstance(p, dict) and p.get("id") is not None:
                verdicts[str(p["id"])] = p

    binary = answer_type in _BINARY_TYPES
    attempted = bool(ans and ans.get("attempted", True))
    points_out, awarded = [], 0.0
    for p in leaf["points"]:
        v = verdicts.get(p["id"], {})
        verdict = str(v.get("verdict") or "none").lower()
        if verdict not in VERDICT_FACTOR:
            verdict = "none"
        if binary and verdict != "full":
            verdict = "none"
        got = _half_floor(p["marks"] * VERDICT_FACTOR[verdict]) if attempted else 0.0
        if verdict == "full":
            got = p["marks"]  # never lose marks to rounding on a fully met point
        awarded += got
        points_out.append({
            "id": p["id"], "text": p.get("text", ""), "expected": p.get("expected", ""),
            "max_marks": p["marks"], "marks_awarded": got, "verdict": verdict if attempted else "none",
            "evidence": str(v.get("evidence") or ""), "reason": str(v.get("reason") or ""),
            "borderline": bool(v.get("borderline")) and attempted,
        })

    awarded = min(awarded, leaf["marks"])
    confidence = str((ans or {}).get("confidence") or ("high" if ans else "low")).lower()
    legibility = bool((ans or {}).get("legibility_issue"))
    review_reasons = []
    if ans is None:
        review_reasons.append("Answer not found on the sheet — please check")
    if confidence == "low":
        review_reasons.append("Low confidence")
    if legibility:
        review_reasons.append("Handwriting hard to read")
    if any(pt["borderline"] for pt in points_out):
        review_reasons.append("Borderline answer — teacher's call")
    if scheme_source == "estimated" and attempted and 0 < awarded < leaf["marks"]:
        review_reasons.append("Marking split was estimated (not in the answer key)")

    # page_index and y_start are best-effort position hints; null is fine if the model couldn't tell.
    raw_pi = (ans or {}).get("page_index")
    raw_ys = (ans or {}).get("y_start")
    page_index = int(raw_pi) if isinstance(raw_pi, (int, float)) and raw_pi >= 0 else None
    y_start = max(0, min(1000, int(raw_ys))) if isinstance(raw_ys, (int, float)) else None

    return {
        "max_marks": leaf["marks"],
        "marks_awarded": round(awarded, 2),
        "student_answer_summary": str((ans or {}).get("student_answer") or ""),
        "key_concepts_found": [pt["text"] or pt["expected"] for pt in points_out if pt["verdict"] in ("full", "mostly")],
        "key_concepts_missed": [pt["text"] or pt["expected"] for pt in points_out if pt["verdict"] in ("partial", "none")],
        "feedback": str((ans or {}).get("feedback") or ("Not attempted" if not attempted else "")),
        "confidence": confidence if confidence in ("high", "medium", "low") else "medium",
        "legibility_issue": legibility,
        "not_attempted": not attempted,
        "points": points_out,
        "needs_review": bool(review_reasons),
        "review_reasons": review_reasons,
        "page_index": page_index,
        "y_start": y_start,
    }


def _score(grading: dict, rubric: dict) -> dict:
    """Turns verdicts into marks. All arithmetic lives here."""
    by_key = {}
    for a in grading.get("answers") or []:
        if not isinstance(a, dict):
            continue
        sub = a.get("sub_id")
        key = str(sub).strip() if sub not in (None, "", "null") else str(a.get("q_no") or "").strip()
        by_key[key] = a

    graded = []
    for q in rubric["questions"]:
        atype = q.get("answer_type", "")
        if q["subparts"]:
            subs = []
            for s in q["subparts"]:
                r = _score_leaf(s, by_key.get(s["sub_id"]), s.get("answer_type") or atype,
                                s.get("scheme_source") or q["scheme_source"])
                r["sub_id"] = s["sub_id"]
                subs.append(r)
            # Position hint: use the first attempted sub-part that has a location.
            first_pos = next(
                (s for s in subs if s.get("page_index") is not None or s.get("y_start") is not None),
                subs[0] if subs else {},
            )
            gq = {
                "q_no": q["q_no"], "max_marks": q["marks"],
                "marks_awarded": round(min(sum(s["marks_awarded"] for s in subs), q["marks"]), 2),
                "student_answer_summary": "See sub-parts",
                "key_concepts_found": [], "key_concepts_missed": [],
                "feedback": "See sub-part feedback",
                "confidence": min((s["confidence"] for s in subs), key=["low", "medium", "high"].index, default="high"),
                "legibility_issue": any(s["legibility_issue"] for s in subs),
                "not_attempted": all(s["not_attempted"] for s in subs),
                "has_subparts": True, "subpart_grades": subs, "points": [],
                "needs_review": any(s["needs_review"] for s in subs),
                "review_reasons": sorted({r for s in subs for r in s["review_reasons"]}),
                "page_index": first_pos.get("page_index"),
                "y_start": first_pos.get("y_start"),
            }
        else:
            gq = _score_leaf(q, by_key.get(q["q_no"]), atype, q["scheme_source"])
            gq.update({"q_no": q["q_no"], "has_subparts": False, "subpart_grades": []})
        gq["or_group"] = q.get("or_group")
        gq["scheme_source"] = q["scheme_source"]
        gq["counted"] = True
        graded.append(gq)

    # OR-groups: only the best `attempt_count` alternatives count
    groups: dict = {}
    for q, gq in zip(rubric["questions"], graded):
        if gq["or_group"]:
            groups.setdefault(gq["or_group"], []).append((q, gq))
    for members in groups.values():
        need = int(_num(members[0][0].get("attempt_count"), 0)) or 1
        ranked = sorted(members, key=lambda m: m[1]["marks_awarded"], reverse=True)
        for _, gq in ranked[need:]:
            gq["counted"] = False
            if gq["not_attempted"]:
                gq["needs_review"] = False
                gq["review_reasons"] = []

    total_paper = float(rubric["total_marks"])
    total = sum(gq["marks_awarded"] for gq in graded if gq["counted"])
    if total > total_paper:
        logger.warning(f"Total {total} > paper max {total_paper}, capping")
        total = total_paper

    return {
        "graded_questions": graded,
        "total_marks_awarded": round(total, 2),
        "max_marks": total_paper,
        "percentage": round(total / total_paper * 100, 1) if total_paper > 0 else 0.0,
        "needs_review_count": sum(1 for gq in graded if gq["needs_review"] and gq["counted"]),
    }


def grade_sheet(
    sheet_path,
    rubric: dict,
    strictness: str = "medium",
    require_identity: bool = True,
    max_retries: int = 2,
    _uploaded_sheet=None,
) -> dict:
    """Pass 2: grade one student's sheet against a rubric from build_rubric().
    sheet_path: one file, or a list of page files in order (phone photos of a multi-page copy)."""
    strictness = strictness if strictness in STRICTNESS_RULES else "medium"
    start = time.time()
    client = _get_client()
    own_upload = _uploaded_sheet is None
    pages = _uploaded_sheet if not own_upload else upload_files(
        sheet_path if isinstance(sheet_path, list) else [sheet_path])
    pages = pages if isinstance(pages, list) else [pages]
    prompt = _build_grading_prompt(rubric, strictness, require_identity)
    if len(pages) > 1:
        prompt = (f"The answer sheet is given as {len(pages)} page images, in order (page 1 first). "
                  f"An answer may continue onto the next page.\n\n") + prompt
    try:
        grading = _generate_json(client, [*pages, prompt], "GRADE", max_retries)
    finally:
        if own_upload:
            delete_files(pages)
    grade_time = round(time.time() - start, 2)

    info = grading.get("student_info") if isinstance(grading.get("student_info"), dict) else {}
    info = {"name": str(info.get("name") or ""), "class": str(info.get("class") or ""),
            "roll_no": str(info.get("roll_no") or ""), "missing_fields": list(info.get("missing_fields") or [])}
    if require_identity:
        # Don't trust is_valid alone: re-derive it from the fields the model actually read
        missing = [f for f, k in (("name", "name"), ("class", "class"), ("roll_no", "roll_no")) if not info[k].strip()]
        info["missing_fields"] = sorted(set(info["missing_fields"]) | set(missing))
        is_valid = bool(grading.get("is_valid")) and not info["missing_fields"]
    else:
        is_valid = True

    base = {
        "student_info": info,
        "overall_remarks": str(grading.get("overall_remarks") or ""),
        "readability_score": str(grading.get("readability_score") or ""),
        "_grade_time_s": grade_time,
    }
    if not is_valid:
        return {**base, "is_valid": False,
                "invalid_reason": str(grading.get("invalid_reason") or "Student identity could not be verified."),
                "missing_fields": info["missing_fields"], "graded_questions": [],
                "total_marks_awarded": 0, "max_marks": rubric["total_marks"], "percentage": 0,
                "overall_remarks": "Assignment cannot be graded: student identity incomplete."}
    return {**base, "is_valid": True, **_score(grading, rubric)}


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE (one sheet, everything uploaded once and in parallel)
# ═══════════════════════════════════════════════════════════════════════

def _public_rubric(rubric: dict) -> dict:
    return {k: v for k, v in rubric.items() if not k.startswith("_")}


def check_assignment(
    question_paper_path: str,
    answer_key_path: str,
    student_sheet_path: str,
    strictness: str = "medium",
    max_retries: int = 2,
) -> dict:
    """
    Question paper + answer key + one student sheet → graded result.
    Response shape stays compatible with v1 (graded_questions, subpart_grades, key_concepts_*),
    plus per-point detail (points[]), needs_review flags and the rubric used.
    """
    pipeline_start = time.time()
    logger.info(f"Smart check: strictness={strictness}")

    files = upload_files([question_paper_path, answer_key_path, student_sheet_path])
    upload_time = round(time.time() - pipeline_start, 2)
    try:
        t = time.time()
        rubric = _build_rubric_from_files(files[0], files[1], max_retries)
        rubric_time = round(time.time() - t, 2)
        result = grade_sheet(student_sheet_path, rubric, strictness, True, max_retries, _uploaded_sheet=files[2])
    finally:
        delete_files(files)

    grade_time = result.pop("_grade_time_s", 0)
    result["paper_info"] = {
        "total_questions": rubric["total_questions"],
        "total_marks": rubric["total_marks"],
        "marks_mismatch": rubric.get("marks_mismatch"),
        "notes": rubric.get("paper_notes", ""),
    }
    result["rubric"] = _public_rubric(rubric)
    result["_meta"] = {
        "pipeline": "2-pass (rubric → grade)" if result["is_valid"] else "2-pass (aborted at identity check)",
        "model": _preferred_model or CHECKER_MODEL,
        "strictness": strictness,
        "upload_time_s": upload_time,
        "rubric_time_s": rubric_time,
        "grade_time_s": grade_time,
        "total_time_s": round(time.time() - pipeline_start, 2),
    }
    logger.info(f"Smart check done: {result.get('total_marks_awarded')}/{result.get('max_marks')} "
                f"in {result['_meta']['total_time_s']}s")
    return result
