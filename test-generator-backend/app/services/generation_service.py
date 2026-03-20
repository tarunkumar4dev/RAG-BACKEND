"""
Generation Service v10 — PRODUCTION

Per-chapter generation, cost optimized.
100 questions ≈ ₹2-5 on gemini-2.5-flash.
"""

import json
import logging
import random
import re
import time
import uuid
from typing import List, Dict, Optional
import math

from google import genai
from google.genai import types as genai_types

from app.core.config import settings
from app.models.test_generator import (
    TestGenerationRequest,
    GeneratedQuestion,
    DifficultyLevel,
    BloomLevel,
    QuestionFormat,
    ChapterSection,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 20
JITTER_RANGE = (0.5, 1.2)

RETRYABLE_KEYWORDS = frozenset([
    "429", "resource_exhausted", "quota", "rate",
    "503", "500", "overloaded", "timeout", "unavailable",
])

VALID_FORMATS = frozenset({
    "mcq", "short_answer", "long_answer", "assertion_reason", "case_based",
})
VALID_BLOOMS = frozenset({
    "remember", "understand", "apply", "analyze", "evaluate", "create",
})

ASSERTION_REASON_OPTIONS = [
    "A) Both A and R are true and R is the correct explanation of A",
    "B) Both A and R are true but R is NOT the correct explanation of A",
    "C) A is true but R is false",
    "D) A is false but R is true",
]

DIFF_INST = {
    "easy": "EASY: 1-step recall. Bloom: remember/understand.",
    "medium": "MEDIUM: 2-3 steps, 1 formula. Bloom: understand/apply.",
    "hard": "HARD: 3+ steps, 2+ concepts combined. Cannot use single formula. Bloom: apply/analyze. 50%+ multi-step calc. Distractors: wrong formula, sign error, misconception.",
    "very_hard": "VERY HARD: Olympiad level, 3+ concepts, non-routine. Non-clean numbers. All options plausible. Bloom: analyze/evaluate/create.",
}

BLOOM_VALID = {
    "easy": {"remember", "understand"},
    "medium": {"understand", "apply"},
    "hard": {"apply", "analyze"},
    "very_hard": {"analyze", "evaluate", "create"},
}
BLOOM_DEFAULT = {
    "easy": "remember",
    "medium": "apply",
    "hard": "analyze",
    "very_hard": "analyze",
}


# ---------------------------------------------------------------------------
# Exception + Client
# ---------------------------------------------------------------------------
class GenerationError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


_client_cache: Optional[genai.Client] = None


def _get_gemini_client() -> genai.Client:
    global _client_cache
    if _client_cache is None:
        if not settings.GEMINI_API_KEY:
            raise GenerationError("GEMINI_API_KEY not configured.", 500)
        _client_cache = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client_cache


# ---------------------------------------------------------------------------
# Compact Prompt Builder (per chapter)
# ---------------------------------------------------------------------------
def _build_chapter_prompt(
    chapter: ChapterSection,
    request: TestGenerationRequest,
    context_chunks: List[Dict],
    count: int,
) -> str:
    ch_name = chapter.chapter.upper()
    ch_chunks = [c for c in context_chunks if c.get("chapter", "").upper() == ch_name]
    if not ch_chunks:
        ch_chunks = context_chunks[:settings.MAX_CONTEXT_CHUNKS]

    ctx = ""
    for i, chunk in enumerate(ch_chunks[:settings.MAX_CONTEXT_CHUNKS], 1):
        content = (chunk.get("content") or "")[:settings.CONTEXT_CHARS_PER_CHUNK]
        if content:
            ctx += f"\n[{i}] {content}\n"

    diff_val = chapter.difficulty.value if hasattr(chapter.difficulty, 'value') else str(chapter.difficulty)
    fmt_val = chapter.format.value if hasattr(chapter.format, 'value') else str(chapter.format)
    is_math = request.subject.lower() in ("mathematics", "maths", "math")

    if fmt_val == "short_answer":
        fmt_line = '"options": null. Answer: 30-60 words.'
    elif fmt_val == "long_answer":
        fmt_line = '"options": null. Answer: 80-150 words.'
    elif fmt_val == "assertion_reason":
        fmt_line = '"text": "Assertion (A):...\\nReason (R):...". 4 standard A&R options.'
    else:
        fmt_line = '4 options A-D. correct_answer = exact option text. Vary position.'

    math_note = 'LaTeX $...$ for math. DOUBLE backslash in JSON: \\\\frac, \\\\sqrt, \\\\theta.' if is_math else ''

    if fmt_val in ("short_answer", "long_answer"):
        tmpl = f'{{"questions":[{{"text":"...","format":"{fmt_val}","options":null,"correct_answer":"answer","explanation":"Given->Solution->Answer","marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"apply","chapter":"{chapter.chapter}","topic":"..."}}]}}'
    else:
        tmpl = f'{{"questions":[{{"text":"...","format":"{fmt_val}","options":["A)...","B)...","C)...","D)..."],"correct_answer":"B)...","explanation":"Given->Solution->Answer","marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"apply","chapter":"{chapter.chapter}","topic":"..."}}]}}'

    return f"""CBSE Class {request.class_grade} {request.subject} — {chapter.chapter}
{DIFF_INST.get(diff_val, DIFF_INST["medium"])}
{fmt_line}
{math_note}
"format"="{fmt_val}" exactly. "chapter"="{chapter.chapter}" exactly.
Explanation: max 3 lines. Self-verify each question.

NCERT:
{ctx}

Generate EXACTLY {count} different questions. ONLY valid JSON:
{tmpl}"""


# ---------------------------------------------------------------------------
# JSON Extraction (with LaTeX fix)
# ---------------------------------------------------------------------------
def _fix_latex_json_escapes(text: str) -> str:
    text = text.replace('\\\\', '\x00DBL\x00')
    for prefix in ['frac', 'forall', 'binom', 'boxed', 'bold', 'not', 'neq',
                    'nabla', 'right', 'rangle', 'rceil', 'rfloor', 'times',
                    'text', 'theta', 'tan', 'therefore', 'triangle', 'tilde']:
        text = re.sub(r'\\(' + prefix + r')(?=[^a-zA-Z]|$)', r'\\\\\1', text)
    text = text.replace('\x00DBL\x00', '\\\\')
    return text


def _extract_json(raw: str) -> dict:
    text = raw.strip().lstrip("\ufeff\u200b")

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fb = text.find("{")
    lb = text.rfind("}")
    if fb == -1 or lb <= fb:
        raise ValueError(f"No JSON found (len={len(raw)})")

    candidate = text[fb:lb + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fixed = _fix_latex_json_escapes(cleaned)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    aggressive = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu{])', r'\\\\', fixed)
    try:
        return json.loads(aggressive)
    except json.JSONDecodeError:
        pass

    nuclear = re.sub(r'[\x00-\x1f]', ' ', aggressive)
    try:
        return json.loads(nuclear)
    except json.JSONDecodeError:
        pass

    logger.error(f"JSON failed 6 attempts. Preview: {candidate[:200]}")
    raise ValueError(f"Could not parse JSON (len={len(raw)})")


# ---------------------------------------------------------------------------
# Question Parser
# ---------------------------------------------------------------------------
def _parse_batch(raw: str, chapter: ChapterSection, request: TestGenerationRequest) -> List[GeneratedQuestion]:
    try:
        data = _extract_json(raw)
        raw_qs = data.get("questions", [])
    except (ValueError, AttributeError) as e:
        logger.error(f"Parse failed ({chapter.chapter}): {e}")
        return []

    if not isinstance(raw_qs, list):
        return []

    questions = []
    seen = set()
    dropped = 0
    diff_val = chapter.difficulty.value if hasattr(chapter.difficulty, 'value') else str(chapter.difficulty)

    for idx, q in enumerate(raw_qs):
        if not isinstance(q, dict):
            continue

        text = (q.get("text") or "").strip()
        if not text or len(text) < 15:
            dropped += 1
            continue

        norm = re.sub(r"\$[^$]*\$", "", text.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if norm in seen:
            dropped += 1
            continue
        seen.add(norm)

        fmt = q.get("format", "mcq")
        if fmt not in VALID_FORMATS:
            fmt = fmt.lower().replace("questionformat.", "").replace(" ", "_").replace(".", "_")
            if fmt not in VALID_FORMATS:
                fmt = "mcq"

        bloom_raw = (q.get("bloom_level") or "").strip().lower()
        valid = BLOOM_VALID.get(diff_val, {"apply"})
        if bloom_raw in valid:
            bloom = BloomLevel(bloom_raw)
        elif bloom_raw in VALID_BLOOMS:
            bloom = BloomLevel(BLOOM_DEFAULT[diff_val])
        else:
            bloom = BloomLevel(BLOOM_DEFAULT[diff_val])

        options = q.get("options")
        correct = (q.get("correct_answer") or "").strip()
        explanation = (q.get("explanation") or "").strip()

        if not explanation or len(explanation) < 10:
            dropped += 1
            continue

        if fmt == "assertion_reason":
            if not options or len(options) != 4:
                options = ASSERTION_REASON_OPTIONS.copy()
            if not correct or correct not in options:
                matched = [o for o in options if len(correct) >= 2 and o[:2].upper() == correct[:2].upper()]
                correct = matched[0] if matched else options[0]

        elif fmt in ("short_answer", "long_answer"):
            options = None
            if not correct or len(correct) < 10:
                dropped += 1
                continue

        elif fmt == "mcq":
            if not isinstance(options, list) or len(options) != 4:
                dropped += 1
                continue
            if correct not in options:
                matched = [o for o in options if len(correct) >= 2 and o[:2].upper() == correct[:2].upper()]
                if matched:
                    correct = matched[0]
                else:
                    dropped += 1
                    continue

        marks = q.get("marks", chapter.marks_per_question)
        if not isinstance(marks, (int, float)) or marks < 1:
            marks = chapter.marks_per_question

        try:
            questions.append(GeneratedQuestion(
                id=str(uuid.uuid4()),
                text=text,
                options=options if fmt in ("mcq", "assertion_reason", "case_based") else None,
                correct_answer=correct,
                explanation=explanation,
                marks=int(marks),
                difficulty=DifficultyLevel(diff_val),
                bloom_level=bloom,
                chapter=chapter.chapter,
                topic=q.get("topic"),
                format=QuestionFormat(fmt),
                validation_status="verified",
            ))
        except Exception as e:
            logger.warning(f"Q{idx} ({chapter.chapter}): {e}")
            dropped += 1

    if dropped:
        logger.info(f"  {chapter.chapter}: dropped {dropped}, kept {len(questions)}")
    return questions


# ---------------------------------------------------------------------------
# Gemini Call with Retry
# ---------------------------------------------------------------------------
def _is_retryable(error_str: str) -> bool:
    return any(kw in error_str.upper() for kw in (k.upper() for k in RETRYABLE_KEYWORDS))


def _call_gemini(client, prompt, model):
    last_exc = None
    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=settings.GENERATION_TEMPERATURE,
                    top_p=0.92,
                    max_output_tokens=settings.MAX_OUTPUT_TOKENS,
                ),
            )
            raw = (resp.text or "").strip()
            if not raw:
                raise GenerationError("Empty response", 502)
            logger.info(f"[{model}] {time.time() - t0:.1f}s ({len(raw)} chars)")
            return raw
        except GenerationError:
            raise
        except Exception as e:
            last_exc = e
            if _is_retryable(str(e)) and attempt < MAX_RETRIES - 1:
                wait = min(BASE_BACKOFF_SECONDS ** (attempt + 1), MAX_BACKOFF_SECONDS) * random.uniform(*JITTER_RANGE)
                logger.warning(f"[{model}] Retry {attempt + 1}/{MAX_RETRIES}: {str(e)[:80]}... {wait:.1f}s")
                time.sleep(wait)
            else:
                break
    raise GenerationError(f"Failed after {MAX_RETRIES} retries: {str(last_exc)[:150]}", 500)


# ---------------------------------------------------------------------------
# Generate for ONE chapter
# ---------------------------------------------------------------------------
def _generate_for_chapter(client, chapter, request, context_chunks, models):
    target = chapter.quantity
    ask = target + settings.OVERSHOOT_PER_CHAPTER
    batch_size = settings.BATCH_SIZE

    logger.info(f"  '{chapter.chapter}': target={target}, fmt={chapter.format}, "
                f"diff={chapter.difficulty}, marks={chapter.marks_per_question}")

    all_qs = []
    remaining = ask
    batch_num = 0

    while remaining > 0 and len(all_qs) < target:
        bc = min(remaining, batch_size)
        batch_num += 1
        prompt = _build_chapter_prompt(chapter, request, context_chunks, bc)

        batch_qs = []
        for m in models:
            try:
                raw = _call_gemini(client, prompt, m)
                batch_qs = _parse_batch(raw, chapter, request)
                if batch_qs:
                    logger.info(f"    Batch {batch_num}: {len(batch_qs)}/{bc} [{m}]")
                    break
            except GenerationError as e:
                if m != models[-1]:
                    logger.warning(f"    {m} failed, fallback...")
                    continue
                logger.error(f"    '{chapter.chapter}' batch {batch_num} failed: {e}")
                break

        all_qs.extend(batch_qs)
        remaining -= bc

        if remaining > 0 and len(all_qs) < target:
            time.sleep(settings.BATCH_DELAY)

    if len(all_qs) > target:
        all_qs = all_qs[:target]

    logger.info(f"  '{chapter.chapter}': {len(all_qs)}/{target}")
    return all_qs


# ---------------------------------------------------------------------------
# Main Entry
# ---------------------------------------------------------------------------
def generate_questions(request, context_chunks, feedback=None):
    if not context_chunks:
        raise GenerationError("No NCERT content found. Verify chapters.", 404)

    client = _get_gemini_client()
    model = settings.GEMINI_GEN_MODEL
    models = [model]
    fallback = getattr(settings, 'GEMINI_FALLBACK_MODEL', None)
    if fallback and fallback != model:
        models.append(fallback)

    total = sum(s.quantity for s in request.chapters)
    logger.info(f"Generation: {len(request.chapters)} chapters, {total} questions, model={model}")

    all_questions = []
    t0 = time.time()
    results = {}

    for ch_idx, chapter in enumerate(request.chapters, 1):
        if chapter.quantity <= 0:
            continue

        logger.info(f"Chapter {ch_idx}/{len(request.chapters)}: {chapter.chapter}")
        ch_qs = _generate_for_chapter(client, chapter, request, context_chunks, models)
        all_questions.extend(ch_qs)
        results[chapter.chapter] = f"{len(ch_qs)}/{chapter.quantity}"

        if ch_idx < len(request.chapters):
            time.sleep(settings.BATCH_DELAY)

    elapsed = time.time() - t0
    summary = " | ".join([f"{ch}: {r}" for ch, r in results.items()])
    logger.info(f"Done: {len(all_questions)}/{total} in {elapsed:.1f}s | {summary}")

    if not all_questions:
        raise GenerationError("All chapters failed.", 500)

    return all_questions