"""
Quiz Question Generator for a4ai Community Quiz
=================================================
Generates MCQs from video transcripts using Gemini 2.0 Flash.

Dependencies:
    pip install google-generativeai

Usage:
    from services.quiz_generator import generate_questions_from_transcript

    questions = generate_questions_from_transcript(
        transcript="...",
        title="Photosynthesis Class 10",
        count=20,
        difficulty="medium",
        focus="mixed",
    )
"""

import json
import logging
import os
import re
from typing import Literal

# Lazy import — only loaded when video flow runs
# Manual flow doesn't need Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configure Gemini (assumes GEMINI_API_KEY in env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY and GEMINI_AVAILABLE:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-flash-latest"  # cost-optimized per current preference


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------
class QuestionGenerationError(Exception):
    def __init__(self, message: str, code: str = "generation_failed"):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
DIFFICULTY_GUIDANCE = {
    "easy": "Direct recall questions. Test if the student watched the video. Use simple language.",
    "medium": "Mix of recall and understanding. Test if the student understood concepts.",
    "hard": "Application and analysis. Test deeper understanding, edge cases, 'why' questions.",
    "mixed": "Mix all three: ~30% easy recall, 50% medium understanding, 20% hard application.",
}

FOCUS_GUIDANCE = {
    "conceptual": "Focus on concepts, definitions, principles, and 'why/how' explanations.",
    "factual": "Focus on specific facts, numbers, names, dates mentioned in the video.",
    "mixed": "Balance conceptual understanding and factual recall.",
}


def _build_prompt(
    transcript: str,
    title: str,
    channel: str,
    count: int,
    difficulty: str,
    focus: str,
) -> str:
    # Clip very long transcripts to keep cost predictable. Gemini Flash handles
    # 1M tokens but most educational videos are <30K tokens of transcript.
    MAX_CHARS = 60_000
    clipped = transcript[:MAX_CHARS]
    if len(transcript) > MAX_CHARS:
        clipped += "\n\n[Transcript clipped for length]"

    return f"""You are an expert teacher creating an MCQ quiz from a YouTube video transcript for Indian students.

VIDEO INFO:
Title: {title}
Channel: {channel}

TRANSCRIPT:
{clipped}

TASK:
Generate exactly {count} multiple-choice questions based on this video.

DIFFICULTY: {difficulty.upper()}
{DIFFICULTY_GUIDANCE.get(difficulty, DIFFICULTY_GUIDANCE["mixed"])}

FOCUS: {focus.upper()}
{FOCUS_GUIDANCE.get(focus, FOCUS_GUIDANCE["mixed"])}

REQUIREMENTS:
- Each question must be answerable ONLY by someone who watched the video
- Exactly 4 options per question
- Exactly ONE correct answer (correct_option is the index 0-3)
- Options should be plausible — no obviously wrong distractors
- Include a 1-2 sentence explanation citing the video content
- Use clear, simple English suitable for Indian school students
- Avoid questions about trivial details (intro music, sponsor mentions, etc.)
- Do NOT include questions whose answer is not actually in the transcript
- Vary question types: "what", "why", "how", "which of the following"

OUTPUT FORMAT:
Return ONLY a valid JSON object. No markdown fences. No preamble. No explanation outside JSON.

{{
  "questions": [
    {{
      "question_text": "What is the primary function of chlorophyll mentioned in the video?",
      "options": [
        "To absorb water from soil",
        "To absorb sunlight for photosynthesis",
        "To release oxygen into the air",
        "To store food in leaves"
      ],
      "correct_option": 1,
      "explanation": "The video explains that chlorophyll's main role is absorbing sunlight, which provides energy for photosynthesis."
    }}
  ]
}}

Generate {count} questions now."""


# ---------------------------------------------------------------------------
# JSON extraction (defensive against Gemini wrapping in markdown / preamble)
# ---------------------------------------------------------------------------
def _extract_json(raw: str) -> dict:
    """Strip markdown fences and extract the first JSON object from response."""
    if not raw or not raw.strip():
        raise QuestionGenerationError("Gemini returned empty response", code="empty_response")

    text = raw.strip()

    # Strip ```json ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Find outermost JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise QuestionGenerationError(
            "Could not find JSON object in Gemini response",
            code="invalid_json",
        )

    json_str = text[start : end + 1]

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        # Try fixing common issues: trailing commas, smart quotes
        cleaned = json_str.replace(""", '"').replace(""", '"').replace("'", "'")
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(f"JSON parse failure. Raw: {raw[:500]}")
            raise QuestionGenerationError(
                f"Gemini returned invalid JSON: {e}",
                code="invalid_json",
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_questions(data: dict, expected_count: int) -> list[dict]:
    if not isinstance(data, dict) or "questions" not in data:
        raise QuestionGenerationError("Response missing 'questions' field", code="invalid_format")

    questions = data["questions"]
    if not isinstance(questions, list):
        raise QuestionGenerationError("'questions' must be a list", code="invalid_format")

    if len(questions) == 0:
        raise QuestionGenerationError("No questions generated", code="empty_questions")

    valid = []
    for i, q in enumerate(questions):
        try:
            assert isinstance(q.get("question_text"), str) and q["question_text"].strip(), "missing question_text"
            assert isinstance(q.get("options"), list) and len(q["options"]) == 4, "options must be list of 4"
            assert all(isinstance(o, str) and o.strip() for o in q["options"]), "all options must be non-empty strings"
            assert isinstance(q.get("correct_option"), int) and 0 <= q["correct_option"] <= 3, "correct_option must be 0-3"

            valid.append({
                "question_text": q["question_text"].strip(),
                "options": [o.strip() for o in q["options"]],
                "correct_option": q["correct_option"],
                "explanation": (q.get("explanation") or "").strip(),
            })
        except AssertionError as e:
            logger.warning(f"Skipping invalid question {i}: {e}")
            continue

    if len(valid) == 0:
        raise QuestionGenerationError("All generated questions failed validation", code="validation_failed")

    # Warn but don't fail if count mismatch (Gemini sometimes generates fewer)
    if len(valid) < expected_count * 0.7:  # got <70% of asked
        logger.warning(f"Only got {len(valid)}/{expected_count} valid questions")

    return valid


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def generate_questions_from_transcript(
    transcript: str,
    title: str = "",
    channel: str = "",
    count: int = 20,
    difficulty: Literal["easy", "medium", "hard", "mixed"] = "mixed",
    focus: Literal["conceptual", "factual", "mixed"] = "mixed",
) -> list[dict]:
    """
    Generate MCQ questions from a video transcript.
    Returns list of validated question dicts ready for DB insertion.
    Each dict: {question_text, options, correct_option, explanation}
    """
    if not GEMINI_AVAILABLE:
        raise QuestionGenerationError(
            "Video quiz generation not available on this server. Use manual quiz instead.",
            code="gemini_not_installed",
        )

    if not GEMINI_API_KEY:
        raise QuestionGenerationError(
            "GEMINI_API_KEY not configured in environment",
            code="config_error",
        )

    if not transcript or len(transcript.strip()) < 50:
        raise QuestionGenerationError(
            "Transcript too short to generate questions",
            code="transcript_too_short",
        )

    if count < 1 or count > 50:
        raise QuestionGenerationError("count must be between 1 and 50", code="invalid_count")

    prompt = _build_prompt(transcript, title, channel, count, difficulty, focus)

    try:
        model = genai.GenerativeModel(
            MODEL_NAME,
            generation_config={
                "temperature": 0.7,
                "response_mime_type": "application/json",  # forces JSON output
            },
        )
        response = model.generate_content(prompt)
        raw_text = response.text

    except Exception as e:
        logger.exception("Gemini API call failed")
        raise QuestionGenerationError(
            f"Gemini API error: {str(e)}",
            code="api_error",
        )

    data = _extract_json(raw_text)
    questions = _validate_questions(data, count)
    return questions