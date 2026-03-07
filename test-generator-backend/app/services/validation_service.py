"""
Validation Service — Gemini Flash verifies factual accuracy of generated questions.
Fixes errors in-place. Saves rejected questions as training data.
"""

import json
import logging
from typing import List, Tuple
from google import genai
from google.genai import types as genai_types
from app.core.config import settings
from app.core.database import get_supabase
from app.models.test_generator import GeneratedQuestion

logger = logging.getLogger(__name__)


def _get_client() -> genai.Client:
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def validate_questions(
    questions: List[GeneratedQuestion],
    subject: str,
    class_grade: str,
) -> List[GeneratedQuestion]:
    """
    Send all questions to Gemini Flash for factual verification.
    Returns validated (and if needed, corrected) questions.
    """
    if not questions:
        return []

    client = _get_client()

    # Serialize questions for prompt
    questions_text = ""
    for i, q in enumerate(questions, 1):
        questions_text += (
            f"\n[Q{i}]\n"
            f"Text: {q.text}\n"
            f"Options: {q.options}\n"
            f"Correct Answer: {q.correct_answer}\n"
            f"Explanation: {q.explanation}\n"
            f"Chapter: {q.chapter}\n"
        )

    prompt = f"""You are a factual accuracy checker for Class {class_grade} {subject} NCERT questions.

Review each question below for:
1. Factual correctness (answer must be verifiably correct)
2. Clarity (question must be unambiguous)
3. Option quality (for MCQ, distractors must be plausible but clearly wrong)
# 4. Review for bloom taxonomy as well in depth

QUESTIONS TO VALIDATE:
{questions_text}

For each question, return:
- "valid": true if correct, false if needs fixing
- "fixed_correct_answer": corrected answer only if wrong (else null)
- "fixed_explanation": corrected explanation only if wrong (else null)
- "notes": brief note if issue found

Return ONLY valid JSON (no markdown):
{{
  "validations": [
    {{
      "index": 1,
      "valid": true,
      "fixed_correct_answer": null,
      "fixed_explanation": null,
      "notes": null
    }}
  ]
}}"""

    try:
        response = client.models.generate_content(
            model=settings.GEMINI_VAL_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=2000,
            ),
        )

        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        validations = data.get("validations", [])

        for val in validations:
            idx = val.get("index", 0) - 1
            if 0 <= idx < len(questions):
                q = questions[idx]
                if not val.get("valid", True):
                    if val.get("fixed_correct_answer"):
                        q.correct_answer = val["fixed_correct_answer"]
                    if val.get("fixed_explanation"):
                        q.explanation = val["fixed_explanation"]
                    q.validation_notes = val.get("notes")
                    q.validation_status = "verified"  # fixed = now verified
                    logger.info(f"Fixed Q{idx+1}: {val.get('notes', '')[:60]}")

        logger.info(f"Validation complete: {len(questions)} questions checked")
        return questions

    except Exception as e:
        logger.error(f"Validation error: {e}")
        # Return questions as-is if validation fails — don't block generation
        for q in questions:
            q.validation_status = "needs_review"
        return questions


def save_training_data(
    test_id: str,
    teacher_id: str,
    question_id: str,
    question_text: str,
    correct_answer: str,
    teacher_feedback: str,
    action: str,  # "reject" | "edit"
):
    """
    Save rejected/edited questions with feedback as training data.
    This is our long-term moat — fine-tuning fuel.
    """
    supabase = get_supabase()
    try:
        supabase.table("training_data").insert({
            "test_id": test_id,
            "teacher_id": teacher_id,
            "question_id": question_id,
            "question_text": question_text,
            "correct_answer": correct_answer,
            "teacher_feedback": teacher_feedback,
            "action": action,
        }).execute()
        logger.info(f"Training data saved: Q {question_id[:8]}... action={action}")
    except Exception as e:
        logger.error(f"Failed to save training data: {e}")