"""
Test Checker Service — v1.1

v1.1 changes:
  - Added extract_answer_key_from_pdf(): sends answer key PDF to Gemini,
    extracts structured question/answer/marks JSON automatically.
  - Teacher no longer needs to create JSON manually — just upload the
    answer key PDF and the system handles conversion.

Core grading engine: sends answer sheet (PDF/image) + answer key to Gemini,
handles order-agnostic question matching, and returns structured results.

Key design decisions:
  - Order-agnostic: prompt explicitly tells Gemini to match by question NUMBER
    found on the sheet, not by position. Student can answer Q10 first — still matches.
  - Confidence gating: low-confidence answers get flagged for teacher review.
  - Retry with backoff: Gemini can be flaky with large PDFs; we retry twice.
  - No hardcoded API key: pulled from app settings.
"""

import json
import logging
import time
from typing import Optional
from pathlib import Path

from google import genai
from app.core.config import settings
import os
CHECKER_API_KEY = os.getenv("CHECKER_GEMINI_API_KEY")

logger = logging.getLogger(__name__)

# ── Gemini client (lazy init) ──────────────────────────────────────
_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        key = CHECKER_API_KEY or settings.GEMINI_API_KEY
        if not key:
            raise ValueError("No Gemini API key found. Set CHECKER_GEMINI_API_KEY in .env")
        _client = genai.Client(api_key=key)
    return _client

# ── Strictness presets ─────────────────────────────────────────────

STRICTNESS_PROMPTS = {
    "easy": (
        "Be lenient. Award partial marks generously. "
        "If the core idea is present, give 40-60% marks even if phrasing is rough. "
        "Ignore minor spelling/grammar mistakes entirely."
    ),
    "medium": (
        "Follow standard CBSE marking scheme. Be fair but not overly strict. "
        "Award partial marks for partially correct answers. "
        "Minor spelling mistakes are acceptable."
    ),
    "hard": (
        "Be strict. Key terms and definitions must be present. "
        "Deduct marks for vague or incomplete answers. "
        "Partial marks only if at least 60% of the expected content is covered."
    ),
    "extreme": (
        "Very strict. Near-perfect answers expected (90%+ accuracy). "
        "Technical terms must be exact. Deduct for any vagueness. "
        "Partial marks only for substantially correct answers."
    ),
}


# ═══════════════════════════════════════════════════════════════════════
# ANSWER KEY EXTRACTION FROM PDF (v1.1)
# ═══════════════════════════════════════════════════════════════════════

def extract_answer_key_from_pdf(file_path: str, max_retries: int = 2) -> dict:
    """
    Upload an answer key PDF/image to Gemini and extract structured
    question-answer-marks data as JSON.

    Returns:
        {
          "questions": [
            {"q_no": 1, "question_text": "...", "correct_answer": "...", "marks": 3, "format": "short_answer"},
            ...
          ],
          "total_marks": 30
        }
    """
    client = _get_client()

    prompt = """You are an expert at reading exam answer keys / marking schemes.

Read this answer key document carefully and extract EVERY question with its correct answer.

INSTRUCTIONS:
1. Find each question number (Q1, Q2, 1., 2., (a), (b), etc.)
2. Extract the correct answer for each question
3. Determine the marks allocated for each question (look for marks in brackets like [3], (3 marks), etc.)
4. Identify the format: "mcq" if it's A/B/C/D options, "short_answer" for 1-3 mark answers, "long_answer" for 4+ marks
5. If marks aren't explicitly mentioned, estimate based on answer length:
   - Single word/letter/number → 1 mark
   - 1-2 sentences → 2 marks  
   - 3-5 sentences → 3 marks
   - Paragraph or more → 5 marks
6. For MCQs, extract both the option letter AND the answer text if visible
7. If questions have sub-parts (a, b, c), treat each sub-part as a separate question
   numbered like 1a, 1b, 1c or 1.1, 1.2, 1.3

IMPORTANT:
- Extract ALL questions, don't skip any
- If you can't read something clearly, include what you can read and note it
- For diagram-based answers, describe the expected diagram in words
- Preserve mathematical notation as-is

Respond ONLY with valid JSON (no markdown, no backticks):
{
  "questions": [
    {
      "q_no": 1,
      "question_text": "Brief question text or topic if visible",
      "correct_answer": "The correct answer extracted from the key",
      "marks": 3,
      "format": "short_answer"
    }
  ],
  "total_marks": 30,
  "total_questions": 10,
  "extraction_notes": "Any issues encountered while reading the document"
}"""

    # Upload file
    logger.info(f"Uploading answer key file: {file_path}")
    try:
        uploaded_file = client.files.upload(file=file_path)
    except Exception as e:
        logger.error(f"Answer key upload failed: {e}")
        raise ValueError(f"Could not upload answer key file: {e}")

    # Extract with retries
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Answer key extraction attempt {attempt}/{max_retries}")
            start = time.time()

            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=[uploaded_file, prompt],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )

            elapsed = round(time.time() - start, 2)
            logger.info(f"Answer key extracted in {elapsed}s")

            result = json.loads(response.text)

            # Validate structure
            if "questions" not in result or not result["questions"]:
                raise ValueError("Gemini returned empty questions list from answer key")

            # Ensure every question has q_no
            for idx, q in enumerate(result["questions"]):
                if "q_no" not in q:
                    q["q_no"] = idx + 1
                if not q.get("marks"):
                    q["marks"] = 1
                if "correct_answer" not in q:
                    q["correct_answer"] = ""

            # Recalculate total_marks (don't trust LLM arithmetic)
            result["total_marks"] = sum(q.get("marks") or 1 for q in result["questions"])
            result["total_questions"] = len(result["questions"])

            logger.info(f"Extracted {result['total_questions']} questions, "
                        f"{result['total_marks']} total marks")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt}: JSON parse failed — {e}")
            last_error = e
            if attempt < max_retries:
                time.sleep(2 * attempt)
        except ValueError as e:
            logger.warning(f"Attempt {attempt}: Validation failed — {e}")
            last_error = e
            if attempt < max_retries:
                time.sleep(2 * attempt)
        except Exception as e:
            logger.warning(f"Attempt {attempt}: Gemini error — {e}")
            last_error = e
            if attempt < max_retries:
                time.sleep(3 * attempt)

    raise ValueError(f"Answer key extraction failed after {max_retries} attempts: {last_error}")


# ═══════════════════════════════════════════════════════════════════════
# GRADING PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════

def _build_prompt(answer_key: dict, strictness: str = "medium") -> str:
    """
    Builds the grading prompt. The answer key dict has shape:
    {
      "questions": [
        {"q_no": 1, "correct_answer": "...", "marks": 3, "format": "short_answer"},
        ...
      ],
      "total_marks": 30,
      "subject": "Science",          # optional metadata
      "class_grade": "10",            # optional
    }
    """
    strictness_instruction = STRICTNESS_PROMPTS.get(strictness, STRICTNESS_PROMPTS["medium"])
    questions_json = json.dumps(answer_key.get("questions", []), indent=2, ensure_ascii=False)
    total_marks = answer_key.get("total_marks", "unknown")

    return f"""You are an experienced Indian school teacher grading student answer sheets.

═══ STRICTNESS: {strictness.upper()} ═══
{strictness_instruction}

═══ ANSWER KEY ({total_marks} total marks) ═══
{questions_json}

═══ GRADING INSTRUCTIONS ═══
1. Read the student's answer sheet carefully (it may be handwritten or printed).
2. **Match by question number**: The student may have answered questions in ANY order.
   Find each question by its number (Q1, Q2, 1., 2., etc.) regardless of where it
   appears on the sheet. Do NOT assume questions are in sequential order.
3. For each question in the answer key, find the student's answer on the sheet.
4. Compare the student's answer against the correct answer and award marks.
5. If a question number cannot be found on the sheet, mark it 0 with feedback
   "Answer not found on sheet".
6. For MCQs: match the option letter/text exactly.
7. For short/long answers: evaluate based on key concepts, not exact wording.
8. For numerical answers: method marks apply even if final answer is wrong.

═══ HANDWRITING NOTES ═══
- Try your best to read handwritten text. If partially illegible, grade what you can read.
- Set legibility_issue=true for any answer you struggled to read.
- Never give 0 solely because handwriting is messy — grade the content you can decipher.

═══ RESPONSE FORMAT ═══
Respond ONLY with valid JSON (no markdown, no backticks):
{{
  "student_answers": [
    {{
      "q_no": 1,
      "matched_on_sheet": true,
      "extracted_answer": "What you read from the student's sheet",
      "marks_awarded": 2,
      "max_marks": 3,
      "confidence": "high",
      "feedback": "Brief reason for marks awarded or deducted",
      "legibility_issue": false
    }}
  ],
  "total_marks_obtained": 18,
  "total_marks_possible": {total_marks},
  "percentage": 60.0,
  "overall_remarks": "Brief overall assessment of the student's performance",
  "readability_score": "good",
  "questions_not_found": [5, 8]
}}

confidence: "high" = clearly readable and grading is certain,
            "medium" = partially readable or answer is ambiguous,
            "low" = barely readable or cannot determine correctness.
readability_score: "good" / "average" / "poor" for overall sheet legibility.
questions_not_found: list of question numbers not found on the sheet."""


# ═══════════════════════════════════════════════════════════════════════
# CORE GRADING FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def grade_answer_sheet(
    file_path: str,
    answer_key: dict,
    strictness: str = "medium",
    max_retries: int = 2,
) -> dict:
    """
    Grade a student's answer sheet against an answer key.

    Args:
        file_path: path to uploaded PDF or image file
        answer_key: dict with "questions" list and optional metadata
        strictness: "easy" | "medium" | "hard" | "extreme"
        max_retries: number of retries on Gemini failure

    Returns:
        Structured result dict with per-question marks and feedback.
    """
    client = _get_client()
    prompt = _build_prompt(answer_key, strictness)

    # Upload file to Gemini
    logger.info(f"Uploading file to Gemini: {file_path}")
    start = time.time()

    try:
        uploaded_file = client.files.upload(file=file_path)
    except Exception as e:
        logger.error(f"File upload to Gemini failed: {e}")
        raise ValueError(f"Could not upload file for grading: {e}")

    upload_time = round(time.time() - start, 2)
    logger.info(f"Upload done in {upload_time}s")

    # Send to Gemini with retries
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Gemini grading attempt {attempt}/{max_retries}")
            gen_start = time.time()

            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=[uploaded_file, prompt],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1,
                },
            )

            gen_time = round(time.time() - gen_start, 2)
            logger.info(f"Gemini responded in {gen_time}s")

            # Parse JSON response
            result = json.loads(response.text)

            # Post-process: ensure all answer key questions are accounted for
            result = _post_process_result(result, answer_key)
            result["_meta"] = {
                "upload_time_s": upload_time,
                "generation_time_s": gen_time,
                "model": settings.GEMINI_MODEL,
                "strictness": strictness,
                "attempt": attempt,
            }
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt}: JSON parse failed — {e}")
            last_error = e
            if attempt < max_retries:
                time.sleep(2 * attempt)
        except Exception as e:
            logger.warning(f"Attempt {attempt}: Gemini error — {e}")
            last_error = e
            if attempt < max_retries:
                time.sleep(3 * attempt)

    logger.error(f"Grading failed after {max_retries} attempts: {last_error}")
    raise ValueError(f"Grading failed after {max_retries} attempts. Please try again.")


# ═══════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ═══════════════════════════════════════════════════════════════════════

def _post_process_result(result: dict, answer_key: dict) -> dict:
    """
    Ensure every question in the answer key appears in the result.
    Recalculate totals for consistency (don't trust LLM math).
    """
    student_answers = result.get("student_answers", [])
    answered_qnos = {a["q_no"] for a in student_answers}
    key_questions = answer_key.get("questions", [])

    # Add missing questions as "not found"
    for kq in key_questions:
        qno = kq.get("q_no") or kq.get("question_number")
        if qno and qno not in answered_qnos:
            student_answers.append({
                "q_no": qno,
                "matched_on_sheet": False,
                "extracted_answer": "",
                "marks_awarded": 0,
                "max_marks": kq.get("marks", 1),
                "confidence": "high",
                "feedback": "Answer not found on sheet",
                "legibility_issue": False,
            })

    # Sort by question number
    student_answers.sort(key=lambda a: a.get("q_no", 0))

    # Recalculate totals (don't trust the LLM's arithmetic)
    total_obtained = sum(a.get("marks_awarded", 0) for a in student_answers)
    total_possible = sum(a.get("max_marks", 0) for a in student_answers)
    percentage = round((total_obtained / total_possible * 100), 1) if total_possible > 0 else 0.0

    result["student_answers"] = student_answers
    result["total_marks_obtained"] = total_obtained
    result["total_marks_possible"] = total_possible
    result["percentage"] = percentage

    # Ensure questions_not_found is accurate
    not_found = [a["q_no"] for a in student_answers if not a.get("matched_on_sheet", True)]
    result["questions_not_found"] = not_found

    # Count low-confidence answers for flagging
    low_conf = [a for a in student_answers if a.get("confidence") == "low"]
    result["needs_review_count"] = len(low_conf)

    return result


# ═══════════════════════════════════════════════════════════════════════
# ANSWER KEY BUILDER FROM SAVED TEST
# ═══════════════════════════════════════════════════════════════════════

def build_answer_key_from_test(test_data: dict, questions: list) -> dict:
    """
    Convert a saved a4ai test (from Supabase) into the answer_key format
    expected by the grading prompt.
    """
    key_questions = []
    for idx, q in enumerate(questions, 1):
        key_questions.append({
            "q_no": q.get("position", idx),
            "question_text": q.get("text", ""),
            "correct_answer": q.get("correct_answer", ""),
            "marks": q.get("marks", 1),
            "format": q.get("format", "short_answer"),
        })

    total_marks = sum(kq["marks"] for kq in key_questions)

    return {
        "questions": key_questions,
        "total_marks": total_marks,
        "subject": test_data.get("subject", ""),
        "class_grade": test_data.get("class_grade", ""),
        "exam_title": test_data.get("exam_title", ""),
    }