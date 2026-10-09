"""
Smart Assignment Checker Service — v1.0

3-Pass Pipeline:
  Pass 1 — PARSE QUESTION PAPER:
            Extract every question, sub-part, and marks from the question paper PDF.
            This defines the MAXIMUM marks structure — never exceeded.

  Pass 2 — PARSE ANSWER KEY:
            Map correct answers / key concepts to each question from Pass 1.

  Pass 3 — GRADE STUDENT SHEET:
            a) Validate student identity: name, class, roll number MUST be present.
               If any missing → return invalid, do not grade.
            b) Extract student's answers (handwritten, any order).
            c) Grade by keyword/meaning relevance against answer key.
            d) Enforce per-question and total marks caps (server-side).

Key rules enforced SERVER-SIDE (not trusted from LLM):
  - marks_awarded[q] <= marks[q] (per question)
  - marks_awarded[sub] <= marks[sub] (per sub-part)
  - sum(marks_awarded) <= total_marks_of_paper
  - All values floored at 0
"""

import json
import logging
import time
import os
from typing import Optional

from google import genai
from google.genai import errors as genai_errors
from app.core.config import settings

logger = logging.getLogger(__name__)

CHECKER_API_KEY = os.getenv("CHECKER_GEMINI_API_KEY")
CHECKER_MODEL = os.getenv("CHECKER_GEMINI_MODEL", "gemini-3.6-flash")

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
    m.strip() for m in os.getenv("CHECKER_GEMINI_FALLBACK_MODELS", "gemini-3.8-flash,gemini-3-flash-preview").split(",")
    if m.strip() and m.strip() != CHECKER_MODEL
]
_OVERLOAD_CODES = {500, 503, 504}
# Give up on a single generate call after this long (Gemini can take 20-50s just to return a 503)
_GENERATE_DEADLINE_S = float(os.getenv("CHECKER_GEMINI_DEADLINE_S", "120"))

# Last model that answered successfully (only used to log when a fallback kicks in)
_preferred_model: Optional[str] = None


def _generate(client: genai.Client, contents: list, tag: str):
    """
    generate_content with resilience against transient Gemini outages.
    Each round tries every model once (primary model first):
      5xx / 429 → move straight to the next model.
    If a whole round fails with 5xx, wait briefly and do another round until the deadline.
    Any other error is raised as-is.
    """
    global _preferred_model
    # Always start with the primary model; fallbacks only when it is down
    models = [CHECKER_MODEL, *CHECKER_FALLBACK_MODELS]

    deadline = time.time() + _GENERATE_DEADLINE_S
    last_error: Optional[Exception] = None
    round_no = 0
    while True:
        round_no += 1
        all_quota = True
        for model in models:
            if time.time() > deadline:
                break
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config={"response_mime_type": "application/json", "temperature": 0.05},
                )
                if model != _preferred_model:
                    logger.info(f"[{tag}] Using {model}")
                _preferred_model = model
                return response
            except genai_errors.APIError as e:
                last_error = e
                if e.code not in _OVERLOAD_CODES and e.code != 429:
                    raise
                all_quota = all_quota and e.code == 429
                logger.warning(f"[{tag}] {model} unavailable ({e.code}), trying next model")

        # Every model is out of quota, or we've run out of time → stop
        if all_quota or time.time() + 5 > deadline:
            break
        logger.warning(f"[{tag}] All models busy (round {round_no}), retrying in 5s")
        time.sleep(5)

    raise last_error


# ═══════════════════════════════════════════════════════════════════════
# PASS 1 — PARSE QUESTION PAPER
# ═══════════════════════════════════════════════════════════════════════

QUESTION_PAPER_PROMPT = """You are reading an exam question paper. Extract its complete structure.

YOUR TASK: Find every question and sub-part with their marks.

RULES:
1. Extract ALL questions — Q1, Q2, 1., 2., etc.
2. If a question has sub-parts (a, b, c or i, ii, iii), extract EACH sub-part separately.
3. Find the marks for each question/sub-part — look for [3], (3 marks), (3M), etc.
   Marks are often given per SECTION instead of per question, e.g. "Section A (1 mark each)",
   "Q6–Q10 carry 3 marks each", "5 × 2 = 10". Apply these to every question in that range.
4. If marks for a sub-part are not explicitly written, distribute parent question marks equally.
5. Internal choice ("OR", "Attempt any 4 of 6"): count marks only for the number of questions
   the student must actually attempt — still list every question.
6. Read the MAXIMUM MARKS printed on the paper header ("Maximum Marks: 50", "M.M. 50",
   "Total Marks: 50", "Full Marks 50") into stated_max_marks. Use null if not printed.
7. Scan the WHOLE paper, every page and every section, so no question is missed.
8. Preserve the question text so we can match student answers correctly.

Respond ONLY with valid JSON (no markdown, no backticks):
{
  "questions": [
    {
      "q_no": "1",
      "question_text": "What is photosynthesis?",
      "marks": 3,
      "has_subparts": false,
      "subparts": []
    },
    {
      "q_no": "2",
      "question_text": "Explain the human digestive system.",
      "marks": 6,
      "has_subparts": true,
      "subparts": [
        {"sub_id": "2a", "text": "Name the organs involved.", "marks": 2},
        {"sub_id": "2b", "text": "Explain the role of the stomach.", "marks": 2},
        {"sub_id": "2c", "text": "What is the function of the small intestine?", "marks": 2}
      ]
    }
  ],
  "stated_max_marks": 50,
  "total_marks": 50,
  "total_questions": 10,
  "paper_notes": "Any observations about the paper format"
}

IMPORTANT:
- sub_id format: parent question number + letter (2a, 2b, 3i, 3ii, etc.)
- marks must be a number, never null. If truly unknown, estimate 1.
- total_marks = sum of ALL top-level question marks (not subparts separately)
- If stated_max_marks is printed, total_marks MUST equal it. If your sum is different,
  you have missed questions or misread marks — re-read the paper and fix it.
"""


def _parse_question_paper(file_path: str, max_retries: int = 2) -> dict:
    """Pass 1: Extract question structure and marks from question paper."""
    client = _get_client()

    logger.info(f"[PASS 1] Parsing question paper: {file_path}")
    uploaded = client.files.upload(file=file_path)

    prompt = QUESTION_PAPER_PROMPT
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[PASS 1] Attempt {attempt}/{max_retries}")
            start = time.time()

            response = _generate(client, [uploaded, prompt], "PASS 1")

            elapsed = round(time.time() - start, 2)
            result = json.loads(response.text)

            # Defensive: ensure marks are numeric and non-null
            total = 0
            for q in result.get("questions", []):
                q["marks"] = max(float(q.get("marks") or 1), 0)
                total += q["marks"]
                for sp in q.get("subparts", []):
                    sp["marks"] = max(float(sp.get("marks") or 1), 0)

            # The max marks printed on the paper is the source of truth.
            # If the extracted questions don't add up to it, questions were missed — retry.
            try:
                stated_max = float(result.get("stated_max_marks") or 0)
            except (TypeError, ValueError):
                stated_max = 0.0

            if stated_max > 0 and abs(total - stated_max) > 0.01:
                found = ", ".join(f"Q{q.get('q_no')}={q['marks']:g}" for q in result.get("questions", []))
                logger.warning(f"[PASS 1] Extracted marks sum to {total:g} but paper states "
                               f"{stated_max:g}. Found: {found}")
                if attempt < max_retries:
                    prompt = (QUESTION_PAPER_PROMPT +
                              f"\n\nCORRECTION: A previous extraction found only these questions/marks: "
                              f"{found} (sum = {total:g}), but the paper states Maximum Marks = "
                              f"{stated_max:g}. Questions or section-wise marks were missed or misread. "
                              f"Re-read every page and section carefully and return the complete structure.")
                    last_error = ValueError(f"marks sum {total:g} != stated {stated_max:g}")
                    continue

            result["total_marks"] = stated_max if stated_max > 0 else total
            result["_extracted_marks_sum"] = total

            result["_parse_time_s"] = elapsed
            result["_uploaded_file"] = uploaded
            logger.info(f"[PASS 1] {result.get('total_questions')} questions, "
                        f"{result['total_marks']} total marks, done in {elapsed}s")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"[PASS 1] Attempt {attempt} JSON error: {e}")
            last_error = e
        except Exception as e:
            logger.warning(f"[PASS 1] Attempt {attempt} error: {e}")
            last_error = e
        if attempt < max_retries:
            time.sleep(2 * attempt)

    raise ValueError(f"Question paper parsing failed: {last_error}")


# ═══════════════════════════════════════════════════════════════════════
# PASS 2 — PARSE ANSWER KEY
# ═══════════════════════════════════════════════════════════════════════

def _build_answer_key_prompt(paper_structure: dict) -> str:
    questions_json = json.dumps(paper_structure.get("questions", []), indent=2, ensure_ascii=False)
    return f"""You are reading an exam answer key / marking scheme.

The question paper has this structure:
{questions_json}

YOUR TASK:
For EACH question and sub-part in the structure above, extract the correct answer
and the KEY CONCEPTS that must be present for full marks.

RULES:
1. Match answers to question numbers exactly as in the structure.
2. Extract the FULL correct answer text.
3. List KEY CONCEPTS — the important words/ideas a student must mention:
   - For definitions: the term + its meaning
   - For processes: the steps in order
   - For numerical: the method + the answer
   - For MCQ: just the correct option
4. If an answer has sub-parts, extract answers for each sub-part separately.
5. Do NOT invent answers — only extract what is in the answer key document.

Respond ONLY with valid JSON (no markdown):
{{
  "answers": [
    {{
      "q_no": "1",
      "correct_answer": "Full answer text from the key",
      "key_concepts": ["concept1", "concept2", "concept3"],
      "answer_type": "definition",
      "subpart_answers": []
    }},
    {{
      "q_no": "2",
      "correct_answer": "See sub-parts",
      "key_concepts": [],
      "answer_type": "descriptive",
      "subpart_answers": [
        {{
          "sub_id": "2a",
          "correct_answer": "Mouth, oesophagus, stomach, small intestine, large intestine",
          "key_concepts": ["mouth", "oesophagus", "stomach", "small intestine", "large intestine"]
        }},
        {{
          "sub_id": "2b",
          "correct_answer": "The stomach churns food and secretes HCl and enzymes...",
          "key_concepts": ["churns", "HCl", "pepsin", "enzymes", "protein digestion"]
        }}
      ]
    }}
  ]
}}

answer_type values: "mcq" | "definition" | "short" | "descriptive" | "numerical" | "diagram"
"""


def _parse_answer_key(file_path: str, paper_structure: dict, max_retries: int = 2) -> dict:
    """Pass 2: Extract correct answers mapped to question structure."""
    client = _get_client()
    prompt = _build_answer_key_prompt(paper_structure)

    logger.info(f"[PASS 2] Parsing answer key: {file_path}")
    uploaded = client.files.upload(file=file_path)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[PASS 2] Attempt {attempt}/{max_retries}")
            start = time.time()

            response = _generate(client, [uploaded, prompt], "PASS 2")

            elapsed = round(time.time() - start, 2)
            result = json.loads(response.text)
            result["_key_time_s"] = elapsed
            logger.info(f"[PASS 2] Extracted {len(result.get('answers', []))} answers in {elapsed}s")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"[PASS 2] Attempt {attempt} JSON error: {e}")
            last_error = e
        except Exception as e:
            logger.warning(f"[PASS 2] Attempt {attempt} error: {e}")
            last_error = e
        if attempt < max_retries:
            time.sleep(2 * attempt)

    raise ValueError(f"Answer key parsing failed: {last_error}")


# ═══════════════════════════════════════════════════════════════════════
# PASS 3 — GRADE STUDENT SHEET
# ═══════════════════════════════════════════════════════════════════════

def _build_grading_prompt(paper_structure: dict, answer_key: dict, strictness: str) -> str:
    strictness_rules = {
        "easy":    "Award marks generously. 40%+ key concept coverage = full marks. Accept synonyms.",
        "medium":  "Standard marking. 60%+ coverage = full marks. Accept near-synonyms. Partial marks for partial coverage.",
        "hard":    "Strict. 80%+ coverage needed. Key terms must be present. Minimal leniency.",
        "extreme": "Very strict. 90%+ exact coverage. Technical terms required.",
    }

    questions_json = json.dumps(paper_structure.get("questions", []), indent=2, ensure_ascii=False)
    answers_json = json.dumps(answer_key.get("answers", []), indent=2, ensure_ascii=False)
    total_marks = paper_structure.get("total_marks", 0)

    return f"""You are a teacher grading a handwritten student assignment/answer sheet.

═══ STRICTNESS: {strictness.upper()} ═══
{strictness_rules.get(strictness, strictness_rules['medium'])}

═══ STEP 1: VALIDATE STUDENT IDENTITY ═══
Before grading ANYTHING, find these three fields on the sheet:
- Student Name
- Class / Section
- Roll Number

These may appear at the top, on a cover page, or in a header.
If ANY of these three is missing or completely illegible → set is_valid = false.
If ALL three are found → set is_valid = true and proceed to grading.

═══ STEP 2: EXTRACT STUDENT ANSWERS ═══
Find each question answer on the sheet (student may answer in ANY order).
Read handwriting carefully. Note illegible parts.

═══ STEP 3: GRADE EACH QUESTION ═══

QUESTION PAPER STRUCTURE (marks per question — these are MAXIMUM marks):
{questions_json}

ANSWER KEY (correct answers and key concepts):
{answers_json}

MARKING RULES:
1. Compare student's answer to the correct answer using KEY CONCEPTS as checkpoints.
2. Count how many key concepts the student mentioned (exact or meaningful synonym).
3. Award marks proportionally: (concepts_covered / total_concepts) × max_marks.
4. HARD RULE: marks_awarded CANNOT exceed max_marks for that question/sub-part.
5. For sub-parts: grade each sub-part independently, then sum for parent question.
6. If question not attempted → 0 marks, feedback = "Not attempted".
7. For MCQ: binary only — correct option = full marks, else 0.
8. For numerical: award method marks even if final answer is wrong.
9. Diagrams: check labels. Unlabeled = 50% of diagram marks.

MARKS GRADING SCALE:
- 90-100% concepts covered → full marks
- 70-89% concepts covered → 80% of marks
- 50-69% concepts covered → 60% of marks
- 30-49% concepts covered → 40% of marks
- 10-29% concepts covered → 20% of marks
- 0-9% concepts covered → 0 marks

Total marks of paper: {total_marks}. Your sum must NEVER exceed this.

Respond ONLY with valid JSON (no markdown):
{{
  "is_valid": true,
  "student_info": {{
    "name": "Rahul Sharma",
    "class": "10B",
    "roll_no": "23",
    "missing_fields": []
  }},
  "invalid_reason": "",
  "graded_questions": [
    {{
      "q_no": "1",
      "max_marks": 3,
      "marks_awarded": 2,
      "student_answer_summary": "What the student wrote (brief)",
      "key_concepts_found": ["photosynthesis", "sunlight", "glucose"],
      "key_concepts_missed": ["chlorophyll", "oxygen release"],
      "feedback": "Covered main idea but missed chlorophyll and oxygen release.",
      "confidence": "high",
      "legibility_issue": false,
      "not_attempted": false,
      "has_subparts": false,
      "subpart_grades": []
    }},
    {{
      "q_no": "2",
      "max_marks": 6,
      "marks_awarded": 5,
      "student_answer_summary": "See sub-parts",
      "key_concepts_found": [],
      "key_concepts_missed": [],
      "feedback": "See sub-part feedback",
      "confidence": "high",
      "legibility_issue": false,
      "not_attempted": false,
      "has_subparts": true,
      "subpart_grades": [
        {{
          "sub_id": "2a",
          "max_marks": 2,
          "marks_awarded": 2,
          "student_answer_summary": "Student listed mouth, stomach, intestines",
          "key_concepts_found": ["mouth", "stomach", "intestines"],
          "key_concepts_missed": [],
          "feedback": "All organs named correctly."
        }},
        {{
          "sub_id": "2b",
          "max_marks": 2,
          "marks_awarded": 1,
          "student_answer_summary": "Stomach mixes food",
          "key_concepts_found": ["mixes food"],
          "key_concepts_missed": ["HCl", "pepsin", "protein digestion"],
          "feedback": "Only mentioned mixing. Missed HCl, pepsin and protein digestion."
        }}
      ]
    }}
  ],
  "total_marks_awarded": 7,
  "max_marks": {total_marks},
  "overall_remarks": "2-3 sentences about the student's overall performance",
  "readability_score": "good"
}}

missing_fields: list of what is missing, e.g. ["roll_no"] or ["name", "class"]
confidence: "high" | "medium" | "low"
readability_score: "good" | "average" | "poor"
"""


def _grade_student_sheet(
    sheet_path: str,
    paper_structure: dict,
    answer_key: dict,
    strictness: str,
    max_retries: int = 2,
) -> dict:
    """Pass 3: Validate student identity and grade the handwritten sheet."""
    client = _get_client()
    prompt = _build_grading_prompt(paper_structure, answer_key, strictness)

    logger.info(f"[PASS 3] Grading student sheet: {sheet_path}")
    uploaded = client.files.upload(file=sheet_path)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[PASS 3] Attempt {attempt}/{max_retries}")
            start = time.time()

            response = _generate(client, [uploaded, prompt], "PASS 3")

            elapsed = round(time.time() - start, 2)
            result = json.loads(response.text)
            result["_grade_time_s"] = elapsed
            logger.info(f"[PASS 3] Graded in {elapsed}s, valid={result.get('is_valid')}")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"[PASS 3] Attempt {attempt} JSON error: {e}")
            last_error = e
        except Exception as e:
            logger.warning(f"[PASS 3] Attempt {attempt} error: {e}")
            last_error = e
        if attempt < max_retries:
            time.sleep(2 * attempt)

    raise ValueError(f"Student sheet grading failed: {last_error}")


# ═══════════════════════════════════════════════════════════════════════
# SERVER-SIDE MARKS ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════

def _enforce_marks_constraints(grading: dict, paper_structure: dict) -> dict:
    """
    CRITICAL: Enforce all marks constraints server-side.
    Never trust LLM arithmetic or caps.
    
    Rules:
    - Per sub-part: awarded <= sub-part max
    - Per question: awarded <= question max
    - Total: sum <= paper total
    """
    total_paper_marks = float(paper_structure.get("total_marks", 0))

    # Build a lookup: q_no -> max_marks, sub_id -> max_marks
    q_max = {}
    sub_max = {}
    for q in paper_structure.get("questions", []):
        q_no = str(q.get("q_no", ""))
        q_max[q_no] = float(q.get("marks") or 0)
        for sp in q.get("subparts", []):
            sub_id = str(sp.get("sub_id", ""))
            sub_max[sub_id] = float(sp.get("marks") or 0)

    total_awarded = 0.0
    graded_questions = grading.get("graded_questions", [])

    for gq in graded_questions:
        q_no = str(gq.get("q_no", ""))
        paper_max = q_max.get(q_no, float(gq.get("max_marks") or 0))

        if gq.get("has_subparts") and gq.get("subpart_grades"):
            # Grade each sub-part independently
            subpart_total = 0.0
            for sp in gq["subpart_grades"]:
                sub_id = str(sp.get("sub_id", ""))
                sub_paper_max = sub_max.get(sub_id, float(sp.get("max_marks") or 0))

                sp_awarded = max(0.0, min(float(sp.get("marks_awarded") or 0), sub_paper_max))
                sp["marks_awarded"] = round(sp_awarded, 1)
                sp["max_marks"] = sub_paper_max
                subpart_total += sp_awarded

            # Parent question marks = sum of sub-parts, capped at paper max
            parent_awarded = min(subpart_total, paper_max)
            gq["marks_awarded"] = round(parent_awarded, 1)
        else:
            # Leaf question — cap at paper max
            awarded = max(0.0, min(float(gq.get("marks_awarded") or 0), paper_max))
            gq["marks_awarded"] = round(awarded, 1)

        gq["max_marks"] = paper_max
        total_awarded += gq["marks_awarded"]

    # Final cap: total awarded cannot exceed paper total
    if total_awarded > total_paper_marks:
        logger.warning(f"Total awarded {total_awarded} > paper max {total_paper_marks}. Scaling down.")
        scale = total_paper_marks / total_awarded if total_awarded > 0 else 1
        for gq in graded_questions:
            gq["marks_awarded"] = round(gq["marks_awarded"] * scale, 1)
            for sp in gq.get("subpart_grades", []):
                sp["marks_awarded"] = round(sp["marks_awarded"] * scale, 1)
        total_awarded = total_paper_marks

    grading["graded_questions"] = graded_questions
    grading["total_marks_awarded"] = round(total_awarded, 1)
    grading["max_marks"] = total_paper_marks
    grading["percentage"] = round((total_awarded / total_paper_marks * 100), 1) if total_paper_marks > 0 else 0.0

    return grading


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def check_assignment(
    question_paper_path: str,
    answer_key_path: str,
    student_sheet_path: str,
    strictness: str = "medium",
    max_retries: int = 2,
) -> dict:
    """
    Full 3-pass assignment checking pipeline.

    Args:
        question_paper_path: path to question paper PDF/image
        answer_key_path:     path to answer key PDF/image
        student_sheet_path:  path to student's answer sheet PDF/image
        strictness:          "easy" | "medium" | "hard" | "extreme"

    Returns:
        Structured result with student info validation + per-question grades
    """
    pipeline_start = time.time()
    logger.info(f"Starting assignment check pipeline: strictness={strictness}")

    # Pass 1: Parse question paper
    paper_structure = _parse_question_paper(question_paper_path, max_retries)
    paper_structure.pop("_uploaded_file", None)

    # Pass 2: Parse answer key
    answer_key = _parse_answer_key(answer_key_path, paper_structure, max_retries)

    # Pass 3: Grade student sheet
    grading = _grade_student_sheet(student_sheet_path, paper_structure, answer_key, strictness, max_retries)

    # If student identity invalid → return early without grading
    if not grading.get("is_valid", False):
        return {
            "is_valid": False,
            "student_info": grading.get("student_info", {}),
            "invalid_reason": grading.get("invalid_reason", "Student identity could not be verified."),
            "missing_fields": grading.get("student_info", {}).get("missing_fields", []),
            "graded_questions": [],
            "total_marks_awarded": 0,
            "max_marks": paper_structure.get("total_marks", 0),
            "percentage": 0,
            "overall_remarks": "Assignment cannot be graded: student identity incomplete.",
            "_meta": {
                "pipeline": "3-pass (aborted at identity check)",
                "model": CHECKER_MODEL,
                "strictness": strictness,
            },
        }

    # Enforce marks constraints server-side
    grading = _enforce_marks_constraints(grading, paper_structure)

    pipeline_time = round(time.time() - pipeline_start, 2)

    return {
        "is_valid": True,
        "student_info": grading.get("student_info", {}),
        "graded_questions": grading.get("graded_questions", []),
        "total_marks_awarded": grading.get("total_marks_awarded", 0),
        "max_marks": grading.get("max_marks", 0),
        "percentage": grading.get("percentage", 0),
        "overall_remarks": grading.get("overall_remarks", ""),
        "readability_score": grading.get("readability_score", ""),
        "paper_info": {
            "total_questions": paper_structure.get("total_questions", 0),
            "total_marks": paper_structure.get("total_marks", 0),
        },
        "_meta": {
            "pipeline": "3-pass (parse paper → parse key → grade sheet)",
            "model": CHECKER_MODEL,
            "strictness": strictness,
            "parse_paper_time_s": paper_structure.get("_parse_time_s", 0),
            "parse_key_time_s": answer_key.get("_key_time_s", 0),
            "grade_time_s": grading.get("_grade_time_s", 0),
            "total_time_s": pipeline_time,
        },
    }