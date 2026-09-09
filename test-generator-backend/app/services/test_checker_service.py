"""
Test Checker Service — v2.0 (Production Grade)

ARCHITECTURE: 2-Pass Grading Pipeline
  Pass 1 — EXTRACT: Read the answer sheet, extract what the student wrote
           for each question. No grading here — just accurate OCR/reading.
  Pass 2 — GRADE: Compare extracted answers against answer key using
           subject-specific rubrics and strict marking parameters.

Why 2-pass?
  - Single-pass asks the model to read + grade simultaneously → both suffer.
  - Separating extraction from grading dramatically improves accuracy.
  - Each prompt is focused on ONE task → better results.

Key design:
  - Subject-aware rubrics (Science ≠ English ≠ Math ≠ Accountancy)
  - Format-aware grading (MCQ vs Short vs Long vs Numerical)
  - Keyword/concept coverage scoring for subjective answers
  - Strict partial marks rules (no guessing)
  - Confidence gating with teacher review flags
  - Server-side total recalculation (never trust LLM math)
"""

import json
import logging
import time
import os
from typing import Optional

from google import genai
from app.core.config import settings

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────
CHECKER_API_KEY = os.getenv("CHECKER_GEMINI_API_KEY")
CHECKER_MODEL = os.getenv("CHECKER_GEMINI_MODEL", "gemini-2.5-pro")

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        key = CHECKER_API_KEY or settings.GEMINI_API_KEY
        if not key:
            raise ValueError("No Gemini API key. Set CHECKER_GEMINI_API_KEY in .env")
        _client = genai.Client(api_key=key)
    return _client


# ═══════════════════════════════════════════════════════════════════════
# STRICTNESS CONFIGS (detailed parameters, not vague instructions)
# ═══════════════════════════════════════════════════════════════════════

STRICTNESS_CONFIGS = {
    "easy": {
        "label": "LENIENT",
        "keyword_coverage_for_full": 40,    # % of key concepts needed for full marks
        "keyword_coverage_for_partial": 20,  # % needed for partial marks
        "partial_marks_ratio": 0.6,          # partial = 60% of max marks
        "spelling_penalty": False,
        "grammar_penalty": False,
        "diagram_label_required": False,
        "mcq_partial_allowed": False,        # MCQ is always binary
        "accept_synonyms": True,
        "accept_examples_as_answer": True,
    },
    "medium": {
        "label": "STANDARD (CBSE)",
        "keyword_coverage_for_full": 60,
        "keyword_coverage_for_partial": 30,
        "partial_marks_ratio": 0.5,
        "spelling_penalty": False,
        "grammar_penalty": False,
        "diagram_label_required": True,
        "mcq_partial_allowed": False,
        "accept_synonyms": True,
        "accept_examples_as_answer": True,
    },
    "hard": {
        "label": "STRICT",
        "keyword_coverage_for_full": 80,
        "keyword_coverage_for_partial": 40,
        "partial_marks_ratio": 0.4,
        "spelling_penalty": True,
        "grammar_penalty": False,
        "diagram_label_required": True,
        "mcq_partial_allowed": False,
        "accept_synonyms": False,
        "accept_examples_as_answer": False,
    },
    "extreme": {
        "label": "VERY STRICT",
        "keyword_coverage_for_full": 90,
        "keyword_coverage_for_partial": 50,
        "partial_marks_ratio": 0.3,
        "spelling_penalty": True,
        "grammar_penalty": True,
        "diagram_label_required": True,
        "mcq_partial_allowed": False,
        "accept_synonyms": False,
        "accept_examples_as_answer": False,
    },
}

# ═══════════════════════════════════════════════════════════════════════
# SUBJECT-SPECIFIC RUBRICS
# ═══════════════════════════════════════════════════════════════════════

SUBJECT_RUBRICS = {
    "science": """
SCIENCE GRADING RULES:
- Definitions: Must contain the EXACT technical term + its meaning. Half marks if meaning is correct but term is missing.
- Diagrams: Check labels, arrows, proportions. Missing labels = deduct 1 mark per missing label (max 50% deduction).
- Chemical equations: Must be balanced. Unbalanced equation = 50% marks only. Missing state symbols (s/l/g/aq) = deduct 0.5 marks.
- Numerical problems: Method marks + Answer marks. Correct method but wrong calculation = 60-70% marks. Wrong method = 0.
- Differences/Comparisons: Each valid point = proportional marks. e.g., 3 marks question needs 3 differences.
- Processes (digestion, photosynthesis): Must follow correct sequence. Missing steps = proportional deduction.
""",
    "mathematics": """
MATHEMATICS GRADING RULES:
- Steps are MANDATORY. Correct answer without steps = maximum 1 mark regardless of total marks.
- Each logical step carries marks. Award step-wise marks even if final answer is wrong.
- Calculation errors in otherwise correct method: deduct only 1 mark from total.
- Units must be mentioned in final answer where applicable. Missing units = deduct 0.5 marks.
- Graphs: Check scale, labels, plotted points, line/curve accuracy. Each missing element = -0.5 marks.
- Proofs/Derivations: Logical flow required. Skipping steps = proportional deduction.
- "Hence proved" / "Hence shown" must be written at end of proofs.
""",
    "english": """
ENGLISH GRADING RULES:
- Comprehension passages: Answer must be derived from the passage, not general knowledge.
- Grammar questions: Only ONE correct answer. Partial marks NOT allowed for grammar MCQs.
- Creative writing (essay/letter/notice): Grade on Content (40%), Expression (30%), Organization (20%), Accuracy (10%).
- Letter writing: Must have correct format (sender address, date, subject, salutation, body, closing). Missing format = deduct 1 mark.
- Poetry questions: Must reference the poem. Generic literary answers = 50% marks max.
- Word limit violations: -1 mark if significantly over/under prescribed word limit.
""",
    "social_science": """
SOCIAL SCIENCE GRADING RULES:
- Dates/Years: Must be accurate. Wrong date in a history answer = deduct 1 mark.
- Map work: Location must be approximately correct (+/- reasonable margin). Label required.
- Reasons/Causes: Each distinct reason = proportional marks. Repetition of same point in different words = counted once.
- Case-based: Answer must reference the given case/source material, not just general knowledge.
- Constitutional/Legal questions: Exact article numbers not required unless specifically asked, but concept must be accurate.
""",
    "accountancy": """
ACCOUNTANCY GRADING RULES:
- Journal entries: Debit-Credit must be correct. Wrong side = 0 marks for that entry.
- Narration is compulsory. Missing narration = deduct 0.5 marks per entry.
- Ledger: Correct posting + balancing required. Wrong balance = deduct 1 mark.
- Trial Balance: Must tally. If it doesn't tally, check individual entries and give step marks.
- Financial statements: Format matters. Wrong format = maximum 50% marks.
- Numerical accuracy: Small arithmetic error = deduct 1 mark only if method is correct.
""",
    "default": """
GENERAL GRADING RULES:
- Definitions: Must contain key terms. Vague definitions = 50% marks.
- Short answers: Must be specific and to the point. Irrelevant content = no marks for that part.
- Long answers: Must cover all sub-points. Each valid point = proportional marks.
- Diagrams: Must be labeled. Unlabeled diagram = 50% marks.
- Examples: Correct relevant example = bonus consideration when answer is borderline.
""",
}


# ═══════════════════════════════════════════════════════════════════════
# PASS 1: EXTRACT STUDENT ANSWERS FROM SHEET
# ═══════════════════════════════════════════════════════════════════════

EXTRACTION_PROMPT = """You are a document reader specializing in reading student answer sheets (handwritten and printed).

YOUR ONLY JOB: Read the answer sheet and extract what the student wrote. Do NOT grade or evaluate anything.

INSTRUCTIONS:
1. Scan the ENTIRE document carefully — students may answer in ANY order.
2. For each answer you find, identify:
   - The question number (Q1, Q2, 1., 2., (a), (b), 1a, 1b, etc.)
   - The COMPLETE text of what the student wrote (every word, number, symbol)
   - Whether it includes a diagram, table, or figure
   - Whether the handwriting is clearly legible
3. If a question has sub-parts, extract each sub-part separately (1a, 1b, 1c).
4. If you cannot read certain words, write them as [illegible] but extract everything else.
5. Preserve mathematical notation, chemical formulas, and symbols exactly as written.
6. Note if an answer appears to be crossed out / cancelled by the student.

CRITICAL:
- Extract ALL answers found on ALL pages. Do not stop early.
- Do NOT skip any answer, even if it seems incomplete or wrong.
- Do NOT add your own interpretation — extract EXACTLY what is written.
- If the student wrote nothing for a question, do not include it.
- Order does not matter — the student may have answered Q5 before Q1.

Respond ONLY with valid JSON (no markdown, no backticks):
{
  "extracted_answers": [
    {
      "q_no": 1,
      "q_no_as_written": "Q1",
      "raw_text": "Exact text the student wrote, preserving their words",
      "has_diagram": false,
      "diagram_description": "",
      "is_crossed_out": false,
      "legibility": "clear",
      "page_found_on": 1
    }
  ],
  "total_answers_found": 8,
  "total_pages_scanned": 3,
  "overall_legibility": "clear",
  "notes": "Any observations about the sheet (e.g., some pages are rotated, ink is faded)"
}

legibility values: "clear" / "partial" / "poor"
"""


def _extract_answers_from_sheet(file_path: str, max_retries: int = 2) -> dict:
    """
    Pass 1: Extract student's answers from the answer sheet.
    No grading — just accurate reading/OCR.
    """
    client = _get_client()

    logger.info(f"[PASS 1] Uploading answer sheet: {file_path}")
    uploaded_file = client.files.upload(file=file_path)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[PASS 1] Extraction attempt {attempt}/{max_retries}")
            start = time.time()

            response = client.models.generate_content(
                model=CHECKER_MODEL,
                contents=[uploaded_file, EXTRACTION_PROMPT],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.05,  # near-zero for faithful extraction
                },
            )

            elapsed = round(time.time() - start, 2)
            result = json.loads(response.text)

            n = len(result.get("extracted_answers", []))
            logger.info(f"[PASS 1] Extracted {n} answers in {elapsed}s")

            result["_extraction_time_s"] = elapsed
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"[PASS 1] Attempt {attempt} JSON error: {e}")
            last_error = e
            if attempt < max_retries:
                time.sleep(2 * attempt)
        except Exception as e:
            logger.warning(f"[PASS 1] Attempt {attempt} error: {e}")
            last_error = e
            if attempt < max_retries:
                time.sleep(3 * attempt)

    raise ValueError(f"Answer extraction failed after {max_retries} attempts: {last_error}")


# ═══════════════════════════════════════════════════════════════════════
# PASS 2: GRADE EXTRACTED ANSWERS AGAINST ANSWER KEY
# ═══════════════════════════════════════════════════════════════════════

def _build_grading_prompt(
    extracted_answers: list,
    answer_key: dict,
    strictness: str = "medium",
    subject: str = "",
) -> str:
    """Build the grading prompt with subject rubrics and strict parameters."""

    config = STRICTNESS_CONFIGS.get(strictness, STRICTNESS_CONFIGS["medium"])

    # Pick subject rubric
    subject_lower = (subject or "").lower().strip()
    rubric = SUBJECT_RUBRICS.get(subject_lower, "")
    if not rubric:
        # Try partial match
        for key in SUBJECT_RUBRICS:
            if key in subject_lower or subject_lower in key:
                rubric = SUBJECT_RUBRICS[key]
                break
        if not rubric:
            rubric = SUBJECT_RUBRICS["default"]

    answer_key_json = json.dumps(answer_key.get("questions", []), indent=2, ensure_ascii=False)
    extracted_json = json.dumps(extracted_answers, indent=2, ensure_ascii=False)
    total_marks = answer_key.get("total_marks", "?")

    return f"""You are a senior examiner grading student answers against an answer key.

═══ STRICTNESS LEVEL: {config['label']} ═══

═══ MARKING PARAMETERS (follow these EXACTLY) ═══

FORMAT-SPECIFIC RULES:
┌─────────────────────────────────────────────────────────────────────┐
│ MCQ / OBJECTIVE:                                                     │
│ - BINARY grading only: full marks if correct, 0 if wrong.           │
│ - No partial marks for MCQ. No "close enough".                      │
│ - Match the option letter OR the answer text.                       │
│                                                                      │
│ SHORT ANSWER (1-3 marks):                                            │
│ - Must contain key terms/concepts from the answer key.              │
│ - {config['keyword_coverage_for_full']}%+ key concepts covered = FULL marks           │
│ - {config['keyword_coverage_for_partial']}%-{config['keyword_coverage_for_full']}% key concepts = PARTIAL marks ({int(config['partial_marks_ratio']*100)}% of max)  │
│ - Below {config['keyword_coverage_for_partial']}% coverage = ZERO marks                       │
│                                                                      │
│ LONG ANSWER (4+ marks):                                              │
│ - Break the correct answer into KEY POINTS.                         │
│ - Each key point carries proportional marks.                        │
│ - Student must hit each point to earn those marks.                  │
│ - Example: 5-mark answer with 5 key points = 1 mark per point.     │
│ - Irrelevant content does NOT earn marks (no marks for padding).    │
│                                                                      │
│ NUMERICAL / CALCULATION:                                             │
│ - Steps carry 60% marks, final answer carries 40% marks.           │
│ - Correct answer with NO steps = maximum 1 mark.                   │
│ - Correct steps but arithmetic error = deduct only 1 mark.         │
│ - Wrong method entirely = 0 marks.                                  │
│                                                                      │
│ DIAGRAM-BASED:                                                       │
│ - Diagram present and labeled = full diagram marks.                 │
│ - Diagram present but unlabeled = 50% diagram marks.               │
│ - No diagram when required = 0 for diagram portion.                 │
└─────────────────────────────────────────────────────────────────────┘

ADDITIONAL PARAMETERS:
- Spelling mistakes penalty: {"YES — deduct 0.5 marks for key technical terms misspelled" if config['spelling_penalty'] else "NO — ignore spelling errors"}
- Grammar penalty: {"YES — deduct for grammatically incoherent answers" if config['grammar_penalty'] else "NO — focus on content only"}
- Accept synonyms for key terms: {"YES" if config['accept_synonyms'] else "NO — exact terms from answer key required"}
- Accept examples as valid answer: {"YES" if config['accept_examples_as_answer'] else "NO — definition/explanation required, not just examples"}

{rubric}

═══ ANSWER KEY ({total_marks} total marks) ═══
{answer_key_json}

═══ STUDENT'S EXTRACTED ANSWERS ═══
{extracted_json}

═══ GRADING TASK ═══

For EACH question in the answer key:
1. Find the matching student answer by question number (q_no).
2. If no matching answer found → marks_awarded = 0, feedback = "Not attempted".
3. If answer found but crossed out → marks_awarded = 0, feedback = "Answer cancelled by student".
4. Compare the student's answer against the correct answer.
5. Apply the format-specific rules above.
6. For subjective answers, identify KEY CONCEPTS in the correct answer and check how many the student covered.

GRADING METHODOLOGY for subjective answers:
- Step A: List the key concepts/points from the correct answer (e.g., ["photosynthesis definition", "sunlight", "CO2", "water", "glucose", "oxygen"])
- Step B: Check which of these the student mentioned (even in different words if synonyms are accepted)
- Step C: Calculate coverage = (concepts found / total concepts) × 100
- Step D: Apply marks based on coverage thresholds above

Respond ONLY with valid JSON (no markdown):
{{
  "graded_answers": [
    {{
      "q_no": 1,
      "max_marks": 3,
      "marks_awarded": 2,
      "key_concepts_total": 4,
      "key_concepts_found": 3,
      "concept_coverage_pct": 75,
      "student_answer_summary": "Brief summary of what student wrote",
      "correct_answer_summary": "Brief summary of expected answer",
      "feedback": "Covered photosynthesis definition and reactants correctly. Missed mentioning chlorophyll as the catalyst. -1 mark.",
      "deductions": [
        {{"reason": "Missing key term: chlorophyll", "marks_deducted": 1}}
      ],
      "confidence": "high",
      "legibility_issue": false,
      "format_type": "short_answer"
    }}
  ],
  "total_marks_obtained": 18,
  "total_marks_possible": {total_marks},
  "percentage": 60.0,
  "grade_summary": {{
    "full_marks_count": 4,
    "partial_marks_count": 3,
    "zero_marks_count": 2,
    "not_attempted_count": 1
  }},
  "overall_remarks": "Detailed 2-3 sentence assessment of student performance, specific strengths and weaknesses",
  "readability_score": "good",
  "questions_not_attempted": [5, 8]
}}

confidence: "high" = grading is certain, "medium" = answer was ambiguous, "low" = could not determine.
IMPORTANT: Be PRECISE with marks. Do not round up generously. Follow the parameters above strictly."""


def _grade_answers(
    extracted_answers: list,
    answer_key: dict,
    strictness: str = "medium",
    subject: str = "",
    max_retries: int = 2,
) -> dict:
    """
    Pass 2: Grade extracted answers against the answer key.
    """
    client = _get_client()
    prompt = _build_grading_prompt(extracted_answers, answer_key, strictness, subject)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"[PASS 2] Grading attempt {attempt}/{max_retries}")
            start = time.time()

            response = client.models.generate_content(
                model=CHECKER_MODEL,
                contents=[prompt],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.05,  # near-zero for consistent grading
                },
            )

            elapsed = round(time.time() - start, 2)
            result = json.loads(response.text)

            logger.info(f"[PASS 2] Graded in {elapsed}s")
            result["_grading_time_s"] = elapsed
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"[PASS 2] Attempt {attempt} JSON error: {e}")
            last_error = e
            if attempt < max_retries:
                time.sleep(2 * attempt)
        except Exception as e:
            logger.warning(f"[PASS 2] Attempt {attempt} error: {e}")
            last_error = e
            if attempt < max_retries:
                time.sleep(3 * attempt)

    raise ValueError(f"Grading failed after {max_retries} attempts: {last_error}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE: Extract → Grade → Post-process
# ═══════════════════════════════════════════════════════════════════════

def grade_answer_sheet(
    file_path: str,
    answer_key: dict,
    strictness: str = "medium",
    max_retries: int = 2,
) -> dict:
    """
    Full grading pipeline:
      1. Extract answers from sheet (Pass 1)
      2. Grade against answer key (Pass 2)
      3. Post-process and validate totals
    """
    subject = answer_key.get("subject", "")

    # ── Pass 1: Extract ──
    logger.info(f"Starting 2-pass grading: strictness={strictness}, subject={subject}")
    extraction = _extract_answers_from_sheet(file_path, max_retries)
    extracted_answers = extraction.get("extracted_answers", [])

    if not extracted_answers:
        logger.warning("No answers extracted from sheet — returning all zeros")
        return _build_empty_result(answer_key, extraction)

    # ── Pass 2: Grade ──
    grading = _grade_answers(extracted_answers, answer_key, strictness, subject, max_retries)

    # ── Post-process ──
    result = _post_process_result(grading, answer_key)

    # Merge metadata
    result["_meta"] = {
        "pipeline": "2-pass (extract → grade)",
        "model": CHECKER_MODEL,
        "strictness": strictness,
        "subject": subject,
        "extraction_time_s": extraction.get("_extraction_time_s", 0),
        "grading_time_s": grading.get("_grading_time_s", 0),
        "total_answers_extracted": len(extracted_answers),
        "sheet_legibility": extraction.get("overall_legibility", "unknown"),
        "pages_scanned": extraction.get("total_pages_scanned", 0),
    }

    return result


def _build_empty_result(answer_key: dict, extraction: dict) -> dict:
    """When no answers were extracted at all."""
    questions = answer_key.get("questions", [])
    total_possible = sum(q.get("marks") or 1 for q in questions)

    return {
        "student_answers": [
            {
                "q_no": q.get("q_no", i + 1),
                "matched_on_sheet": False,
                "extracted_answer": "",
                "marks_awarded": 0,
                "max_marks": q.get("marks") or 1,
                "confidence": "high",
                "feedback": "Not attempted — answer not found on sheet",
                "legibility_issue": False,
            }
            for i, q in enumerate(questions)
        ],
        "total_marks_obtained": 0,
        "total_marks_possible": total_possible,
        "percentage": 0.0,
        "overall_remarks": "No answers could be extracted from the submitted sheet.",
        "readability_score": extraction.get("overall_legibility", "poor"),
        "questions_not_found": [q.get("q_no", i + 1) for i, q in enumerate(questions)],
        "needs_review_count": 0,
        "_meta": {
            "pipeline": "2-pass (extraction returned empty)",
            "model": CHECKER_MODEL,
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# POST-PROCESSING (server-side validation)
# ═══════════════════════════════════════════════════════════════════════

def _post_process_result(result: dict, answer_key: dict) -> dict:
    """
    - Normalize field names for frontend compatibility
    - Ensure every answer key question is accounted for
    - Recalculate totals server-side
    - Cap marks (never exceed max)
    - Flag low-confidence answers
    """
    graded = result.get("graded_answers", [])
    key_questions = answer_key.get("questions", [])

    # Build normalized student_answers list
    student_answers = []
    graded_qnos = set()

    for g in graded:
        qno = g.get("q_no", 0)
        max_m = g.get("max_marks") or 1
        awarded = g.get("marks_awarded") or 0

        # Cap: never exceed max marks
        awarded = min(awarded, max_m)
        # Floor: never go below 0
        awarded = max(awarded, 0)

        graded_qnos.add(qno)

        student_answers.append({
            "q_no": qno,
            "matched_on_sheet": True,
            "extracted_answer": g.get("student_answer_summary", ""),
            "marks_awarded": awarded,
            "max_marks": max_m,
            "confidence": g.get("confidence", "medium"),
            "feedback": g.get("feedback", ""),
            "legibility_issue": g.get("legibility_issue", False),
            "key_concepts_total": g.get("key_concepts_total"),
            "key_concepts_found": g.get("key_concepts_found"),
            "concept_coverage_pct": g.get("concept_coverage_pct"),
            "deductions": g.get("deductions", []),
            "format_type": g.get("format_type", ""),
        })

    # Add any missing questions (not attempted)
    for kq in key_questions:
        qno = kq.get("q_no") or kq.get("question_number")
        if qno and qno not in graded_qnos:
            student_answers.append({
                "q_no": qno,
                "matched_on_sheet": False,
                "extracted_answer": "",
                "marks_awarded": 0,
                "max_marks": kq.get("marks") or 1,
                "confidence": "high",
                "feedback": "Not attempted — answer not found on sheet",
                "legibility_issue": False,
                "deductions": [],
            })

    # Sort by question number
    student_answers.sort(key=lambda a: (
        int(str(a.get("q_no", 0)).replace("a", "").replace("b", "").replace("c", "")[:3] or 0),
        str(a.get("q_no", ""))
    ))

    # Server-side totals (NEVER trust LLM arithmetic)
    total_obtained = sum(a["marks_awarded"] for a in student_answers)
    total_possible = sum(a["max_marks"] for a in student_answers)
    percentage = round((total_obtained / total_possible * 100), 1) if total_possible > 0 else 0.0

    not_found = [a["q_no"] for a in student_answers if not a["matched_on_sheet"]]
    low_conf = [a for a in student_answers if a.get("confidence") == "low"]

    return {
        "student_answers": student_answers,
        "total_marks_obtained": total_obtained,
        "total_marks_possible": total_possible,
        "percentage": percentage,
        "overall_remarks": result.get("overall_remarks", ""),
        "readability_score": result.get("readability_score", "unknown"),
        "questions_not_found": not_found,
        "needs_review_count": len(low_conf),
        "grade_summary": result.get("grade_summary", {}),
    }


# ═══════════════════════════════════════════════════════════════════════
# ANSWER KEY EXTRACTION FROM PDF
# ═══════════════════════════════════════════════════════════════════════

ANSWER_KEY_EXTRACTION_PROMPT = """You are an expert at reading exam answer keys and marking schemes.

Read this document and extract EVERY question with its correct answer and marks.

EXTRACTION RULES:
1. Find each question number (Q1, Q2, 1., 2., (a), (b), etc.)
2. Extract the COMPLETE correct answer for each question.
3. Determine marks: look for [3], (3 marks), (3M), etc. 
4. If marks not mentioned, estimate:
   - Single word/letter/option → 1 mark (MCQ)
   - 1-2 sentences → 2 marks (short answer)
   - 3-5 sentences → 3 marks (short answer)
   - Paragraph+ → 5 marks (long answer)
5. Identify format: "mcq" (A/B/C/D), "short_answer" (1-3 marks), "long_answer" (4+), "numerical" (calculation)
6. For MCQs: extract BOTH the option letter AND answer text
7. Sub-parts (a,b,c) = separate questions numbered 1a, 1b, etc.
8. Extract KEY CONCEPTS from each answer (these will be used for grading):
   - For "What is photosynthesis?": key concepts = ["process", "plants", "sunlight", "CO2", "water", "glucose", "oxygen"]

Respond ONLY with valid JSON (no markdown):
{
  "questions": [
    {
      "q_no": 1,
      "question_text": "Question text if visible, or topic",
      "correct_answer": "Full correct answer",
      "key_concepts": ["concept1", "concept2", "concept3"],
      "marks": 3,
      "format": "short_answer"
    }
  ],
  "total_marks": 30,
  "total_questions": 10,
  "subject": "Detected subject if identifiable, else empty string",
  "extraction_notes": "Any issues reading the document"
}"""


def extract_answer_key_from_pdf(file_path: str, max_retries: int = 2) -> dict:
    """Extract structured answer key from PDF/image."""
    client = _get_client()

    logger.info(f"Extracting answer key from: {file_path}")
    uploaded_file = client.files.upload(file=file_path)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Answer key extraction attempt {attempt}/{max_retries}")
            start = time.time()

            response = client.models.generate_content(
                model=CHECKER_MODEL,
                contents=[uploaded_file, ANSWER_KEY_EXTRACTION_PROMPT],
                config={
                    "response_mime_type": "application/json",
                    "temperature": 0.05,
                },
            )

            elapsed = round(time.time() - start, 2)
            result = json.loads(response.text)

            if "questions" not in result or not result["questions"]:
                raise ValueError("Empty questions list from answer key extraction")

            # Defensive: ensure marks are numeric
            for q in result["questions"]:
                if not q.get("q_no"):
                    q["q_no"] = result["questions"].index(q) + 1
                q["marks"] = int(q.get("marks") or 1)
                if not q.get("correct_answer"):
                    q["correct_answer"] = ""
                if not q.get("key_concepts"):
                    q["key_concepts"] = []

            result["total_marks"] = sum(q["marks"] for q in result["questions"])
            result["total_questions"] = len(result["questions"])

            logger.info(f"Extracted {result['total_questions']} questions, "
                        f"{result['total_marks']} marks in {elapsed}s")
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"Attempt {attempt}: JSON parse failed — {e}")
            last_error = e
            if attempt < max_retries:
                time.sleep(2 * attempt)
        except ValueError as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(2 * attempt)
        except Exception as e:
            logger.warning(f"Attempt {attempt}: error — {e}")
            last_error = e
            if attempt < max_retries:
                time.sleep(3 * attempt)

    raise ValueError(f"Answer key extraction failed after {max_retries} attempts: {last_error}")


# ═══════════════════════════════════════════════════════════════════════
# ANSWER KEY FROM SAVED TEST
# ═══════════════════════════════════════════════════════════════════════

def build_answer_key_from_test(test_data: dict, questions: list) -> dict:
    """Convert a saved a4ai test into the answer_key format."""
    key_questions = []
    for idx, q in enumerate(questions, 1):
        key_questions.append({
            "q_no": q.get("position", idx),
            "question_text": q.get("text", ""),
            "correct_answer": q.get("correct_answer", ""),
            "marks": int(q.get("marks") or 1),
            "format": q.get("format", "short_answer"),
            "key_concepts": [],  # not available from saved tests
        })

    return {
        "questions": key_questions,
        "total_marks": sum(kq["marks"] for kq in key_questions),
        "subject": test_data.get("subject", ""),
        "class_grade": test_data.get("class_grade", ""),
        "exam_title": test_data.get("exam_title", ""),
    }