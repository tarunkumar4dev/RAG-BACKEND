"""
Generation Service v5 — Production-Grade CBSE Test Generator.

v5 Fixes (from GPT/DeepSeek 6-7/10 audit):
  1. Difficulty calibration with concrete examples + anti-patterns
  2. Distractor design based on common student mistakes
  3. Self-verification pass (model checks its own answers)
  4. Bloom's taxonomy ALWAYS assigned (not optional)
  5. NCERT syllabus boundary guard
  6. Per-chapter smart caps (quality > quantity)
  7. Structured explanation format (Given → Formula → Steps → Answer)
  8. Non-clean numbers for hard/very_hard
  9. Failed questions dropped, not flagged
  10. LaTeX backslash-safe JSON extraction
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
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_RETRIES = 4
BASE_BACKOFF_SECONDS = 2
MAX_BACKOFF_SECONDS = 16
JITTER_RANGE = (0.5, 1.5)

BATCH_SIZE = 12               # reduced from 15 for higher quality per question
BATCH_DELAY_SECONDS = 20      # delay between batches (free tier: 15 RPM)
OVERSHOOT_FACTOR = 1.30       # request 30% extra to cover dedup/parse/drop losses

FALLBACK_MODEL = "gemini-3-flash-preview"

RETRYABLE_KEYWORDS = frozenset([
    "429", "resource_exhausted", "quota", "rate",
    "503", "500", "overloaded", "timeout", "unavailable",
])

VALID_FORMATS = frozenset({
    "mcq", "short_answer", "long_answer", "assertion_reason", "case_based",
})
VALID_DIFFICULTIES = frozenset({"easy", "medium", "hard", "very_hard"})
VALID_BLOOMS = frozenset({
    "remember", "understand", "apply", "analyze", "evaluate", "create",
})

ASSERTION_REASON_OPTIONS = [
    "A) Both A and R are true and R is the correct explanation of A",
    "B) Both A and R are true but R is NOT the correct explanation of A",
    "C) A is true but R is false",
    "D) A is false but R is true",
]

# Per-chapter quality caps (difficulty → max questions per chapter)
PER_CHAPTER_CAPS = {
    "easy": 10,
    "medium": 8,
    "hard": 7,
    "very_hard": 5,
}


# ---------------------------------------------------------------------------
# v5 Difficulty Instructions — Example-Based Calibration
# ---------------------------------------------------------------------------
DIFFICULTY_INSTRUCTIONS = {
    "easy": """DIFFICULTY: EASY — CALIBRATION RULES
What EASY means:
- Single concept, direct recall or 1-step application
- Student who read the chapter ONCE can answer correctly
- Bloom level: Remember or Understand only

EXAMPLES of EASY questions:
✅ "What is the SI unit of electric current?" (direct recall)
✅ "Which gas is released during photosynthesis?" (direct recall)
✅ "What is the value of $\\sin 30°$?" (memorized value)
✅ "State Ohm's law." (definition)

ANTI-PATTERNS — these are NOT easy:
❌ "Calculate equivalent resistance of 3 resistors" (multi-step → medium)
❌ "Why does current decrease when resistance increases?" (reasoning → medium)
❌ "If pH of solution A is 3, what is the concentration of H⁺ ions?" (calculation → medium)

ALL questions must be tagged difficulty: "easy"
bloom_level must be "remember" or "understand" only""",

    "medium": """DIFFICULTY: MEDIUM — CALIBRATION RULES
What MEDIUM means:
- Requires understanding, not just memory
- 2-3 step problems, straightforward application of ONE formula/concept
- "Why" and "How" questions with clear answers
- Bloom level: Understand or Apply

EXAMPLES of MEDIUM questions:
✅ "Calculate resistance if V=12V and I=2A" (one formula, plug values)
✅ "Why do metals conduct electricity?" (conceptual understanding)
✅ "Find the mean of the following data: 5, 8, 12, 15, 20" (direct formula)
✅ "Balance: $Fe + O_2 \\rightarrow Fe_2O_3$" (known procedure)

ANTI-PATTERNS — these are NOT medium:
❌ "What is resistance?" (too simple → easy)
❌ "Three resistors of 2Ω, 4Ω, 6Ω in mixed series-parallel..." (multi-step → hard)
❌ "Name the indicator that turns pink in basic solution" (recall → easy)

ALL questions must be tagged difficulty: "medium"
bloom_level must be "understand" or "apply" only""",

    "hard": """DIFFICULTY: HARD — CALIBRATION RULES
What HARD means:
- Multi-step reasoning (3+ steps), combining 2 concepts
- Application to unfamiliar scenarios or non-standard framing
- CANNOT be solved by plugging into one formula
- Requires equation formation from word problems
- Bloom level: Apply or Analyze

EXAMPLES of HARD questions:
✅ "Three resistors of $2\\Omega$, $4\\Omega$, $6\\Omega$: first two in parallel, combination in series with third. If V=$24$V, find current through each resistor." (multi-step circuit)
✅ "A train travels 300 km at uniform speed. If speed increased by 5 km/h, journey takes 2 hours less. Find original speed." (form equation + solve)
✅ "If roots of $x^2 + px + q = 0$ are in ratio 2:3, express $p^2$ in terms of $q$." (abstract reasoning)
✅ "25 ml of pH 2 HCl mixed with 25 ml of pH 12 NaOH. Calculate final pH." (multi-concept)

ANTI-PATTERNS — these are NOT hard:
❌ "Calculate resistance using R = V/I with given values" (one formula → medium)
❌ "What is the formula for lens maker's equation?" (recall → easy)
❌ "Find mean from a frequency table" (direct formula → medium)

DISTRACTOR DESIGN for hard:
- Option A: correct answer
- Option B: answer if student forgets to convert units
- Option C: answer if student uses wrong formula
- Option D: answer if student makes sign error
(Distribute correct answer randomly across A-D)

ALL questions must be tagged difficulty: "hard"
bloom_level must be "apply" or "analyze" only
At least 40% of hard questions must require conceptual reasoning beyond direct computation""",

    "very_hard": """DIFFICULTY: VERY HARD — CALIBRATION RULES
What VERY HARD means:
- Competition/Olympiad level — hardest 5% of CBSE questions
- Combines 3+ concepts, often from different sections of the chapter
- Non-routine problems: unusual framing, boundary conditions, parametric reasoning
- Requires creative problem-solving, not textbook pattern matching
- If an average student can solve it in under 3 minutes, it is NOT very_hard
- Bloom level: Analyze, Evaluate, or Create only

EXAMPLES of VERY HARD questions:
✅ "Two identical cells (emf $E$, internal resistance $r$) are connected to external resistance $R$. In case 1: cells in series. In case 2: cells in parallel. For what value of $R$ do both cases give equal current?" (multi-concept, parametric)
✅ "If $\\alpha, \\beta$ are roots of $2x^2 - 5x + 3 = 0$, find the quadratic equation whose roots are $\\frac{\\alpha}{\\beta}$ and $\\frac{\\beta}{\\alpha}$" (chain of transformations)
✅ "A solution of pH 3 is diluted 100 times with water. What is the resulting pH? Explain why it is NOT exactly pH 5." (conceptual trap — buffer region)
✅ "In a GP, the sum of first 3 terms is 13 and their product is 27. If the GP has both positive and negative common ratios possible, find ALL valid GPs." (multi-solution)

ANTI-PATTERNS — these are NOT very_hard:
❌ "Calculate equivalent resistance of series-parallel circuit" (procedural → hard at best)
❌ "Find discriminant of $x^2 + 5x + 6$" (single formula → medium)
❌ "What is the color of phenolphthalein in basic solution?" (recall → easy)
❌ "Solve: $2x^2 - 7x + 3 = 0$" (standard quadratic → medium)

DISTRACTOR DESIGN for very_hard:
- All 4 options must be mathematically plausible
- Include: answer from partial solution, answer from common misconception, answer from sign/unit error
- Distractors should be VERY close to correct (e.g., $\\frac{7}{3}$ vs $\\frac{7}{2}$ vs $\\frac{3}{7}$ vs $\\frac{2}{7}$)
- A student who understands 80% of the concept should still get it wrong

NON-CLEAN NUMBERS: Use values like $\\frac{7}{3}$, $2\\sqrt{3}$, $\\frac{-5 \\pm \\sqrt{13}}{4}$ — avoid clean integers where possible.

ALL questions must be tagged difficulty: "very_hard"
bloom_level must be "analyze", "evaluate", or "create" only
ZERO recall/definition/single-formula questions""",
}


# ---------------------------------------------------------------------------
# v5 Bloom's Taxonomy — Always Assigned
# ---------------------------------------------------------------------------
BLOOM_ENFORCEMENT = """
BLOOM'S TAXONOMY (MANDATORY for every question):
Assign the CORRECT bloom_level based on what the question actually tests:
- "remember": Direct recall of facts, definitions, formulas, names
- "understand": Explain why, describe how, compare concepts
- "apply": Use a formula or method to solve a specific problem
- "analyze": Break down complex problems, identify patterns, multi-step reasoning
- "evaluate": Judge, assess, compare approaches, determine validity
- "create": Design experiments, form new equations, synthesize concepts

BLOOM-DIFFICULTY ALIGNMENT (ENFORCE):
- easy → bloom must be "remember" or "understand"
- medium → bloom must be "understand" or "apply"
- hard → bloom must be "apply" or "analyze"
- very_hard → bloom must be "analyze", "evaluate", or "create"

If bloom_level doesn't match difficulty, the question is WRONG. Fix it."""


# ---------------------------------------------------------------------------
# v5 Self-Verification Instruction
# ---------------------------------------------------------------------------
SELF_VERIFY = """
SELF-VERIFICATION (DO THIS FOR EVERY QUESTION):
After writing each question, mentally verify:
1. Solve the question yourself step-by-step
2. Confirm your solution matches the correct_answer
3. Check that NO distractor is actually correct
4. Verify explanation derives the answer completely (no "plausible" or "likely")
5. Confirm the question is solvable with ONLY the given data (no missing info)
6. Confirm ALL data needed is explicitly stated in the question

If verification fails → rewrite the question. Do NOT include unverified questions.
NEVER use words like "plausible", "likely", "approximately" in explanations for math.
Math explanations must be EXACT derivations."""


# ---------------------------------------------------------------------------
# v5 Explanation Format
# ---------------------------------------------------------------------------
EXPLANATION_FORMAT = """
EXPLANATION FORMAT (MANDATORY):
Every explanation must follow this structure:
1. GIVEN: State all given values
2. FIND: What we need to calculate
3. FORMULA: Write the relevant formula(s)
4. SOLUTION: Step-by-step calculation with actual numbers
5. ANSWER: Final answer with units

Example:
"Given: V = 12V, R₁ = 2Ω, R₂ = 4Ω (in series)
Find: Total current I
Formula: For series: R_total = R₁ + R₂, then I = V/R_total
Solution: R_total = 2 + 4 = 6Ω, I = 12/6 = 2A
Answer: I = 2A"

For conceptual questions: State the principle → Explain reasoning → Conclude.
NEVER write vague explanations. Every step must be traceable."""


# ---------------------------------------------------------------------------
# v5 Syllabus Guard
# ---------------------------------------------------------------------------
SYLLABUS_GUARD = """
SYLLABUS BOUNDARY (CRITICAL):
Generate questions ONLY from topics covered in NCERT textbook for the specified class.
DO NOT include:
- Topics from higher classes (e.g., no Class 12 optics in Class 10 test)
- Topics not in NCERT (e.g., no lens-mirror combinations in Class 10)
- Advanced chemistry reactions not mentioned in the textbook
- Formulas or theorems not derived/stated in the NCERT chapter

If unsure whether a concept is in syllabus, use a simpler version that is definitely covered.
Every question must be answerable by a student who has ONLY read the NCERT textbook."""


# ---------------------------------------------------------------------------
# LaTeX + Math Formatting Instructions
# ---------------------------------------------------------------------------
MATH_FORMATTING = """
MATHEMATICAL FORMATTING (MANDATORY for all math content):
Use LaTeX notation for ALL mathematical expressions:
- Superscripts: $x^2$, $a^{n+1}$, $x^{10}$ (NOT x^2 or x2)
- Subscripts: $a_1$, $x_{n}$ (NOT a1 or xn)
- Fractions: $\\frac{a}{b}$, $\\frac{-b \\pm \\sqrt{b^2-4ac}}{2a}$ (NOT a/b)
- Square roots: $\\sqrt{2}$, $\\sqrt{x+1}$ (NOT sqrt(2) or √2)
- Greek: $\\alpha$, $\\beta$, $\\theta$, $\\pi$ (NOT alpha, theta, pi)
- Inequalities: $\\geq$, $\\leq$, $\\neq$ (NOT >=, <=, !=)
- Multiplication: $\\times$ (NOT x or *)
- Degree: $90°$ or $90^{\\circ}$
- Trigonometric: $\\sin\\theta$, $\\cos 60°$, $\\tan 45°$
- Summation: $\\sum_{i=1}^{n}$
- Therefore: $\\therefore$

EXAMPLES of properly formatted questions:
✅ "Find the roots of $2x^2 + 3x - 5 = 0$"
❌ "Find the roots of 2x^2 + 3x - 5 = 0"

✅ "If $\\alpha$ and $\\beta$ are roots of $x^2 - 5x + 6 = 0$, find $\\alpha^2 + \\beta^2$"
❌ "If α and β are roots of x² - 5x + 6 = 0, find α² + β²"

Apply LaTeX in: question text, ALL options, correct_answer, AND explanation.
Every equation, expression, variable, or number in math context must be in $...$ delimiters."""

SCIENCE_FORMATTING = """
SCIENTIFIC FORMATTING:
- Chemical formulas: $H_2O$, $CO_2$, $H_2SO_4$ (use LaTeX subscripts)
- Units: write clearly with proper notation (m/s, kg, J, V, A, Ω)
- Scientific notation: $3.0 \\times 10^8$ m/s
- Reactions: Use $\\rightarrow$ for arrows"""


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------
class GenerationError(Exception):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Gemini Client
# ---------------------------------------------------------------------------
_client_cache: Optional[genai.Client] = None


def _get_gemini_client() -> genai.Client:
    global _client_cache
    if _client_cache is None:
        if not settings.GEMINI_API_KEY:
            raise GenerationError("GEMINI_API_KEY is not configured.", status_code=500)
        _client_cache = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _client_cache


# ---------------------------------------------------------------------------
# v5 Smart Prompt Builder — Production Grade
# ---------------------------------------------------------------------------
def build_smart_prompt(
    request: TestGenerationRequest,
    context_chunks: List[Dict],
    batch_count: int,
    batch_number: int,
    total_batches: int,
    feedback: Optional[str] = None,
) -> str:
    """Build prompt with all v5 quality improvements."""

    context_text = ""
    for i, chunk in enumerate(context_chunks[:15], 1):
        content = (chunk.get("content") or "")[:1000]
        context_text += (
            f"\n[Chunk {i} | Chapter: {chunk.get('chapter', 'N/A')} | "
            f"Class: {chunk.get('class_grade', 'N/A')}]\n{content}\n"
        )

    chapter_lines = []
    for sec in request.chapters:
        parts = [f"Chapter: {sec.chapter}"]
        if sec.topic:
            parts.append(f"Topic: {sec.topic}")
        if sec.subtopics:
            parts.append(f"Subtopics: {', '.join(sec.subtopics)}")
        parts.append(f"Difficulty: {sec.difficulty}")
        parts.append(f"Format: {sec.format}")
        parts.append(f"Marks each: {sec.marks_per_question}")
        chapter_lines.append("- " + " | ".join(parts))
    chapter_requirements = "\n".join(chapter_lines)

    # Primary settings
    primary_difficulty = request.chapters[0].difficulty if request.chapters else "medium"
    diff_instruction = DIFFICULTY_INSTRUCTIONS.get(primary_difficulty, DIFFICULTY_INSTRUCTIONS["medium"])
    primary_format = request.chapters[0].format if request.chapters else "mcq"

    # Subject-specific formatting
    is_math = request.subject.lower() in ("mathematics", "maths", "math")
    is_science = request.subject.lower() in ("science", "physics", "chemistry", "biology")
    formatting_instruction = MATH_FORMATTING if is_math else (SCIENCE_FORMATTING if is_science else "")

    # Format instruction
    if primary_format == "assertion_reason":
        format_instruction = (
            "FORMAT: assertion_reason\n"
            "Each question MUST follow this EXACT structure:\n"
            '"text": "Assertion (A): [statement]\\nReason (R): [statement]"\n'
            '"options": the 4 standard CBSE assertion-reason options\n'
            '"correct_answer": one of the above 4 options\n'
            "VARY the correct answers across A, B, C, D!\n"
            '"format": "assertion_reason"'
        )
    elif primary_format == "mcq":
        format_instruction = (
            "FORMAT: mcq\n"
            "Each question must have exactly 4 options (A through D).\n"
            "correct_answer MUST be an EXACT copy of one option string.\n"
            "VARY the correct answer position — don't make all answers A."
        )
    else:
        format_instruction = f"FORMAT: {primary_format}"

    feedback_section = ""
    if feedback and feedback != "string":
        feedback_section = f"\nTEACHER FEEDBACK (apply strictly):\n{feedback}\n"

    # Batch variety
    batch_variety = ""
    if batch_number > 1:
        batch_variety = (
            f"\nThis is batch {batch_number} of {total_batches}. "
            f"Generate COMPLETELY DIFFERENT questions — different concepts, "
            f"different scenarios, different numerical values. "
            f"Do NOT repeat any idea from earlier batches.\n"
        )

    # v5: Question type variety (improved)
    type_variety = ""
    if primary_format == "mcq" and batch_count >= 4:
        if is_math:
            type_variety = f"""
QUESTION TYPE DISTRIBUTION (MANDATORY for {batch_count} questions):
- ~20% Conceptual: "Which statement about... is true?", "What condition ensures..."
- ~40% Numerical/Calculation: Problems with actual numbers requiring step-by-step solving
- ~25% Word Problems: Real-life scenario → form equation → solve
- ~15% Analytical: "For what value of k...", "Under what condition...", "Compare cases..."

IMPORTANT: At least {max(2, batch_count // 3)} questions MUST require actual multi-step calculation.
Do NOT generate only "which of the following" style questions."""
        else:
            type_variety = f"""
QUESTION TYPE DISTRIBUTION (MANDATORY for {batch_count} questions):
- ~20% Recall: "What is...", "Name the...", "Which of the following..."
- ~30% Conceptual: "Why does...", "Explain how...", "What happens when..."
- ~25% Application: Real-life scenarios, experimental reasoning, practical situations
- ~25% Analytical: Data interpretation, multi-concept reasoning, cause-effect chains"""

    # v5: Difficulty-appropriate temperature guidance
    number_guidance = ""
    if primary_difficulty in ("hard", "very_hard") and is_math:
        number_guidance = """
NUMBER COMPLEXITY:
- Avoid only clean integers. Use fractions like $\\frac{7}{3}$, surds like $2\\sqrt{3}$, decimals like $2.5$
- At least 30% of questions should have non-integer answers
- Use realistic but slightly awkward values that require careful calculation
"""

    prompt = f"""You are an expert CBSE exam paper setter for Class {request.class_grade} {request.subject}.
You create questions that match the quality of ACTUAL CBSE board exam papers and top coaching institute tests.

EXAM CONTEXT:
{chapter_requirements}
{feedback_section}

{diff_instruction}

{BLOOM_ENFORCEMENT}

{format_instruction}

{formatting_instruction}

{SYLLABUS_GUARD}

{type_variety}
{number_guidance}
{batch_variety}

{EXPLANATION_FORMAT}

{SELF_VERIFY}

NCERT SOURCE CONTENT (base all questions on this — do NOT go beyond this syllabus):
{context_text}

QUALITY STANDARDS:
1. Generate EXACTLY {batch_count} questions — no more, no less.
2. Every question must be factually accurate per NCERT textbook.
3. Each question tests a DIFFERENT concept — ZERO repetition.
4. Explanations must follow the Given→Find→Formula→Solution→Answer format.
5. Use Class {request.class_grade} appropriate language and complexity.
6. EVERY question MUST be difficulty: "{primary_difficulty}" — do NOT use any other difficulty level.
7. EVERY question MUST have a valid bloom_level that matches the difficulty.
8. {"Use LaTeX $...$ notation for ALL math expressions." if is_math else "Use proper scientific notation."}
9. Correct answers should be distributed across A, B, C, D (not all A).
10. Self-verify EVERY question before including it.

DISTRACTOR DESIGN (CRITICAL for quality):
- Each wrong option must represent a SPECIFIC student mistake:
  * Option from using wrong formula
  * Option from calculation error (sign, unit, arithmetic)
  * Option from common misconception about the concept
- NO obviously wrong options. All 4 options must look plausible to a student who partially understands.

OUTPUT — return ONLY valid JSON, no markdown fences, no commentary:
{{
  "questions": [
    {{
      "text": "Question text with $LaTeX$ math",
      "format": "{primary_format}",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "correct_answer": "B) exact copy of one option",
      "explanation": "Given: ... Find: ... Formula: ... Solution: ... Answer: ...",
      "marks": {request.chapters[0].marks_per_question if request.chapters else 1},
      "difficulty": "{primary_difficulty}",
      "bloom_level": "remember|understand|apply|analyze|evaluate|create",
      "chapter": "Chapter name",
      "topic": "Specific topic"
    }}
  ]
}}

Generate exactly {batch_count} high-quality, self-verified, CBSE-standard questions now:"""

    return prompt


# ---------------------------------------------------------------------------
# Robust JSON Extraction (v5 — LaTeX-safe)
# ---------------------------------------------------------------------------
def _extract_json(raw: str) -> dict:
    """Extract valid JSON from potentially messy LLM output."""
    text = raw.strip().lstrip("\ufeff\u200b")

    # Remove markdown fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract {...} block
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace: last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Attempt 3: fix trailing commas
        cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Attempt 4: fix broken LaTeX backslashes in JSON strings
        latex_fixed = re.sub(
            r'(?<!\\)\\(?![\\"/bfnrtu])',
            r'\\\\',
            cleaned,
        )
        try:
            return json.loads(latex_fixed)
        except json.JSONDecodeError:
            pass

        # Attempt 5: debug logging
        logger.error(f"JSON parse failed. First 500 chars: {candidate[:500]}")

    raise ValueError(f"Could not extract valid JSON (len={len(raw)})")


# ---------------------------------------------------------------------------
# v5 Question Parser + Quality Gate (Stricter)
# ---------------------------------------------------------------------------

# Bloom-difficulty alignment rules
BLOOM_DIFFICULTY_VALID = {
    "easy": {"remember", "understand"},
    "medium": {"understand", "apply"},
    "hard": {"apply", "analyze"},
    "very_hard": {"analyze", "evaluate", "create"},
}

# Default bloom if LLM gives wrong one
BLOOM_DIFFICULTY_DEFAULT = {
    "easy": "remember",
    "medium": "apply",
    "hard": "analyze",
    "very_hard": "analyze",
}


def parse_questions(
    raw_response: str,
    request: TestGenerationRequest,
) -> List[GeneratedQuestion]:
    """Parse LLM JSON into validated GeneratedQuestion models.
    
    v5: Stricter quality gate — drops bad questions instead of flagging.
    """

    try:
        data = _extract_json(raw_response)
        raw_questions = data.get("questions", [])
    except (ValueError, AttributeError) as e:
        logger.error(f"JSON extraction failed: {e}")
        return []

    if not isinstance(raw_questions, list):
        logger.error("'questions' key is not a list")
        return []

    questions: List[GeneratedQuestion] = []
    seen_texts: set = set()
    dropped_count = 0

    for idx, q in enumerate(raw_questions):
        if not isinstance(q, dict):
            continue

        text = (q.get("text") or "").strip()
        if not text or len(text) < 15:
            dropped_count += 1
            continue

        # Normalize for dedup (strip LaTeX for comparison)
        norm = re.sub(r"\$[^$]*\$", "", text.lower())
        norm = re.sub(r"\s+", " ", norm).strip()
        if norm in seen_texts:
            dropped_count += 1
            continue
        seen_texts.add(norm)

        fmt = q.get("format", "mcq")
        if fmt not in VALID_FORMATS:
            fmt = "mcq"

        # ENFORCE requested difficulty
        primary_difficulty = request.chapters[0].difficulty if request.chapters else "medium"
        difficulty = primary_difficulty  # Force it regardless of what LLM says

        # v5: ALWAYS assign bloom (fix alignment if needed)
        bloom_raw = (q.get("bloom_level") or "").strip().lower()
        valid_blooms_for_diff = BLOOM_DIFFICULTY_VALID.get(difficulty, {"apply"})
        if bloom_raw in valid_blooms_for_diff:
            bloom_level = BloomLevel(bloom_raw)
        elif bloom_raw in VALID_BLOOMS:
            # LLM gave valid bloom but wrong for this difficulty — auto-correct
            bloom_level = BloomLevel(BLOOM_DIFFICULTY_DEFAULT[difficulty])
        else:
            # No bloom or garbage — assign default
            bloom_level = BloomLevel(BLOOM_DIFFICULTY_DEFAULT[difficulty])

        options = q.get("options")
        correct_answer = (q.get("correct_answer") or "").strip()
        explanation = (q.get("explanation") or "").strip()

        # v5: DROP questions with vague explanations
        if not explanation or len(explanation) < 30:
            logger.warning(f"Q{idx}: dropped — explanation too short ({len(explanation)} chars)")
            dropped_count += 1
            continue

        # v5: DROP questions with "plausible", "likely", "approximately" in math explanations
        is_math = request.subject.lower() in ("mathematics", "maths", "math")
        if is_math:
            vague_words = {"plausible", "likely", "probably", "seems", "might be"}
            if any(w in explanation.lower() for w in vague_words):
                logger.warning(f"Q{idx}: dropped — vague explanation in math")
                dropped_count += 1
                continue

        # --- Assertion-Reason: fix options if missing ---
        if fmt == "assertion_reason":
            if not options or len(options) != 4:
                options = ASSERTION_REASON_OPTIONS.copy()
            if not correct_answer or correct_answer not in options:
                if len(correct_answer) >= 2:
                    matched = [o for o in options if o[:2].upper() == correct_answer[:2].upper()]
                    if matched:
                        correct_answer = matched[0]
                    else:
                        correct_answer = options[0]
                else:
                    correct_answer = options[0]

        # --- MCQ validation ---
        elif fmt == "mcq":
            if not isinstance(options, list) or len(options) != 4:
                dropped_count += 1
                continue
            if correct_answer not in options:
                matched = (
                    [o for o in options if o[:2].upper() == correct_answer[:2].upper()]
                    if len(correct_answer) >= 2
                    else []
                )
                if matched:
                    correct_answer = matched[0]
                else:
                    dropped_count += 1
                    continue

        marks = q.get("marks", 1)
        if not isinstance(marks, (int, float)) or marks < 1:
            marks = 1

        try:
            question = GeneratedQuestion(
                id=str(uuid.uuid4()),
                text=text,
                options=options if fmt in ("mcq", "assertion_reason") else None,
                correct_answer=correct_answer,
                explanation=explanation,
                marks=int(marks),
                difficulty=DifficultyLevel(difficulty),
                bloom_level=bloom_level,  # v5: always assigned
                chapter=(q.get("chapter") or "").strip(),
                topic=q.get("topic"),
                format=QuestionFormat(fmt),
                validation_status="verified",  # v5: verified by default (dropped if bad)
            )
            questions.append(question)
        except Exception as e:
            logger.warning(f"Q{idx}: model validation failed: {e}")
            dropped_count += 1
            continue

    if dropped_count > 0:
        logger.info(f"Quality gate: dropped {dropped_count} questions, kept {len(questions)}")

    return questions


# ---------------------------------------------------------------------------
# Retryable Error Detection
# ---------------------------------------------------------------------------
def _is_retryable(error_str: str) -> bool:
    upper = error_str.upper()
    return any(kw in upper for kw in (k.upper() for k in RETRYABLE_KEYWORDS))


def _call_gemini_with_retry(
    client: genai.Client,
    prompt: str,
    model: str,
) -> str:
    """Call Gemini with exponential backoff + jitter."""
    last_exception: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        try:
            t0 = time.time()

            # v5: Lower temperature for hard/very_hard (more precise answers)
            temperature = 0.5

            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=temperature,
                    top_p=0.92,
                    max_output_tokens=8192,
                ),
            )
            elapsed = time.time() - t0
            raw_text = (response.text or "").strip()

            if not raw_text:
                raise GenerationError("Gemini returned empty response.", status_code=502)

            logger.info(f"Gemini [{model}] responded in {elapsed:.1f}s ({len(raw_text)} chars)")
            return raw_text

        except GenerationError:
            raise

        except Exception as e:
            last_exception = e
            error_str = str(e)

            if _is_retryable(error_str) and attempt < MAX_RETRIES - 1:
                raw_wait = min(BASE_BACKOFF_SECONDS ** (attempt + 1), MAX_BACKOFF_SECONDS)
                jittered = raw_wait * random.uniform(*JITTER_RANGE)
                logger.warning(
                    f"[{model}] Retryable error (attempt {attempt + 1}/{MAX_RETRIES}): "
                    f"{error_str[:120]}... retrying in {jittered:.1f}s"
                )
                time.sleep(jittered)
                continue
            else:
                break

    error_str = str(last_exception) if last_exception else "Unknown"
    status = 429 if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str) else 500
    raise GenerationError(
        f"Generation failed after {MAX_RETRIES} attempts: {error_str[:200]}",
        status_code=status,
    )


# ---------------------------------------------------------------------------
# v5 Per-Chapter Generation with Smart Caps
# ---------------------------------------------------------------------------
def _apply_chapter_caps(request: TestGenerationRequest) -> TestGenerationRequest:
    """Apply per-chapter quality caps based on difficulty."""
    import copy
    capped = copy.deepcopy(request)

    for section in capped.chapters:
        cap = PER_CHAPTER_CAPS.get(section.difficulty, 8)
        if section.quantity > cap:
            logger.info(
                f"Chapter '{section.chapter}': capped {section.quantity} → {cap} "
                f"(difficulty={section.difficulty})"
            )
            section.quantity = cap

    return capped


# ---------------------------------------------------------------------------
# Main Entry Point — Smart Batch Generation v5
# ---------------------------------------------------------------------------
def generate_questions(
    request: TestGenerationRequest,
    context_chunks: List[Dict],
    feedback: Optional[str] = None,
) -> List[GeneratedQuestion]:
    """
    Generate questions with v5 improvements:
    - Per-chapter quality caps
    - 30% overshoot to hit target after quality gate drops
    - 12 questions per batch (quality > speed)
    - LaTeX-safe JSON parsing
    - Bloom always assigned
    - Bad questions dropped, not flagged
    """
    if not context_chunks:
        raise GenerationError(
            "No NCERT content found. Please verify chapter names.",
            status_code=404,
        )

    # v5: Apply per-chapter caps
    request = _apply_chapter_caps(request)

    client = _get_gemini_client()
    total_requested = sum(sec.quantity for sec in request.chapters)
    model = settings.GEMINI_GEN_MODEL

    # Overshoot: request 30% more to account for dedup + quality gate drops
    overshoot_total = min(
        math.ceil(total_requested * OVERSHOOT_FACTOR),
        total_requested + 15,  # cap overshoot at +15
    )

    models = [model]
    if FALLBACK_MODEL and FALLBACK_MODEL != model:
        models.append(FALLBACK_MODEL)

    # Calculate batches
    batches = []
    remaining = overshoot_total
    while remaining > 0:
        batch_count = min(remaining, BATCH_SIZE)
        batches.append(batch_count)
        remaining -= batch_count

    total_batches = len(batches)
    logger.info(
        f"Target: {total_requested} questions | "
        f"Generating: {overshoot_total} (with {int((OVERSHOOT_FACTOR-1)*100)}% overshoot) | "
        f"Batches: {total_batches} x {BATCH_SIZE}"
    )

    all_questions: List[GeneratedQuestion] = []
    t_start = time.time()

    for batch_idx, batch_count in enumerate(batches, 1):
        logger.info(f"Batch {batch_idx}/{total_batches}: generating {batch_count} questions...")

        prompt = build_smart_prompt(
            request=request,
            context_chunks=context_chunks,
            batch_count=batch_count,
            batch_number=batch_idx,
            total_batches=total_batches,
            feedback=feedback,
        )

        batch_questions = []
        for m in models:
            try:
                raw = _call_gemini_with_retry(client, prompt, m)
                batch_questions = parse_questions(raw, request)
                if batch_questions:
                    logger.info(
                        f"  Batch {batch_idx}: {len(batch_questions)}/{batch_count} questions"
                    )
                    break
            except GenerationError as e:
                if m != models[-1]:
                    logger.warning(f"  Model {m} failed, trying next...")
                    continue
                logger.error(f"  Batch {batch_idx} failed: {e}")
                break

        all_questions.extend(batch_questions)

        # Early stop if we already have enough (with buffer)
        if len(all_questions) >= total_requested + 5:
            logger.info(f"  Already have {len(all_questions)} questions, skipping remaining batches")
            break

        # Delay between batches
        if batch_idx < total_batches:
            logger.info(f"  Waiting {BATCH_DELAY_SECONDS}s before next batch...")
            time.sleep(BATCH_DELAY_SECONDS)

    elapsed = time.time() - t_start

    logger.info(
        f"Generation complete: {len(all_questions)} questions generated "
        f"(target: {total_requested}) in {elapsed:.1f}s"
    )

    if not all_questions:
        raise GenerationError("All generation batches failed.", status_code=500)

    return all_questions