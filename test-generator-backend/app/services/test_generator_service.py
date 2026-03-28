"""
Generation Service v12 — PRODUCTION + ACCOUNTANCY

Changes from v11:
  - Accountancy table formats: journal_entry, ledger, trial_balance
  - Specialized Gemini prompts for commerce subjects
  - answer_table field in GeneratedQuestion
  - All v11 features retained (CBSE sections, Unicode math, etc.)
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
    AnswerTable,
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
    "journal_entry", "ledger", "trial_balance",
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
    "hard": "HARD: 3+ steps, 2+ concepts combined. Bloom: apply/analyze. Distractors: wrong formula, sign error, misconception.",
    "very_hard": "VERY HARD: Olympiad level, 3+ concepts, non-routine. All options plausible. Bloom: analyze/evaluate/create.",
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
# Accountancy Constants (NEW v12)
# ---------------------------------------------------------------------------
ACCOUNTANCY_SUBJECTS = {"accountancy", "accounts", "accounting"}

ACCOUNTANCY_TABLE_FORMATS = {"journal_entry", "ledger", "trial_balance"}

ACCOUNTANCY_PROMPT_TEMPLATES = {
    "journal_entry": """Generate a Journal Entry question for CBSE Class {class_grade} Accountancy.

The question should describe 3-5 business transactions that the student must journalize.
The answer MUST include an "answer_table" with:
- type: "journal_entry"
- headers: ["Date", "Particulars", "L.F.", "Debit (₹)", "Credit (₹)"]
- rows: Each row is a list of 5 strings. For the credit entry, indent the Particulars with "  To " prefix.
  After each transaction pair, add a narration row like ["", "(Being ...)", "", "", ""]
- total_row: null (journal entries don't have totals)

Example answer_table:
{{
  "type": "journal_entry",
  "headers": ["Date", "Particulars", "L.F.", "Debit (₹)", "Credit (₹)"],
  "rows": [
    ["2024-04-01", "Cash A/c  Dr.", "", "50,000", ""],
    ["", "  To Capital A/c", "", "", "50,000"],
    ["", "(Being capital introduced in cash)", "", "", ""],
    ["2024-04-03", "Purchases A/c  Dr.", "", "20,000", ""],
    ["", "  To Cash A/c", "", "", "20,000"],
    ["", "(Being goods purchased for cash)", "", "", ""]
  ],
  "total_row": null
}}""",

    "ledger": """Generate a Ledger preparation question for CBSE Class {class_grade} Accountancy.

The question should give transactions and ask to prepare a specific ledger account (T-account format).
The answer MUST include an "answer_table" with:
- type: "ledger"
- headers: ["Date", "Particulars", "J.F.", "Amount (₹)", "Date", "Particulars", "J.F.", "Amount (₹)"]
  (Left 4 columns = Debit side, Right 4 columns = Credit side)
- rows: Each row has 8 strings. Use "" for empty cells.
- total_row: 8 strings with totals on both sides

Example answer_table:
{{
  "type": "ledger",
  "headers": ["Date", "Particulars", "J.F.", "Amount (₹)", "Date", "Particulars", "J.F.", "Amount (₹)"],
  "rows": [
    ["2024-04-01", "To Capital A/c", "", "50,000", "2024-04-05", "By Purchases A/c", "", "20,000"],
    ["2024-04-10", "To Sales A/c", "", "30,000", "2024-04-15", "By Rent A/c", "", "5,000"],
    ["", "", "", "", "2024-04-30", "By Balance c/d", "", "55,000"]
  ],
  "total_row": ["", "", "", "80,000", "", "", "", "80,000"]
}}""",

    "trial_balance": """Generate a Trial Balance preparation question for CBSE Class {class_grade} Accountancy.

Give a list of ledger balances and ask the student to prepare a Trial Balance.
The answer MUST include an "answer_table" with:
- type: "trial_balance"
- headers: ["S.No.", "Account Name", "L.F.", "Debit (₹)", "Credit (₹)"]
- rows: Each row has 5 strings.
- total_row: ["", "Total", "", "X,XXX", "X,XXX"] (both sides must match)

Example answer_table:
{{
  "type": "trial_balance",
  "headers": ["S.No.", "Account Name", "L.F.", "Debit (₹)", "Credit (₹)"],
  "rows": [
    ["1", "Cash A/c", "", "50,000", ""],
    ["2", "Capital A/c", "", "", "1,00,000"],
    ["3", "Purchases A/c", "", "40,000", ""],
    ["4", "Sales A/c", "", "", "60,000"],
    ["5", "Rent A/c", "", "5,000", ""],
    ["6", "Furniture A/c", "", "65,000", ""]
  ],
  "total_row": ["", "Total", "", "1,60,000", "1,60,000"]
}}""",
}


# ---------------------------------------------------------------------------
# CBSE Section Templates
# ---------------------------------------------------------------------------
CBSE_SECTIONS = {
    "A": {
        "title": "Section A",
        "subtitle": "Multiple Choice Questions / Assertion-Reason",
        "marks_per_q": 1,
        "count": 20,
        "total_marks": 20,
        "formats": ["mcq", "assertion_reason"],
        "mcq_count": 16,
        "ar_count": 4,
        "difficulty": "easy",
        "bloom": ["remember", "understand"],
        "instruction": "All questions are compulsory. Each question carries 1 mark.",
    },
    "B": {
        "title": "Section B",
        "subtitle": "Very Short Answer Type Questions",
        "marks_per_q": 2,
        "count": 5,
        "total_marks": 10,
        "formats": ["short_answer"],
        "difficulty": "medium",
        "bloom": ["understand", "apply"],
        "instruction": "All questions are compulsory. Each question carries 2 marks.",
    },
    "C": {
        "title": "Section C",
        "subtitle": "Short Answer Type Questions",
        "marks_per_q": 3,
        "count": 6,
        "total_marks": 18,
        "formats": ["short_answer"],
        "difficulty": "medium",
        "bloom": ["apply", "analyze"],
        "instruction": "All questions are compulsory. Each question carries 3 marks.",
    },
    "D": {
        "title": "Section D",
        "subtitle": "Long Answer Type Questions",
        "marks_per_q": 5,
        "count": 4,
        "total_marks": 20,
        "formats": ["long_answer"],
        "difficulty": "hard",
        "bloom": ["analyze", "evaluate"],
        "instruction": "All questions are compulsory. Each question carries 5 marks.",
    },
    "E": {
        "title": "Section E",
        "subtitle": "Case Study / Source Based Questions",
        "marks_per_q": 4,
        "count": 3,
        "total_marks": 12,
        "formats": ["case_based"],
        "difficulty": "hard",
        "bloom": ["apply", "analyze", "evaluate"],
        "instruction": "All questions are compulsory. Each question carries 4 marks. Each case study has sub-parts.",
    },
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
# MATH FORMATTING INSTRUCTIONS
# ---------------------------------------------------------------------------
MATH_FORMAT_INSTRUCTION = """MATH FORMATTING RULES (CRITICAL — follow exactly):
• Use UNICODE symbols directly: α β γ θ π σ φ ω ε δ λ μ Σ Π Δ
• Fractions: write as (numerator/denominator), e.g. (3/4), (x+1/x-1)
• Square root: √(x), cube root: ∛(x)
• Powers: x², x³, xⁿ, x^(n+1) for complex exponents
• Subscripts: a₁, a₂, xₙ or a_n for complex subscripts
• Inequalities: ≤ ≥ ≠ ≈ 
• Set notation: ∈ ∉ ∪ ∩ ⊂ ⊃ ⊆ ⊇ ∅ ℝ ℤ ℕ ℚ
• Arrows: → ⇒ ⇔ ←
• Logical: ∀ ∃ ∴ ∵
• Operations: × ÷ ± ∓ · ∞
• Limits: lim(x→2), lim(n→∞)
• Summation: Σ(k=1 to n), Integral: ∫(0 to π)
• Combinations: C(n,r) or ⁿCᵣ, Permutations: P(n,r) or ⁿPᵣ
• Absolute value: |x|, floor: ⌊x⌋, ceil: ⌈x⌉

DO NOT use LaTeX commands like \\frac, \\sqrt, \\theta, \\left, \\right, \\mathbb.
DO NOT use $ delimiters.
Write clean readable text that a teacher can read directly."""


# ---------------------------------------------------------------------------
# Compact Prompt Builder (per chapter, per section)
# ---------------------------------------------------------------------------
def _build_chapter_prompt(
    chapter: ChapterSection,
    request: TestGenerationRequest,
    context_chunks: List[Dict],
    count: int,
    section_key: str = None,
    section_info: dict = None,
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

    section_ctx = ""
    if section_key and section_info:
        section_ctx = f"""
This is for {section_info['title']} ({section_info['subtitle']}).
Each question: {section_info['marks_per_q']} marks. {section_info.get('instruction', '')}"""

    if fmt_val == "short_answer":
        if section_info and section_info.get("marks_per_q", 2) == 2:
            fmt_line = '"options": null. Answer: 30-50 words. Show 2 clear steps.'
        else:
            fmt_line = '"options": null. Answer: 50-80 words. Show 3 clear steps with working.'
    elif fmt_val == "long_answer":
        # Accountancy long answer branch
        is_acc = request.subject.lower() in ACCOUNTANCY_SUBJECTS
        if is_acc:
            fmt_line = (
                '"options": null. '
                'This is an Accountancy question — the answer MUST include an "answer_table" field. '
                'Choose the most appropriate table type for the question: '
                '"journal_entry", "ledger", or "trial_balance". '
                'Follow the exact answer_table structure from your Accountancy training. '
                '"correct_answer" should be a 2-3 line text summary. '
                '"explanation" must explain each accounting entry step-by-step.'
            )
        else:
            fmt_line = '"options": null. Answer: 100-150 words. Show complete step-by-step solution with diagrams description if needed.'
    elif fmt_val == "assertion_reason":
        fmt_line = '"text": "Assertion (A): [statement]\\nReason (R): [statement]". Use these 4 options exactly:\n"A) Both A and R are true and R is the correct explanation of A"\n"B) Both A and R are true but R is NOT the correct explanation of A"\n"C) A is true but R is false"\n"D) A is false but R is true"'
    elif fmt_val == "case_based":
        fmt_line = '"text": Start with a real-world case/scenario (3-4 lines), then ask 3 sub-parts labeled (i), (ii), (iii) within the text. "options": provide 4 options for each sub-part OR set null if subjective sub-parts. Answer all sub-parts in correct_answer.'
    else:
        fmt_line = '4 options labeled A) B) C) D). correct_answer = exact full option text including label. Vary correct answer position (not always A or B). All 4 options must be plausible.'

    section_field = f', "section": "{section_key}"' if section_key else ''

    if fmt_val in ("short_answer", "long_answer"):
        is_acc_long = (
            fmt_val == "long_answer"
            and request.subject.lower() in ACCOUNTANCY_SUBJECTS
        )
        if is_acc_long:
            tmpl = (
                f'{{"questions":[{{"text":"...","format":"long_answer","options":null,'
                f'"correct_answer":"Summary of accounting entries...","explanation":"Step 1:... Step 2:...","answer_table":{{"type":"journal_entry","headers":[...],"rows":[[...]],"total_row":null}},'
                f'"marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"apply","chapter":"{chapter.chapter}","topic":"specific topic"{section_field}}}]}}'
            )
        else:
            tmpl = f'{{"questions":[{{"text":"...","format":"{fmt_val}","options":null,"correct_answer":"full detailed answer","explanation":"Step 1:... Step 2:... Final:...","marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"apply","chapter":"{chapter.chapter}","topic":"specific topic"{section_field}}}]}}'
    else:
        tmpl = f'{{"questions":[{{"text":"...","format":"{fmt_val}","options":["A) ...","B) ...","C) ...","D) ..."],"correct_answer":"B) exact option","explanation":"Step 1:... Step 2:... Answer:...","marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"apply","chapter":"{chapter.chapter}","topic":"specific topic"{section_field}}}]}}'

    return f"""You are an expert CBSE Class {request.class_grade} {request.subject} paper setter.
{section_ctx}
Chapter: {chapter.chapter}
Difficulty: {DIFF_INST.get(diff_val, DIFF_INST["medium"])}

FORMAT RULES:
{fmt_line}

{MATH_FORMAT_INSTRUCTION}

"format" must be exactly "{fmt_val}". "chapter" must be exactly "{chapter.chapter}".
Every question MUST have a detailed explanation (min 3 lines showing complete working).
Each question must test a DIFFERENT concept/topic from this chapter. No repeated concepts.

NCERT Reference:
{ctx}

Generate EXACTLY {count} unique questions. Return ONLY valid JSON, no extra text:
{tmpl}"""


# ---------------------------------------------------------------------------
# Accountancy Prompt Builder (NEW v12)
# ---------------------------------------------------------------------------
def _build_accountancy_prompt(
    chapter,
    request,
    context_chunks,
    count,
    table_format,
    section_key=None,
    section_info=None,
):
    """Build a specialized prompt for Accountancy table-based questions."""
    ch_name = chapter.chapter.upper()
    ch_chunks = [c for c in context_chunks if c.get("chapter", "").upper() == ch_name]
    if not ch_chunks:
        ch_chunks = context_chunks[:8]

    ctx = ""
    for i, chunk in enumerate(ch_chunks[:8], 1):
        content = (chunk.get("content") or "")[:2000]
        if content:
            ctx += f"\n[{i}] {content}\n"

    diff_val = chapter.difficulty.value if hasattr(chapter.difficulty, 'value') else str(chapter.difficulty)

    section_ctx = ""
    if section_key and section_info:
        section_ctx = f"""
This is for {section_info['title']} ({section_info['subtitle']}).
Each question: {section_info['marks_per_q']} marks. {section_info.get('instruction', '')}"""

    template_instruction = ACCOUNTANCY_PROMPT_TEMPLATES.get(table_format, "")
    template_instruction = template_instruction.format(class_grade=request.class_grade)

    section_field = f', "section": "{section_key}"' if section_key else ''

    return f"""You are an expert CBSE Class {request.class_grade} Accountancy paper setter.
{section_ctx}
Chapter: {chapter.chapter}
Difficulty: {DIFF_INST.get(diff_val, DIFF_INST["medium"])}

{template_instruction}

IMPORTANT RULES:
- All amounts must use Indian number format (e.g., 1,00,000 not 100,000)
- Use realistic business scenarios relevant to the chapter
- Each question must test a DIFFERENT concept
- "correct_answer" should be a text summary of the answer
- "answer_table" is the structured table (MANDATORY for this format)
- "explanation" must explain the accounting principle and each entry

NCERT Reference:
{ctx}

Generate EXACTLY {count} unique questions. Return ONLY valid JSON:
{{"questions":[{{"text":"...","format":"{table_format}","options":null,"correct_answer":"Summary of entries...","explanation":"Step-by-step accounting logic...","answer_table":{{"type":"{table_format}","headers":[...],"rows":[[...]],"total_row":null}},"marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"apply","chapter":"{chapter.chapter}","topic":"specific topic"{section_field}}}]}}"""


# ---------------------------------------------------------------------------
# JSON Extraction (improved)
# ---------------------------------------------------------------------------
def _fix_latex_json_escapes(text: str) -> str:
    text = text.replace('\\\\', '\x00DBL\x00')
    for prefix in ['frac', 'forall', 'binom', 'boxed', 'bold', 'not', 'neq',
                    'nabla', 'right', 'rangle', 'rceil', 'rfloor', 'times',
                    'text', 'theta', 'tan', 'therefore', 'triangle', 'tilde',
                    'left', 'sqrt', 'mathbb', 'mathrm', 'mathbf', 'overline',
                    'underline', 'sin', 'cos', 'log', 'lim', 'sum', 'int',
                    'infty', 'alpha', 'beta', 'gamma', 'delta', 'pi', 'sigma',
                    'cup', 'cap', 'subset', 'supset', 'in', 'notin', 'emptyset',
                    'leq', 'geq', 'cdot', 'pm', 'circ', 'degree']:
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
# Post-process: clean any LaTeX that Gemini still sneaks in
# ---------------------------------------------------------------------------
UNICODE_REPLACEMENTS = {
    r'\times': '×', r'\div': '÷', r'\pm': '±', r'\cdot': '·',
    r'\leq': '≤', r'\geq': '≥', r'\neq': '≠', r'\approx': '≈',
    r'\infty': '∞', r'\therefore': '∴', r'\because': '∵',
    r'\cup': '∪', r'\cap': '∩', r'\in': '∈', r'\notin': '∉',
    r'\subset': '⊂', r'\emptyset': '∅', r'\forall': '∀', r'\exists': '∃',
    r'\rightarrow': '→', r'\Rightarrow': '⇒', r'\to': '→',
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\theta': 'θ', r'\pi': 'π', r'\sigma': 'σ', r'\phi': 'φ',
    r'\omega': 'ω', r'\lambda': 'λ', r'\mu': 'μ', r'\epsilon': 'ε',
    r'\left': '', r'\right': '',
    r'\bigl': '', r'\bigr': '',
    r'\Bigl': '', r'\Bigr': '',
    r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan',
    r'\log': 'log', r'\ln': 'ln', r'\lim': 'lim',
    r'\sec': 'sec', r'\csc': 'csc', r'\cot': 'cot',
}


def _clean_gemini_text(text: str) -> str:
    if not text:
        return text

    result = text
    result = result.replace('₹', 'Rs.')
    result = re.sub(r'\$([^$]+)\$', r'\1', result)

    for latex, uni in sorted(UNICODE_REPLACEMENTS.items(), key=lambda x: -len(x[0])):
        result = result.replace(latex, uni)

    for _ in range(3):
        result = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1/\2)', result)
    for _ in range(3):
        result = re.sub(r'(?<![a-zA-Z])frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1/\2)', result)

    result = re.sub(r'\\sqrt\[([^]]*)\]\{([^}]*)\}', r'\1√(\2)', result)
    result = re.sub(r'\\sqrt\{([^}]*)\}', r'√(\1)', result)

    mathbb_map = {'R': 'ℝ', 'Z': 'ℤ', 'N': 'ℕ', 'Q': 'ℚ', 'C': 'ℂ'}
    for letter, symbol in mathbb_map.items():
        result = result.replace(f'\\mathbb{{{letter}}}', symbol)
        result = result.replace(f'mathbb{{{letter}}}', symbol)  
        result = re.sub(rf'(?<![a-zA-Z])mathbb\s*{letter}(?![a-zA-Z])', symbol, result)

    result = re.sub(r'\\(?:text|mathrm|mathbf|textbf)\{([^}]*)\}', r'\1', result)
    result = result.replace('\\setminus', ' \\ ')
    result = result.replace('setminus', ' \\ ')
    result = re.sub(r'\\binom\{([^}]*)\}\{([^}]*)\}', r'C(\1,\2)', result)
    result = re.sub(r'\\(?:overline|underline|bar|hat|tilde|vec)\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\([a-zA-Z]+)\{([^}]*)\}', r'\2', result)
    result = re.sub(r'\\([a-zA-Z]+)', '', result)
    result = result.replace('{', '').replace('}', '')
    lines = result.split('\n')
    result = '\n'.join(re.sub(r' +', ' ', line).strip() for line in lines).strip()

    return result


# ---------------------------------------------------------------------------
# Question Parser (v12 — with answer_table support)
# ---------------------------------------------------------------------------
def _parse_batch(raw: str, chapter: ChapterSection, request: TestGenerationRequest,
                 section_key: str = None) -> List[GeneratedQuestion]:
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
    is_accountancy = request.subject.lower() in ACCOUNTANCY_SUBJECTS

    for idx, q in enumerate(raw_qs):
        if not isinstance(q, dict):
            continue

        text = _clean_gemini_text((q.get("text") or "").strip())
        if not text or len(text) < 15:
            dropped += 1
            continue

        norm = re.sub(r'[^a-z0-9]', '', text.lower())
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
        correct = _clean_gemini_text((q.get("correct_answer") or "").strip())
        explanation = _clean_gemini_text((q.get("explanation") or "").strip())

        if not explanation or len(explanation) < 10:
            dropped += 1
            continue

        if isinstance(options, list):
            options = [_clean_gemini_text(o) for o in options]

        # ── Parse answer_table for Accountancy (NEW v12) ──
        answer_table = None
        raw_table = q.get("answer_table")
        if raw_table and isinstance(raw_table, dict):
            try:
                table_type = raw_table.get("type", "")
                headers = raw_table.get("headers", [])
                rows = raw_table.get("rows", [])
                total_row = raw_table.get("total_row")

                if headers and rows and table_type:
                    expected_cols = len(headers)
                    clean_rows = []
                    for row in rows:
                        if isinstance(row, list):
                            padded = (row + [""] * expected_cols)[:expected_cols]
                            clean_rows.append([str(cell) for cell in padded])

                    clean_total = None
                    if total_row and isinstance(total_row, list):
                        clean_total = ([str(c) for c in total_row] + [""] * expected_cols)[:expected_cols]

                    answer_table = AnswerTable(
                        type=table_type,
                        headers=[str(h) for h in headers],
                        rows=clean_rows,
                        total_row=clean_total,
                    )
                    logger.info(f"  Parsed {table_type} table: {len(clean_rows)} rows")
            except Exception as e:
                logger.warning(f"  answer_table parse failed: {e}")
                answer_table = None

        # ── Format-specific validation ──
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

        elif fmt in ACCOUNTANCY_TABLE_FORMATS:
            options = None
            if not answer_table and (not correct or len(correct) < 10):
                dropped += 1
                continue

        elif fmt == "case_based":
            if not correct or len(correct) < 15:
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

        q_section = q.get("section") or section_key

        try:
            gq = GeneratedQuestion(
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
                answer_table=answer_table,
            )
            gq._section = q_section
            questions.append(gq)
        except Exception as e:
            logger.warning(f"Q{idx} ({chapter.chapter}): {e}")
            dropped += 1

    if dropped:
        logger.info(f"  {chapter.chapter}: dropped {dropped}, kept {len(questions)}")
    return questions


# ---------------------------------------------------------------------------
# Gemini Call with Retry (FIX: Added response_mime_type="application/json")
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
                    response_mime_type="application/json",  # ← ADDED: Force JSON output
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
# Generate for ONE chapter (v12 — routes Accountancy to special prompt)
# ---------------------------------------------------------------------------
def _generate_for_chapter(client, chapter, request, context_chunks, models,
                          section_key=None, section_info=None):
    target = chapter.quantity
    ask = target + settings.OVERSHOOT_PER_CHAPTER
    batch_size = settings.BATCH_SIZE
    is_accountancy = request.subject.lower() in ACCOUNTANCY_SUBJECTS
    fmt_val = chapter.format.value if hasattr(chapter.format, 'value') else str(chapter.format)

    logger.info(f"  '{chapter.chapter}': target={target}, fmt={fmt_val}, "
                f"diff={chapter.difficulty}, marks={chapter.marks_per_question}"
                + (f", section={section_key}" if section_key else "")
                + (" [ACCOUNTANCY TABLE]" if is_accountancy and fmt_val in ACCOUNTANCY_TABLE_FORMATS else ""))

    all_qs = []
    remaining = ask
    batch_num = 0

    while remaining > 0 and len(all_qs) < target:
        bc = min(remaining, batch_size)
        batch_num += 1

        # Route to Accountancy prompt for table formats
        if is_accountancy and fmt_val in ACCOUNTANCY_TABLE_FORMATS:
            prompt = _build_accountancy_prompt(
                chapter, request, context_chunks, bc,
                fmt_val, section_key, section_info
            )
        else:
            prompt = _build_chapter_prompt(
                chapter, request, context_chunks, bc,
                section_key, section_info
            )

        batch_qs = []
        for m in models:
            try:
                raw = _call_gemini(client, prompt, m)
                batch_qs = _parse_batch(raw, chapter, request, section_key)
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
# CBSE Section-based Paper Generation
# ---------------------------------------------------------------------------
def _distribute_chapters_to_sections(chapters: List[ChapterSection]) -> Dict[str, List[dict]]:
    ch_names = [ch.chapter for ch in chapters]
    num_chapters = len(ch_names)

    distribution = {}

    for sec_key, sec_info in CBSE_SECTIONS.items():
        total_q = sec_info["count"]
        base = total_q // num_chapters
        remainder = total_q % num_chapters

        sec_dist = []
        for i, ch_name in enumerate(ch_names):
            count = base + (1 if i < remainder else 0)
            if count > 0:
                sec_dist.append({"chapter": ch_name, "count": count})

        distribution[sec_key] = sec_dist

    return distribution


def generate_cbse_paper(request, context_chunks, feedback=None):
    if not context_chunks:
        raise GenerationError("No NCERT content found. Verify chapters.", 404)

    client = _get_gemini_client()
    model = settings.GEMINI_GEN_MODEL
    models = [model]
    fallback = getattr(settings, 'GEMINI_FALLBACK_MODEL', None)
    if fallback and fallback != model:
        models.append(fallback)

    distribution = _distribute_chapters_to_sections(request.chapters)

    total_expected = sum(sec["count"] for sec in CBSE_SECTIONS.values())
    logger.info(f"CBSE Paper: {len(request.chapters)} chapters, {total_expected} questions, model={model}")

    all_questions = []
    t0 = time.time()

    for sec_key, sec_info in CBSE_SECTIONS.items():
        sec_chapters = distribution.get(sec_key, [])
        logger.info(f"\n{'='*50}")
        logger.info(f"{sec_info['title']}: {sec_info['count']} questions × {sec_info['marks_per_q']} marks")

        for ch_entry in sec_chapters:
            ch_name = ch_entry["chapter"]
            count = ch_entry["count"]

            orig_ch = next((c for c in request.chapters if c.chapter == ch_name), None)
            if not orig_ch:
                continue

            formats = sec_info["formats"]
            if sec_key == "A":
                mcq_count = sec_info.get("mcq_count", 16)
                ar_count = sec_info.get("ar_count", 4)
                total_a = mcq_count + ar_count

                ch_mcq = max(1, round(count * mcq_count / total_a))
                ch_ar = count - ch_mcq

                if ch_mcq > 0:
                    mcq_chapter = ChapterSection(
                        chapter=ch_name,
                        difficulty=DifficultyLevel(sec_info["difficulty"]),
                        format=QuestionFormat("mcq"),
                        marks_per_question=sec_info["marks_per_q"],
                        quantity=ch_mcq,
                        topics=orig_ch.topics if hasattr(orig_ch, 'topics') else None,
                    )
                    qs = _generate_for_chapter(client, mcq_chapter, request, context_chunks,
                                               models, sec_key, sec_info)
                    all_questions.extend(qs)
                    time.sleep(settings.BATCH_DELAY)

                if ch_ar > 0:
                    ar_chapter = ChapterSection(
                        chapter=ch_name,
                        difficulty=DifficultyLevel(sec_info["difficulty"]),
                        format=QuestionFormat("assertion_reason"),
                        marks_per_question=sec_info["marks_per_q"],
                        quantity=ch_ar,
                        topics=orig_ch.topics if hasattr(orig_ch, 'topics') else None,
                    )
                    qs = _generate_for_chapter(client, ar_chapter, request, context_chunks,
                                               models, sec_key, sec_info)
                    all_questions.extend(qs)
                    time.sleep(settings.BATCH_DELAY)
            else:
                fmt = formats[0]
                sec_chapter = ChapterSection(
                    chapter=ch_name,
                    difficulty=DifficultyLevel(sec_info["difficulty"]),
                    format=QuestionFormat(fmt),
                    marks_per_question=sec_info["marks_per_q"],
                    quantity=count,
                    topics=orig_ch.topics if hasattr(orig_ch, 'topics') else None,
                )
                qs = _generate_for_chapter(client, sec_chapter, request, context_chunks,
                                           models, sec_key, sec_info)
                all_questions.extend(qs)
                time.sleep(settings.BATCH_DELAY)

    elapsed = time.time() - t0
    logger.info(f"CBSE Paper Done: {len(all_questions)}/{total_expected} in {elapsed:.1f}s")

    if not all_questions:
        raise GenerationError("All sections failed.", 500)

    return all_questions


# ---------------------------------------------------------------------------
# Main Entry — supports both modes
# ---------------------------------------------------------------------------
def generate_questions(request, context_chunks, feedback=None, cbse_pattern: bool = False):
    if cbse_pattern:
        return generate_cbse_paper(request, context_chunks, feedback)

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


# ---------------------------------------------------------------------------
# Aliases for endpoint backward compatibility
# ---------------------------------------------------------------------------
generate_test = generate_questions


def handle_feedback(*args, **kwargs):
    """Placeholder — feedback not yet implemented."""
    return None