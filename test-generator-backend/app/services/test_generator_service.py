"""
Test Generator Service v14 — STATISTICS SUPPORT + ROBUST JSON

v14 changes (additive only, no breaking changes):
  - Added QuestionTable support for Statistics / Data-handling questions
  - Added STATISTICS_PROMPT_TEMPLATES — forces inline markdown table + structured JSON
  - Added _build_statistics_prompt() — dedicated prompt builder for Statistics chapters
  - Added validation gate in _parse_batch — drops broken Stats questions
    (text mentions "following table" but no table data present)
  - Added robust JSON extraction (handles truncated responses, unescaped newlines
    inside strings — critical for markdown tables in question text)
  - Routes Statistics chapters to specialized prompt automatically

v13 features retained:
  - Accountancy table formats (journal_entry, ledger, trial_balance)
  - CBSE Accountancy paper pattern (Part A + Part B with OR alternatives)
  - Unicode modifier letter cleanup (15ᵗʰ → 15th)
  - Generic CBSE 5-section pattern for Science/Maths
  - Model fallback chain
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
    QuestionTable,  # v14: new
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 2
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
# Accountancy Constants
# ---------------------------------------------------------------------------
ACCOUNTANCY_SUBJECTS = {"accountancy", "accounts", "accounting"}

ACCOUNTANCY_TABLE_FORMATS = {"journal_entry", "ledger", "trial_balance"}

ACCOUNTANCY_PROMPT_TEMPLATES = {
    "journal_entry": """Generate a Journal Entry question for CBSE Class {class_grade} Accountancy.

The question should describe 3-5 business transactions that the student must journalize.
The answer MUST include an "answer_table" with:
- type: "journal_entry"
- headers: ["Date", "Particulars", "L.F.", "Debit (Rs.)", "Credit (Rs.)"]
- rows: Each row is a list of 5 strings. For the credit entry, indent the Particulars with "  To " prefix.
  After each transaction pair, add a narration row like ["", "(Being ...)", "", "", ""]
- total_row: null (journal entries don't have totals)

Example answer_table:
{{
  "type": "journal_entry",
  "headers": ["Date", "Particulars", "L.F.", "Debit (Rs.)", "Credit (Rs.)"],
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
- headers: ["Date", "Particulars", "J.F.", "Amount (Rs.)", "Date", "Particulars", "J.F.", "Amount (Rs.)"]
  (Left 4 columns = Debit side, Right 4 columns = Credit side)
- rows: Each row has 8 strings. Use "" for empty cells.
- total_row: 8 strings with totals on both sides

Example answer_table:
{{
  "type": "ledger",
  "headers": ["Date", "Particulars", "J.F.", "Amount (Rs.)", "Date", "Particulars", "J.F.", "Amount (Rs.)"],
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
- headers: ["S.No.", "Account Name", "L.F.", "Debit (Rs.)", "Credit (Rs.)"]
- rows: Each row has 5 strings.
- total_row: ["", "Total", "", "X,XXX", "X,XXX"] (both sides must match)

Example answer_table:
{{
  "type": "trial_balance",
  "headers": ["S.No.", "Account Name", "L.F.", "Debit (Rs.)", "Credit (Rs.)"],
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


# ═══════════════════════════════════════════════════════════════════════════
# v14 NEW: Statistics / Data-handling Constants
# ═══════════════════════════════════════════════════════════════════════════

# Chapter names that typically need data tables in questions
STATISTICS_CHAPTERS = {
    "statistics",
    "data handling",
    "data analysis",
    "probability",  # often involves frequency data
}

# Topic keywords that indicate table-driven questions
STATISTICS_TOPIC_KEYWORDS = {
    "frequency", "mean", "median", "mode",
    "histogram", "ogive", "cumulative",
    "class interval", "grouped data",
    "frequency distribution", "frequency polygon",
}

# Phrases in question text that REQUIRE a table to be present
TABLE_REQUIRED_TRIGGERS = (
    "following table",
    "following frequency distribution",
    "following frequency",
    "following data",
    "table shows",
    "data given below",
    "given below",
    "the table below",
    "from the table",
    "in the table",
    "based on the data",
    "the data:",
    "calculate the mean of the following",
    "calculate the median of the following",
    "calculate the mode of the following",
    "find the mean of the following",
    "find the median of the following",
    "find the mode of the following",
)


def _is_statistics_question(chapter_name: str, topic: Optional[str] = None) -> bool:
    """Detect if a chapter/topic typically requires data tables in questions."""
    ch_lower = (chapter_name or "").lower().strip()

    for stat_ch in STATISTICS_CHAPTERS:
        if stat_ch in ch_lower:
            return True

    combined = f"{ch_lower} {(topic or '').lower()}"
    for kw in STATISTICS_TOPIC_KEYWORDS:
        if kw in combined:
            return True

    return False


def _has_inline_table(text: str) -> bool:
    """
    Check if text contains a markdown pipe table.
    Looks for at least 2 pipe-separated rows with a header separator line.
    """
    if not text:
        return False
    # Need pipes
    if text.count("|") < 4:
        return False
    # Need a separator line like |---|---|  or  |:---:|---:|
    if not re.search(r"\|[\s\-:]+\|", text):
        return False
    return True


def _references_table(text: str) -> bool:
    """Check if text references a table (trigger phrase)."""
    if not text:
        return False
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in TABLE_REQUIRED_TRIGGERS)


# ═══════════════════════════════════════════════════════════════════════════
# v16: Inline Data Detection + Recovery
#
# Gemini sometimes bypasses both markdown and structured table by writing data
# as comma-separated lists inline:
#   "xi: 10, 20, 36, 40, 50  fi: 1, 1, 3, 4, 3"
#   "Class Interval: 0-10, 10-20, 20-30  Frequency: 5, 12, 15"
#
# We detect these and try to recover into a proper QuestionTable using regex
# patterns rather than fixed keyword lists (more robust to variations).
# ═══════════════════════════════════════════════════════════════════════════


def _extract_inline_data(text: str) -> Optional[dict]:
    """
    Try to extract inline data into a structured table. Tries 3 strategies:

    1. Single-line markdown pipes: "Class | Frequency ---|--- 0-10 | 5 10-20 | 12"
    2. Label-colon pattern: "Class Interval: 0-10, 10-20  Frequency: 5, 12"
    3. xi/fi shorthand: "xi: 10, 20, 36  fi: 1, 1, 3"

    Returns dict like {"headers": [...], "rows": [[...]], "type": "..."} or None.
    """
    if not text:
        return None

    # Strategy 1: Single-line markdown pipes
    result = _extract_singleline_markdown(text)
    if result:
        return result

    # Strategy 2/3: "Label: data, data..." patterns (works for both Class Interval and xi/fi)
    result = _extract_label_colon_pattern(text)
    if result:
        return result

    return None


def _extract_singleline_markdown(text: str) -> Optional[dict]:
    """Extract from single-line markdown like 'Class | Freq ---|--- 0-10 | 5 10-20 | 12'."""
    # Need at least 4 pipes and a separator dash sequence
    if text.count("|") < 4:
        return None
    if not re.search(r'-{3,}', text):
        return None

    # Split on the separator (---|---)
    sep_match = re.search(r'\|?[\s\-]*-{3,}[\s\-|]*-{2,}\s*\|?', text)
    if not sep_match:
        return None

    before = text[:sep_match.start()]
    after = text[sep_match.end():]

    # Trim before to the section after the last sentence boundary
    # (avoids grabbing question prefix like "Find the mean of...Class Interval")
    last_break = max(
        before.rfind('. '),
        before.rfind('? '),
        before.rfind('! '),
        before.rfind('\n'),
        -1,
    )
    if last_break >= 0:
        before = before[last_break + 1:].lstrip()

    # Extract "Header1 | Header2" — limit each header to ~30 chars
    header_match = re.search(
        r'([A-Za-z][^|]{1,40}?)\s*\|\s*([A-Za-z][^|]{1,40}?)\s*$',
        before,
    )
    if not header_match:
        return None

    header1 = _clean_label(header_match.group(1).strip())
    header2 = _clean_label(header_match.group(2).strip())

    # Data rows from after — pattern is "val1 | val2 val3 | val4 ..."
    row_pattern = re.compile(
        r'(\d+\s*-\s*\d+|\d+\.?\d*|[a-zA-Z]\d?)\s*\|\s*(\d+\.?\d*|[a-zA-Z]\d?)'
    )
    rows = []
    for m in row_pattern.finditer(after):
        c1 = re.sub(r'\s*-\s*', '-', m.group(1).strip())
        c2 = m.group(2).strip()
        rows.append([c1, c2])

    if len(rows) < 3:
        return None

    return {
        "type": "frequency_distribution",
        "headers": [header1, header2],
        "rows": rows,
        "caption": None,
    }


def _extract_label_colon_pattern(text: str) -> Optional[dict]:
    """
    Find all "label: data, data, ..." segments and pair the first two
    as column1 and column2 of a frequency distribution.
    """
    # Pattern: a label (letters, parens, _) then colon then comma-separated values
    # Values can be: numeric ranges (0-10), numbers (5, 5.2), or single-letter placeholders (p, x, f1)
    pattern = re.compile(
        r'([A-Za-z][A-Za-z0-9\s\(\)_/]{2,40}?)\s*[:|]\s*'
        r'((?:\d+\s*-\s*\d+|\d+\.?\d*|[a-zA-Z]\d?)'
        r'(?:\s*,\s*(?:\d+\s*-\s*\d+|\d+\.?\d*|[a-zA-Z]\d?))+)',
    )

    matches = []
    for m in pattern.finditer(text):
        label = m.group(1).strip()
        data_str = m.group(2)
        values = _parse_inline_values(": " + data_str)
        if len(values) >= 3:
            matches.append({
                "label": label,
                "values": values,
                "pos": m.start(),
            })

    if len(matches) < 2:
        return None

    # Take the first two label-data pairs
    col1, col2 = matches[0], matches[1]

    # Match lengths if needed (truncate to shorter)
    min_len = min(len(col1["values"]), len(col2["values"]))
    if min_len < 3:
        return None
    col1_values = col1["values"][:min_len]
    col2_values = col2["values"][:min_len]

    # Sanity: at least one column should look like ranges OR numbers
    col1_has_ranges = any("-" in v for v in col1_values)
    col1_all_numeric = all(
        re.match(r'^-?\d+\.?\d*$|^[a-zA-Z]\d?$', v) for v in col1_values
    )
    if not (col1_has_ranges or col1_all_numeric):
        return None

    # Clean labels
    col1_label = _clean_label(col1["label"],
                               default="Class Interval" if col1_has_ranges else "xi")
    col2_label = _clean_label(col2["label"], default="Frequency")

    return {
        "type": "frequency_distribution",
        "headers": [col1_label, col2_label],
        "rows": [[a, b] for a, b in zip(col1_values, col2_values)],
        "caption": None,
    }


def _clean_label(raw_label: str, default: str = "Value") -> str:
    """Clean up a label: strip parens, shorthand markers, normalize."""
    if not raw_label:
        return default

    # Strip parenthetical content like "(fi)" or "(xi)"
    cleaned = re.sub(r'\s*\([^\)]*\)\s*', '', raw_label).strip()
    # Remove leading/trailing punctuation
    cleaned = re.sub(r'^[:.,;\s]+|[:.,;\s]+$', '', cleaned).strip()

    if not cleaned or len(cleaned) < 2:
        return default

    # Normalize common labels
    lower = cleaned.lower()
    label_map = {
        "fi": "Frequency",
        "f_i": "Frequency",
        "number of students": "Number of Students",
        "number of workers": "Number of Workers",
        "number of patients": "Number of Patients",
        "number of persons": "Number of Persons",
        "number of households": "Number of Households",
        "number of families": "Number of Families",
        "frequency": "Frequency",
        "marks obtained": "Marks Obtained",
        "class interval": "Class Interval",
        "daily wages": "Daily Wages",
        "age": "Age",
        "daily income": "Daily Income",
        "lifetimes": "Lifetimes",
        "lifetime": "Lifetime",
        "family size": "Family Size",
        "absentees": "Number of Absentees",
        "number of absentees": "Number of Absentees",
        "daily expenditure": "Daily Expenditure",
        "xi": "xi",
        "x_i": "xi",
    }

    if lower in label_map:
        return label_map[lower]

    # Title-case the cleaned label as fallback
    return cleaned.title()


def _parse_inline_values(segment: str) -> List[str]:
    """
    Parse a segment like ": 0-10, 10-20, 20-30, 30-40" or "10, 20, 36, 40"
    into a list of value strings.
    """
    if not segment:
        return []

    # Strip leading punctuation/colons/parens
    segment = re.sub(r'^[\s:;\(\)]+', '', segment)
    # Strip trailing punctuation
    segment = re.sub(r'[\s:;\(\)\.]+$', '', segment)

    # Split on commas
    parts = [p.strip() for p in segment.split(",")]

    # Each part should look like a number or a range like "0-10"
    valid = []
    for p in parts:
        # Range like "0-10" or "100-120"
        if re.match(r'^-?\d+\s*-\s*\d+$', p):
            valid.append(re.sub(r'\s*-\s*', '-', p))
        # Pure number (int or decimal, possibly with trailing letter for missing freq like "p", "f")
        elif re.match(r'^-?\d+\.?\d*$', p):
            valid.append(p)
        elif re.match(r'^[a-zA-Z]\d?$', p):  # missing-frequency placeholders like "p", "f", "f1", "x"
            valid.append(p)
        else:
            # Stop at first invalid token (likely text after data ends)
            break

    return valid


def _has_inline_data_leak(text: str) -> bool:
    """
    Detect if text contains comma-separated data that should have been in a
    structured table. Used to drop questions that bypassed the structured-only rule.
    """
    if not text:
        return False
    # Look for sequences of 4+ comma-separated numbers/ranges
    # e.g. "10, 20, 30, 40, 50" or "0-10, 10-20, 20-30, 30-40"
    pattern = re.compile(
        r'(?:\d+\s*-\s*\d+|\d+)(?:\s*,\s*(?:\d+\s*-\s*\d+|\d+|[a-zA-Z]\d?)){3,}'
    )
    return bool(pattern.search(text))


def _strip_inline_data_from_text(text: str, recovered_table: dict) -> str:
    """
    After extracting inline data into structured table, remove the data portion
    from text so it doesn't appear twice in PDF.
    """
    if not text or not recovered_table:
        return text

    # Find earliest position where data/headers start
    earliest = len(text)

    # Match label-colon pattern (e.g. "Class Interval: 0-10, ...")
    label_pattern = re.compile(
        r'([A-Za-z][A-Za-z0-9\s\(\)_/]{2,40}?)\s*[:|]\s*'
        r'((?:\d+\s*-\s*\d+|\d+\.?\d*|[a-zA-Z]\d?)'
        r'(?:\s*,\s*(?:\d+\s*-\s*\d+|\d+\.?\d*|[a-zA-Z]\d?))+)',
    )
    m = label_pattern.search(text)
    if m and m.start() < earliest:
        earliest = m.start()

    # Match markdown pipe pattern. For markdown we need to walk back from the
    # pipe to the LAST sentence break, so we keep "Find median." but drop
    # "Marks Obtained | Number of Students ---|--- ..."
    pipe_match = re.search(
        r'[A-Za-z][^|\n]{0,40}\|\s*[A-Za-z][^|\n]{0,40}\s*\|?[\s\-]*-{3,}',
        text,
    )
    if pipe_match:
        # Walk back from pipe_match.start() to find the prior sentence boundary
        prior_text = text[:pipe_match.start()]
        last_break = max(
            prior_text.rfind('. '),
            prior_text.rfind('? '),
            prior_text.rfind('! '),
            prior_text.rfind('\n'),
            -1,
        )
        # Position to cut at is just after the sentence break (or pipe start if no break)
        cut_pos = (last_break + 2) if last_break >= 0 else pipe_match.start()
        if cut_pos < earliest:
            earliest = cut_pos

    if earliest >= len(text):
        return text

    cleaned = text[:earliest].strip()

    # Remove trailing connectives like "as follows:", "given below:", etc.
    cleaned = re.sub(
        r'(?:as follows|the following|given below|below|here|are given|are as follows)\s*[:.]?\s*$',
        '', cleaned, flags=re.IGNORECASE,
    ).strip()
    cleaned = re.sub(r'[:;,]+$', '', cleaned).strip()

    if len(cleaned) < 15:
        cleaned = "Find the answer based on the given data."

    if not cleaned.endswith(('.', '?', '!')):
        cleaned += "."

    return cleaned


STATISTICS_PROMPT_TEMPLATES = {
    "frequency_distribution": """STATISTICS QUESTION — STRUCTURED TABLE FORMAT (STRICT)

═══ ABSOLUTE REQUIREMENT ═══

The "question_table" field is MANDATORY. It is THE PRIMARY source of table data.
Questions WITHOUT a valid "question_table" field WILL BE REJECTED automatically.

DO NOT put the data table inside "text" as pipes/dashes. Keep "text" CLEAN — only
the question prompt, no data. The data goes in "question_table" only.

═══ EXACT OUTPUT FORMAT — COPY THIS STRUCTURE ═══

{{
  "text": "Find the mean of the given frequency distribution.",
  "question_table": {{
    "type": "frequency_distribution",
    "headers": ["Class Interval", "Frequency"],
    "rows": [
      ["0-10", "5"],
      ["10-20", "8"],
      ["20-30", "15"],
      ["30-40", "12"],
      ["40-50", "7"],
      ["50-60", "3"]
    ],
    "caption": "Marks scored by 50 students"
  }},
  "correct_answer": "xi values: 5, 15, 25, 35, 45, 55. Σfi = 50, Σfixi = 1510. Mean = 1510/50 = 30.2",
  "explanation": "Step 1: Find xi (mid-point) of each class. Step 2: Compute fixi for each row. Step 3: Mean = Σfixi / Σfi = 1510/50 = 30.2"
}}

═══ REJECTION CRITERIA — auto-dropped if any of these occur ═══

❌ "question_table" is null, missing, or empty
❌ "question_table" has empty "headers" or empty "rows"
❌ "text" contains inline table pipes/dashes (data must be in question_table only)
❌ "rows" contains anything other than strings (numbers must be quoted: "5" not 5)
❌ "text" contains comma-separated number lists (4+ values like "5, 12, 15, 8, 6")
❌ "text" contains labels like "Class Interval:", "xi:", "fi:", "Frequency:" followed by data

═══ FORBIDDEN PATTERNS (these are real failures we've seen) ═══

❌ FORBIDDEN — Comma-separated lists in text:
   "text": "Find mean. Class Interval: 0-10, 10-20, 20-30 Frequency: 5, 12, 15"

❌ FORBIDDEN — xi/fi shorthand:
   "text": "Find mean. xi: 10, 20, 36 fi: 1, 1, 3"

❌ FORBIDDEN — Single-line markdown:
   "text": "Find mean. Class | Frequency ---|--- 0-10 | 5 10-20 | 12"

❌ FORBIDDEN — Data appended after question:
   "text": "Find median of: 5, 8, 12, 15, 20, 25"

✅ CORRECT — Clean text + structured question_table:
   "text": "Find the mean of the given frequency distribution.",
   "question_table": {{"headers": [...], "rows": [[...], ...]}}

═══ DATA REALISM RULES ═══

- Use 5-7 class intervals (NOT less, NOT more)
- Class intervals must be of EQUAL width (all 10, all 5, all 20 etc.)
- Frequencies must vary realistically — not all same, not extreme outliers
- Σfi (total frequency) should be a round number (30, 40, 50, 60, 80, 100)
- Use realistic CBSE contexts: marks scored, daily wages, heights, weights,
  ages, family incomes, distances, time spent, etc.
- Numbers must be plausible for the context (Indian classroom)

═══ THE GOLDEN RULE ═══

Clean text + structured question_table = ACCEPTED
Inline table in text + no question_table = REJECTED

Always emit both: clean text describing what to find, and structured question_table
with all the data. Never put data in text, never skip question_table.
"""
}


def _build_statistics_prompt(
    chapter: ChapterSection,
    request: TestGenerationRequest,
    context_chunks: List[Dict],
    count: int,
    section_key: Optional[str] = None,
    section_info: Optional[dict] = None,
) -> str:
    """Build a Statistics-specific prompt that forces table data in questions."""
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

    # Format-specific answer instructions
    if fmt_val == "mcq":
        fmt_line = (
            'Provide 4 options labeled A) B) C) D). '
            'correct_answer = exact full option text. '
            'Distractors should be plausible (e.g. mode/median confusion, '
            'arithmetic errors in calculation, off-by-one mistakes).'
        )
    elif fmt_val == "short_answer":
        fmt_line = (
            '"options": null. Show full calculation: tabular working with '
            'xi (mid-point), fi, fixi columns. Then Σfi, Σfixi, and final answer.'
        )
    elif fmt_val == "long_answer":
        fmt_line = (
            '"options": null. Show complete tabular working with all required '
            'columns (xi, fi, di, fidi, etc. as needed for the method). '
            'Apply the formula step-by-step. State the final answer clearly with units.'
        )
    elif fmt_val == "assertion_reason":
        fmt_line = (
            '"text": "Assertion (A): [statement about a statistical concept]\\n'
            'Reason (R): [related principle]". '
            'Use 4 standard AR options. NOTE: For pure assertion-reason questions, '
            'a question_table is NOT required — only include if the assertion '
            'references specific numerical data.'
        )
    else:
        fmt_line = '4 options labeled A) B) C) D).'

    template_instruction = STATISTICS_PROMPT_TEMPLATES["frequency_distribution"]

    section_field = f', "section": "{section_key}"' if section_key else ''

    # JSON template — for AR/conceptual questions, question_table can be null
    if fmt_val == "assertion_reason":
        json_tmpl = (
            f'{{"questions":[{{'
            f'"text":"Assertion (A): ...\\nReason (R): ...",'
            f'"format":"assertion_reason",'
            f'"options":["A) ...","B) ...","C) ...","D) ..."],'
            f'"correct_answer":"A) ...",'
            f'"explanation":"...",'
            f'"question_table":null,'
            f'"marks":{chapter.marks_per_question},'
            f'"difficulty":"{diff_val}","bloom_level":"apply",'
            f'"chapter":"{chapter.chapter}","topic":"specific topic"{section_field}'
            f'}}]}}'
        )
    elif fmt_val == "mcq":
        json_tmpl = (
            f'{{"questions":[{{'
            f'"text":"Find the mean of the given frequency distribution.",'
            f'"format":"mcq",'
            f'"options":["A) 28.4","B) 30.2","C) 32.6","D) 26.8"],'
            f'"correct_answer":"B) 30.2",'
            f'"explanation":"Σfi = 50, Σfixi = 1510, Mean = 1510/50 = 30.2",'
            f'"question_table":{{"type":"frequency_distribution","headers":["Class Interval","Frequency"],"rows":[["0-10","5"],["10-20","8"],["20-30","15"],["30-40","12"],["40-50","7"],["50-60","3"]],"caption":"Marks scored by 50 students"}},'
            f'"marks":{chapter.marks_per_question},'
            f'"difficulty":"{diff_val}","bloom_level":"apply",'
            f'"chapter":"{chapter.chapter}","topic":"specific topic"{section_field}'
            f'}}]}}'
        )
    else:
        json_tmpl = (
            f'{{"questions":[{{'
            f'"text":"Find the mean of the given frequency distribution.",'
            f'"format":"{fmt_val}",'
            f'"options":null,'
            f'"correct_answer":"xi: 5,15,25,35,45,55. Σfi=50, Σfixi=1510. Mean = 1510/50 = 30.2",'
            f'"explanation":"Step 1: Find xi (mid-point) of each class. Step 2: Compute fixi. Step 3: Mean = Σfixi / Σfi = 30.2",'
            f'"question_table":{{"type":"frequency_distribution","headers":["Class Interval","Frequency"],"rows":[["0-10","5"],["10-20","8"],["20-30","15"],["30-40","12"],["40-50","7"],["50-60","3"]],"caption":"Marks scored by 50 students"}},'
            f'"marks":{chapter.marks_per_question},'
            f'"difficulty":"{diff_val}","bloom_level":"apply",'
            f'"chapter":"{chapter.chapter}","topic":"specific topic"{section_field}'
            f'}}]}}'
        )

    return f"""You are an expert CBSE Class {request.class_grade} {request.subject} paper setter
specializing in Statistics and Data-handling questions.
{section_ctx}
Chapter: {chapter.chapter}
Difficulty: {DIFF_INST.get(diff_val, DIFF_INST["medium"])}

{template_instruction}

FORMAT RULES:
{fmt_line}

{MATH_FORMAT_INSTRUCTION}

"format" must be exactly "{fmt_val}". "chapter" must be exactly "{chapter.chapter}".
Every question MUST have detailed explanation showing complete working.
Each question must test a DIFFERENT concept (mean, median, mode, cumulative, etc.).
Vary the contexts and data — no two questions should use the same scenario.

NCERT Reference:
{ctx}

Generate EXACTLY {count} unique questions. Return ONLY valid JSON, no extra text:
{json_tmpl}"""


# ---------------------------------------------------------------------------
# CBSE Section Templates (Generic — Science/Maths/etc.)
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
# CBSE Accountancy Class 12 — Pattern Config (SQP 2025-26)
# ---------------------------------------------------------------------------
CBSE_ACCOUNTANCY_PATTERN = {
    "total_questions": 34,
    "total_marks": 80,
    "parts": {
        "A": {
            "title": "Part A",
            "subtitle": "Accounting for Partnership Firms and Companies",
            "marks": 60,
            "instruction": "Question 1 to 16 carry 1 mark each. Questions 17 to 20 carry 3 marks each. Questions 21-22 carry 4 marks each. Questions 23 to 26 carry 6 marks each.",
            "groups": [
                {
                    "id": "A1",
                    "marks_per_q": 1,
                    "count": 16,
                    "or_count": 4,
                    "formats": ["mcq", "assertion_reason"],
                    "mcq_count": 12,
                    "ar_count": 4,
                    "difficulty": "easy",
                    "blooms": ["remember", "understand"],
                },
                {
                    "id": "A3",
                    "marks_per_q": 3,
                    "count": 4,
                    "or_count": 2,
                    "formats": ["short_answer"],
                    "difficulty": "medium",
                    "blooms": ["understand", "apply"],
                },
                {
                    "id": "A4",
                    "marks_per_q": 4,
                    "count": 2,
                    "or_count": 1,
                    "formats": ["short_answer", "journal_entry"],
                    "difficulty": "medium",
                    "blooms": ["apply", "analyze"],
                },
                {
                    "id": "A6",
                    "marks_per_q": 6,
                    "count": 4,
                    "or_count": 2,
                    "formats": ["long_answer", "journal_entry", "ledger"],
                    "difficulty": "hard",
                    "blooms": ["analyze", "evaluate"],
                },
            ],
        },
        "B1": {
            "title": "Part B (Option I)",
            "subtitle": "Analysis of Financial Statements",
            "marks": 20,
            "instruction": "Question 27 to 30 carry 1 mark each. Questions 31-32 carry 3 marks each. Question 33 carries 4 marks. Question 34 carries 6 marks.",
            "groups": [
                {
                    "id": "B1_1",
                    "marks_per_q": 1,
                    "count": 4,
                    "or_count": 2,
                    "formats": ["mcq", "assertion_reason"],
                    "mcq_count": 3,
                    "ar_count": 1,
                    "difficulty": "easy",
                    "blooms": ["remember", "understand"],
                },
                {
                    "id": "B1_3",
                    "marks_per_q": 3,
                    "count": 2,
                    "or_count": 1,
                    "formats": ["short_answer"],
                    "difficulty": "medium",
                    "blooms": ["understand", "apply"],
                },
                {
                    "id": "B1_4",
                    "marks_per_q": 4,
                    "count": 1,
                    "or_count": 1,
                    "formats": ["short_answer"],
                    "difficulty": "medium",
                    "blooms": ["apply", "analyze"],
                },
                {
                    "id": "B1_6",
                    "marks_per_q": 6,
                    "count": 1,
                    "or_count": 0,
                    "formats": ["long_answer"],
                    "difficulty": "hard",
                    "blooms": ["analyze", "evaluate"],
                },
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Accountancy Topic → Part Classification
# ---------------------------------------------------------------------------
ACCOUNTANCY_PART_A_TOPICS = {
    "accounting for partnership", "reconstitution of partnership",
    "admission of a partner", "retirement and death of a partner",
    "retirement of a partner", "death of a partner",
    "dissolution of partnership", "dissolution of partnership firm",
    "accounting for companies", "accounting for share capital",
    "issue of shares", "forfeiture and reissue of shares",
    "issue of debentures", "redemption of debentures",
    "issue and redemption of debentures", "goodwill",
    "goodwill nature and valuation", "fundamentals of partnership",
    "change in profit sharing ratio", "reconstitution of a partnership firm",
    "accounting for partnership firms",
}

ACCOUNTANCY_PART_B_TOPICS = {
    "analysis of financial statements", "financial statements analysis",
    "ratio analysis", "accounting ratios", "cash flow statement", "cash flow",
    "comparative statements", "common size statements",
    "tools of financial statements analysis",
    "financial statements of a company", "statement of profit and loss",
    "balance sheet of a company",
}


def _classify_chapter_part(chapter_name: str) -> str:
    ch_lower = chapter_name.lower().strip()
    for topic in ACCOUNTANCY_PART_B_TOPICS:
        if topic in ch_lower or ch_lower in topic:
            return "B1"
    for topic in ACCOUNTANCY_PART_A_TOPICS:
        if topic in ch_lower or ch_lower in topic:
            return "A"
    if any(kw in ch_lower for kw in ["partner", "share", "debenture", "company", "goodwill",
                                       "admission", "retire", "death", "dissolut", "forfeit"]):
        return "A"
    if any(kw in ch_lower for kw in ["ratio", "cash flow", "financial statement", "comparative",
                                       "common size", "balance sheet"]):
        return "B1"
    return "A"


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
• Ordinals: write as plain text: 1st, 2nd, 3rd, 4th, 15th (NOT superscript modifiers)

CRITICAL JSON RULES:
- Do NOT use unescaped newlines inside JSON string values — always use \\n instead
- Do NOT use LaTeX commands like \\frac, \\sqrt, \\theta, \\left, \\right, \\mathbb
- Do NOT use $ delimiters around math
- Do NOT use Unicode modifier letters for ordinals (like ᵗʰ, ˢᵗ). Write "15th" not "15ᵗʰ".
- Keep each question's JSON compact

Write clean readable text that a teacher can read directly."""


# ---------------------------------------------------------------------------
# Unicode Modifier Letter Cleanup
# ---------------------------------------------------------------------------
MODIFIER_LETTERS = {
    '\u1D57': 't',  # ᵗ
    '\u02B0': 'h',  # ʰ
    '\u02E2': 's',  # ˢ
    '\u1D48': 'd',  # ᵈ
    '\u02B3': 'r',  # ʳ
    '\u02E1': 'l',  # ˡ
    '\u1D43': 'a',  # ᵃ
    '\u1D49': 'e',  # ᵉ
    '\u1D52': 'o',  # ᵒ
}


def _fix_modifier_letters(text: str) -> str:
    """Convert Unicode modifier letters to plain text (15ᵗʰ → 15th)."""
    if not text:
        return text
    result = text
    for mod, plain in MODIFIER_LETTERS.items():
        result = result.replace(mod, plain)
    return result


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
        is_acc = request.subject.lower() in ACCOUNTANCY_SUBJECTS
        if is_acc:
            fmt_line = (
                '"options": null. '
                'This is an Accountancy question — the answer MUST include an "answer_table" field. '
                'Choose the most appropriate table type: "journal_entry", "ledger", or "trial_balance". '
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
# Accountancy Prompt Builder
# ---------------------------------------------------------------------------
def _build_accountancy_prompt(
    chapter, request, context_chunks, count,
    table_format, section_key=None, section_info=None,
):
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
- Use Rs. for currency (NOT the ₹ symbol)
- Use realistic business scenarios relevant to the chapter
- Each question must test a DIFFERENT concept
- "correct_answer" should be a text summary of the answer
- "answer_table" is the structured table (MANDATORY for this format)
- "explanation" must explain the accounting principle and each entry
- Write ordinals as plain text: "15th" not "15ᵗʰ"

NCERT Reference:
{ctx}

Generate EXACTLY {count} unique questions. Return ONLY valid JSON:
{{"questions":[{{"text":"...","format":"{table_format}","options":null,"correct_answer":"Summary of entries...","explanation":"Step-by-step accounting logic...","answer_table":{{"type":"{table_format}","headers":[...],"rows":[[...]],"total_row":null}},"marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"apply","chapter":"{chapter.chapter}","topic":"specific topic"{section_field}}}]}}"""


# ---------------------------------------------------------------------------
# Accountancy CBSE Pattern Prompt Builder (for OR questions)
# ---------------------------------------------------------------------------
ACCOUNTANCY_PROMPT_RULES = """ACCOUNTANCY QUESTION RULES (CRITICAL):
• Every numerical question MUST include GIVEN DATA — amounts in Rs., dates, ratios, account balances.
• Use Rs. for currency (not the ₹ symbol). Example: Rs. 5,00,000 (with commas for Indian numbering).
• Partnership questions: include capital amounts, profit-sharing ratios, dates of admission/retirement.
• Company accounts: include share details (face value, premium/discount, payment schedule).
• Journal entry questions: the answer must show Date, Particulars, L.F., Debit (Rs.), Credit (Rs.).
• For 3+ mark questions, include a scenario/case with at least 3-4 numerical data points.
• For 6 mark questions, include detailed scenarios with balance sheet extracts or multiple transactions.
• Financial statement questions: include actual financial data (Revenue, Expenses, Assets, Liabilities with amounts).
• Cash flow questions: include opening and closing balances of relevant items.
• Ratio questions: provide the necessary financial data to calculate the ratio.
• NEVER give vague questions — always provide specific numbers, dates, and names of companies/partners.
• Use realistic Indian company names (e.g., Priya Ltd., Raman Enterprises) and partner names.
• All amounts should be in round figures suitable for manual calculation.
• Write ordinals as plain text: "15th" not "15ᵗʰ"."""


def _build_accountancy_cbse_prompt(
    chapter, request, context_chunks, count,
    group_info, part_key, generate_or=False,
):
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
    marks = chapter.marks_per_question

    part_info = CBSE_ACCOUNTANCY_PATTERN["parts"].get(part_key, {})
    part_ctx = f"""This is for {part_info.get('title', '')} — {part_info.get('subtitle', '')}.
Each question carries {marks} marks."""

    or_instruction = ""
    if generate_or:
        or_instruction = """
IMPORTANT: For EACH question, also generate an OR alternative question.
The OR question must test a DIFFERENT concept but carry the SAME marks and difficulty.
Format the OR question with the field "is_or": true.
So for each question slot, output TWO questions: the main one and its OR alternative."""

    if fmt_val == "mcq":
        fmt_line = '4 options labeled A) B) C) D). correct_answer = exact full option text. All options must be plausible with numerical values where applicable. Vary correct answer position.'
    elif fmt_val == "assertion_reason":
        fmt_line = '"text": "Assertion (A): [statement]\\nReason (R): [related principle]". Use 4 standard AR options.'
    elif marks == 3:
        fmt_line = '"options": null. Answer in 50-80 words with complete journal entries or calculations.'
    elif marks == 4:
        fmt_line = '"options": null. Answer in 80-120 words. Include journal entries with proper format or detailed calculations.'
    elif marks == 6:
        fmt_line = '"options": null. Answer in 120-200 words. Include complete journal entries, ledger accounts, or detailed calculations with all working notes.'
    else:
        fmt_line = '"options": null. Provide a clear, detailed answer with proper accounting format.'

    section_tag = f"{part_key}_{marks}m"

    if fmt_val in ("mcq", "assertion_reason"):
        json_template = f'{{"questions":[{{"text":"...","format":"{fmt_val}","options":["A) ...","B) ...","C) ...","D) ..."],"correct_answer":"B) exact option","explanation":"...","marks":{marks},"difficulty":"{diff_val}","bloom_level":"apply","chapter":"{chapter.chapter}","topic":"specific topic","section":"{section_tag}","is_or":false}}]}}'
    else:
        json_template = f'{{"questions":[{{"text":"[question with given numerical data]","format":"{fmt_val}","options":null,"correct_answer":"[complete answer]","explanation":"[detailed working]","marks":{marks},"difficulty":"{diff_val}","bloom_level":"apply","chapter":"{chapter.chapter}","topic":"specific topic","section":"{section_tag}","is_or":false}}]}}'

    return f"""You are an expert CBSE Class 12 Accountancy paper setter following the latest CBSE pattern.

{part_ctx}
{or_instruction}

Chapter: {chapter.chapter}
Difficulty: {DIFF_INST.get(diff_val, DIFF_INST["medium"])}

{ACCOUNTANCY_PROMPT_RULES}

FORMAT RULES:
{fmt_line}

{MATH_FORMAT_INSTRUCTION}

"format" must be exactly "{fmt_val}". "chapter" must be exactly "{chapter.chapter}".
Every question MUST have a detailed explanation showing complete working.
Each question must test a DIFFERENT concept from this chapter. No repeated concepts.

NCERT Reference:
{ctx}

Generate EXACTLY {count} {'pairs of questions (main + OR alternative)' if generate_or else 'unique questions'}. Return ONLY valid JSON:
{json_template}"""


# ═══════════════════════════════════════════════════════════════════════════
# JSON Extraction — v14 ROBUST
# Handles: truncated responses, unescaped newlines inside strings,
#          bad LaTeX escapes, control chars, trailing commas, partial extraction
# ═══════════════════════════════════════════════════════════════════════════
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


def _escape_control_chars_in_strings(text: str) -> str:
    """
    Walk through text and escape raw newlines/tabs/CRs that appear INSIDE JSON strings.
    Gemini sometimes emits literal newlines inside string values which is invalid JSON.
    Critical for Statistics where markdown tables in question text use newlines.
    """
    result = []
    in_string = False
    escape_next = False

    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue

        if ch == '\\':
            result.append(ch)
            escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue

        if in_string:
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            elif ord(ch) < 0x20:
                result.append(f'\\u{ord(ch):04x}')
            else:
                result.append(ch)
        else:
            result.append(ch)

    return ''.join(result)


def _extract_questions_individually(text: str) -> list:
    """
    Last-resort: extract individual question objects by matching balanced braces.
    Saves partial batches when the outer JSON is truncated or has one bad question.
    """
    questions = []
    depth = 0
    start = -1
    in_string = False
    escape_next = False

    qs_match = re.search(r'"questions"\s*:\s*\[', text)
    if not qs_match:
        return []

    i = qs_match.end()
    while i < len(text):
        ch = text[i]

        if escape_next:
            escape_next = False
            i += 1
            continue

        if ch == '\\':
            escape_next = True
            i += 1
            continue

        if ch == '"':
            in_string = not in_string

        if not in_string:
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start != -1:
                    candidate = text[start:i+1]
                    try:
                        cleaned = _escape_control_chars_in_strings(candidate)
                        cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
                        obj = json.loads(cleaned)
                        questions.append(obj)
                    except (json.JSONDecodeError, ValueError):
                        try:
                            fixed = _fix_latex_json_escapes(cleaned)
                            obj = json.loads(fixed)
                            questions.append(obj)
                        except (json.JSONDecodeError, ValueError):
                            pass  # skip broken question, continue
                    start = -1

        i += 1

    return questions


def _extract_json(raw: str) -> dict:
    """
    Robust JSON extractor with multiple fallback strategies.
    """
    text = raw.strip().lstrip("\ufeff\u200b")

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()

    # Early detection: is this truncated?
    open_braces = text.count("{")
    close_braces = text.count("}")
    open_brackets = text.count("[")
    close_brackets = text.count("]")

    is_truncated = (
        open_braces > close_braces + 1
        or open_brackets > close_brackets
        or not text.rstrip().endswith(("}", "]"))
    )

    if is_truncated:
        logger.warning(
            f"JSON appears truncated (braces: {open_braces}/{close_braces}, "
            f"brackets: {open_brackets}/{close_brackets}). "
            f"Skipping to per-question extraction."
        )
        individual_qs = _extract_questions_individually(text)
        if individual_qs:
            logger.info(f"✓ Recovered {len(individual_qs)} questions from truncated response")
            return {"questions": individual_qs}

    # Attempt 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fb = text.find("{")
    lb = text.rfind("}")
    if fb == -1:
        raise ValueError(f"No JSON found (len={len(raw)})")

    if lb <= fb:
        candidate = text[fb:]
    else:
        candidate = text[fb:lb + 1]

    # Attempt 2: as-is
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Attempt 3: strip trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Attempt 4: escape control chars inside strings (CRITICAL for markdown tables)
    escaped_ctrl = _escape_control_chars_in_strings(cleaned)
    try:
        return json.loads(escaped_ctrl)
    except json.JSONDecodeError:
        pass

    # Attempt 5: fix LaTeX escapes
    fixed = _fix_latex_json_escapes(escaped_ctrl)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 6: aggressive backslash escaping
    aggressive = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu{])', r'\\\\', fixed)
    try:
        return json.loads(aggressive)
    except json.JSONDecodeError:
        pass

    # Attempt 7: per-question extraction — last resort BEFORE nuclear
    logger.warning(
        f"All bulk parse attempts failed, trying per-question extraction (len={len(raw)})"
    )
    individual_qs = _extract_questions_individually(text)
    if individual_qs:
        logger.info(f"✓ Recovered {len(individual_qs)} questions via per-question extraction")
        return {"questions": individual_qs}

    logger.error(f"JSON failed all attempts. Preview: {candidate[:200]}")
    raise ValueError(f"Could not parse JSON (len={len(raw)})")


# ---------------------------------------------------------------------------
# Post-process: clean Gemini text
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
    """
    Clean Gemini output: replace LaTeX with Unicode, fix modifier letters, etc.
    IMPORTANT: This function PRESERVES newlines and pipe characters so that
    inline markdown tables (used in Statistics questions) survive intact.
    """
    if not text:
        return text

    result = text
    result = result.replace('₹', 'Rs.')

    # v13: Fix Unicode modifier letters (15ᵗʰ → 15th)
    result = _fix_modifier_letters(result)

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

    # Preserve newlines (critical for markdown tables) — only collapse spaces within lines
    lines = result.split('\n')
    result = '\n'.join(re.sub(r' +', ' ', line).strip() for line in lines).strip()

    return result


# ---------------------------------------------------------------------------
# v14: Question Table Parser
# ---------------------------------------------------------------------------
def _parse_question_table(raw_table: dict) -> Optional[QuestionTable]:
    """Safely parse a question_table dict from Gemini into QuestionTable model."""
    if not raw_table or not isinstance(raw_table, dict):
        return None

    try:
        table_type = raw_table.get("type", "frequency_distribution")
        headers = raw_table.get("headers", [])
        rows = raw_table.get("rows", [])
        caption = raw_table.get("caption")

        if not headers or not rows:
            return None

        expected_cols = len(headers)
        clean_rows = []
        for row in rows:
            if isinstance(row, list):
                padded = (row + [""] * expected_cols)[:expected_cols]
                clean_rows.append([str(cell) for cell in padded])

        if not clean_rows:
            return None

        return QuestionTable(
            type=str(table_type),
            headers=[str(h) for h in headers],
            rows=clean_rows,
            caption=str(caption) if caption else None,
        )
    except Exception as e:
        logger.warning(f"  question_table parse failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Question Parser (with answer_table + question_table + OR support + validation)
# ---------------------------------------------------------------------------
def _parse_batch(raw: str, chapter: ChapterSection, request: TestGenerationRequest,
                 section_key: str = None) -> List[GeneratedQuestion]:
    try:
        data = _extract_json(raw)
    except (ValueError, AttributeError) as e:
        logger.error(f"Parse failed ({chapter.chapter}): {e}")
        return []

    # Normalize: accept both {"questions": [...]} and direct [...]
    if isinstance(data, dict):
        raw_qs = data.get("questions", [])
    elif isinstance(data, list):
        raw_qs = data
    else:
        logger.error(f"Parse failed ({chapter.chapter}): unexpected type {type(data).__name__}")
        return []

    if not isinstance(raw_qs, list):
        logger.error(f"Parse failed ({chapter.chapter}): questions field not a list")
        return []

    if not isinstance(raw_qs, list):
        return []

    questions = []
    seen = set()
    dropped = 0
    dropped_table_missing = 0
    diff_val = chapter.difficulty.value if hasattr(chapter.difficulty, 'value') else str(chapter.difficulty)
    is_accountancy = request.subject.lower() in ACCOUNTANCY_SUBJECTS
    is_stats = _is_statistics_question(chapter.chapter, getattr(chapter, 'topic', None))
    last_main_id = None

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

        # Parse answer_table for Accountancy
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

        # v14: Parse question_table for Statistics
        question_table = _parse_question_table(q.get("question_table"))

        # ═══════════════════════════════════════════════════════════════
        # v16: STRICT VALIDATION GATE WITH RECOVERY
        #
        # For Statistics chapters:
        #   1. If structured question_table present → ACCEPT
        #   2. If text has inline data leak → try RECOVERY (extract to structured)
        #      - If recovery succeeds → strip inline data from text, use recovered table
        #      - If recovery fails → DROP
        #   3. If references table but no data anywhere → DROP
        #
        # For non-Stats chapters: lenient — accept inline markdown OR structured.
        # ═══════════════════════════════════════════════════════════════
        stats_chapter = _is_statistics_question(
            chapter.chapter, getattr(chapter, 'topic', None)
        )

        if stats_chapter:
            # CASE 1: Already has structured table — verify text doesn't leak data
            if question_table is not None:
                if _has_inline_data_leak(text):
                    # Sometimes Gemini sends BOTH structured AND inline data.
                    # Strip the inline portion, keep the structured table.
                    cleaned_text = _strip_inline_data_from_text(text, {
                        "headers": question_table.headers,
                        "rows": question_table.rows,
                    })
                    if cleaned_text and cleaned_text != text:
                        text = cleaned_text
                        logger.info(f"  ✓ Cleaned inline data leak from text (structured table preserved)")

            # CASE 2: No structured table — try to recover from inline data
            elif _has_inline_data_leak(text) or _references_table(text):
                recovered = _extract_inline_data(text)
                if recovered:
                    try:
                        question_table = QuestionTable(
                            type=recovered.get("type", "frequency_distribution"),
                            headers=recovered["headers"],
                            rows=recovered["rows"],
                            caption=recovered.get("caption"),
                        )
                        cleaned_text = _strip_inline_data_from_text(text, recovered)
                        if cleaned_text:
                            text = cleaned_text
                        logger.info(f"  ✓ Recovered inline data into structured table ({len(recovered['rows'])} rows)")
                    except Exception as e:
                        logger.warning(f"  ⚠ Recovery failed: {e}")
                        question_table = None

                # Still no table after recovery attempt → DROP
                if question_table is None:
                    logger.warning(
                        f"  ⚠ Dropped Stats Q (no usable table data): {text[:80]}..."
                    )
                    dropped_table_missing += 1
                    dropped += 1
                    continue
        else:
            # Non-Stats: lenient mode (accept inline markdown OR structured)
            if _references_table(text):
                has_inline = _has_inline_table(text)
                if not has_inline and question_table is None:
                    logger.warning(
                        f"  ⚠ Dropped Q (table referenced but data missing): {text[:80]}..."
                    )
                    dropped_table_missing += 1
                    dropped += 1
                    continue

        # Format-specific validation
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
        is_or = q.get("is_or", False)

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
                question_table=question_table,  # v14
            )
            gq._section = q_section
            gq._is_or = is_or
            gq._or_of = last_main_id if is_or else None

            if not is_or:
                last_main_id = gq.id

            questions.append(gq)
        except Exception as e:
            logger.warning(f"Q{idx} ({chapter.chapter}): {e}")
            dropped += 1

    if dropped:
        extra = f" (incl. {dropped_table_missing} with missing table data)" if dropped_table_missing else ""
        logger.info(f"  {chapter.chapter}: dropped {dropped}{extra}, kept {len(questions)}")

    if is_stats and questions:
        with_table = sum(1 for q in questions if q.question_table is not None)
        logger.info(f"  {chapter.chapter}: {with_table}/{len(questions)} questions have structured question_table")

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
                    response_mime_type="application/json",
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
# Generate for ONE chapter — v14: routes Statistics questions
# ---------------------------------------------------------------------------
def _generate_for_chapter(client, chapter, request, context_chunks, models,
                          section_key=None, section_info=None):
    target = chapter.quantity
    ask = target + settings.OVERSHOOT_PER_CHAPTER
    batch_size = settings.BATCH_SIZE
    is_accountancy = request.subject.lower() in ACCOUNTANCY_SUBJECTS
    is_stats = _is_statistics_question(chapter.chapter, getattr(chapter, 'topic', None))
    fmt_val = chapter.format.value if hasattr(chapter.format, 'value') else str(chapter.format)

    routing_tag = ""
    if is_accountancy and fmt_val in ACCOUNTANCY_TABLE_FORMATS:
        routing_tag = " [ACCOUNTANCY TABLE]"
    elif is_stats:
        routing_tag = " [STATISTICS]"

    logger.info(f"  '{chapter.chapter}': target={target}, fmt={fmt_val}, "
                f"diff={chapter.difficulty}, marks={chapter.marks_per_question}"
                + (f", section={section_key}" if section_key else "")
                + routing_tag)

    all_qs = []
    remaining = ask
    batch_num = 0

    while remaining > 0 and len(all_qs) < target:
        bc = min(remaining, batch_size)
        batch_num += 1

        # v14: Routing logic — Stats > Accountancy table > Generic
        if is_accountancy and fmt_val in ACCOUNTANCY_TABLE_FORMATS:
            prompt = _build_accountancy_prompt(
                chapter, request, context_chunks, bc,
                fmt_val, section_key, section_info
            )
        elif is_stats:
            prompt = _build_statistics_prompt(
                chapter, request, context_chunks, bc,
                section_key, section_info
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
# CBSE Section-based Paper Generation (Generic)
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
                        topic=orig_ch.topic if hasattr(orig_ch, 'topic') else None,
                    )
                    qs = _generate_for_chapter(client, mcq_chapter, request, context_chunks, models, sec_key, sec_info)
                    all_questions.extend(qs)
                    time.sleep(settings.BATCH_DELAY)

                if ch_ar > 0:
                    ar_chapter = ChapterSection(
                        chapter=ch_name,
                        difficulty=DifficultyLevel(sec_info["difficulty"]),
                        format=QuestionFormat("assertion_reason"),
                        marks_per_question=sec_info["marks_per_q"],
                        quantity=ch_ar,
                        topic=orig_ch.topic if hasattr(orig_ch, 'topic') else None,
                    )
                    qs = _generate_for_chapter(client, ar_chapter, request, context_chunks, models, sec_key, sec_info)
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
                    topic=orig_ch.topic if hasattr(orig_ch, 'topic') else None,
                )
                qs = _generate_for_chapter(client, sec_chapter, request, context_chunks, models, sec_key, sec_info)
                all_questions.extend(qs)
                time.sleep(settings.BATCH_DELAY)

    elapsed = time.time() - t0
    logger.info(f"CBSE Paper Done: {len(all_questions)}/{total_expected} in {elapsed:.1f}s")

    if not all_questions:
        raise GenerationError("All sections failed.", 500)

    return all_questions


# ---------------------------------------------------------------------------
# CBSE Accountancy Paper Generation (34q, 80 marks, Part A + Part B)
# ---------------------------------------------------------------------------
def _distribute_accountancy_chapters(chapters, pattern):
    part_a_chapters = []
    part_b_chapters = []

    for ch in chapters:
        part = _classify_chapter_part(ch.chapter)
        if part == "B1":
            part_b_chapters.append(ch)
        else:
            part_a_chapters.append(ch)

    if not part_a_chapters and part_b_chapters:
        part_a_chapters = part_b_chapters
        part_b_chapters = []

    distribution = {}

    for part_key, part_info in pattern["parts"].items():
        ch_list = part_a_chapters if part_key == "A" else part_b_chapters
        if not ch_list:
            distribution[part_key] = {}
            continue

        part_dist = {}
        num_chapters = len(ch_list)

        for group in part_info["groups"]:
            group_id = group["id"]
            total_q = group["count"]
            or_count = group.get("or_count", 0)

            base = total_q // num_chapters
            remainder = total_q % num_chapters
            or_base = or_count // num_chapters
            or_remainder = or_count % num_chapters

            group_dist = []
            for i, ch in enumerate(ch_list):
                count = base + (1 if i < remainder else 0)
                ch_or = or_base + (1 if i < or_remainder else 0)
                if count > 0:
                    group_dist.append({
                        "chapter": ch.chapter,
                        "count": count,
                        "or_count": min(ch_or, count),
                    })

            part_dist[group_id] = group_dist
        distribution[part_key] = part_dist

    return distribution


def generate_cbse_accountancy_paper(request, context_chunks, feedback=None):
    if not context_chunks:
        raise GenerationError("No NCERT content found for Accountancy.", 404)

    client = _get_gemini_client()
    model = settings.GEMINI_GEN_MODEL
    models = [model]
    fallback = getattr(settings, 'GEMINI_FALLBACK_MODEL', None)
    if fallback and fallback != model:
        models.append(fallback)

    pattern = CBSE_ACCOUNTANCY_PATTERN
    distribution = _distribute_accountancy_chapters(request.chapters, pattern)

    logger.info(f"CBSE Accountancy Paper: {len(request.chapters)} chapters, "
                f"target={pattern['total_questions']} questions, {pattern['total_marks']} marks")

    all_questions = []
    t0 = time.time()

    for part_key, part_info in pattern["parts"].items():
        part_groups = distribution.get(part_key, {})
        if not part_groups:
            logger.info(f"Skipping {part_info['title']} — no chapters selected")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"{part_info['title']}: {part_info['subtitle']}")

        for group in part_info["groups"]:
            group_id = group["id"]
            group_chapters = part_groups.get(group_id, [])
            if not group_chapters:
                continue

            marks = group["marks_per_q"]
            formats = group["formats"]
            difficulty = group["difficulty"]

            logger.info(f"\n  Group {group_id}: {group['count']} × {marks}m (or={group.get('or_count', 0)})")

            for ch_entry in group_chapters:
                ch_name = ch_entry["chapter"]
                count = ch_entry["count"]
                or_count = ch_entry.get("or_count", 0)

                orig_ch = next((c for c in request.chapters if c.chapter == ch_name), None)
                if not orig_ch:
                    continue

                if group_id in ("A1", "B1_1"):
                    mcq_total = group.get("mcq_count", 12)
                    ar_total = group.get("ar_count", 4)
                    total_group = mcq_total + ar_total
                    ch_mcq = max(1, round(count * mcq_total / total_group))
                    ch_ar = count - ch_mcq

                    if ch_mcq > 0:
                        mcq_chapter = ChapterSection(
                            chapter=ch_name,
                            difficulty=DifficultyLevel(difficulty),
                            format=QuestionFormat("mcq"),
                            marks_per_question=marks,
                            quantity=ch_mcq + min(or_count, ch_mcq),
                        )
                        prompt = _build_accountancy_cbse_prompt(
                            mcq_chapter, request, context_chunks,
                            ch_mcq + min(or_count, ch_mcq), group, part_key,
                            generate_or=(or_count > 0),
                        )
                        for m in models:
                            try:
                                raw = _call_gemini(client, prompt, m)
                                batch_qs = _parse_batch(raw, mcq_chapter, request, f"{part_key}_{marks}m")
                                if batch_qs:
                                    main_qs = [q for q in batch_qs if not getattr(q, '_is_or', False)]
                                    or_qs = [q for q in batch_qs if getattr(q, '_is_or', False)]
                                    all_questions.extend(main_qs[:ch_mcq])
                                    all_questions.extend(or_qs[:min(or_count, ch_mcq)])
                                    logger.info(f"    MCQ {ch_name}: {len(main_qs[:ch_mcq])} + {len(or_qs[:min(or_count, ch_mcq)])} OR")
                                    break
                            except GenerationError as e:
                                if m != models[-1]:
                                    continue
                                logger.error(f"    MCQ {ch_name} failed: {e}")
                        time.sleep(settings.BATCH_DELAY)

                    if ch_ar > 0:
                        ar_or = max(0, or_count - min(or_count, ch_mcq))
                        ar_chapter = ChapterSection(
                            chapter=ch_name,
                            difficulty=DifficultyLevel(difficulty),
                            format=QuestionFormat("assertion_reason"),
                            marks_per_question=marks,
                            quantity=ch_ar + ar_or,
                        )
                        prompt = _build_accountancy_cbse_prompt(
                            ar_chapter, request, context_chunks,
                            ch_ar + ar_or, group, part_key,
                            generate_or=(ar_or > 0),
                        )
                        for m in models:
                            try:
                                raw = _call_gemini(client, prompt, m)
                                batch_qs = _parse_batch(raw, ar_chapter, request, f"{part_key}_{marks}m")
                                if batch_qs:
                                    main_qs = [q for q in batch_qs if not getattr(q, '_is_or', False)]
                                    or_qs = [q for q in batch_qs if getattr(q, '_is_or', False)]
                                    all_questions.extend(main_qs[:ch_ar])
                                    all_questions.extend(or_qs[:ar_or])
                                    logger.info(f"    AR {ch_name}: {len(main_qs[:ch_ar])} + {len(or_qs[:ar_or])} OR")
                                    break
                            except GenerationError as e:
                                if m != models[-1]:
                                    continue
                                logger.error(f"    AR {ch_name} failed: {e}")
                        time.sleep(settings.BATCH_DELAY)

                else:
                    fmt = formats[0]
                    if marks >= 4 and "journal_entry" in formats:
                        fmt = "journal_entry"
                    elif marks >= 6 and "long_answer" in formats:
                        fmt = "long_answer"

                    sec_chapter = ChapterSection(
                        chapter=ch_name,
                        difficulty=DifficultyLevel(difficulty),
                        format=QuestionFormat(fmt),
                        marks_per_question=marks,
                        quantity=count + or_count,
                    )
                    prompt = _build_accountancy_cbse_prompt(
                        sec_chapter, request, context_chunks,
                        count + or_count, group, part_key,
                        generate_or=(or_count > 0),
                    )
                    for m in models:
                        try:
                            raw = _call_gemini(client, prompt, m)
                            batch_qs = _parse_batch(raw, sec_chapter, request, f"{part_key}_{marks}m")
                            if batch_qs:
                                main_qs = [q for q in batch_qs if not getattr(q, '_is_or', False)]
                                or_qs = [q for q in batch_qs if getattr(q, '_is_or', False)]
                                all_questions.extend(main_qs[:count])
                                all_questions.extend(or_qs[:or_count])
                                logger.info(f"    {fmt} {ch_name}: {len(main_qs[:count])} + {len(or_qs[:or_count])} OR")
                                break
                        except GenerationError as e:
                            if m != models[-1]:
                                continue
                            logger.error(f"    {fmt} {ch_name} failed: {e}")
                    time.sleep(settings.BATCH_DELAY)

    elapsed = time.time() - t0
    main_count = len([q for q in all_questions if not getattr(q, '_is_or', False)])
    or_count_total = len([q for q in all_questions if getattr(q, '_is_or', False)])
    logger.info(f"\nCBSE Accountancy Done: {main_count} main + {or_count_total} OR = {len(all_questions)} in {elapsed:.1f}s")

    if not all_questions:
        raise GenerationError("All Accountancy generation failed.", 500)

    return all_questions


# ---------------------------------------------------------------------------
# Main Entry — routes by subject + pattern
# ---------------------------------------------------------------------------
def generate_questions(request, context_chunks, feedback=None, cbse_pattern: bool = False):
    subject_lower = (request.subject or "").lower()
    is_accountancy = subject_lower in ACCOUNTANCY_SUBJECTS

    if cbse_pattern and is_accountancy:
        return generate_cbse_accountancy_paper(request, context_chunks, feedback)

    if cbse_pattern:
        return generate_cbse_paper(request, context_chunks, feedback)

    # Original per-chapter mode
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
# Aliases for backward compatibility
# ---------------------------------------------------------------------------
generate_test = generate_questions


def handle_feedback(*args, **kwargs):
    return None