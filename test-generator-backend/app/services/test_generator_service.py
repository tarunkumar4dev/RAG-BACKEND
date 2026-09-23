"""
Test Generator Service v18 — ROBUST HISTORY SUBTOPIC MATCHING

v18 changes (only one function changed):
  - _get_subtopic_context() now uses aggressive normalization +
    substring + word-overlap matching, so chapter name variations
    like "Nationalism in India", "The Rise of Nationalism in Europe",
    "Rise of Nationalism Europe" all match correctly

v18.1 additions:
  - English Writing Skills prompt builderr
  - English Grammar prompt builder
  - Routing in _generate_for_chapter for writing/grammar pseudo-chapters
"""

import json
import logging
import random
import re
import time
import uuid
from typing import List, Dict, Optional
import math
import concurrent.futures

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
    QuestionTable,
    SubPart,
    MarkingStep,
)

logger = logging.getLogger(__name__)

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


def _ch_match(chunk_ch: str, target: str) -> bool:
    """Check if a chunk's chapter name matches the target chapter name."""
    a = (chunk_ch or "").upper().strip()
    b = (target or "").upper().strip()
    if not a or not b:
        return False
    return a == b or (a in b) or (b in a)


ASSERTION_REASON_OPTIONS = [
    "A) Both A and R are true and R is the correct explanation of A",
    "B) Both A and R are true but R is NOT the correct explanation of A",
    "C) A is true but R is false",
    "D) A is false but R is true",
]

DIFF_INST = {
    "easy": "EASY: Direct conceptual question testing understanding. AVOID pure historical/biographical recall (no 'who discovered when' questions). Bloom: understand.",
    "medium": "MEDIUM: 2-3 steps applying concepts to real-world scenarios. Use 'Give reasons', 'Why does', or 'Explain how' framing. Bloom: understand/apply.",
    "hard": "HARD: 3+ steps, scenario-based reasoning, compare-contrast across concepts. Distractors: wrong concept, common misconception. Bloom: apply/analyze.",
    "very_hard": "VERY HARD: Multi-concept analysis, experimental reasoning, evaluate trade-offs. All options plausible. Bloom: analyze/evaluate/create.",
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

# ═══════════════════════════════════════════════════════════════════════════
# PEDAGOGY_DIRECTIVES — Per-format pedagogical rules
# This is the core quality lever. Each format gets its own instruction block
# that forces Gemini to produce board-exam-quality questions.
# ═══════════════════════════════════════════════════════════════════════════

PEDAGOGY_DIRECTIVES = {
    "mcq": """═══ MCQ PEDAGOGY — MANDATORY RULES ═══

QUESTION STEM TEMPLATES (rotate across questions, never repeat same template in batch):
  • "A student observes [scenario]. What will happen and why?"
  • "Which of the following statements is INCORRECT about [concept]?"
  • "In the given experimental setup, [observation]. The reason is:"
  • "[Two contrasting situations]. The difference arises because:"
  • "Identify the correct sequence of steps in [process]."
  • "Consider the following statements: (I)... (II)... (III)... Which are correct?"
  • "Match the following: Column A... Column B... Choose the correct option."

DISTRACTOR RULES (all 4 mandatory):
  • One distractor = common student MISCONCEPTION (the mistake students actually make)
  • One distractor = partially correct (right concept, wrong application or scope)
  • One distractor = computational/unit/sign ERROR trap (for numerical subjects)
  • Correct answer position MUST vary: distribute roughly 25% each across A/B/C/D in the batch

FORBIDDEN STEMS (instant rejection):
  • "What is the definition of..."
  • "Who discovered / Who invented..."
  • "In which year..."
  • "What is the full form of..."
  • "Define [term]"
  • "[Name] is also known as..."
""",

    "assertion_reason": """═══ ASSERTION-REASON PEDAGOGY ═══

AR CONSTRUCTION RULES:
  • Assertion (A) must state a CONCEPTUAL claim — NOT a trivial fact
  • Reason (R) must state a RELEVANT scientific/mathematical/logical principle
  • A and R must BOTH be testable claims that require understanding
  • In a batch of 4 AR questions, distribute answers:
    - At least 1 must be option B (both true, R NOT explanation of A)
    - At least 1 must be option C or D (one statement is false)
    - Do NOT make all answers option A

FORBIDDEN AR patterns:
  • A = "Water boils at 100°C", R = "Heat increases temperature" (trivial)
  • A = definition of X, R = definition of Y (recall-only)
  • A and R that are completely unrelated (makes R irrelevant)

REQUIRED AR patterns:
  • A = "[Conceptual phenomenon]", R = "[Underlying principle that may or may not cause A]"
  • A = "[Observable effect]", R = "[Mechanism — student must judge if R explains A]"
""",

    "short_answer": """═══ SHORT ANSWER PEDAGOGY (2-3 marks) ═══

QUESTION CONSTRUCTION:
  • NUMERICAL questions MUST include "Given:" data block with values
  • NON-NUMERICAL must ask for reasoning: "Give reasons", "Explain why", "Differentiate"
  • AVOID bare "Define X" — instead ask "Define X and give one example where it applies"

ANSWER QUALITY:
  • "correct_answer" = complete MODEL ANSWER with each step numbered
    Example: "Step 1: Identify that... \\nStep 2: Apply formula... \\nStep 3: Calculate..."
  • "explanation" = MARKING SCHEME breakdown
    Example: "Step 1 (1m): Correct formula. Step 2 (1m): Substitution. Step 3 (1m): Final answer with unit."
  • Answer length: 30-80 words (proportional to marks)
""",

    "long_answer": """═══ LONG ANSWER PEDAGOGY (5 marks) ═══

QUESTION STRUCTURE (mandatory):
  • Structure as: (a) Derivation/Description [3m] + (b) Application/Diagram [2m]
  • OR Structure: (a) Explain concept [3m] + (b) Compare with related concept [2m]
  • Include "Explain with the help of a diagram" where applicable (describe diagram in text)

ANSWER QUALITY:
  • "correct_answer" = complete MODEL ANSWER (150-200 words) with:
    - Clear paragraph structure
    - Technical terms properly used
    - Diagrams described in text: "[Diagram: Label X, Y, Z showing...]"
  • "explanation" = detailed MARKING SCHEME:
    - "Introduction (1m): Define the concept. Body (2m): Explain mechanism with examples. Application (1m): Real-world relevance. Diagram (1m): Correct labeling."
  • Include "marking_scheme" field: [{"step": "Definition", "marks": 1}, {"step": "Explanation with mechanism", "marks": 2}, ...]
""",

    "case_based": """═══ CASE STUDY PEDAGOGY (4 marks) ═══

PASSAGE RULES:
  • Passage: 80-120 words describing a REAL-WORLD scenario
  • Use Indian contexts: news, sports, daily life, school experiments, NCERT activities
  • The passage must contain enough information to answer ALL sub-parts

SUB-PART RULES (EXACTLY 3 sub-parts mandatory):
  • (i) [1 mark] — Recall/Identify: "What is X mentioned in the passage?"
  • (ii) [1 mark] — Apply/Explain: "Why does Y happen in this scenario?"
  • (iii) [2 marks] — Analyze/Evaluate: "What would happen if Z was changed? Explain."
  • Sub-parts MUST escalate in cognitive difficulty
  • Include sub-parts in the "sub_parts" JSON field:
    [{"label": "i", "text": "...", "marks": 1, "answer": "..."}, ...]
  • Also include all sub-parts in the "text" field as: "(i) ... (ii) ... (iii) ..."

ANSWER QUALITY:
  • "correct_answer" = combined answer for ALL sub-parts with labels (i)/(ii)/(iii)
  • Each sub-answer must be proportional to marks (1m = 1-2 sentences, 2m = 3-4 sentences)
""",
}

# Subject-specific difficulty overrides
DIFF_INST_OVERRIDES = {
    "mathematics": {
        "easy": "EASY: Direct formula application, 1-step calculation. No multi-step. Bloom: remember/understand.",
        "medium": "MEDIUM: 2-step problem requiring formula recall + substitution. Real-world word problem. Bloom: apply.",
        "hard": "HARD: Multi-concept (e.g., combine AP+GP, or geometry+algebra). Prove/derive. Bloom: analyze.",
        "very_hard": "VERY HARD: Competition-style. Novel problem combining 3+ concepts. Bloom: evaluate/create.",
    },
    "science": {
        "easy": "EASY: Identify/name/state a concept or phenomenon. Bloom: remember.",
        "medium": "MEDIUM: Explain WHY a phenomenon occurs. Use 'Give reasons'. Bloom: understand/apply.",
        "hard": "HARD: Experimental reasoning or compare-contrast across concepts. Bloom: apply/analyze.",
        "very_hard": "VERY HARD: Design experiment, predict outcomes, evaluate trade-offs. Bloom: analyze/evaluate.",
    },
    "social science": {
        "easy": "EASY: Identify key facts, events, or features. Bloom: remember.",
        "medium": "MEDIUM: Explain significance, describe causes/effects. Bloom: understand/apply.",
        "hard": "HARD: Analyze causes, compare movements/events, evaluate outcomes. Bloom: analyze/evaluate.",
        "very_hard": "VERY HARD: Critical analysis of sources, debate perspectives. Bloom: evaluate/create.",
    },
    "history": {
        "easy": "EASY: Identify key events, movements, or leaders. Bloom: remember.",
        "medium": "MEDIUM: Explain significance of an event, describe 3 features/reforms. Bloom: understand.",
        "hard": "HARD: Compare two movements, analyze causes of failure/success. Bloom: analyze.",
        "very_hard": "VERY HARD: Evaluate historiographical perspectives, source-based critical analysis. Bloom: evaluate.",
    },
    "english": {
        "easy": "EASY: Identify grammar rule, recall literary device. Bloom: remember.",
        "medium": "MEDIUM: Apply grammar in context, interpret a passage. Bloom: apply.",
        "hard": "HARD: Analyze author's intent, evaluate argument quality. Bloom: analyze.",
        "very_hard": "VERY HARD: Create original response (letter/paragraph), synthesize themes. Bloom: create.",
    },
}

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
- total_row: null""",

    "ledger": """Generate a Ledger preparation question for CBSE Class {class_grade} Accountancy.

The question should give transactions and ask to prepare a specific ledger account.
The answer MUST include an "answer_table" with:
- type: "ledger"
- headers: ["Date", "Particulars", "J.F.", "Amount (Rs.)", "Date", "Particulars", "J.F.", "Amount (Rs.)"]
- rows: Each row has 8 strings.
- total_row: 8 strings with totals on both sides""",

    "trial_balance": """Generate a Trial Balance preparation question for CBSE Class {class_grade} Accountancy.

The answer MUST include an "answer_table" with:
- type: "trial_balance"
- headers: ["S.No.", "Account Name", "L.F.", "Debit (Rs.)", "Credit (Rs.)"]
- rows: Each row has 5 strings.
- total_row: ["", "Total", "", "X,XXX", "X,XXX"]""",
}

STATISTICS_CHAPTERS = {"statistics", "data handling", "data analysis"}
STATISTICS_TOPIC_KEYWORDS = {
    "frequency", "mean", "median", "mode", "histogram", "ogive", "cumulative",
    "class interval", "grouped data", "frequency distribution", "frequency polygon",
}
TABLE_REQUIRED_TRIGGERS = (
    "following table", "following frequency distribution", "following frequency",
    "following data", "table shows", "data given below", "given below",
    "the table below", "from the table", "in the table", "based on the data",
    "the data:", "calculate the mean of the following", "calculate the median of the following",
    "calculate the mode of the following", "find the mean of the following",
    "find the median of the following", "find the mode of the following",
)


# ═══════════════════════════════════════════════════════════════════════════
# English Writing & Grammar Prompts
# ═══════════════════════════════════════════════════════════════════════════

ENGLISH_WRITING_TYPES = {
    "letter": "Formal Letter (100-120 words) — letter to editor, complaint, enquiry, request to authority, application",
    "paragraph": "Analytical Paragraph (100-120 words) — based on a chart/graph/data/cue/situation",
    "notice": "Notice Writing (50-60 words) — school notice board announcements, events, lost & found, meetings",
    "invitation": "Invitation (50-60 words) — formal/informal invitation cards for events, ceremonies, functions",
    "poster": "Poster Making (50-60 words) — awareness campaigns, social issues, school events with visual layout described",
    "any": "Either a Formal Letter, Analytical Paragraph, Notice, or Invitation (as per marks). Vary across questions.",
}

ENGLISH_GRAMMAR_TOPICS = [
    "Tenses (present/past/future, perfect, continuous)",
    "Modals (can, could, may, might, must, should, would, will, shall)",
    "Determiners (a, an, the, this, that, some, any, much, many, few, little)",
    "Subject-Verb Concord",
    "Reported Speech (statements, questions, commands, requests)",
    "Gap-Filling exercises",
    "Editing/Omission exercises",
    "Sentence Transformation",
]


def _build_english_writing_prompt(chapter, request, count, section_key=None, section_info=None):
    """Build prompt for CBSE English Writing Skills questions."""
    if hasattr(chapter, 'difficulty'):
        diff_val = chapter.difficulty.value if hasattr(chapter.difficulty, 'value') else str(chapter.difficulty)
    elif isinstance(chapter, dict) and 'difficulty' in chapter:
        diff_obj = chapter['difficulty']
        diff_val = diff_obj.value if hasattr(diff_obj, 'value') else str(diff_obj)
    else:
        diff_val = "medium"

    if hasattr(chapter, 'format'):
        fmt_val = chapter.format.value if hasattr(chapter.format, 'value') else str(chapter.format)
    elif isinstance(chapter, dict) and 'format' in chapter:
        fmt_obj = chapter['format']
        fmt_val = fmt_obj.value if hasattr(fmt_obj, 'value') else str(fmt_obj)
    else:
        fmt_val = "long_answer"

    if hasattr(chapter, 'marks_per_question'):
        marks = chapter.marks_per_question
    elif isinstance(chapter, dict) and 'marks_per_question' in chapter:
        marks = chapter['marks_per_question']
    else:
        marks = 5

    # Decide writing type from topic field, fallback to "any"
    if hasattr(chapter, 'topic'):
        raw_topic = getattr(chapter, 'topic', None) or ""
    elif isinstance(chapter, dict):
        raw_topic = chapter.get('topic') or ""
    elif isinstance(chapter, str):
        raw_topic = chapter
    else:
        raw_topic = ""

    topic = raw_topic.lower()
    if "letter" in topic:
        writing_type = ENGLISH_WRITING_TYPES["letter"]
    elif "paragraph" in topic or "analytical" in topic:
        writing_type = ENGLISH_WRITING_TYPES["paragraph"]
    elif "notice" in topic:
        writing_type = ENGLISH_WRITING_TYPES["notice"]
    elif "invitation" in topic or "invite" in topic:
        writing_type = ENGLISH_WRITING_TYPES["invitation"]
    elif "poster" in topic:
        writing_type = ENGLISH_WRITING_TYPES["poster"]
    else:
        writing_type = ENGLISH_WRITING_TYPES["any"]

    if any(k in writing_type.lower() for k in ("notice", "invitation", "poster")):
        word_limit = "50-60 words"
    else:
        word_limit = "100-120 words"

    section_ctx = ""
    if section_key and section_info:
        section_ctx = f"\nThis is for {section_info.get('title', '')} ({section_info.get('subtitle', '')}).\nEach question: {marks} marks."

    section_field = f', "section": "{section_key}"' if section_key else ''

    return f"""You are an expert CBSE Class {request.class_grade} English paper setter for the Writing Skills section.
{section_ctx}

Generate {count} unique CBSE-style WRITING questions.

WRITING TYPE: {writing_type}

EACH QUESTION MUST:
- Present a clear real-world scenario (current/relatable to Indian students)
- Specify exact word limit: {word_limit}
- For Letters: give all required details (sender, receiver, purpose, key points to include)
- For Analytical Paragraphs: provide a chart/graph description, data table, or visual cue in words
- For Notices/Invitations/Posters: include issuing authority/organizer, target audience, date/time/venue, and key guidelines
- Be answerable based on the cue given (no external knowledge needed)
- The "correct_answer" should be a complete model answer ({word_limit}) showing proper format
- The "explanation" should explain the marking scheme: format (1m), content ({max(1, marks - 2)}m), expression/grammar ({1 if marks <= 3 else 2}m)

CBSE LETTER FORMAT REMINDER (in your model answer):
- Sender's address
- Date
- Receiver's address
- Subject line
- Salutation
- Body (3 paragraphs: introduction, main content, closing)
- Complimentary close + name

CBSE ANALYTICAL PARAGRAPH REMINDER (in your model answer):
- Opening sentence stating what the data shows
- 2-3 sentences analyzing trends/patterns/comparisons
- Closing sentence with conclusion or inference

CBSE NOTICE / INVITATION / POSTER FORMAT REMINDER (in your model answer):
- Organization / Institution Name in bold/caps
- Heading / Title (e.g. NOTICE / INVITATION)
- Date of issue
- Event name, Date, Time, Venue, and Chief Guest / Contact Person details
- Signatory with Name and Designation

"format" must be exactly "long_answer". "options": null.

Return ONLY valid JSON:
{{"questions":[{{"text":"[full scenario with all cue details]","format":"long_answer","options":null,"correct_answer":"[complete {word_limit} model answer with proper format]","explanation":"Marking: Format (1m) + Content (2m) + Expression (2m) = {marks}m. Key points expected: ...","marks":{marks},"difficulty":"{diff_val}","bloom_level":"create","chapter":"{chapter.chapter}","topic":"{getattr(chapter, 'topic', None) or 'writing'}"{section_field}}}]}}"""


def _build_english_grammar_prompt(chapter, request, count, section_key=None, section_info=None):
    """Build prompt for CBSE English Grammar questions."""
    diff_val = chapter.difficulty.value if hasattr(chapter.difficulty, 'value') else str(chapter.difficulty)
    fmt_val = chapter.format.value if hasattr(chapter.format, 'value') else str(chapter.format)
    marks = chapter.marks_per_question

    section_ctx = ""
    if section_key and section_info:
        section_ctx = f"\nThis is for {section_info.get('title', '')}.\nEach question: {marks} marks."

    section_field = f', "section": "{section_key}"' if section_key else ''

    topics_list = "\n".join(f"  • {t}" for t in ENGLISH_GRAMMAR_TOPICS)

    if fmt_val == "mcq":
        fmt_line = '"options": 4 options labeled A) B) C) D). correct_answer = exact full option text.'
        json_tmpl = (
            f'{{"questions":[{{"text":"Fill in the blank: She ___ to school every day. Choose the correct option.",'
            f'"format":"mcq","options":["A) go","B) goes","C) going","D) gone"],'
            f'"correct_answer":"B) goes","explanation":"Subject-verb agreement: singular subject \\"She\\" takes singular verb \\"goes\\".",'
            f'"marks":{marks},"difficulty":"{diff_val}","bloom_level":"apply",'
            f'"chapter":"{chapter.chapter}","topic":"specific grammar topic"{section_field}}}]}}'
        )
    elif fmt_val == "short_answer":
        fmt_line = '"options": null. Provide direct answer (filled blank, corrected sentence, or transformed sentence).'
        json_tmpl = (
            f'{{"questions":[{{"text":"Fill in the blank with the correct form of verb: She ___ (go) to school every day.",'
            f'"format":"short_answer","options":null,'
            f'"correct_answer":"goes","explanation":"Simple present tense, third person singular requires \\"goes\\".",'
            f'"marks":{marks},"difficulty":"{diff_val}","bloom_level":"apply",'
            f'"chapter":"{chapter.chapter}","topic":"tenses"{section_field}}}]}}'
        )
    else:
        fmt_line = 'Provide 4 options A) B) C) D).'
        json_tmpl = (
            f'{{"questions":[{{"text":"...","format":"{fmt_val}","options":["A) ...","B) ...","C) ...","D) ..."],'
            f'"correct_answer":"B) ...","explanation":"...",'
            f'"marks":{marks},"difficulty":"{diff_val}","bloom_level":"apply",'
            f'"chapter":"{chapter.chapter}","topic":"specific grammar topic"{section_field}}}]}}'
        )

    return f"""You are an expert CBSE Class {request.class_grade} English paper setter for the Grammar section.
{section_ctx}

Generate {count} unique CBSE-style GRAMMAR questions.

GRAMMAR TOPICS TO COVER (rotate across questions, don't repeat):
{topics_list}

EACH QUESTION MUST:
- Test ONE specific grammar concept
- Use Gap-Fill / Editing / Transformation / Reported Speech format
- Be at Class {request.class_grade} difficulty level
- Have an unambiguous correct answer
- Vary topics across questions — DO NOT repeat the same grammar topic

QUESTION TYPES (rotate):
1. Gap-fill: "She ___ (go) to school every day."
2. Editing: "Spot the error: He don't like coffee."
3. Transformation: "Change to passive: They built the house."
4. Reported speech: "Change to indirect: He said, 'I am tired.'"
5. Modal/Determiner choice: "Choose the correct option..."

FORMAT RULES:
{fmt_line}

Return ONLY valid JSON:
{json_tmpl}"""


def _is_statistics_question(chapter_name: str, topic: Optional[str] = None) -> bool:
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
    if not text:
        return False
    if text.count("|") < 4:
        return False
    if not re.search(r"\|[\s\-:]+\|", text):
        return False
    return True


def _references_table(text: str) -> bool:
    if not text:
        return False
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in TABLE_REQUIRED_TRIGGERS)


def _extract_inline_data(text: str) -> Optional[dict]:
    if not text:
        return None
    result = _extract_singleline_markdown(text)
    if result:
        return result
    result = _extract_label_colon_pattern(text)
    if result:
        return result
    return None


def _extract_singleline_markdown(text: str) -> Optional[dict]:
    if text.count("|") < 4:
        return None
    if not re.search(r'-{3,}', text):
        return None
    sep_match = re.search(r'\|?[\s\-]*-{3,}[\s\-|]*-{2,}\s*\|?', text)
    if not sep_match:
        return None
    before = text[:sep_match.start()]
    after = text[sep_match.end():]
    last_break = max(before.rfind('. '), before.rfind('? '), before.rfind('! '), before.rfind('\n'), -1)
    if last_break >= 0:
        before = before[last_break + 1:].lstrip()
    header_match = re.search(r'([A-Za-z][^|]{1,40}?)\s*\|\s*([A-Za-z][^|]{1,40}?)\s*$', before)
    if not header_match:
        return None
    header1 = _clean_label(header_match.group(1).strip())
    header2 = _clean_label(header_match.group(2).strip())
    row_pattern = re.compile(r'(\d+\s*-\s*\d+|\d+\.?\d*|[a-zA-Z]\d?)\s*\|\s*(\d+\.?\d*|[a-zA-Z]\d?)')
    rows = []
    for m in row_pattern.finditer(after):
        c1 = re.sub(r'\s*-\s*', '-', m.group(1).strip())
        c2 = m.group(2).strip()
        rows.append([c1, c2])
    if len(rows) < 3:
        return None
    return {"type": "frequency_distribution", "headers": [header1, header2], "rows": rows, "caption": None}


def _extract_label_colon_pattern(text: str) -> Optional[dict]:
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
            matches.append({"label": label, "values": values, "pos": m.start()})
    if len(matches) < 2:
        return None
    col1, col2 = matches[0], matches[1]
    min_len = min(len(col1["values"]), len(col2["values"]))
    if min_len < 3:
        return None
    col1_values = col1["values"][:min_len]
    col2_values = col2["values"][:min_len]
    col1_has_ranges = any("-" in v for v in col1_values)
    col1_all_numeric = all(re.match(r'^-?\d+\.?\d*$|^[a-zA-Z]\d?$', v) for v in col1_values)
    if not (col1_has_ranges or col1_all_numeric):
        return None
    col1_label = _clean_label(col1["label"], default="Class Interval" if col1_has_ranges else "xi")
    col2_label = _clean_label(col2["label"], default="Frequency")
    return {
        "type": "frequency_distribution",
        "headers": [col1_label, col2_label],
        "rows": [[a, b] for a, b in zip(col1_values, col2_values)],
        "caption": None,
    }


def _clean_label(raw_label: str, default: str = "Value") -> str:
    if not raw_label:
        return default
    cleaned = re.sub(r'\s*\([^\)]*\)\s*', '', raw_label).strip()
    cleaned = re.sub(r'^[:.,;\s]+|[:.,;\s]+$', '', cleaned).strip()
    if not cleaned or len(cleaned) < 2:
        return default
    lower = cleaned.lower()
    label_map = {
        "fi": "Frequency", "f_i": "Frequency",
        "number of students": "Number of Students",
        "number of workers": "Number of Workers",
        "number of patients": "Number of Patients",
        "number of persons": "Number of Persons",
        "number of households": "Number of Households",
        "number of families": "Number of Families",
        "frequency": "Frequency",
        "marks obtained": "Marks Obtained",
        "class interval": "Class Interval",
        "daily wages": "Daily Wages", "age": "Age",
        "daily income": "Daily Income",
        "lifetimes": "Lifetimes", "lifetime": "Lifetime",
        "family size": "Family Size",
        "absentees": "Number of Absentees",
        "number of absentees": "Number of Absentees",
        "daily expenditure": "Daily Expenditure",
        "xi": "xi", "x_i": "xi",
    }
    if lower in label_map:
        return label_map[lower]
    return cleaned.title()


def _parse_inline_values(segment: str) -> List[str]:
    if not segment:
        return []
    segment = re.sub(r'^[\s:;\(\)]+', '', segment)
    segment = re.sub(r'[\s:;\(\)\.]+$', '', segment)
    parts = [p.strip() for p in segment.split(",")]
    valid = []
    for p in parts:
        if re.match(r'^-?\d+\s*-\s*\d+$', p):
            valid.append(re.sub(r'\s*-\s*', '-', p))
        elif re.match(r'^-?\d+\.?\d*$', p):
            valid.append(p)
        elif re.match(r'^[a-zA-Z]\d?$', p):
            valid.append(p)
        else:
            break
    return valid


def _has_inline_data_leak(text: str) -> bool:
    if not text:
        return False
    pattern = re.compile(r'(?:\d+\s*-\s*\d+|\d+)(?:\s*,\s*(?:\d+\s*-\s*\d+|\d+|[a-zA-Z]\d?)){3,}')
    return bool(pattern.search(text))


def _strip_inline_data_from_text(text: str, recovered_table: dict) -> str:
    if not text or not recovered_table or not recovered_table.get("rows"):
        return text
    earliest = len(text)
    label_pattern = re.compile(
        r'([A-Za-z][A-Za-z0-9\s\(\)_/]{2,40}?)\s*[:|]\s*'
        r'((?:\d+\s*-\s*\d+|\d+\.?\d*|[a-zA-Z]\d?)'
        r'(?:\s*,\s*(?:\d+\s*-\s*\d+|\d+\.?\d*|[a-zA-Z]\d?))+)',
    )
    m = label_pattern.search(text)
    if m and m.start() < earliest:
        earliest = m.start()
    pipe_match = re.search(r'[A-Za-z][^|\n]{0,40}\|\s*[A-Za-z][^|\n]{0,40}\s*\|?[\s\-]*-{3,}', text)
    if pipe_match:
        prior_text = text[:pipe_match.start()]
        last_break = max(prior_text.rfind('. '), prior_text.rfind('? '), prior_text.rfind('! '), prior_text.rfind('\n'), -1)
        cut_pos = (last_break + 2) if last_break >= 0 else pipe_match.start()
        if cut_pos < earliest:
            earliest = cut_pos
    if earliest >= len(text):
        return text
    cleaned = text[:earliest].strip()
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


STATISTICS_TABLE_TYPES = {
    "mean": {"type": "frequency_distribution", "headers": ["Class Interval", "Frequency"], "caption": "Marks scored by 50 students"},
    "median": {"type": "cumulative_frequency", "headers": ["Class Interval", "Frequency", "Cumulative Frequency"], "caption": "Cumulative frequency table"},
    "mode": {"type": "frequency_distribution", "headers": ["Class Interval", "Frequency"], "caption": "Frequency distribution table"},
    "ogive": {"type": "cumulative_frequency", "headers": ["Class Interval", "Cumulative Frequency"], "caption": "Cumulative frequency distribution for ogive"},
    "histogram": {"type": "frequency_distribution", "headers": ["Class Interval", "Frequency"], "caption": "Frequency distribution for histogram"},
}


def _detect_statistics_subtopic(topic: Optional[str]) -> str:
    topic_lower = (topic or "").lower()
    if "median" in topic_lower:
        return "median"
    if "mode" in topic_lower:
        return "mode"
    if "ogive" in topic_lower:
        return "ogive"
    if "histogram" in topic_lower:
        return "histogram"
    return "mean"


STATISTICS_PROMPT_TEMPLATES = {
    "frequency_distribution": """STATISTICS QUESTION — STRUCTURED TABLE FORMAT (STRICT)

The "question_table" field is MANDATORY. Questions WITHOUT a valid "question_table" will be REJECTED.
DO NOT put the data table inside "text" as pipes/dashes. Keep "text" CLEAN.

EXACT OUTPUT FORMAT:
{{
  "text": "Find the mean of the given frequency distribution.",
  "question_table": {{
    "type": "frequency_distribution",
    "headers": ["Class Interval", "Frequency"],
    "rows": [["0-10","5"],["10-20","8"],["20-30","15"],["30-40","12"],["40-50","7"],["50-60","3"]],
    "caption": "Marks scored by 50 students"
  }},
  "correct_answer": "...",
  "explanation": "..."
}}

REJECTION CRITERIA:
- "question_table" is null/missing/empty
- "text" contains comma-separated number lists or labels like "Class Interval:", "xi:", "fi:" followed by data
- "rows" contains anything other than strings

DATA RULES:
- Use 5-7 class intervals of EQUAL width
- Σfi should be a round number (30, 40, 50, 60, 80, 100)
- Use realistic CBSE contexts (marks, wages, heights, ages)
"""
}


def _build_statistics_prompt(chapter, request, context_chunks, count, section_key=None, section_info=None):
    ch_chunks = [c for c in context_chunks if _ch_match(c.get("chapter", ""), chapter.chapter)]
    if not ch_chunks:
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

    stat_subtopic = _detect_statistics_subtopic(getattr(chapter, 'topic', None))
    stat_spec = STATISTICS_TABLE_TYPES.get(stat_subtopic, STATISTICS_TABLE_TYPES["mean"])
    tbl_type = stat_spec["type"]
    tbl_headers = json.dumps(stat_spec["headers"])
    tbl_caption = stat_spec["caption"]

    section_ctx = ""
    if section_key and section_info:
        section_ctx = f"\nThis is for {section_info['title']} ({section_info['subtitle']}).\nEach question: {section_info['marks_per_q']} marks. {section_info.get('instruction', '')}"

    if fmt_val == "mcq":
        fmt_line = 'Provide 4 options labeled A) B) C) D). correct_answer = exact full option text.'
    elif fmt_val in ("short_answer", "long_answer"):
        fmt_line = '"options": null. Show full tabular working.'
    elif fmt_val == "assertion_reason":
        fmt_line = '"text": "Assertion (A): ...\\nReason (R): ...". For pure AR, question_table can be null.'
    else:
        fmt_line = '4 options labeled A) B) C) D).'

    template_instruction = STATISTICS_PROMPT_TEMPLATES["frequency_distribution"]
    section_field = f', "section": "{section_key}"' if section_key else ''

    if fmt_val == "assertion_reason":
        json_tmpl = (
            f'{{"questions":[{{"text":"Assertion (A): ...\\nReason (R): ...","format":"assertion_reason",'
            f'"options":["A) ...","B) ...","C) ...","D) ..."],"correct_answer":"A) ...","explanation":"...",'
            f'"question_table":null,"marks":{chapter.marks_per_question},"difficulty":"{diff_val}",'
            f'"bloom_level":"apply","chapter":"{chapter.chapter}","topic":"{stat_subtopic}"{section_field}}}]}}'
        )
    elif fmt_val == "mcq":
        json_tmpl = (
            f'{{"questions":[{{"text":"Find the {stat_subtopic} of the given distribution.","format":"mcq",'
            f'"options":["A) 28.4","B) 30.2","C) 32.6","D) 26.8"],"correct_answer":"B) 30.2",'
            f'"explanation":"Calculation: Mean/Median/Mode = 30.2",'
            f'"question_table":{{"type":"{tbl_type}","headers":{tbl_headers},'
            f'"rows":[["0-10","5"],["10-20","8"],["20-30","15"],["30-40","12"],["40-50","7"],["50-60","3"]],'
            f'"caption":"{tbl_caption}"}},'
            f'"marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"apply",'
            f'"chapter":"{chapter.chapter}","topic":"{stat_subtopic}"{section_field}}}]}}'
        )
    else:
        json_tmpl = (
            f'{{"questions":[{{"text":"Find the {stat_subtopic} of the given distribution.","format":"{fmt_val}",'
            f'"options":null,"correct_answer":"...","explanation":"...",'
            f'"question_table":{{"type":"{tbl_type}","headers":{tbl_headers},'
            f'"rows":[["0-10","5"],["10-20","8"],["20-30","15"],["30-40","12"],["40-50","7"],["50-60","3"]],'
            f'"caption":"{tbl_caption}"}},'
            f'"marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"apply",'
            f'"chapter":"{chapter.chapter}","topic":"{stat_subtopic}"{section_field}}}]}}'
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
Each question must test a DIFFERENT concept ({stat_subtopic}). Vary contexts and data.

NCERT Reference:
{ctx}

Generate EXACTLY {count} unique questions. Return ONLY valid JSON:
{json_tmpl}"""


CBSE_SECTIONS = {
    "A": {"title": "Section A", "subtitle": "Multiple Choice Questions / Assertion-Reason", "marks_per_q": 1, "count": 20, "total_marks": 20, "formats": ["mcq", "assertion_reason"], "mcq_count": 16, "ar_count": 4, "difficulty": "easy", "bloom": ["remember", "understand"], "instruction": "All questions are compulsory. Each question carries 1 mark."},
    "B": {"title": "Section B", "subtitle": "Very Short Answer Type Questions", "marks_per_q": 2, "count": 5, "total_marks": 10, "formats": ["short_answer"], "difficulty": "medium", "bloom": ["understand", "apply"], "instruction": "All questions are compulsory. Each question carries 2 marks."},
    "C": {"title": "Section C", "subtitle": "Short Answer Type Questions", "marks_per_q": 3, "count": 6, "total_marks": 18, "formats": ["short_answer"], "difficulty": "medium", "bloom": ["apply", "analyze"], "instruction": "All questions are compulsory. Each question carries 3 marks."},
    "D": {"title": "Section D", "subtitle": "Long Answer Type Questions", "marks_per_q": 5, "count": 4, "total_marks": 20, "formats": ["long_answer"], "difficulty": "hard", "bloom": ["analyze", "evaluate"], "instruction": "All questions are compulsory. Each question carries 5 marks."},
    "E": {"title": "Section E", "subtitle": "Case Study / Source Based Questions", "marks_per_q": 4, "count": 3, "total_marks": 12, "formats": ["case_based"], "difficulty": "hard", "bloom": ["apply", "analyze", "evaluate"], "instruction": "All questions are compulsory. Each question carries 4 marks."},
}

CBSE_ACCOUNTANCY_PATTERN = {
    "total_questions": 34, "total_marks": 80,
    "parts": {
        "A": {
            "title": "Part A", "subtitle": "Accounting for Partnership Firms and Companies", "marks": 60,
            "instruction": "Question 1 to 16 carry 1 mark each. Questions 17 to 20 carry 3 marks each. Questions 21-22 carry 4 marks each. Questions 23 to 26 carry 6 marks each.",
            "groups": [
                {"id": "A1", "marks_per_q": 1, "count": 16, "or_count": 4, "formats": ["mcq", "assertion_reason"], "mcq_count": 12, "ar_count": 4, "difficulty": "easy", "blooms": ["remember", "understand"]},
                {"id": "A3", "marks_per_q": 3, "count": 4, "or_count": 2, "formats": ["short_answer"], "difficulty": "medium", "blooms": ["understand", "apply"]},
                {"id": "A4", "marks_per_q": 4, "count": 2, "or_count": 1, "formats": ["short_answer", "journal_entry"], "difficulty": "medium", "blooms": ["apply", "analyze"]},
                {"id": "A6", "marks_per_q": 6, "count": 4, "or_count": 2, "formats": ["long_answer", "journal_entry", "ledger"], "difficulty": "hard", "blooms": ["analyze", "evaluate"]},
            ],
        },
        "B1": {
            "title": "Part B (Option I)", "subtitle": "Analysis of Financial Statements", "marks": 20,
            "instruction": "Question 27 to 30 carry 1 mark each. Questions 31-32 carry 3 marks each. Question 33 carries 4 marks. Question 34 carries 6 marks.",
            "groups": [
                {"id": "B1_1", "marks_per_q": 1, "count": 4, "or_count": 2, "formats": ["mcq", "assertion_reason"], "mcq_count": 3, "ar_count": 1, "difficulty": "easy", "blooms": ["remember", "understand"]},
                {"id": "B1_3", "marks_per_q": 3, "count": 2, "or_count": 1, "formats": ["short_answer"], "difficulty": "medium", "blooms": ["understand", "apply"]},
                {"id": "B1_4", "marks_per_q": 4, "count": 1, "or_count": 1, "formats": ["short_answer"], "difficulty": "medium", "blooms": ["apply", "analyze"]},
                {"id": "B1_6", "marks_per_q": 6, "count": 1, "or_count": 0, "formats": ["long_answer"], "difficulty": "hard", "blooms": ["analyze", "evaluate"]},
            ],
        },
    },
}

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
    if any(kw in ch_lower for kw in ["partner", "share", "debenture", "company", "goodwill", "admission", "retire", "death", "dissolut", "forfeit"]):
        return "A"
    if any(kw in ch_lower for kw in ["ratio", "cash flow", "financial statement", "comparative", "common size", "balance sheet"]):
        return "B1"
    return "A"


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


MATH_FORMAT_INSTRUCTION = """MATH FORMATTING RULES:
• Use UNICODE: α β γ θ π σ φ ω ε δ λ μ Σ Π Δ
• Fractions: (numerator/denominator)
• Square root: √(x)
• Powers: x², x³, xⁿ
• Subscripts: a₁, a₂, xₙ
• Inequalities: ≤ ≥ ≠ ≈
• Set: ∈ ∉ ∪ ∩ ⊂ ⊃ ⊆ ⊇ ∅ ℝ ℤ ℕ ℚ
• Arrows: -> => <= <-
• Logical: ∀ ∃ ∴ ∵
• Operations: × ÷ ± ∓ · ∞
• Ordinals: "1st", "2nd", "15th" (NOT superscript)

CRITICAL JSON RULES:
- Do NOT use unescaped newlines inside strings — use \\n
- Do NOT use LaTeX commands like \\frac, \\sqrt, \\theta
- Do NOT use $ delimiters
- Do NOT use Unicode modifier letters (ᵗʰ, ˢᵗ)"""


CBSE_QUALITY_DIRECTIVES = """═══ CBSE BOARD PATTERN — STRICT QUALITY RULES ═══

❌ FORBIDDEN — DO NOT generate:
  • "Who first discovered X and in what year?"
  • "When did [scientist] make [discovery]?"
  • "What material did [scientist] use for first observation?"
  • Pure biographical/historical recall about scientists
  • Single-fact recall on dates, names, founding years
  • Questions answerable from chapter introduction page only

✅ REQUIRED — Generate CBSE board-style questions:
  • SCENARIO-BASED: "Renuka kept onion peel in hypotonic solution; Sahil kept
    RBC in same. Onion peel swelled, RBC burst. Why? Explain."
  • "GIVE REASONS" / "EXPLAIN WHY": "Why is mitochondria called the powerhouse?"
  • COMPARE & CONTRAST: "Write 3 differences between plant cell and animal cell."
  • DIAGRAM-BASED: "Draw a plant cell. Label parts that (a) synthesize protein..."
  • APPLY-A-CONCEPT: "Why do vegetables release water when salted?"
  • EXPERIMENTAL REASONING: "Two beakers with raisins — explain difference."
  • ASSERTION-REASON ON CONCEPTS (not history)

═══ COGNITIVE DISTRIBUTION (mandatory) ═══
  • PURE RECALL — MAXIMUM 20%
  • CONCEPTUAL UNDERSTANDING — ~30%
  • APPLIED SCENARIOS + REASONING — ~50%

═══ CONTENT FOCUS ═══
Even if intro chunks discuss discovery history (Robert Hooke 1665, etc.),
DO NOT make those the subject. Use as background only. Generate questions
on CORE CONCEPTS and APPLICATIONS that CBSE actually tests.
"""


CBSE_HISTORY_SUBTOPICS = {
    "the rise of nationalism in europe": [
        "French Revolution and the idea of nation (La Patrie, Le Citoyen)",
        "Napoleonic Code and administrative reforms (Civil Code of 1804)",
        "Congress of Vienna 1815 and Metternich's conservative regime",
        "Revolutions of 1830 and 1848 — liberal nationalism",
        "Greek War of Independence (1821-1832)",
        "Unification of Germany — Bismarck, Prussia, wars with Denmark/Austria/France",
        "Unification of Italy — Mazzini, Cavour, Garibaldi, Victor Emmanuel II",
        "Unification of Britain — Act of Union 1707, bloodless revolution 1688",
        "Visualising the nation — Marianne, Germania, female allegories",
        "Nationalism and culture — Romanticism, Herder, Grimm Brothers, folk culture",
        "Hunger, hardship and popular revolt — 1830s economic crisis, weavers' revolt",
        "The Frankfurt Parliament 1848 — demands and failure",
        "Role of language and folklore in nationalism (Poland, Greece)",
        "Balkan nationalism and the lead-up to World War I",
        "Frédéric Sorrieu's utopian vision (1848 prints)"
    ],
    "nationalism in india": [
        "Non-Cooperation Movement (1920-22) — causes, spread, withdrawal",
        "Civil Disobedience Movement (1930-34) — Salt March, Dandi, response",
        "Rowlatt Act and Jallianwala Bagh massacre",
        "Khilafat Movement and Hindu-Muslim unity",
        "Differing strands: Swaraj vs separate electorates",
        "Role of Mahatma Gandhi — Satyagraha philosophy",
        "Simon Commission and boycott (1928)",
        "Lahore Congress and Purna Swaraj (1929)",
        "Quit India Movement (1942)",
        "Role of business classes, industrialists in nationalism",
        "Peasant movements — Awadh, Bardoli, Champaran",
        "Cultural processes — Bharat Mata, Bankim Chandra, Tagore",
        "Dalit politics and Ambedkar's role",
        "Muslim League, Two-Nation Theory, and partition",
        "Limits of civil disobedience — participation by caste/class/gender"
    ],
}

CBSE_QUALITY_DIRECTIVES_HUMANITIES = """═══ CBSE BOARD PATTERN — HISTORY / HUMANITIES ═══

❌ FORBIDDEN — DO NOT generate:
  • "Who was..." ending with just a name
  • "In which year did X happen?" — isolated date recall
  • "Define absolutism/utopian/nation-state" — bare definitions
  • Random, out-of-context questions not in NCERT
  • Questions testing the same sub-topic twice

✅ REQUIRED — Generate CBSE board-style questions:
  • "Explain the significance of..." (multiple points, cause-effect)
  • "Analyze the role of X in..." (critical thinking)
  • "How did X contribute to Y? Explain with examples."
  • "Describe any three features/measures/reforms of..."
  • "Compare and contrast..." (German vs Italian unification)
  • ASSERTION-REASON: Test conceptual understanding, not trivia
  • SOURCE-BASED / CASE-STUDY: Excerpt + 2-3 sub-questions

═══ TOPIC DIVERSITY — STRICTLY ENFORCED ═══
Each batch MUST cover DIFFERENT sub-topics. No two questions should test
the same sub-topic. Use this list:

TOPICS TO COVER (pick unique ones):
{subtopic_list}

Do NOT generate two questions on the same bullet point above.
"""


# ═══════════════════════════════════════════════════════════════════════════
# v18 FIX: Robust subtopic matching with normalization + word-overlap
# ═══════════════════════════════════════════════════════════════════════════
def _get_subtopic_context(chapter_name: str) -> Optional[str]:
    """Return CBSE sub-topics list with robust chapter name matching."""
    if not chapter_name:
        return None
    
    # Aggressive normalization: lowercase, strip punctuation, collapse spaces
    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = re.sub(r"[\u2013\u2014\-:,;()\[\]\/&]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    
    stopwords = {"the", "of", "in", "a", "an", "and", "to", "for"}
    ch_norm = _norm(chapter_name)
    ch_words = set(ch_norm.split()) - stopwords
    
    best_match = None
    best_score = 0
    
    for key, topics in CBSE_HISTORY_SUBTOPICS.items():
        key_norm = _norm(key)
        # Strategy 1: exact normalized match
        if key_norm == ch_norm:
            return "\n".join(f"  • {t}" for t in topics)
        # Strategy 2: substring match (either direction)
        if key_norm in ch_norm or ch_norm in key_norm:
            return "\n".join(f"  • {t}" for t in topics)
        # Strategy 3: word-overlap (>= 2 significant words)
        key_words = set(key_norm.split()) - stopwords
        overlap = len(ch_words & key_words)
        if overlap >= 2 and overlap > best_score:
            best_score = overlap
            best_match = topics
    
    if best_match:
        return "\n".join(f"  • {t}" for t in best_match)
    return None


MODIFIER_LETTERS = {
    '\u1D57': 't', '\u02B0': 'h', '\u02E2': 's', '\u1D48': 'd',
    '\u02B3': 'r', '\u02E1': 'l', '\u1D43': 'a', '\u1D49': 'e',
    '\u1D52': 'o',
}


def _fix_modifier_letters(text: str) -> str:
    if not text:
        return text
    result = text
    for mod, plain in MODIFIER_LETTERS.items():
        result = result.replace(mod, plain)
    return result


def _build_chapter_prompt(chapter, request, context_chunks, count, section_key=None, section_info=None):
    ch_chunks = [c for c in context_chunks if _ch_match(c.get("chapter", ""), chapter.chapter)]
    if not ch_chunks:
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
        section_ctx = f"\nThis is for {section_info['title']} ({section_info['subtitle']}).\nEach question: {section_info['marks_per_q']} marks. {section_info.get('instruction', '')}"

    if fmt_val == "short_answer":
        if section_info and section_info.get("marks_per_q", 2) == 2:
            fmt_line = '"options": null. Answer: 30-50 words. Show 2 clear steps.'
        else:
            fmt_line = '"options": null. Answer: 50-80 words. Show 3 clear steps with working.'
    elif fmt_val == "long_answer":
        is_acc = request.subject.lower() in ACCOUNTANCY_SUBJECTS
        if is_acc:
            fmt_line = '"options": null. This is Accountancy — answer MUST include "answer_table" field.'
        else:
            fmt_line = '"options": null. Answer: 100-150 words. Step-by-step solution.'
    elif fmt_val == "assertion_reason":
        fmt_line = '"text": "Assertion (A): ...\\nReason (R): ...". Use 4 standard AR options.'
    elif fmt_val == "case_based":
        fmt_line = '"text": Real-world case (3-4 lines), then 3 sub-parts (i),(ii),(iii). "options": 4 each OR null.'
    else:
        fmt_line = '4 options labeled A) B) C) D). Vary correct answer position. All plausible.'

    bloom_default_val = BLOOM_DEFAULT.get(diff_val, "apply")

    section_field = f', "section": "{section_key}"' if section_key else ''
    is_or_field = ', "is_or": false' if section_key in ("D", "E") else ''

    # Include marking_scheme and sub_parts in template for long_answer and case_based
    if fmt_val == "case_based":
        tmpl = (
            f'{{"questions":[{{"text":"...","format":"case_based","options":null,'
            f'"correct_answer":"(i) ... (ii) ... (iii) ...","explanation":"...",'
            f'"sub_parts":[{{"label":"i","text":"...","marks":1,"answer":"..."}},{{"label":"ii","text":"...","marks":1,"answer":"..."}},{{"label":"iii","text":"...","marks":2,"answer":"..."}}],'
            f'"marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"{bloom_default_val}","chapter":"{chapter.chapter}","topic":"specific topic"{is_or_field}{section_field}}}]}}'
        )
    elif fmt_val in ("short_answer", "long_answer"):
        is_acc_long = (fmt_val == "long_answer" and request.subject.lower() in ACCOUNTANCY_SUBJECTS)
        if is_acc_long:
            tmpl = (
                f'{{"questions":[{{"text":"...","format":"long_answer","options":null,'
                f'"correct_answer":"Summary...","explanation":"...","answer_table":{{"type":"journal_entry","headers":[...],"rows":[[...]],"total_row":null}},'
                f'"marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"{bloom_default_val}","chapter":"{chapter.chapter}","topic":"specific topic"{is_or_field}{section_field}}}]}}'
            )
        elif fmt_val == "long_answer":
            tmpl = (
                f'{{"questions":[{{"text":"...","format":"long_answer","options":null,'
                f'"correct_answer":"Full model answer (150-200 words)...","explanation":"Marking scheme...",'
                f'"marking_scheme":[{{"step":"...","marks":1}},{{"step":"...","marks":2}}],'
                f'"marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"{bloom_default_val}","chapter":"{chapter.chapter}","topic":"specific topic"{is_or_field}{section_field}}}]}}'
            )
        else:
            tmpl = f'{{"questions":[{{"text":"...","format":"{fmt_val}","options":null,"correct_answer":"Step 1:... Step 2:...","explanation":"Marking: Step 1 (1m)... Step 2 (1m)...","marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"{bloom_default_val}","chapter":"{chapter.chapter}","topic":"specific topic"{is_or_field}{section_field}}}]}}'
    else:
        tmpl = f'{{"questions":[{{"text":"...","format":"{fmt_val}","options":["A) ...","B) ...","C) ...","D) ..."],"correct_answer":"B) exact option","explanation":"...","marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"{bloom_default_val}","chapter":"{chapter.chapter}","topic":"specific topic"{is_or_field}{section_field}}}]}}'

    # Pick subject-specific directives
    subtopic_list = _get_subtopic_context(chapter.chapter)
    quality_text = CBSE_QUALITY_DIRECTIVES_HUMANITIES.format(subtopic_list=subtopic_list) if subtopic_list else CBSE_QUALITY_DIRECTIVES

    # Inject per-format pedagogy (the core quality lever)
    pedagogy = PEDAGOGY_DIRECTIVES.get(fmt_val, "")

    # Use subject-specific difficulty instruction if available
    subject_lower = (request.subject or "").lower()
    subj_diff = DIFF_INST_OVERRIDES.get(subject_lower, DIFF_INST)
    diff_instruction = subj_diff.get(diff_val, DIFF_INST.get(diff_val, DIFF_INST["medium"]))

    # Add OR instruction for sections D and E
    or_instruction = ""
    if section_key in ("D", "E"):
        or_instruction = '\nFor the LAST question in this batch, also generate an OR alternative testing a DIFFERENT concept from the same chapter. Mark it with "is_or": true in the JSON.'

    return f"""You are an expert CBSE Class {request.class_grade} {request.subject} paper setter.
{section_ctx}
Chapter: {chapter.chapter}
Difficulty: {diff_instruction}

{quality_text}

{pedagogy}

FORMAT RULES:
{fmt_line}

{MATH_FORMAT_INSTRUCTION}

"format" must be exactly "{fmt_val}". "chapter" must be exactly "{chapter.chapter}".
Each question must test a DIFFERENT sub-topic. Use the topic list above where given.
{or_instruction}

NCERT Reference:
{ctx}

Generate EXACTLY {count} unique questions. Return ONLY valid JSON:
{tmpl}"""


def _build_accountancy_prompt(chapter, request, context_chunks, count, table_format, section_key=None, section_info=None):
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
        section_ctx = f"\nThis is for {section_info['title']} ({section_info['subtitle']}).\nEach question: {section_info['marks_per_q']} marks."

    template_instruction = ACCOUNTANCY_PROMPT_TEMPLATES.get(table_format, "")
    template_instruction = template_instruction.format(class_grade=request.class_grade)
    section_field = f', "section": "{section_key}"' if section_key else ''

    return f"""You are an expert CBSE Class {request.class_grade} Accountancy paper setter.
{section_ctx}
Chapter: {chapter.chapter}
Difficulty: {DIFF_INST.get(diff_val, DIFF_INST["medium"])}

{template_instruction}

RULES:
- Indian number format (1,00,000 not 100,000), Rs. not ₹
- Realistic business scenarios
- Each question tests DIFFERENT concept
- Ordinals as plain text: "15th" not "15ᵗʰ"

NCERT Reference:
{ctx}

Generate EXACTLY {count} unique questions. Return ONLY valid JSON:
{{"questions":[{{"text":"...","format":"{table_format}","options":null,"correct_answer":"...","explanation":"...","answer_table":{{"type":"{table_format}","headers":[...],"rows":[[...]],"total_row":null}},"marks":{chapter.marks_per_question},"difficulty":"{diff_val}","bloom_level":"apply","chapter":"{chapter.chapter}","topic":"specific topic"{section_field}}}]}}"""


ACCOUNTANCY_PROMPT_RULES = """ACCOUNTANCY RULES (CRITICAL):
• Every numerical question MUST include GIVEN DATA — amounts in Rs., dates, ratios.
• Use Rs. (not ₹). Format: Rs. 5,00,000.
• Partnership: capital amounts, profit-sharing ratios, dates.
• Companies: share details (face value, premium, payment schedule).
• 3+ mark questions: scenario with 3-4 numerical data points.
• 6 mark questions: detailed scenarios with balance sheet extracts.
• NEVER vague — always specific numbers, dates, names.
• Use realistic Indian names (Priya Ltd., Raman Enterprises).
• Round figures suitable for manual calculation.
• Ordinals as plain text: "15th" not "15ᵗʰ"."""


def _build_accountancy_cbse_prompt(chapter, request, context_chunks, count, group_info, part_key, generate_or=False):
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
    part_ctx = f"This is for {part_info.get('title', '')} — {part_info.get('subtitle', '')}.\nEach question carries {marks} marks."

    or_instruction = ""
    if generate_or:
        or_instruction = "\nFor EACH question, also generate an OR alternative testing DIFFERENT concept. Mark with \"is_or\": true."

    if fmt_val == "mcq":
        fmt_line = '4 options A) B) C) D). correct_answer = exact full text. Vary correct position.'
    elif fmt_val == "assertion_reason":
        fmt_line = '"text": "Assertion (A): ...\\nReason (R): ...". Use 4 standard AR options.'
    elif marks == 3:
        fmt_line = '"options": null. Answer 50-80 words with journal entries or calculations.'
    elif marks == 4:
        fmt_line = '"options": null. Answer 80-120 words.'
    elif marks == 6:
        fmt_line = '"options": null. Answer 120-200 words with complete entries/ledgers.'
    else:
        fmt_line = '"options": null. Clear, detailed answer.'

    section_tag = f"{part_key}_{marks}m"

    if fmt_val in ("mcq", "assertion_reason"):
        json_template = f'{{"questions":[{{"text":"...","format":"{fmt_val}","options":["A) ...","B) ...","C) ...","D) ..."],"correct_answer":"B) exact option","explanation":"...","marks":{marks},"difficulty":"{diff_val}","bloom_level":"apply","chapter":"{chapter.chapter}","topic":"specific topic","section":"{section_tag}","is_or":false}}]}}'
    else:
        json_template = f'{{"questions":[{{"text":"[question with given data]","format":"{fmt_val}","options":null,"correct_answer":"[answer]","explanation":"[working]","marks":{marks},"difficulty":"{diff_val}","bloom_level":"apply","chapter":"{chapter.chapter}","topic":"specific topic","section":"{section_tag}","is_or":false}}]}}'

    return f"""You are an expert CBSE Class 12 Accountancy paper setter.

{part_ctx}
{or_instruction}

Chapter: {chapter.chapter}
Difficulty: {DIFF_INST.get(diff_val, DIFF_INST["medium"])}

{ACCOUNTANCY_PROMPT_RULES}

FORMAT RULES:
{fmt_line}

{MATH_FORMAT_INSTRUCTION}

"format" must be exactly "{fmt_val}". "chapter" must be exactly "{chapter.chapter}".

NCERT Reference:
{ctx}

Generate EXACTLY {count} {'pairs (main + OR)' if generate_or else 'unique questions'}. Return ONLY valid JSON:
{json_template}"""


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
                        try:
                            obj = json.loads(cleaned)
                            questions.append(obj)
                        except json.JSONDecodeError:
                            fixed = _fix_latex_json_escapes(cleaned)
                            obj = json.loads(fixed)
                            questions.append(obj)
                    except (json.JSONDecodeError, ValueError, NameError):
                        pass
                    start = -1
        i += 1
    return questions


def _extract_json(raw: str) -> dict:
    text = raw.strip().lstrip("\ufeff\u200b")
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()

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
        logger.warning(f"JSON truncated, trying per-question extraction")
        individual_qs = _extract_questions_individually(text)
        if individual_qs:
            logger.info(f"✓ Recovered {len(individual_qs)} questions")
            return {"questions": individual_qs}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fb = text.find("{")
    lb = text.rfind("}")
    if fb == -1:
        raise ValueError(f"No JSON found")

    candidate = text[fb:] if lb <= fb else text[fb:lb + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r",\s*([}\]])", r"\1", candidate)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    escaped_ctrl = _escape_control_chars_in_strings(cleaned)
    try:
        return json.loads(escaped_ctrl)
    except json.JSONDecodeError:
        pass

    fixed = _fix_latex_json_escapes(escaped_ctrl)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    aggressive = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu{])', r'\\\\', fixed)
    try:
        return json.loads(aggressive)
    except json.JSONDecodeError:
        pass

    individual_qs = _extract_questions_individually(text)
    if individual_qs:
        return {"questions": individual_qs}

    raise ValueError(f"Could not parse JSON")


UNICODE_REPLACEMENTS = {
    r'\times': '×', r'\div': '÷', r'\pm': '±', r'\cdot': '·',
    r'\leq': '≤', r'\geq': '≥', r'\neq': '≠', r'\approx': '≈',
    r'\infty': '∞', r'\therefore': '∴', r'\because': '∵',
    r'\cup': '∪', r'\cap': '∩', r'\in': '∈', r'\notin': '∉',
    r'\subset': '⊂', r'\emptyset': '∅', r'\forall': '∀', r'\exists': '∃',
    r'\rightarrow': '->', r'\Rightarrow': '=>', r'\to': '->',
    r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
    r'\theta': 'θ', r'\pi': 'π', r'\sigma': 'σ', r'\phi': 'φ',
    r'\omega': 'ω', r'\lambda': 'λ', r'\mu': 'μ', r'\epsilon': 'ε',
    r'\left': '', r'\right': '',
    r'\bigl': '', r'\bigr': '', r'\Bigl': '', r'\Bigr': '',
    r'\sin': 'sin', r'\cos': 'cos', r'\tan': 'tan',
    r'\log': 'log', r'\ln': 'ln', r'\lim': 'lim',
    r'\sec': 'sec', r'\csc': 'csc', r'\cot': 'cot',
}


def _clean_gemini_text(text: str) -> str:
    if not text:
        return text
    result = text
    result = result.replace('₹', 'Rs.')
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
    lines = result.split('\n')
    result = '\n'.join(re.sub(r' +', ' ', line).strip() for line in lines).strip()
    return result


def _parse_question_table(raw_table: dict) -> Optional[QuestionTable]:
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


# ═══════════════════════════════════════════════════════════════════════════
# POST-GENERATION QUALITY VALIDATORS
# These filter out low-quality questions that passed structural validation
# but fail pedagogical standards.
# ═══════════════════════════════════════════════════════════════════════════

# Trivia stem patterns — questions starting with these are low-quality recall
_TRIVIA_STEMS = [
    r"^what is the definition of\b",
    r"^define\s+\w",
    r"^who (discovered|invented|founded|proposed|formulated)\b",
    r"^in which year\b",
    r"^when (did|was)\b",
    r"^what is the full form of\b",
    r"^\w+ is also known as\b",
    r"^name the\b",
    r"^what is the other name\b",
]
_TRIVIA_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _TRIVIA_STEMS]


def _validate_question_quality(q, fmt, diff_val):
    """Validate a single question's pedagogical quality. Returns (ok, reason)."""
    text = q.text if hasattr(q, 'text') else ""
    text_lower = text.lower().strip()

    # 1. Reject trivia stems for apply+ difficulty
    if diff_val not in ("easy",):
        for pattern in _TRIVIA_PATTERNS:
            if pattern.search(text_lower):
                return False, f"trivia_stem: {text[:50]}"

    # 2. MCQ: Check that options aren't all suspiciously long templated text
    if fmt == "mcq":
        options = getattr(q, 'options', None) or []
        if len(options) == 4:
            cleaned = [re.sub(r'^[A-Fa-f][).\s]+\s*', '', str(o)).strip() for o in options]
            lengths = [len(c) for c in cleaned]
            if max(lengths) > 30 and (max(lengths) - min(lengths)) < 5:
                return False, "mcq_template_distractors"

    # 3. Subjective: Check explanation quality
    if fmt in ("short_answer", "long_answer"):
        explanation = getattr(q, 'explanation', "") or ""
        if len(explanation) < 30:
            return False, f"weak_explanation: len={len(explanation)}"

    # 4. Case-based: check sub_parts in text OR in sub_parts field
    if fmt == "case_based":
        has_in_text = "(i)" in text or "(a)" in text or "(1)" in text
        has_in_field = bool(getattr(q, 'sub_parts', None))
        if not has_in_text and not has_in_field:
            return False, "case_no_subparts"

    return True, ""


def _validate_batch_quality(questions, chapter, request):
    """Batch-level quality checks. Returns filtered list."""
    if not questions:
        return questions

    if hasattr(chapter, 'difficulty'):
        diff_val = chapter.difficulty.value if hasattr(chapter.difficulty, 'value') else str(chapter.difficulty)
    elif isinstance(chapter, dict) and 'difficulty' in chapter:
        diff_obj = chapter['difficulty']
        diff_val = diff_obj.value if hasattr(diff_obj, 'value') else str(diff_obj)
    else:
        diff_val = "medium"

    filtered = []
    rejected = 0

    for q in questions:
        if hasattr(q, 'format'):
            q_fmt = q.format.value if hasattr(q.format, 'value') else str(q.format)
        elif isinstance(q, dict):
            q_fmt = str(q.get('format', 'mcq'))
        else:
            q_fmt = 'mcq'

        ok, reason = _validate_question_quality(q, q_fmt, diff_val)
        if ok:
            filtered.append(q)
        else:
            rejected += 1
            logger.info(f"  Quality reject: {reason}")

    # 2. MCQ answer position distribution check (per-MCQ filter, not first question format)
    def _get_fmt(item):
        if hasattr(item, 'format'):
            return item.format.value if hasattr(item.format, 'value') else str(item.format)
        elif isinstance(item, dict):
            return str(item.get('format', ''))
        return ''

    def _get_ans(item):
        if hasattr(item, 'correct_answer'):
            return getattr(item, 'correct_answer', '') or ''
        elif isinstance(item, dict):
            return item.get('correct_answer') or ''
        return ''

    def _get_top(item):
        if hasattr(item, 'topic'):
            return getattr(item, 'topic', '') or ''
        elif isinstance(item, dict):
            return item.get('topic') or ''
        return ''

    mcq_qs = [q for q in filtered if _get_fmt(q) == "mcq"]
    if len(mcq_qs) >= 4:
        positions = {"A": 0, "B": 0, "C": 0, "D": 0}
        for q in mcq_qs:
            correct = _get_ans(q)[:1].upper()
            if correct in positions:
                positions[correct] += 1
        total = sum(positions.values())
        if total > 0:
            for pos, count in positions.items():
                if count / total > 0.6:
                    logger.warning(f"  Answer skew: {pos}={count}/{total} ({count/total:.0%})")

    # 3. Topic diversity check (log warning)
    if len(filtered) >= 3:
        topics = [_get_top(q) for q in filtered]
        topic_set = set(t.lower().strip() for t in topics if t)
        if len(topic_set) > 0 and len(topic_set) < len(filtered) // 2:
            logger.warning(f"  Low topic diversity: {len(topic_set)} unique topics for {len(filtered)} questions")

    if rejected:
        logger.info(f"  Quality filter: kept {len(filtered)}/{len(filtered)+rejected}")

    return filtered


def _parse_batch(raw, chapter, request, section_key=None):
    try:
        data = _extract_json(raw)
    except (ValueError, AttributeError) as e:
        logger.error(f"Parse failed ({chapter.chapter}): {e}")
        return []

    if isinstance(data, dict):
        raw_qs = data.get("questions", [])
    elif isinstance(data, list):
        raw_qs = data
    else:
        return []

    if not isinstance(raw_qs, list):
        return []

    questions = []
    seen = set()
    dropped = 0
    dropped_table_missing = 0
    diff_val = chapter.difficulty.value if hasattr(chapter.difficulty, 'value') else str(chapter.difficulty)
    is_stats = _is_statistics_question(chapter.chapter, getattr(chapter, 'topic', None))
    is_english_writing = (request.subject or "").lower() == "english" and "writing" in chapter.chapter.lower()
    is_english_grammar = (request.subject or "").lower() == "english" and "grammar" in chapter.chapter.lower()
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

        # For writing/grammar questions, be more lenient on explanation length
        min_explanation = 10
        if is_english_writing or is_english_grammar:
            min_explanation = 5
        if not explanation or len(explanation) < min_explanation:
            dropped += 1
            continue

        if isinstance(options, list):
            options = [_clean_gemini_text(o) for o in options]

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
                    answer_table = AnswerTable(type=table_type, headers=[str(h) for h in headers], rows=clean_rows, total_row=clean_total)
            except Exception as e:
                logger.warning(f"  answer_table parse failed: {e}")

        question_table = _parse_question_table(q.get("question_table"))

        # Statistics: require question_table
        if is_stats:
            if question_table is not None:
                if _has_inline_data_leak(text):
                    cleaned_text = _strip_inline_data_from_text(text, {"headers": question_table.headers, "rows": question_table.rows})
                    if cleaned_text and cleaned_text != text:
                        text = cleaned_text
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
                    except Exception:
                        question_table = None
                if question_table is None:
                    dropped_table_missing += 1
                    dropped += 1
                    continue
        elif not is_english_writing and not is_english_grammar:
            if _references_table(text):
                has_inline = _has_inline_table(text)
                if not has_inline and question_table is None:
                    dropped_table_missing += 1
                    dropped += 1
                    continue

        # ── Format-specific validation ──
        if fmt == "assertion_reason":
            if not options or len(options) != 4:
                options = ASSERTION_REASON_OPTIONS.copy()
            if not correct or correct not in options:
                matched = [o for o in options if len(correct) >= 2 and o[:2].upper() == correct[:2].upper()]
                correct = matched[0] if matched else options[0]
        elif fmt in ("short_answer", "long_answer"):
            options = None
            # Writing + Grammar: be lenient on correct_answer length
            min_correct_len = 10
            if is_english_writing or is_english_grammar:
                min_correct_len = 3  # "goes", "an", "was" etc.
            if not correct or len(correct) < min_correct_len:
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

        # Marks: enforce section marks in CBSE, trust Gemini in Simple mode (with bounds check)
        if section_key:
            marks = chapter.marks_per_question
        else:
            raw_marks = q.get("marks", chapter.marks_per_question)
            if isinstance(raw_marks, (int, float)) and 1 <= raw_marks <= 10:
                marks = int(raw_marks)
            else:
                marks = chapter.marks_per_question

        q_section = q.get("section") or section_key
        is_or = q.get("is_or", False)

        # Parse rich answer fields (new in v19)
        raw_marking_scheme = q.get("marking_scheme")
        if isinstance(raw_marking_scheme, list):
            marking_scheme = []
            for ms in raw_marking_scheme:
                if isinstance(ms, dict) and "step" in ms:
                    try:
                        raw_step_marks = ms.get("marks", 1)
                        step_marks = int(raw_step_marks) if isinstance(raw_step_marks, (int, float)) else 1
                        marking_scheme.append(MarkingStep(
                            step=str(ms.get("step", "")),
                            marks=step_marks,
                            description=str(ms.get("description", "")) if ms.get("description") else None,
                        ))
                    except Exception:
                        pass
            if not marking_scheme:
                marking_scheme = None
        else:
            marking_scheme = None

        raw_sub_parts = q.get("sub_parts")
        if isinstance(raw_sub_parts, list):
            sub_parts = []
            for sp in raw_sub_parts:
                if isinstance(sp, dict) and "text" in sp:
                    try:
                        raw_sp_marks = sp.get("marks", 1)
                        sp_marks = int(raw_sp_marks) if isinstance(raw_sp_marks, (int, float)) else 1
                        sub_parts.append(SubPart(
                            label=str(sp.get("label", "")),
                            text=str(sp.get("text", "")),
                            marks=sp_marks,
                            answer=str(sp.get("answer", "")) if sp.get("answer") else None,
                            correct_answer=str(sp.get("correct_answer", "")) if sp.get("correct_answer") else None,
                            options=sp.get("options") if isinstance(sp.get("options"), list) else None,
                        ))
                    except Exception:
                        pass
            if not sub_parts:
                sub_parts = None
        else:
            sub_parts = None

        common_mistakes = q.get("common_mistakes")
        if not isinstance(common_mistakes, list):
            common_mistakes = None

        model_answer = q.get("model_answer")
        if model_answer and not isinstance(model_answer, str):
            model_answer = None

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
                question_table=question_table,
                marking_scheme=marking_scheme,
                sub_parts=sub_parts,
                common_mistakes=common_mistakes,
                model_answer=model_answer,
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
        extra = f" (incl. {dropped_table_missing} missing table)" if dropped_table_missing else ""
        logger.info(f"  {chapter.chapter}: dropped {dropped}{extra}, kept {len(questions)}")

    # Post-parse quality validation
    questions = _validate_batch_quality(questions, chapter, request)
    return questions

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
                raise GenerationError("Empty response from model", 502)
            logger.info(f"[{model}] {time.time() - t0:.1f}s ({len(raw)} chars)")
            return raw
        except GenerationError:
            raise
        except Exception as e:
            last_exc = e
            err = str(e)
            if not _is_retryable(err):
                # Fail fast with actual reason
                raise GenerationError(f"{model}: {err[:200]}", 502)
            if attempt < MAX_RETRIES - 1:
                wait = min(BASE_BACKOFF_SECONDS ** (attempt + 1), MAX_BACKOFF_SECONDS) * random.uniform(*JITTER_RANGE)
                logger.warning(f"[{model}] Retry {attempt + 1}: {err[:80]}... waiting {wait:.1f}s")
                time.sleep(wait)
    raise GenerationError(f"{model}: failed after {MAX_RETRIES} attempts: {str(last_exc)[:150]}", 503)


def _generate_for_chapter(client, chapter, request, context_chunks, models, section_key=None, section_info=None, errors=None):
    target = chapter.quantity
    ask = target + settings.OVERSHOOT_PER_CHAPTER
    batch_size = settings.BATCH_SIZE
    is_accountancy = request.subject.lower() in ACCOUNTANCY_SUBJECTS
    is_stats = _is_statistics_question(chapter.chapter, getattr(chapter, 'topic', None))
    is_english = (request.subject or "").lower() == "english"
    is_writing = is_english and "writing" in chapter.chapter.lower()
    is_grammar = is_english and "grammar" in chapter.chapter.lower()
    fmt_val = chapter.format.value if hasattr(chapter.format, 'value') else str(chapter.format)

    routing_tag = ""
    if is_accountancy and fmt_val in ACCOUNTANCY_TABLE_FORMATS:
        routing_tag = " [ACCOUNTANCY]"
    elif is_stats:
        routing_tag = " [STATISTICS]"
    elif is_writing:
        routing_tag = " [ENG-WRITING]"
    elif is_grammar:
        routing_tag = " [ENG-GRAMMAR]"

    logger.info(f"  '{chapter.chapter}': target={target}, fmt={fmt_val}, diff={chapter.difficulty}, marks={chapter.marks_per_question}" + (f", section={section_key}" if section_key else "") + routing_tag)

    all_qs = []
    remaining = ask
    batch_num = 0

    while remaining > 0 and len(all_qs) < target:
        bc = min(remaining, batch_size)
        batch_num += 1

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
        elif is_writing:
            prompt = _build_english_writing_prompt(
                chapter, request, bc, section_key, section_info
            )
        elif is_grammar:
            prompt = _build_english_grammar_prompt(
                chapter, request, bc, section_key, section_info
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
                if errors is not None:
                    errors.append(f"{chapter.chapter}/{section_key or 'simple'}: {str(e)}")
                if m != models[-1]:
                    continue
                logger.error(f"    '{chapter.chapter}' batch {batch_num} failed: {e}")
                break

        all_qs.extend(batch_qs)
        produced = len(batch_qs)
        remaining -= bc

        # If quality filter or parser rejected questions, retry for shortfall
        if produced < bc and len(all_qs) < target and batch_num < 4:
            shortfall = bc - produced
            remaining += shortfall

        if remaining > 0 and len(all_qs) < target:
            time.sleep(settings.BATCH_DELAY)

    if len(all_qs) > target:
        all_qs = all_qs[:target]
    logger.info(f"  '{chapter.chapter}': {len(all_qs)}/{target}")
    return all_qs


def _distribute_chapters_to_sections(chapters):
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
        raise GenerationError("No NCERT content found.", 404)

    client = _get_gemini_client()
    model = settings.GEMINI_GEN_MODEL
    models = [model]
    fallback = getattr(settings, 'GEMINI_FALLBACK_MODEL', None)
    if fallback and fallback != model:
        models.append(fallback)

    distribution = _distribute_chapters_to_sections(request.chapters)
    total_expected = sum(sec["count"] for sec in CBSE_SECTIONS.values())
    logger.info(f"CBSE Paper: {len(request.chapters)} chapters, {total_expected} questions")

    all_questions = []
    errors = []
    t0 = time.time()

    for sec_key, sec_info in CBSE_SECTIONS.items():
        sec_chapters = distribution.get(sec_key, [])
        logger.info(f"\n{sec_info['title']}: {sec_info['count']} × {sec_info['marks_per_q']}m")

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
                    mcq_chapter = ChapterSection(chapter=ch_name, difficulty=DifficultyLevel(sec_info["difficulty"]), format=QuestionFormat("mcq"), marks_per_question=sec_info["marks_per_q"], quantity=ch_mcq, topic=getattr(orig_ch, 'topic', None))
                    qs = _generate_for_chapter(client, mcq_chapter, request, context_chunks, models, sec_key, sec_info, errors=errors)
                    all_questions.extend(qs)
                    time.sleep(settings.BATCH_DELAY)

                if ch_ar > 0:
                    ar_chapter = ChapterSection(chapter=ch_name, difficulty=DifficultyLevel(sec_info["difficulty"]), format=QuestionFormat("assertion_reason"), marks_per_question=sec_info["marks_per_q"], quantity=ch_ar, topic=getattr(orig_ch, 'topic', None))
                    qs = _generate_for_chapter(client, ar_chapter, request, context_chunks, models, sec_key, sec_info, errors=errors)
                    all_questions.extend(qs)
                    time.sleep(settings.BATCH_DELAY)
            elif sec_key in ("D", "E"):
                fmt = formats[0]
                sec_chapter = ChapterSection(
                    chapter=ch_name,
                    difficulty=DifficultyLevel(sec_info["difficulty"]),
                    format=QuestionFormat(fmt),
                    marks_per_question=sec_info["marks_per_q"],
                    quantity=count + 1,  # +1 for OR alternative
                    topic=getattr(orig_ch, 'topic', None)
                )
                qs = _generate_for_chapter(client, sec_chapter, request, context_chunks, models, sec_key, sec_info, errors=errors)
                main_qs = [q for q in qs if not getattr(q, '_is_or', False)]
                or_qs = [q for q in qs if getattr(q, '_is_or', False)]
                all_questions.extend(main_qs[:count])
                if or_qs:
                    all_questions.append(or_qs[0])
                elif len(main_qs) > count:
                    extra = main_qs[count]
                    extra._is_or = True
                    all_questions.append(extra)
                time.sleep(settings.BATCH_DELAY)
            else:
                fmt = formats[0]
                sec_chapter = ChapterSection(chapter=ch_name, difficulty=DifficultyLevel(sec_info["difficulty"]), format=QuestionFormat(fmt), marks_per_question=sec_info["marks_per_q"], quantity=count, topic=getattr(orig_ch, 'topic', None))
                qs = _generate_for_chapter(client, sec_chapter, request, context_chunks, models, sec_key, sec_info, errors=errors)
                all_questions.extend(qs)
                time.sleep(settings.BATCH_DELAY)

    elapsed = time.time() - t0
    logger.info(f"CBSE Paper Done: {len(all_questions)}/{total_expected} in {elapsed:.1f}s")
    if not all_questions:
        detail = errors[0] if errors else "unknown cause"
        raise GenerationError(f"CBSE paper generation failed — {detail}", 500)
    return all_questions


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
                    group_dist.append({"chapter": ch.chapter, "count": count, "or_count": min(ch_or, count)})
            part_dist[group_id] = group_dist
        distribution[part_key] = part_dist
    return distribution


def generate_cbse_accountancy_paper(request, context_chunks, feedback=None):
    if not context_chunks:
        raise GenerationError("No NCERT content for Accountancy.", 404)

    client = _get_gemini_client()
    model = settings.GEMINI_GEN_MODEL
    models = [model]
    fallback = getattr(settings, 'GEMINI_FALLBACK_MODEL', None)
    if fallback and fallback != model:
        models.append(fallback)

    pattern = CBSE_ACCOUNTANCY_PATTERN
    distribution = _distribute_accountancy_chapters(request.chapters, pattern)
    logger.info(f"CBSE Accountancy: {len(request.chapters)} chapters, target={pattern['total_questions']}")

    all_questions = []
    errors = []
    t0 = time.time()

    for part_key, part_info in pattern["parts"].items():
        part_groups = distribution.get(part_key, {})
        if not part_groups:
            continue
        logger.info(f"\n{part_info['title']}: {part_info['subtitle']}")

        for group in part_info["groups"]:
            group_id = group["id"]
            group_chapters = part_groups.get(group_id, [])
            if not group_chapters:
                continue
            marks = group["marks_per_q"]
            formats = group["formats"]
            difficulty = group["difficulty"]
            logger.info(f"  Group {group_id}: {group['count']} × {marks}m")

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
                        mcq_chapter = ChapterSection(chapter=ch_name, difficulty=DifficultyLevel(difficulty), format=QuestionFormat("mcq"), marks_per_question=marks, quantity=ch_mcq + min(or_count, ch_mcq))
                        prompt = _build_accountancy_cbse_prompt(mcq_chapter, request, context_chunks, ch_mcq + min(or_count, ch_mcq), group, part_key, generate_or=(or_count > 0))
                        for m in models:
                            try:
                                raw = _call_gemini(client, prompt, m)
                                batch_qs = _parse_batch(raw, mcq_chapter, request, f"{part_key}_{marks}m")
                                if batch_qs:
                                    main_qs = [q for q in batch_qs if not getattr(q, '_is_or', False)]
                                    or_qs = [q for q in batch_qs if getattr(q, '_is_or', False)]
                                    all_questions.extend(main_qs[:ch_mcq])
                                    all_questions.extend(or_qs[:min(or_count, ch_mcq)])
                                    break
                            except GenerationError as e:
                                errors.append(f"{ch_name}/A1: {str(e)}")
                                if m != models[-1]:
                                    continue
                        time.sleep(settings.BATCH_DELAY)

                    if ch_ar > 0:
                        ar_or = max(0, or_count - min(or_count, ch_mcq))
                        ar_chapter = ChapterSection(chapter=ch_name, difficulty=DifficultyLevel(difficulty), format=QuestionFormat("assertion_reason"), marks_per_question=marks, quantity=ch_ar + ar_or)
                        prompt = _build_accountancy_cbse_prompt(ar_chapter, request, context_chunks, ch_ar + ar_or, group, part_key, generate_or=(ar_or > 0))
                        for m in models:
                            try:
                                raw = _call_gemini(client, prompt, m)
                                batch_qs = _parse_batch(raw, ar_chapter, request, f"{part_key}_{marks}m")
                                if batch_qs:
                                    main_qs = [q for q in batch_qs if not getattr(q, '_is_or', False)]
                                    or_qs = [q for q in batch_qs if getattr(q, '_is_or', False)]
                                    all_questions.extend(main_qs[:ch_ar])
                                    all_questions.extend(or_qs[:ar_or])
                                    break
                            except GenerationError as e:
                                errors.append(f"{ch_name}/AR: {str(e)}")
                                if m != models[-1]:
                                    continue
                        time.sleep(settings.BATCH_DELAY)
                else:
                    fmt = formats[0]
                    if marks >= 4 and "journal_entry" in formats:
                        fmt = "journal_entry"
                    elif marks >= 6 and "long_answer" in formats:
                        fmt = "long_answer"
                    sec_chapter = ChapterSection(chapter=ch_name, difficulty=DifficultyLevel(difficulty), format=QuestionFormat(fmt), marks_per_question=marks, quantity=count + or_count)
                    prompt = _build_accountancy_cbse_prompt(sec_chapter, request, context_chunks, count + or_count, group, part_key, generate_or=(or_count > 0))
                    for m in models:
                        try:
                            raw = _call_gemini(client, prompt, m)
                            batch_qs = _parse_batch(raw, sec_chapter, request, f"{part_key}_{marks}m")
                            if batch_qs:
                                main_qs = [q for q in batch_qs if not getattr(q, '_is_or', False)]
                                or_qs = [q for q in batch_qs if getattr(q, '_is_or', False)]
                                all_questions.extend(main_qs[:count])
                                all_questions.extend(or_qs[:or_count])
                                break
                        except GenerationError as e:
                            errors.append(f"{ch_name}/{fmt}: {str(e)}")
                            if m != models[-1]:
                                continue
                    time.sleep(settings.BATCH_DELAY)

    elapsed = time.time() - t0
    main_count = len([q for q in all_questions if not getattr(q, '_is_or', False)])
    logger.info(f"\nCBSE Accountancy: {main_count} main + {len(all_questions) - main_count} OR = {len(all_questions)} in {elapsed:.1f}s")
    if not all_questions:
        detail = errors[0] if errors else "unknown cause"
        raise GenerationError(f"All Accountancy generation failed — {detail}", 500)
    return all_questions


def generate_questions(request, context_chunks, feedback=None, cbse_pattern: bool = False):
    subject_lower = (request.subject or "").lower()
    is_accountancy = subject_lower in ACCOUNTANCY_SUBJECTS

    if cbse_pattern and is_accountancy:
        return generate_cbse_accountancy_paper(request, context_chunks, feedback)
    if cbse_pattern:
        return generate_cbse_paper(request, context_chunks, feedback)

    if not context_chunks:
        raise GenerationError("No NCERT content found.", 404)

    client = _get_gemini_client()
    model = settings.GEMINI_GEN_MODEL
    models = [model]
    fallback = getattr(settings, 'GEMINI_FALLBACK_MODEL', None)
    if fallback and fallback != model:
        models.append(fallback)

    valid_chapters = [c for c in request.chapters if c.quantity > 0]
    total = sum(s.quantity for s in valid_chapters)
    logger.info(f"Generation: {len(valid_chapters)} chapters, {total} questions")

    all_questions = []
    errors = []
    t0 = time.time()

    if len(valid_chapters) <= 1:
        for ch_idx, chapter in enumerate(valid_chapters, 1):
            logger.info(f"Chapter {ch_idx}/{len(valid_chapters)}: {chapter.chapter}")
            ch_qs = _generate_for_chapter(client, chapter, request, context_chunks, models, errors=errors)
            all_questions.extend(ch_qs)
    else:
        max_workers = min(len(valid_chapters), 4)
        logger.info(f"Generating {len(valid_chapters)} chapters in parallel with {max_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(_generate_for_chapter, client, chapter, request, context_chunks, models, errors=errors): idx
                for idx, chapter in enumerate(valid_chapters)
            }
            results = [None] * len(valid_chapters)
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    ch_qs = future.result()
                    results[idx] = ch_qs
                except Exception as exc:
                    logger.error(f"Parallel chapter generation error: {exc}")
                    errors.append(str(exc))
                    results[idx] = []
            for qs in results:
                if qs:
                    all_questions.extend(qs)

    elapsed = time.time() - t0
    logger.info(f"Done: {len(all_questions)}/{total} in {elapsed:.1f}s")
    if not all_questions:
        detail = errors[0] if errors else "unknown cause"
        raise GenerationError(f"All chapters failed — {detail}", 500)
    return all_questions


generate_test = generate_questions


def handle_feedback(*args, **kwargs):
    return None


    