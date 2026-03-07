"""
Question Planner — Decides WHAT types of questions to generate BEFORE calling LLM.

Instead of letting the LLM decide everything, we enforce a distribution:
  - Recall (definitions, facts)
  - Conceptual (explain why, compare, differentiate)
  - Application (real-life scenarios, diagrams)
  - Numerical (formula-based calculations)
  - Assertion-Reason (CBSE standard format)

This is the single biggest quality lever.
"""

import logging
import math
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distribution Profiles
# ---------------------------------------------------------------------------
# Each profile maps question_type → percentage of total
DISTRIBUTION_PROFILES = {
    "balanced": {
        "recall": 0.20,
        "conceptual": 0.25,
        "application": 0.25,
        "numerical": 0.15,
        "assertion_reason": 0.15,
    },
    "cbse_standard": {
        "recall": 0.15,
        "conceptual": 0.20,
        "application": 0.25,
        "numerical": 0.20,
        "assertion_reason": 0.20,
    },
    "easy_test": {
        "recall": 0.40,
        "conceptual": 0.30,
        "application": 0.15,
        "numerical": 0.10,
        "assertion_reason": 0.05,
    },
    "competitive": {
        "recall": 0.10,
        "conceptual": 0.15,
        "application": 0.30,
        "numerical": 0.25,
        "assertion_reason": 0.20,
    },
}

# Subjects that typically don't have numerical questions
NON_NUMERICAL_SUBJECTS = {"english", "hindi", "history", "geography", "civics", "political science"}


@dataclass
class QuestionBatch:
    """A batch of questions to generate with specific type constraints."""
    question_type: str          # recall, conceptual, application, numerical, assertion_reason
    count: int
    chapter: str
    topic: Optional[str] = None
    subtopics: List[str] = field(default_factory=list)
    difficulty: str = "medium"
    marks_per_question: int = 1
    bloom_levels: List[str] = field(default_factory=list)

    # Type-specific instructions for the LLM
    type_instruction: str = ""
    format_override: Optional[str] = None  # e.g. "assertion_reason" → format


@dataclass
class GenerationPlan:
    """Complete plan for generating a test."""
    batches: List[QuestionBatch]
    total_questions: int
    distribution_profile: str
    distribution_summary: Dict[str, int]


# ---------------------------------------------------------------------------
# Bloom Mapping per Question Type
# ---------------------------------------------------------------------------
TYPE_TO_BLOOM = {
    "recall": ["remember"],
    "conceptual": ["understand", "analyze"],
    "application": ["apply", "analyze"],
    "numerical": ["apply", "evaluate"],
    "assertion_reason": ["analyze", "evaluate"],
}


# ---------------------------------------------------------------------------
# Type-Specific LLM Instructions
# ---------------------------------------------------------------------------
TYPE_INSTRUCTIONS = {
    "recall": (
        "Generate RECALL questions that test direct factual knowledge.\n"
        "These should ask: definitions, identify, name, list, state.\n"
        "Example: 'What is the SI unit of electric current?'\n"
        "Example: 'Name the device used to measure potential difference.'\n"
        "Keep them straightforward but not trivially obvious."
    ),
    "conceptual": (
        "Generate CONCEPTUAL questions that test deep understanding.\n"
        "These should ask: explain why, differentiate between, compare, "
        "what would happen if, reason behind.\n"
        "Example: 'Why does a compass needle deflect when brought near a current-carrying wire?'\n"
        "Example: 'Differentiate between a solenoid and a bar magnet.'\n"
        "NEVER ask simple definitions. Every question must require REASONING."
    ),
    "application": (
        "Generate APPLICATION questions based on real-life scenarios.\n"
        "These should present a situation and ask students to apply concepts.\n"
        "Example: 'A student connects a 100W bulb and a 60W bulb in series to 220V. "
        "Which bulb glows brighter and why?'\n"
        "Example: 'In a house, the fuse keeps blowing when the AC and geyser are turned on together. "
        "Explain why and suggest a solution.'\n"
        "Every question MUST have a real-world context or scenario."
    ),
    "numerical": (
        "Generate NUMERICAL questions that require mathematical calculation.\n"
        "Use formulas from the chapter (V=IR, P=VI, P=I²R, etc.).\n"
        "Provide realistic values. Include step-by-step solution in explanation.\n"
        "Example: 'An electric heater of resistance 44Ω is connected to a 220V supply. "
        "Calculate: (i) the current drawn (ii) the power consumed.'\n"
        "Example: 'A wire of resistance 20Ω is bent to form a closed circle. "
        "What is the effective resistance between two diametrically opposite points?'\n"
        "EVERY question MUST involve calculation. No pure theory questions."
    ),
    "assertion_reason": (
        "Generate ASSERTION-REASON questions in standard CBSE format.\n"
        "MANDATORY FORMAT for each question:\n"
        "- Text must contain: 'Assertion (A): [statement]\\nReason (R): [statement]'\n"
        "- Options MUST be exactly these 4:\n"
        '  A) Both A and R are true and R is the correct explanation of A\n'
        '  B) Both A and R are true but R is NOT the correct explanation of A\n'
        '  C) A is true but R is false\n'
        '  D) A is false but R is true\n'
        "- correct_answer must be one of these options\n"
        "- Vary the correct answers across A, B, C, D — don't make all answers 'A'\n"
        "Example:\n"
        "  Assertion (A): The resistance of a conductor increases with increase in temperature.\n"
        "  Reason (R): On increasing temperature, the average speed of free electrons increases.\n"
        "  Correct: A) Both A and R are true and R is the correct explanation of A"
    ),
}


# ---------------------------------------------------------------------------
# Difficulty Definitions
# ---------------------------------------------------------------------------
DIFFICULTY_INSTRUCTIONS = {
    "easy": (
        "EASY difficulty means:\n"
        "- Direct from textbook, single concept\n"
        "- No multi-step reasoning required\n"
        "- Student who read the chapter once can answer"
    ),
    "medium": (
        "MEDIUM difficulty means:\n"
        "- Requires understanding, not just memorization\n"
        "- May combine 2 concepts from same topic\n"
        "- Student needs to think before answering"
    ),
    "hard": (
        "HARD difficulty means:\n"
        "- Multi-step reasoning or multi-concept integration\n"
        "- Application to unfamiliar scenarios\n"
        "- Requires deep understanding + logical deduction\n"
        "- NO pure definitions or one-line facts allowed at hard level\n"
        "- For MCQs: distractors should be plausible and tricky"
    ),
}


# ---------------------------------------------------------------------------
# Planner Logic
# ---------------------------------------------------------------------------
def _get_distribution(
    difficulty: str,
    subject: str,
    profile: str = "balanced",
) -> Dict[str, float]:
    """Get question type distribution, adjusted for subject and difficulty."""
    base = DISTRIBUTION_PROFILES.get(profile, DISTRIBUTION_PROFILES["balanced"]).copy()

    # Non-numerical subjects: redistribute numerical quota
    if subject.lower() in NON_NUMERICAL_SUBJECTS:
        numerical_share = base.pop("numerical", 0)
        base["conceptual"] = base.get("conceptual", 0) + numerical_share * 0.5
        base["application"] = base.get("application", 0) + numerical_share * 0.5

    # Hard tests: shift toward application + numerical
    if difficulty == "hard":
        base["recall"] = max(0.05, base.get("recall", 0) - 0.10)
        base["application"] = base.get("application", 0) + 0.05
        base["numerical"] = base.get("numerical", 0) + 0.05

    # Easy tests: shift toward recall
    if difficulty == "easy":
        base["recall"] = base.get("recall", 0) + 0.10
        base["application"] = max(0.05, base.get("application", 0) - 0.05)
        base["numerical"] = max(0.05, base.get("numerical", 0) - 0.05)

    # Normalize to sum=1
    total = sum(base.values())
    if total > 0:
        base = {k: v / total for k, v in base.items()}

    return base


def _distribute_count(total: int, distribution: Dict[str, float]) -> Dict[str, int]:
    """Convert percentage distribution into integer counts that sum to total."""
    counts = {}
    remaining = total

    # First pass: floor values
    for qtype, pct in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
        count = math.floor(total * pct)
        counts[qtype] = count
        remaining -= count

    # Second pass: distribute remainder to largest types
    for qtype in sorted(counts, key=lambda x: distribution[x], reverse=True):
        if remaining <= 0:
            break
        counts[qtype] += 1
        remaining -= 1

    # Ensure at least 1 of each type if total allows
    if total >= len(counts):
        for qtype in counts:
            if counts[qtype] == 0:
                # Steal from the largest bucket
                largest = max(counts, key=lambda x: counts[x])
                if counts[largest] > 1:
                    counts[largest] -= 1
                    counts[qtype] = 1

    return {k: v for k, v in counts.items() if v > 0}


def create_generation_plan(
    chapters: list,
    subject: str,
    class_grade: str,
    bloom_enabled: bool = False,
    distribution_profile: str = "balanced",
) -> GenerationPlan:
    """
    Create a structured plan for question generation.

    Args:
        chapters: List of chapter config dicts from the request
                  (each has: chapter, topic, subtopics, quantity, difficulty,
                   format, marks_per_question, bloom_levels)
        subject: Subject name
        class_grade: Class grade
        bloom_enabled: Whether to tag Bloom levels
        distribution_profile: One of the DISTRIBUTION_PROFILES keys

    Returns:
        GenerationPlan with typed batches ready for generation
    """
    all_batches: List[QuestionBatch] = []
    total_questions = 0
    overall_distribution: Dict[str, int] = {}

    for chapter_config in chapters:
        chapter_name = chapter_config.chapter
        quantity = chapter_config.quantity
        difficulty = chapter_config.difficulty
        topic = chapter_config.topic
        subtopics = chapter_config.subtopics or []
        marks = chapter_config.marks_per_question
        user_bloom = chapter_config.bloom_levels or []

        # Check if user explicitly requested a specific format
        user_format = chapter_config.format
        if user_format == "assertion_reason":
            # All questions should be assertion_reason
            batch = QuestionBatch(
                question_type="assertion_reason",
                count=quantity,
                chapter=chapter_name,
                topic=topic,
                subtopics=subtopics,
                difficulty=difficulty,
                marks_per_question=marks,
                bloom_levels=user_bloom or TYPE_TO_BLOOM.get("assertion_reason", []),
                type_instruction=TYPE_INSTRUCTIONS["assertion_reason"],
                format_override="assertion_reason",
            )
            all_batches.append(batch)
            overall_distribution["assertion_reason"] = (
                overall_distribution.get("assertion_reason", 0) + quantity
            )
            total_questions += quantity
            continue

        # For mixed formats: compute distribution
        dist = _get_distribution(difficulty, subject, distribution_profile)
        counts = _distribute_count(quantity, dist)

        for qtype, count in counts.items():
            if count <= 0:
                continue

            # Determine format for this batch
            if qtype == "assertion_reason":
                fmt_override = "assertion_reason"
            elif qtype == "numerical":
                fmt_override = "short_answer" if user_format != "mcq" else "mcq"
            else:
                fmt_override = user_format  # respect user's format choice

            bloom = user_bloom if user_bloom else TYPE_TO_BLOOM.get(qtype, [])

            batch = QuestionBatch(
                question_type=qtype,
                count=count,
                chapter=chapter_name,
                topic=topic,
                subtopics=subtopics,
                difficulty=difficulty,
                marks_per_question=marks,
                bloom_levels=bloom if bloom_enabled else [],
                type_instruction=TYPE_INSTRUCTIONS.get(qtype, ""),
                format_override=fmt_override,
            )
            all_batches.append(batch)
            overall_distribution[qtype] = overall_distribution.get(qtype, 0) + count
            total_questions += count

    plan = GenerationPlan(
        batches=all_batches,
        total_questions=total_questions,
        distribution_profile=distribution_profile,
        distribution_summary=overall_distribution,
    )

    logger.info(
        f"📋 Generation plan: {total_questions} questions | "
        f"Distribution: {overall_distribution} | Profile: {distribution_profile}"
    )

    return plan