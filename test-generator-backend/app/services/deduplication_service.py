"""
Deduplication Service — Lightweight (no ML models)

Removes duplicate questions using text similarity (SequenceMatcher).
No numpy, no sentence-transformers, no torch needed.
Vercel-compatible.
"""

import logging
from difflib import SequenceMatcher
from typing import List, Tuple

from app.models.test_generator import GeneratedQuestion

logger = logging.getLogger(__name__)

DEFAULT_SIMILARITY_THRESHOLD = 0.82


def _text_similarity(a: str, b: str) -> float:
    """Compute text similarity using SequenceMatcher (0.0 to 1.0)."""
    if not a or not b:
        return 0.0
    # Normalize
    a = a.strip().lower()
    b = b.strip().lower()
    return SequenceMatcher(None, a, b).ratio()


def _question_quality_score(q: GeneratedQuestion) -> float:
    score = 0.0
    score += min(len(q.text) / 200, 1.0) * 2.0
    if q.explanation and len(q.explanation) > 20:
        score += 2.0
    if q.options and len(q.options) == 4:
        score += 1.0
    if q.correct_answer and len(q.correct_answer) > 2:
        score += 1.0
    if q.bloom_level:
        score += 0.5
    if q.topic:
        score += 0.5
    return score


def deduplicate_questions(
    questions: List[GeneratedQuestion],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Tuple[List[GeneratedQuestion], int]:
    if len(questions) <= 1:
        return questions, 0

    keep = [True] * len(questions)

    for i in range(len(questions)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(questions)):
            if not keep[j]:
                continue

            similarity = _text_similarity(questions[i].text, questions[j].text)
            if similarity >= similarity_threshold:
                score_i = _question_quality_score(questions[i])
                score_j = _question_quality_score(questions[j])

                if score_i >= score_j:
                    keep[j] = False
                    logger.info(f"Dedup: Removed Q{j} (sim={similarity:.2f} with Q{i})")
                else:
                    keep[i] = False
                    logger.info(f"Dedup: Removed Q{i} (sim={similarity:.2f} with Q{j})")
                    break

    deduplicated = [q for q, k in zip(questions, keep) if k]
    removed_count = len(questions) - len(deduplicated)

    if removed_count > 0:
        logger.info(f"Deduplication: {len(questions)} -> {len(deduplicated)} (removed {removed_count})")
    else:
        logger.info("No duplicates found")

    return deduplicated, removed_count