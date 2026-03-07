"""
Deduplication Service — Removes semantically similar questions.

Uses the same all-MiniLM-L6-v2 model already loaded for RAG embeddings,
so no additional model loading overhead.

Pipeline:
  1. Encode all question texts → 384-dim embeddings
  2. Compute pairwise cosine similarity
  3. If similarity > threshold → keep the better one (longer explanation, has options)
  4. Return deduplicated list
"""

import logging
import numpy as np
from typing import List, Tuple

from sentence_transformers import SentenceTransformer

from app.models.test_generator import GeneratedQuestion

logger = logging.getLogger(__name__)

# Reuse the same model as RAG service (cached by sentence-transformers)
_model = SentenceTransformer("all-MiniLM-L6-v2")

# Similarity threshold — questions above this are considered duplicates
DEFAULT_SIMILARITY_THRESHOLD = 0.82


def _compute_similarity_matrix(texts: List[str]) -> np.ndarray:
    """Compute pairwise cosine similarity matrix for a list of texts."""
    if not texts:
        return np.array([])

    embeddings = _model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    # Cosine similarity = dot product of normalized vectors
    similarity_matrix = np.dot(embeddings, embeddings.T)
    return similarity_matrix


def _question_quality_score(q: GeneratedQuestion) -> float:
    """
    Score a question's quality for deciding which duplicate to keep.
    Higher = better quality.
    """
    score = 0.0

    # Longer text = more detailed question
    score += min(len(q.text) / 200, 1.0) * 2.0

    # Has explanation
    if q.explanation and len(q.explanation) > 20:
        score += 2.0

    # Has proper options (for MCQ)
    if q.options and len(q.options) == 4:
        score += 1.0

    # Has correct answer
    if q.correct_answer and len(q.correct_answer) > 2:
        score += 1.0

    # Has bloom level
    if q.bloom_level:
        score += 0.5

    # Has topic
    if q.topic:
        score += 0.5

    # Assertion-reason with proper options gets bonus
    if q.format and q.format.value == "assertion_reason" and q.options:
        score += 1.5

    return score


def deduplicate_questions(
    questions: List[GeneratedQuestion],
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Tuple[List[GeneratedQuestion], int]:
    """
    Remove semantically duplicate questions.

    Args:
        questions: List of generated questions
        similarity_threshold: Cosine similarity threshold (default 0.82)

    Returns:
        Tuple of (deduplicated questions, number removed)
    """
    if len(questions) <= 1:
        return questions, 0

    texts = [q.text for q in questions]
    sim_matrix = _compute_similarity_matrix(texts)

    # Track which questions to keep
    keep = [True] * len(questions)
    duplicate_pairs: List[Tuple[int, int, float]] = []

    for i in range(len(questions)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(questions)):
            if not keep[j]:
                continue

            similarity = sim_matrix[i][j]
            if similarity >= similarity_threshold:
                duplicate_pairs.append((i, j, similarity))

                # Keep the higher-quality question
                score_i = _question_quality_score(questions[i])
                score_j = _question_quality_score(questions[j])

                if score_i >= score_j:
                    keep[j] = False
                    logger.info(
                        f"🔁 Dedup: Removed Q{j} (sim={similarity:.2f} with Q{i}) "
                        f"— kept Q{i} (score {score_i:.1f} vs {score_j:.1f})"
                    )
                else:
                    keep[i] = False
                    logger.info(
                        f"🔁 Dedup: Removed Q{i} (sim={similarity:.2f} with Q{j}) "
                        f"— kept Q{j} (score {score_j:.1f} vs {score_i:.1f})"
                    )
                    break  # i is removed, no need to check more pairs for i

    deduplicated = [q for q, k in zip(questions, keep) if k]
    removed_count = len(questions) - len(deduplicated)

    if removed_count > 0:
        logger.info(
            f"🧹 Deduplication: {len(questions)} → {len(deduplicated)} "
            f"(removed {removed_count} duplicates, threshold={similarity_threshold})"
        )
    else:
        logger.info("✅ No duplicates found")

    return deduplicated, removed_count