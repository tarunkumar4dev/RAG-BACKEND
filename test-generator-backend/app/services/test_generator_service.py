"""
Test Generator Service — SYNC Pipeline (Production)

Pipeline: RAG → Generate → Dedup → Validate → Save Draft → Return
All sync — no async wrappers, no event loop issues.
"""

import uuid
import time
import logging
from typing import List
from datetime import datetime

from app.core.config import settings
from app.core.database import get_supabase
from app.models.test_generator import (
    TestGenerationRequest,
    TestGenerationResponse,
    TestFeedbackRequest,
    GeneratedQuestion,
)
from app.services.rag_service import retrieve_context
from app.services.generation_service import generate_questions, GenerationError
from app.services.deduplication_service import deduplicate_questions
from app.services.validation_service import validate_questions, save_training_data

logger = logging.getLogger(__name__)


def generate_test(request: TestGenerationRequest) -> TestGenerationResponse:
    """Main pipeline — fully synchronous."""
    start = time.time()

    # ── 1. Iteration guard ──────────────────────────────────────────
    if request.iteration >= settings.MAX_ITERATIONS:
        raise ValueError(f"Max regeneration attempts ({settings.MAX_ITERATIONS}) reached.")

    # ── 2. RAG retrieval ────────────────────────────────────────────
    all_chapters = [s.chapter for s in request.chapters]
    all_topics = []
    for s in request.chapters:
        if s.topic:
            all_topics.append(s.topic)
        all_topics.extend(s.subtopics)

    try:
        context_chunks = retrieve_context(
            chapters=all_chapters,
            topics=all_topics,
            subject=request.subject,
            class_grade=request.class_grade,
            max_chunks=settings.MAX_CHUNKS or 25,
        )
    except Exception as e:
        logger.error(f"RAG failed: {e}")
        raise ValueError(f"Failed to retrieve content: {str(e)}")

    if not context_chunks:
        raise ValueError("No NCERT content found for selected chapters/topics.")

    logger.info(f"RAG returned {len(context_chunks)} chunks")

    # ── 3. Generate questions ───────────────────────────────────────
    questions = generate_questions(
        request=request,
        context_chunks=context_chunks,
        feedback=request.teacher_feedback,
    )

    if not questions:
        raise ValueError("Question generation failed. Please try again.")

    logger.info(f"Generated {len(questions)} raw questions")

    # ── 4. Deduplication ────────────────────────────────────────────
    questions, removed_count = deduplicate_questions(
        questions=questions,
        similarity_threshold=settings.DEDUP_THRESHOLD,
    )

    if removed_count > 0:
        logger.info(f"Dedup removed {removed_count} duplicates -> {len(questions)} remaining")

    # ── 5. Trim to requested count ─────────────────────────────────
    total_requested = sum(s.quantity for s in request.chapters)
    if len(questions) > total_requested:
        questions = _select_best_questions(questions, total_requested)
        logger.info(f"Trimmed to {total_requested} best questions")

    # ── 6. Validation ───────────────────────────────────────────────
    try:
        questions = validate_questions(
            questions=questions,
            subject=request.subject,
            class_grade=request.class_grade,
            parallel=True,
        )
    except Exception as e:
        logger.warning(f"Validation error: {e}")
        # Continue with unvalidated questions rather than failing

    # ── 7. Save draft ───────────────────────────────────────────────
    test_id = str(uuid.uuid4())
    try:
        _save_draft(test_id, request, questions)
    except Exception as e:
        logger.warning(f"Draft save failed: {e}")
        # Continue — user still gets their questions

    # ── 8. Return ───────────────────────────────────────────────────
    total_marks = sum(q.marks for q in questions)
    elapsed = round(time.time() - start, 2)

    logger.info(f"Test {test_id[:8]} generated: {len(questions)} questions, {total_marks} marks, {elapsed}s")

    return TestGenerationResponse(
        test_id=test_id,
        exam_title=request.exam_title,
        questions=questions,
        total_marks=total_marks,
        total_questions=len(questions),
        iteration=request.iteration,
        generation_time_seconds=elapsed,
        status="preview",
    )


def _select_best_questions(questions: List[GeneratedQuestion], target: int) -> List[GeneratedQuestion]:
    """Select best quality questions by scoring."""
    if len(questions) <= target:
        return questions

    scored = []
    for q in questions:
        score = 0
        if q.explanation:
            score += min(len(q.explanation) / 50, 5)
        if q.options and len(q.options) == 4:
            score += min(sum(len(o) for o in q.options) / 120, 3)
        if q.text and len(q.text) > 20:
            score += 2
        if q.text and len(q.text) < 15:
            score -= 3
        scored.append((score, q))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [q for _, q in scored[:target]]


def _save_draft(test_id: str, request: TestGenerationRequest, questions: list):
    """Save test draft to Supabase."""
    supabase = get_supabase()

    teacher_id = request.teacher_id
    try:
        uuid.UUID(teacher_id)
    except (ValueError, AttributeError):
        teacher_id = str(uuid.uuid4())
        logger.warning(f"Generated placeholder teacher_id: {teacher_id}")

    supabase.table("tests").insert({
        "id": test_id,
        "teacher_id": teacher_id,
        "exam_title": request.exam_title,
        "board": request.board,
        "class_grade": request.class_grade,
        "subject": request.subject,
        "iteration": request.iteration,
        "status": "draft",
        "request_payload": request.model_dump(),
        "total_questions": len(questions),
        "total_marks": sum(q.marks for q in questions),
        "created_at": datetime.utcnow().isoformat(),
    }).execute()

    question_rows = [
        {
            "id": q.id,
            "test_id": test_id,
            "text": q.text,
            "options": q.options,
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
            "marks": q.marks,
            "difficulty": q.difficulty.value if hasattr(q.difficulty, "value") else q.difficulty,
            "bloom_level": q.bloom_level.value if q.bloom_level and hasattr(q.bloom_level, "value") else q.bloom_level,
            "chapter": q.chapter,
            "topic": q.topic,
            "format": q.format.value if hasattr(q.format, "value") else q.format,
            "validation_status": q.validation_status,
            "position": i,
        }
        for i, q in enumerate(questions)
    ]

    # Insert in batches of 50
    for i in range(0, len(question_rows), 50):
        batch = question_rows[i:i + 50]
        supabase.table("questions").insert(batch).execute()

    logger.info(f"Draft saved: test {test_id[:8]}, {len(questions)} questions")


def handle_feedback(feedback_request: TestFeedbackRequest) -> TestGenerationResponse:
    """Handle teacher feedback — regenerate rejected questions."""
    supabase = get_supabase()

    result = supabase.table("tests").select("*").eq("id", feedback_request.test_id).single().execute()
    if not result.data:
        raise ValueError(f"Test {feedback_request.test_id} not found")

    test_data = result.data
    current_iteration = test_data.get("iteration", 0)

    if current_iteration >= settings.MAX_ITERATIONS:
        raise ValueError(f"Max iterations ({settings.MAX_ITERATIONS}) reached")

    # Save feedback
    rejected_count = 0
    for fb in feedback_request.feedbacks:
        if fb.action in ("reject", "edit"):
            try:
                save_training_data(
                    test_id=feedback_request.test_id,
                    teacher_id=feedback_request.teacher_id,
                    question_id=fb.question_id,
                    question_text="",
                    correct_answer="",
                    teacher_feedback=fb.comment or "",
                    action=fb.action,
                )
            except Exception as e:
                logger.warning(f"Training data save failed: {e}")
            if fb.action == "reject":
                rejected_count += 1

    # Create new request
    original_request = TestGenerationRequest(**test_data["request_payload"])
    original_request.iteration = current_iteration + 1
    original_request.teacher_feedback = feedback_request.global_comment
    original_request.previous_test_id = feedback_request.test_id

    # Adjust quantities
    if rejected_count > 0 and original_request.chapters:
        total = sum(s.quantity for s in original_request.chapters)
        for s in original_request.chapters:
            proportion = s.quantity / total
            s.quantity = max(1, int(s.quantity + rejected_count * proportion))

    return generate_test(original_request)