"""
Test Generator Service — Orchestrates the full pipeline:
Input → RAG → Generate (smart batches) → Deduplicate → Validate → Save

Pipeline:
  1. RAG retrieval (NCERT chunks)
  2. Smart batch generation (15 questions per API call, 20s delays)
  3. Deduplication (semantic similarity removal)
  4. Validation (Gemini Flash)
  5. Save draft + Return preview
"""

import uuid
import time
import logging
from typing import Optional

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
    """
    Full generation pipeline.
    """
    start = time.time()

    # ── 1. Iteration guard ──────────────────────────────────────────────────
    if request.iteration >= settings.MAX_ITERATIONS:
        raise ValueError(
            f"Max regeneration attempts ({settings.MAX_ITERATIONS}) reached. "
            "Please edit questions manually or start a new test."
        )

    # ── 2. RAG retrieval ────────────────────────────────────────────────────
    all_chapters = [s.chapter for s in request.chapters]
    all_topics = [s.topic for s in request.chapters if s.topic]
    all_topics += [st for s in request.chapters for st in s.subtopics]

    context_chunks = retrieve_context(
        chapters=all_chapters,
        topics=all_topics,
        subject=request.subject,
        class_grade=request.class_grade,
        max_chunks=settings.MAX_CHUNKS,
    )

    if not context_chunks:
        raise ValueError("No NCERT content found for the selected chapters/topics.")

    logger.info(f"RAG returned {len(context_chunks)} chunks")

    # ── 3. Generate (smart batches) ─────────────────────────────────────────
    questions = generate_questions(
        request=request,
        context_chunks=context_chunks,
        feedback=request.teacher_feedback,
    )

    if not questions:
        raise ValueError("Question generation failed. Please try again.")

    logger.info(f"Generated {len(questions)} raw questions")

    # ── 4. Deduplicate ──────────────────────────────────────────────────────
    questions, removed_count = deduplicate_questions(
        questions=questions,
        similarity_threshold=0.82,
    )

    if removed_count > 0:
        logger.info(f"Dedup removed {removed_count} duplicates -> {len(questions)} remaining")

    # ── 4b. Trim to requested count ────────────────────────────────────────
    total_requested = sum(s.quantity for s in request.chapters)
    if len(questions) > total_requested:
        logger.info(f"Trimming {len(questions)} -> {total_requested} (requested)")
        questions = questions[:total_requested]

    # ── 5. Validate ─────────────────────────────────────────────────────────
    questions = validate_questions(
        questions=questions,
        subject=request.subject,
        class_grade=request.class_grade,
    )

    # ── 6. Save draft ───────────────────────────────────────────────────────
    test_id = str(uuid.uuid4())
    _save_draft(test_id, request, questions)

    # ── 7. Return preview ───────────────────────────────────────────────────
    total_marks = sum(q.marks for q in questions)
    elapsed = round(time.time() - start, 2)

    logger.info(
        f"Test {test_id[:8]} generated: {len(questions)} questions, "
        f"{total_marks} marks, {elapsed}s"
    )

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


def handle_feedback(feedback_request: TestFeedbackRequest) -> TestGenerationResponse:
    """
    Process teacher feedback and regenerate rejected questions.
    """
    supabase = get_supabase()

    result = (
        supabase.table("tests")
        .select("*")
        .eq("id", feedback_request.test_id)
        .single()
        .execute()
    )
    if not result.data:
        raise ValueError(f"Test {feedback_request.test_id} not found")

    test_data = result.data

    current_iteration = test_data.get("iteration", 0)
    if current_iteration >= settings.MAX_ITERATIONS:
        raise ValueError(
            f"Max iterations ({settings.MAX_ITERATIONS}) reached for this test."
        )

    rejected_question_ids = []
    for fb in feedback_request.feedbacks:
        if fb.action in ("reject", "edit"):
            save_training_data(
                test_id=feedback_request.test_id,
                teacher_id=feedback_request.teacher_id,
                question_id=fb.question_id,
                question_text="",
                correct_answer="",
                teacher_feedback=fb.comment or "",
                action=fb.action,
            )
            if fb.action == "reject":
                rejected_question_ids.append(fb.question_id)

    original_request = TestGenerationRequest(**test_data["request_payload"])
    original_request.iteration = current_iteration + 1
    original_request.teacher_feedback = feedback_request.global_comment
    original_request.previous_test_id = feedback_request.test_id

    return generate_test(original_request)


def _save_draft(
    test_id: str,
    request: TestGenerationRequest,
    questions: list,
):
    """Persist test draft to Supabase."""
    supabase = get_supabase()

    # Validate teacher_id is a proper UUID
    teacher_id = request.teacher_id
    try:
        uuid.UUID(teacher_id)
    except (ValueError, AttributeError):
        logger.warning(f"Invalid teacher_id '{teacher_id}', generating placeholder UUID")
        teacher_id = str(uuid.uuid4())

    try:
        supabase.table("tests").insert(
            {
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
            }
        ).execute()

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
        supabase.table("questions").insert(question_rows).execute()
        logger.info(f"Draft saved: test {test_id[:8]}, {len(questions)} questions")

    except Exception as e:
        logger.error(f"Failed to save draft: {e}")