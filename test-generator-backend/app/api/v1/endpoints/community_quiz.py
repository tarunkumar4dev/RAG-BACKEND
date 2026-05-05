# app/api/v1/endpoints/community_quiz.py
# ──────────────────────────────────────────────────────────
# Community Quiz API — Video-based quizzes by creators (Aman Bhaiya use case)
#
# POST   /community-quizzes/preview-video    → Teacher pastes URL, sees preview
# POST   /community-quizzes                  → Create quiz (fetch + generate)
# GET    /community-quizzes                  → List teacher's quizzes
# GET    /community-quizzes/q/{slug}         → Public landing page data
# POST   /community-quizzes/q/{slug}/start   → Public: register, get questions
# POST   /community-quizzes/q/{slug}/submit  → Public: submit, get score+rank
# GET    /community-quizzes/{quiz_id}/leaderboard → Teacher: ranked list
# POST   /community-quizzes/test-generate-no-auth  → 🧪 TEMP test (delete in prod)
# ──────────────────────────────────────────────────────────

import hashlib
import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Optional, List

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from app.core.database import get_supabase
from app.services.youtube_transcript_service import (
    fetch_video_data,
    TranscriptFetchError,
)
from app.services.community_quiz_generator import (
    generate_questions_from_transcript,
    QuestionGenerationError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/community-quizzes", tags=["Community Quizzes"])


# ═══════════════════════════════════════════════════════════
# AUTH HELPERS (verifies Supabase JWT from Authorization header)
# ═══════════════════════════════════════════════════════════

def get_optional_user_id(request: Request) -> Optional[str]:
    """Extract user ID from Bearer token. Returns None if not authenticated."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    try:
        token = auth_header.split(" ")[1]
        sb = get_supabase()
        user_response = sb.auth.get_user(token)
        if user_response and user_response.user:
            return user_response.user.id
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
    return None


def get_required_user_id(request: Request) -> str:
    user_id = get_optional_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id


# ═══════════════════════════════════════════════════════════
# PYDANTIC MODELS (kept inline for simplicity)
# ═══════════════════════════════════════════════════════════

class PreviewVideoRequest(BaseModel):
    url: str


class CreateQuizRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    subject: str
    chapter: Optional[str] = None
    class_level: Optional[str] = None

    source_type: str = Field(..., pattern="^(video|ncert|manual|bank)$")
    source_url: Optional[str] = None  # required if source_type='video'

    duration_minutes: int = Field(20, ge=1, le=240)
    duration_window_hours: int = Field(168, ge=1, le=720)

    question_count: int = Field(20, ge=1, le=50)
    difficulty: str = Field("mixed", pattern="^(easy|medium|hard|mixed)$")
    focus: str = Field("mixed", pattern="^(conceptual|factual|mixed)$")

    creator_name: Optional[str] = None
    creator_logo_url: Optional[str] = None
    creator_channel_url: Optional[str] = None

    show_leaderboard_to_participants: bool = False
    show_correct_answers_after_submit: bool = True

    @field_validator("source_url")
    @classmethod
    def _check_url(cls, v, info):
        if info.data.get("source_type") == "video" and not v:
            raise ValueError("source_url is required when source_type='video'")
        return v


class StartAttemptRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    phone: str = Field(..., min_length=7, max_length=20)
    email: Optional[str] = None
    class_level: Optional[str] = None


class SubmitAnswer(BaseModel):
    question_id: str
    selected_option: Optional[int] = Field(None, ge=0, le=3)
    time_taken_ms: Optional[int] = None


class SubmitAttemptRequest(BaseModel):
    attempt_id: str
    answers: List[SubmitAnswer]
    tab_switch_count: int = 0


class TestGenerateRequest(BaseModel):
    """🧪 For temporary no-auth test endpoint only."""
    url: str
    count: int = 5
    difficulty: str = "medium"


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

SLUG_CHARS = "abcdefghijkmnpqrstuvwxyz23456789"  # no ambiguous 0/O/1/l


def _generate_slug(length: int = 7) -> str:
    return "".join(random.choices(SLUG_CHARS, k=length))


def _hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode()).hexdigest()[:32]


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _normalize_phone(phone: str) -> str:
    return "".join(c for c in phone if c.isdigit() or c == "+")


# ═══════════════════════════════════════════════════════════
# 1. POST /preview-video — teacher pastes URL, sees preview
# ═══════════════════════════════════════════════════════════

@router.post("/preview-video")
async def preview_video(payload: PreviewVideoRequest, request: Request):
    """Fetches transcript + metadata so teacher can confirm before generating."""
    teacher_id = get_required_user_id(request)

    try:
        data = fetch_video_data(payload.url)
    except TranscriptFetchError as e:
        raise HTTPException(status_code=400, detail={"code": e.code, "message": e.message})

    return {
        "video_id": data["video_id"],
        "url": data["url"],
        "title": data["title"],
        "channel": data["channel"],
        "thumbnail": data["thumbnail"],
        "language": data["language"],
        "transcript_word_count": data["transcript_word_count"],
        "transcript_preview": data["transcript"][:300] + ("..." if len(data["transcript"]) > 300 else ""),
    }


# ═══════════════════════════════════════════════════════════
# 2. POST / — create quiz (sync, takes 60-90s)
# ═══════════════════════════════════════════════════════════

@router.post("", status_code=201)
async def create_quiz(payload: CreateQuizRequest, request: Request):
    """Full flow: fetch transcript → generate questions → save quiz."""
    teacher_id = get_required_user_id(request)
    sb = get_supabase()
    started_at = datetime.now(timezone.utc)
    transcript_text = None
    source_metadata = {}

    # Step 1: Fetch transcript if video source
    if payload.source_type == "video":
        try:
            video_data = fetch_video_data(payload.source_url)
            transcript_text = video_data["transcript"]
            source_metadata = {
                "video_id": video_data["video_id"],
                "title": video_data["title"],
                "channel": video_data["channel"],
                "thumbnail": video_data["thumbnail"],
                "language": video_data["language"],
                "word_count": video_data["transcript_word_count"],
            }
        except TranscriptFetchError as e:
            raise HTTPException(status_code=400, detail={"code": e.code, "message": e.message})
    else:
        raise HTTPException(
            status_code=501,
            detail={
                "code": "not_implemented",
                "message": "Only video source is implemented in this build. NCERT/manual flows coming soon.",
            },
        )

    # Step 2: Generate questions
    try:
        questions = generate_questions_from_transcript(
            transcript=transcript_text,
            title=source_metadata.get("title", ""),
            channel=source_metadata.get("channel", ""),
            count=payload.question_count,
            difficulty=payload.difficulty,
            focus=payload.focus,
        )
    except QuestionGenerationError as e:
        raise HTTPException(status_code=500, detail={"code": e.code, "message": e.message})

    # Step 3: Generate unique slug
    slug = None
    for _ in range(5):
        candidate = _generate_slug()
        existing = sb.table("community_quizzes").select("id").eq("share_slug", candidate).execute()
        if not existing.data:
            slug = candidate
            break
    if not slug:
        raise HTTPException(status_code=500, detail={"code": "slug_collision", "message": "Could not generate unique slug"})

    # Step 4: Insert quiz
    ends_at = started_at + timedelta(hours=payload.duration_window_hours)
    total_marks = len(questions)

    quiz_insert = sb.table("community_quizzes").insert({
        "teacher_id": teacher_id,
        "title": payload.title,
        "description": payload.description,
        "subject": payload.subject,
        "chapter": payload.chapter,
        "class_level": payload.class_level,
        "share_slug": slug,
        "duration_minutes": payload.duration_minutes,
        "starts_at": started_at.isoformat(),
        "ends_at": ends_at.isoformat(),
        "total_questions": len(questions),
        "total_marks": total_marks,
        "status": "live",
        "source_type": payload.source_type,
        "source_url": payload.source_url,
        "source_transcript": transcript_text,
        "source_metadata": source_metadata,
        "creator_name": payload.creator_name,
        "creator_logo_url": payload.creator_logo_url,
        "creator_channel_url": payload.creator_channel_url,
        "show_leaderboard_to_participants": payload.show_leaderboard_to_participants,
        "show_correct_answers_after_submit": payload.show_correct_answers_after_submit,
    }).execute()

    if not quiz_insert.data:
        raise HTTPException(status_code=500, detail={"code": "db_insert_failed", "message": "Could not save quiz"})

    quiz_id = quiz_insert.data[0]["id"]

    # Step 5: Insert questions
    question_rows = [
        {
            "quiz_id": quiz_id,
            "question_text": q["question_text"],
            "options": q["options"],
            "correct_option": q["correct_option"],
            "explanation": q["explanation"],
            "marks": 1,
            "order_index": i,
            "source": "auto",
        }
        for i, q in enumerate(questions)
    ]
    sb.table("community_quiz_questions").insert(question_rows).execute()

    # Step 6: Log job for analytics
    duration = int((datetime.now(timezone.utc) - started_at).total_seconds())
    try:
        sb.table("quiz_generation_jobs").insert({
            "quiz_id": quiz_id,
            "teacher_id": teacher_id,
            "source_type": payload.source_type,
            "source_url": payload.source_url,
            "status": "completed",
            "duration_seconds": duration,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        logger.warning(f"Could not log generation job: {e}")  # non-fatal

    logger.info(f"Community quiz created: {quiz_id} ({slug}) with {len(questions)} questions in {duration}s")

    return {
        "quiz_id": quiz_id,
        "share_slug": slug,
        "share_link": f"{request.headers.get('origin', 'https://a4ai.in')}/q/{slug}",
        "total_questions": len(questions),
        "ends_at": ends_at.isoformat(),
        "generation_seconds": duration,
    }


# ═══════════════════════════════════════════════════════════
# 3. GET / — list teacher's quizzes
# ═══════════════════════════════════════════════════════════

@router.get("")
async def list_my_quizzes(request: Request):
    teacher_id = get_required_user_id(request)
    sb = get_supabase()

    result = sb.table("community_quizzes").select(
        "id, title, subject, chapter, share_slug, status, "
        "starts_at, ends_at, total_questions, total_attempts, total_completions, "
        "source_type, source_metadata, created_at"
    ).eq("teacher_id", teacher_id).order("created_at", desc=True).execute()

    return {"quizzes": result.data or []}


# ═══════════════════════════════════════════════════════════
# 4. GET /q/{slug} — public landing page data
# ═══════════════════════════════════════════════════════════

@router.get("/q/{slug}")
async def get_public_quiz(slug: str):
    sb = get_supabase()

    result = sb.table("community_quizzes").select(
        "id, title, description, subject, chapter, class_level, "
        "duration_minutes, ends_at, total_questions, total_marks, status, "
        "creator_name, creator_logo_url, creator_channel_url, source_metadata"
    ).eq("share_slug", slug).execute()

    if not result.data:
        raise HTTPException(status_code=404, detail={"code": "quiz_not_found", "message": "Quiz not found"})

    quiz = result.data[0]
    ends_at = datetime.fromisoformat(quiz["ends_at"].replace("Z", "+00:00"))
    if ends_at < datetime.now(timezone.utc) or quiz["status"] != "live":
        raise HTTPException(status_code=410, detail={"code": "quiz_ended", "message": "This quiz has ended"})

    return quiz


# ═══════════════════════════════════════════════════════════
# 5. POST /q/{slug}/start — register participant + return questions
# ═══════════════════════════════════════════════════════════

@router.post("/q/{slug}/start")
async def start_attempt(slug: str, payload: StartAttemptRequest, request: Request):
    sb = get_supabase()

    # Fetch quiz
    quiz_result = sb.table("community_quizzes").select("*").eq("share_slug", slug).execute()
    if not quiz_result.data:
        raise HTTPException(status_code=404, detail={"code": "quiz_not_found", "message": "Quiz not found"})

    quiz = quiz_result.data[0]
    ends_at = datetime.fromisoformat(quiz["ends_at"].replace("Z", "+00:00"))
    if ends_at < datetime.now(timezone.utc) or quiz["status"] != "live":
        raise HTTPException(status_code=410, detail={"code": "quiz_ended", "message": "This quiz has ended"})

    phone = _normalize_phone(payload.phone)
    ip = _get_client_ip(request)

    # Try to insert. UNIQUE(quiz_id, phone) constraint prevents duplicates.
    try:
        attempt_insert = sb.table("community_quiz_attempts").insert({
            "quiz_id": quiz["id"],
            "participant_name": payload.name.strip(),
            "participant_phone": phone,
            "participant_email": payload.email,
            "participant_class": payload.class_level,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "in_progress",
            "ip_hash": _hash_ip(ip),
            "user_agent": request.headers.get("User-Agent", "")[:500],
        }).execute()
    except Exception as e:
        err_str = str(e).lower()
        if "duplicate" in err_str or "23505" in err_str or "unique" in err_str:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "already_attempted",
                    "message": "Is phone number se quiz already attempt ho chuka hai. One attempt per phone allowed.",
                },
            )
        logger.exception("Attempt creation failed")
        raise HTTPException(status_code=500, detail={"code": "attempt_failed", "message": "Could not start attempt"})

    attempt_id = attempt_insert.data[0]["id"]

    # Fetch questions WITHOUT correct_option/explanation
    q_result = sb.table("community_quiz_questions").select(
        "id, question_text, options, marks, order_index"
    ).eq("quiz_id", quiz["id"]).order("order_index").execute()

    # Increment total_attempts (best-effort)
    try:
        sb.table("community_quizzes").update({
            "total_attempts": (quiz.get("total_attempts") or 0) + 1
        }).eq("id", quiz["id"]).execute()
    except Exception:
        pass

    return {
        "attempt_id": attempt_id,
        "quiz": {
            "title": quiz["title"],
            "duration_minutes": quiz["duration_minutes"],
            "total_questions": quiz["total_questions"],
            "total_marks": quiz["total_marks"],
        },
        "questions": q_result.data or [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


# ═══════════════════════════════════════════════════════════
# 6. POST /q/{slug}/submit — server-side scoring
# ═══════════════════════════════════════════════════════════

@router.post("/q/{slug}/submit")
async def submit_attempt(slug: str, payload: SubmitAttemptRequest):
    sb = get_supabase()

    # Fetch attempt
    attempt_result = sb.table("community_quiz_attempts").select("*").eq("id", payload.attempt_id).execute()
    if not attempt_result.data:
        raise HTTPException(status_code=404, detail={"code": "attempt_not_found", "message": "Attempt not found"})

    attempt = attempt_result.data[0]
    if attempt["status"] != "in_progress":
        raise HTTPException(status_code=409, detail={"code": "already_submitted", "message": "Already submitted"})

    # Verify slug matches
    quiz_result = sb.table("community_quizzes").select("*").eq("id", attempt["quiz_id"]).execute()
    if not quiz_result.data or quiz_result.data[0]["share_slug"] != slug:
        raise HTTPException(status_code=400, detail={"code": "quiz_mismatch", "message": "Invalid attempt for this quiz"})

    quiz = quiz_result.data[0]

    # Fetch questions WITH correct answers
    questions_result = sb.table("community_quiz_questions").select(
        "id, correct_option, marks, explanation, order_index, question_text, options"
    ).eq("quiz_id", attempt["quiz_id"]).execute()

    correct_map = {q["id"]: q for q in questions_result.data}

    # Evaluate
    total_score = 0
    correct_count = 0
    attempted_count = 0
    answer_rows = []
    answers_review = []

    for ans in payload.answers:
        q = correct_map.get(ans.question_id)
        if not q:
            continue

        is_correct = False
        if ans.selected_option is not None:
            attempted_count += 1
            is_correct = ans.selected_option == q["correct_option"]
            if is_correct:
                correct_count += 1
                total_score += q["marks"]

        answer_rows.append({
            "attempt_id": payload.attempt_id,
            "question_id": ans.question_id,
            "selected_option": ans.selected_option,
            "is_correct": is_correct,
            "time_taken_ms": ans.time_taken_ms,
        })

        if quiz.get("show_correct_answers_after_submit"):
            answers_review.append({
                "question_id": ans.question_id,
                "question_text": q["question_text"],
                "options": q["options"],
                "selected_option": ans.selected_option,
                "correct_option": q["correct_option"],
                "is_correct": is_correct,
                "explanation": q.get("explanation"),
            })

    # Compute time taken
    started_at = datetime.fromisoformat(attempt["started_at"].replace("Z", "+00:00"))
    submitted_at = datetime.now(timezone.utc)
    time_taken = int((submitted_at - started_at).total_seconds())

    # Save answers
    if answer_rows:
        sb.table("community_quiz_answers").insert(answer_rows).execute()

    # Update attempt
    sb.table("community_quiz_attempts").update({
        "submitted_at": submitted_at.isoformat(),
        "time_taken_seconds": time_taken,
        "total_score": total_score,
        "correct_count": correct_count,
        "attempted_count": attempted_count,
        "status": "submitted",
        "tab_switch_count": payload.tab_switch_count,
    }).eq("id", payload.attempt_id).execute()

    # Increment completions (best-effort)
    try:
        sb.table("community_quizzes").update({
            "total_completions": (quiz.get("total_completions") or 0) + 1
        }).eq("id", quiz["id"]).execute()
    except Exception:
        pass

    # Get rank from leaderboard view
    rank = None
    total_participants = 0
    try:
        rank_result = sb.table("community_quiz_leaderboard").select("rank, attempt_id").eq("quiz_id", attempt["quiz_id"]).execute()
        rank = next((r["rank"] for r in rank_result.data if r["attempt_id"] == payload.attempt_id), None)
        total_participants = len(rank_result.data)
    except Exception as e:
        logger.warning(f"Could not fetch rank: {e}")

    response = {
        "total_score": total_score,
        "total_marks": quiz["total_marks"],
        "correct_count": correct_count,
        "attempted_count": attempted_count,
        "total_questions": quiz["total_questions"],
        "time_taken_seconds": time_taken,
        "rank": rank,
        "total_participants": total_participants,
        "show_leaderboard": quiz.get("show_leaderboard_to_participants", False),
    }

    if quiz.get("show_correct_answers_after_submit"):
        response["answers_review"] = answers_review

    return response


# ═══════════════════════════════════════════════════════════
# 7. GET /{quiz_id}/leaderboard — teacher view
# ═══════════════════════════════════════════════════════════

@router.get("/{quiz_id}/leaderboard")
async def get_leaderboard(quiz_id: str, request: Request):
    teacher_id = get_required_user_id(request)
    sb = get_supabase()

    # Verify ownership
    quiz_result = sb.table("community_quizzes").select(
        "id, teacher_id, title, total_questions, total_marks"
    ).eq("id", quiz_id).execute()

    if not quiz_result.data:
        raise HTTPException(status_code=404, detail={"code": "quiz_not_found", "message": "Quiz not found"})

    quiz = quiz_result.data[0]
    if quiz["teacher_id"] != teacher_id:
        raise HTTPException(status_code=403, detail={"code": "forbidden", "message": "Not your quiz"})

    leaderboard = sb.table("community_quiz_leaderboard").select("*").eq("quiz_id", quiz_id).order("rank").execute()

    return {
        "quiz": quiz,
        "total_participants": len(leaderboard.data or []),
        "leaderboard": leaderboard.data or [],
    }


# ═══════════════════════════════════════════════════════════
# 🧪 8. TEMPORARY TEST ENDPOINT — DELETE BEFORE PRODUCTION
# Tests transcript fetch + question generation without auth
# ═══════════════════════════════════════════════════════════

@router.post("/test-generate-no-auth")
async def test_generate_no_auth(payload: TestGenerateRequest):
    """⚠️ TEMP — no auth, for local testing only. DELETE before deploying."""
    try:
        video_data = fetch_video_data(payload.url)
    except TranscriptFetchError as e:
        raise HTTPException(status_code=400, detail={"code": e.code, "message": e.message})

    try:
        questions = generate_questions_from_transcript(
            transcript=video_data["transcript"],
            title=video_data["title"],
            channel=video_data["channel"],
            count=payload.count,
            difficulty=payload.difficulty,
            focus="mixed",
        )
    except QuestionGenerationError as e:
        raise HTTPException(status_code=500, detail={"code": e.code, "message": e.message})

    return {
        "video": {
            "title": video_data["title"],
            "channel": video_data["channel"],
            "language": video_data["language"],
            "word_count": video_data["transcript_word_count"],
        },
        "questions_generated": len(questions),
        "questions": questions,
    }