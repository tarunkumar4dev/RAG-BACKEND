# app/services/contest_service.py
# ──────────────────────────────────────────────────────────
# Business logic for Contest CRUD + Submission + Scoring
# Uses Supabase Python client (not asyncpg)
# ──────────────────────────────────────────────────────────

import string
import random
import json
import logging
from typing import Optional, List
from datetime import datetime, timezone

from app.core.database import get_supabase

from app.schemas.contest import (
    CreateContestRequest,
    CreateContestResponse,
    ContestInfoResponse,
    ContestDataResponse,
    ContestQuestionOut,
    ContestQuestionWithAnswer,
    SubmitContestRequest,
    SubmitContestResponse,
    AttemptSummary,
    ContestLeaderboardResponse,
    StartAttemptRequest,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _generate_short_code(length: int = 8) -> str:
    """Generate a random alphanumeric short code for contest URL"""
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choices(chars, k=length))


def _calculate_score(
    answers: List[dict],
    questions: List[dict],
) -> tuple:
    """
    Calculate score from student answers vs correct answers.
    Returns (score, total_marks)
    """
    q_map = {}
    for q in questions:
        q_map[str(q["id"])] = {
            "correct": q.get("correct_answer"),
            "marks": q.get("marks", 1),
            "type": q.get("question_type", "MCQ"),
        }

    score = 0
    total = sum(q.get("marks", 1) for q in questions)

    for ans in answers:
        qid = ans.get("questionId") or ans.get("question_id")
        selected = ans.get("selected") or ans.get("selectedOption")
        if not qid or not selected:
            continue

        q_info = q_map.get(str(qid))
        if not q_info:
            continue

        if q_info["type"] == "MCQ":
            if selected.strip().upper() == (q_info["correct"] or "").strip().upper():
                score += q_info["marks"]

    return score, total


# ═══════════════════════════════════════════════════════════
# CREATE CONTEST
# ═══════════════════════════════════════════════════════════

def create_contest(
    request: CreateContestRequest,
    teacher_id: Optional[str] = None,
) -> CreateContestResponse:
    """Create a new contest with questions and return share link"""

    db = get_supabase()

    short_code = _generate_short_code()

    # Ensure uniqueness (retry if collision)
    for _ in range(5):
        existing = db.table("contests").select("id").eq("short_code", short_code).execute()
        if not existing.data:
            break
        short_code = _generate_short_code()

    total_marks = sum(q.marks for q in request.questions)

    # 1. Insert contest
    contest_data = {
        "teacher_id": teacher_id,
        "title": request.title,
        "subject": request.subject,
        "class_grade": request.class_grade,
        "board": request.board,
        "logo_base64": request.logo_base64,
        "short_code": short_code,
        "duration_minutes": request.duration_minutes,
        "answer_mode": request.answer_mode,
        "show_explanation": request.show_explanation,
        "enable_camera": request.enable_camera,
        "enable_tab_detection": request.enable_tab_detection,
        "allow_back_navigation": request.allow_back_navigation,
        "max_warnings": request.max_warnings,
        "max_attempts": request.max_attempts,
        "scheduled_at": request.scheduled_at.isoformat() if request.scheduled_at else None,
        "expires_at": request.expires_at.isoformat() if request.expires_at else None,
        "total_questions": len(request.questions),
        "total_marks": total_marks,
        "status": "active",
    }

    result = db.table("contests").insert(contest_data).execute()

    if not result.data:
        raise Exception("Failed to insert contest")

    contest_id = result.data[0]["id"]

    # 2. Insert questions
    questions_to_insert = []
    for idx, q in enumerate(request.questions, start=1):
        questions_to_insert.append({
            "contest_id": contest_id,
            "question_number": idx,
            "question_text": q.question_text,
            "question_type": q.question_type,
            "options": q.options if q.options else None,
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
            "marks": q.marks,
            "difficulty": q.difficulty,
            "chapter": q.chapter,
        })

    if questions_to_insert:
        db.table("contest_questions").insert(questions_to_insert).execute()

    share_link = f"https://a4ai.in/contest/{short_code}"

    logger.info(f"Contest created: {contest_id} ({short_code}) with {len(request.questions)} questions")

    return CreateContestResponse(
        contest_id=contest_id,
        short_code=short_code,
        share_link=share_link,
        total_questions=len(request.questions),
        total_marks=total_marks,
    )


# ═══════════════════════════════════════════════════════════
# GET CONTEST INFO (public — for landing/preview page)
# ═══════════════════════════════════════════════════════════

def get_contest_info(short_code: str) -> Optional[ContestInfoResponse]:
    """Get public contest info by short code (student landing page)"""

    db = get_supabase()

    result = db.table("contests").select(
        "id, title, subject, class_grade, board, logo_base64, "
        "duration_minutes, total_questions, total_marks, "
        "enable_camera, enable_tab_detection, allow_back_navigation, "
        "max_warnings, status, scheduled_at, expires_at"
    ).eq("short_code", short_code).execute()

    if not result.data:
        return None

    row = result.data[0]

    # Check expiry
    if row.get("expires_at"):
        expires = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
        if expires < datetime.now(timezone.utc):
            row["status"] = "ended"

    return ContestInfoResponse(
        contest_id=str(row["id"]),
        title=row["title"],
        subject=row["subject"],
        class_grade=row["class_grade"],
        board=row["board"],
        logo_base64=row.get("logo_base64"),
        duration_minutes=row["duration_minutes"],
        total_questions=row["total_questions"],
        total_marks=row["total_marks"],
        enable_camera=row["enable_camera"],
        enable_tab_detection=row["enable_tab_detection"],
        allow_back_navigation=row["allow_back_navigation"],
        max_warnings=row["max_warnings"],
        status=row["status"],
    )


# ═══════════════════════════════════════════════════════════
# START ATTEMPT (student clicks "Start Test")
# ═══════════════════════════════════════════════════════════

def start_attempt(
    short_code: str,
    request: StartAttemptRequest,
    student_id: Optional[str] = None,
) -> Optional[ContestDataResponse]:
    """Start a new attempt: creates attempt record, returns questions (without answers)"""

    db = get_supabase()

    # 1. Get contest
    contest_result = db.table("contests").select("*").eq(
        "short_code", short_code
    ).eq("status", "active").execute()

    if not contest_result.data:
        return None

    contest = contest_result.data[0]

    # 2. Check max attempts
    if student_id:
        attempts_result = db.table("contest_attempts").select(
            "id", count="exact"
        ).eq("contest_id", contest["id"]).eq("student_id", student_id).execute()

        attempt_count = attempts_result.count or 0
        if attempt_count >= contest["max_attempts"]:
            raise ValueError(f"Maximum attempts ({contest['max_attempts']}) reached")

    # 3. Create attempt record
    attempt_data = {
        "contest_id": contest["id"],
        "student_id": student_id,
        "student_name": request.student_name,
        "student_email": request.student_email,
        "status": "in_progress",
    }

    attempt_result = db.table("contest_attempts").insert(attempt_data).execute()

    if not attempt_result.data:
        raise Exception("Failed to create attempt")

    attempt_id = attempt_result.data[0]["id"]

    # 4. Fetch questions (WITHOUT correct_answer and explanation)
    questions_result = db.table("contest_questions").select(
        "id, question_number, question_text, question_type, "
        "options, marks, difficulty, chapter"
    ).eq("contest_id", contest["id"]).order("question_number").execute()

    question_list = []
    for q in questions_result.data:
        opts = q.get("options")
        if isinstance(opts, str):
            opts = json.loads(opts)

        question_list.append(
            ContestQuestionOut(
                id=str(q["id"]),
                question_number=q["question_number"],
                question_text=q["question_text"],
                question_type=q["question_type"],
                options=opts,
                marks=q["marks"],
                difficulty=q["difficulty"],
                chapter=q.get("chapter"),
            )
        )

    # 5. Increment total_attempts
    db.table("contests").update({
        "total_attempts": contest["total_attempts"] + 1
    }).eq("id", contest["id"]).execute()

    return ContestDataResponse(
        contest_id=str(contest["id"]),
        attempt_id=str(attempt_id),
        title=contest["title"],
        subject=contest["subject"],
        class_grade=contest["class_grade"],
        duration_minutes=contest["duration_minutes"],
        enable_camera=contest["enable_camera"],
        enable_tab_detection=contest["enable_tab_detection"],
        allow_back_navigation=contest["allow_back_navigation"],
        max_warnings=contest["max_warnings"],
        answer_mode=contest["answer_mode"],
        show_explanation=contest["show_explanation"],
        questions=question_list,
    )


# ═══════════════════════════════════════════════════════════
# SUBMIT ATTEMPT
# ═══════════════════════════════════════════════════════════

def submit_attempt(
    contest_id: str,
    attempt_id: str,
    request: SubmitContestRequest,
) -> SubmitContestResponse:
    """Submit student answers, calculate score, return results"""

    db = get_supabase()

    # 1. Verify attempt exists and is in_progress
    attempt_result = db.table("contest_attempts").select("*").eq(
        "id", attempt_id
    ).eq("contest_id", contest_id).execute()

    if not attempt_result.data:
        raise ValueError("Attempt not found")

    attempt = attempt_result.data[0]
    if attempt["status"] not in ("in_progress",):
        raise ValueError("Attempt already submitted")

    # 2. Get contest settings
    contest_result = db.table("contests").select("*").eq("id", contest_id).execute()
    if not contest_result.data:
        raise ValueError("Contest not found")

    contest = contest_result.data[0]

    # 3. Get all questions WITH correct answers
    questions_result = db.table("contest_questions").select("*").eq(
        "contest_id", contest_id
    ).order("question_number").execute()

    questions_data = []
    for q in questions_result.data:
        opts = q.get("options")
        if isinstance(opts, str):
            opts = json.loads(opts)

        questions_data.append({
            "id": str(q["id"]),
            "question_number": q["question_number"],
            "question_text": q["question_text"],
            "question_type": q["question_type"],
            "options": opts,
            "correct_answer": q.get("correct_answer"),
            "explanation": q.get("explanation"),
            "marks": q.get("marks", 1),
            "difficulty": q.get("difficulty"),
            "chapter": q.get("chapter"),
        })

    # 4. Calculate score
    score, total_marks = _calculate_score(request.answers, questions_data)
    percentage = round((score / total_marks * 100), 2) if total_marks > 0 else 0

    # Determine status
    status = "submitted"
    if request.warning_count >= contest["max_warnings"]:
        status = "auto_submitted"

    # Count answered
    answered_count = sum(
        1 for a in request.answers
        if a.get("selected") or a.get("selectedOption") or a.get("textAnswer")
    )

    # 5. Update attempt
    update_data = {
        "status": status,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "score": score,
        "total_marks": total_marks,
        "percentage": percentage,
        "time_taken_seconds": request.time_taken_seconds,
        "warning_count": request.warning_count,
        "warning_log": request.warning_log,
        "answers": request.answers,
    }

    db.table("contest_attempts").update(update_data).eq("id", attempt_id).execute()

    logger.info(
        f"Contest {contest_id} attempt {attempt_id} submitted: "
        f"score={score}/{total_marks} ({percentage}%) warnings={request.warning_count}"
    )

    # 6. Build response
    questions_with_answers = None
    if contest["answer_mode"] == "after_test":
        questions_with_answers = [
            ContestQuestionWithAnswer(
                id=q["id"],
                question_number=q["question_number"],
                question_text=q["question_text"],
                question_type=q["question_type"],
                options=q["options"],
                correct_answer=q["correct_answer"],
                explanation=q["explanation"] if contest["show_explanation"] else None,
                marks=q["marks"],
                difficulty=q["difficulty"],
                chapter=q.get("chapter"),
            )
            for q in questions_data
        ]

    return SubmitContestResponse(
        attempt_id=attempt_id,
        status=status,
        score=score,
        total_marks=total_marks,
        percentage=percentage,
        answered_count=answered_count,
        total_questions=len(questions_data),
        questions_with_answers=questions_with_answers,
    )


# ═══════════════════════════════════════════════════════════
# GET LEADERBOARD (for teacher)
# ═══════════════════════════════════════════════════════════

def get_leaderboard(
    contest_id: str,
    teacher_id: Optional[str] = None,
) -> Optional[ContestLeaderboardResponse]:
    """Get all attempts for a contest (teacher view)"""

    db = get_supabase()

    # Verify contest exists
    query = db.table("contests").select("id, title").eq("id", contest_id)
    if teacher_id:
        query = query.eq("teacher_id", teacher_id)

    contest_result = query.execute()
    if not contest_result.data:
        return None

    contest = contest_result.data[0]

    # Get all attempts ordered by percentage desc
    attempts_result = db.table("contest_attempts").select(
        "id, student_name, student_email, status, score, "
        "total_marks, percentage, time_taken_seconds, "
        "warning_count, submitted_at"
    ).eq("contest_id", contest_id).order(
        "percentage", desc=True
    ).execute()

    attempt_list = [
        AttemptSummary(
            attempt_id=str(a["id"]),
            student_name=a.get("student_name"),
            student_email=a.get("student_email"),
            status=a["status"],
            score=a.get("score"),
            total_marks=a.get("total_marks", 0),
            percentage=float(a["percentage"]) if a.get("percentage") else None,
            time_taken_seconds=a.get("time_taken_seconds"),
            warning_count=a.get("warning_count", 0),
            submitted_at=a.get("submitted_at"),
        )
        for a in attempts_result.data
    ]

    return ContestLeaderboardResponse(
        contest_id=contest_id,
        title=contest["title"],
        total_attempts=len(attempt_list),
        attempts=attempt_list,
    )


# ═══════════════════════════════════════════════════════════
# LIST CONTESTS (for teacher dashboard)
# ═══════════════════════════════════════════════════════════

def list_teacher_contests(teacher_id: str) -> List[dict]:
    """List all contests created by a teacher"""

    db = get_supabase()

    result = db.table("contests").select(
        "id, title, subject, class_grade, short_code, "
        "total_questions, total_marks, total_attempts, "
        "duration_minutes, answer_mode, status, "
        "created_at, updated_at"
    ).eq("teacher_id", teacher_id).order("created_at", desc=True).execute()

    return [
        {
            "contest_id": str(r["id"]),
            "title": r["title"],
            "subject": r["subject"],
            "class_grade": r["class_grade"],
            "short_code": r["short_code"],
            "share_link": f"https://a4ai.in/contest/{r['short_code']}",
            "total_questions": r["total_questions"],
            "total_marks": r["total_marks"],
            "total_attempts": r["total_attempts"],
            "duration_minutes": r["duration_minutes"],
            "answer_mode": r["answer_mode"],
            "status": r["status"],
            "created_at": r["created_at"],
        }
        for r in result.data
    ]