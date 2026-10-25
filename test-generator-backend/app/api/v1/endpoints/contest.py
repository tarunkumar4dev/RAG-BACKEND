# app/api/v1/endpoints/contest.py
# ──────────────────────────────────────────────────────────
# Contest API Endpoints
#
# POST   /api/v1/contests              → Create contest (teacher)
# GET    /api/v1/contests/:code/info   → Get contest info (student landing)
# POST   /api/v1/contests/:code/start  → Start attempt (student)
# POST   /api/v1/contests/:id/submit   → Submit answers (student)
# PUT    /api/v1/contests/:id/attempts/:attempt_id/progress → Save progress (student, every ~15 s)
# GET    /api/v1/contests/:id/attempts                       → Every attempt (contest's teacher)
# POST   /api/v1/contests/:id/attempts/:attempt_id/allow-retry → One more attempt after this one (teacher)
# POST   /api/v1/contests/:id/publish-results                → Release results (teacher, answer_mode teacher_release)
# GET    /api/v1/contests/:id/leaderboard → View results (teacher)
# GET    /api/v1/contests/my            → List teacher's contests
# ──────────────────────────────────────────────────────────

import logging
from datetime import timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from app.schemas.contest import (
    CreateContestRequest,
    CreateContestResponse,
    ContestInfoResponse,
    ContestDataResponse,
    SubmitContestRequest,
    SubmitContestResponse,
    StartAttemptRequest,
    ProgressRequest,
    ContestLeaderboardResponse,
)
from app.services import contest_service

# If you have auth dependency, import it:
# from app.core.auth import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/contests", tags=["Contests"])


def _already_submitted(e: contest_service.AlreadySubmitted) -> JSONResponse:
    """409 for a caller who is done; `detail` keeps the old error text for old frontends."""
    return JSONResponse(status_code=409, content={
        "code": "ALREADY_SUBMITTED", "attempt_id": e.attempt_id, "detail": str(e),
    })


def _not_started(e: contest_service.ContestNotStarted) -> JSONResponse:
    """403 before contests.scheduled_at; `detail` too, for frontends that only show `detail`."""
    return JSONResponse(status_code=403, content={
        "code": "CONTEST_NOT_STARTED",
        "scheduled_at": e.scheduled_at.astimezone(timezone.utc).isoformat(),
        "detail": "This contest has not started yet.",
        # count down against the server clock, not the phone's (a fast phone clock looked like a late opening)
        "server_now": contest_service._now().astimezone(timezone.utc).isoformat(),
    })


# ═══════════════════════════════════════════════════════════
# Helper: Extract user ID from request (optional auth)
# ═══════════════════════════════════════════════════════════

async def get_optional_user_id(request: Request) -> Optional[str]:
    """
    Extract user ID from Authorization header if present.
    Returns None if not authenticated (anonymous student).
    Replace this with your actual auth logic.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None

    try:
        from app.core.auth import verify_token
        token = auth_header.split(" ")[1]
        user = await verify_token(token)
        return user.id if user else None
    except Exception:
        return None


async def get_required_user_id(request: Request) -> str:
    """Same as above but raises 401 if not authenticated"""
    user_id = await get_optional_user_id(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id


# ═══════════════════════════════════════════════════════════
# POST /contests — Create a new contest
# ═══════════════════════════════════════════════════════════

@router.post("", response_model=CreateContestResponse)
async def create_contest(
    body: CreateContestRequest,
    request: Request,
):
    """
    Teacher creates a contest from generated test questions.
    Returns a shareable link.
    """
    try:
        teacher_id = await get_optional_user_id(request)
        if not teacher_id and request.headers.get("Authorization", "").startswith("Bearer "):
            # unchanged behaviour (old clients), but no longer silent: this contest gets no teacher_id
            logger.warning("Create contest: a bearer token was sent but did not verify; the contest has no teacher")

        if not body.questions or len(body.questions) == 0:
            raise HTTPException(status_code=400, detail="At least 1 question is required")

        result = contest_service.create_contest(
            request=body,
            teacher_id=teacher_id,
        )

        logger.info(f"Contest created: {result.short_code} by teacher {teacher_id}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create contest error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create contest: {str(e)}")


# ═══════════════════════════════════════════════════════════
# GET /contests/:code/info — Public contest info
# ═══════════════════════════════════════════════════════════

@router.get("/{short_code}/info", response_model=ContestInfoResponse)
async def get_contest_info(short_code: str):
    """
    Get public contest info for student landing page.
    No authentication required.
    """
    try:
        info = contest_service.get_contest_info(short_code)
        if not info:
            raise HTTPException(status_code=404, detail="Contest not found")

        if info.status == "ended":
            raise HTTPException(status_code=410, detail="This contest has ended")

        if info.status == "paused":
            raise HTTPException(status_code=403, detail="This contest is paused")

        return info

    except contest_service.ContestNotStarted as e:
        return _not_started(e)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get contest info error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load contest")


# ═══════════════════════════════════════════════════════════
# POST /contests/:code/start — Start an attempt
# ═══════════════════════════════════════════════════════════

@router.post("/{short_code}/start", response_model=ContestDataResponse)
async def start_contest_attempt(
    short_code: str,
    body: StartAttemptRequest,
    request: Request,
):
    """
    Student starts a contest attempt.
    Creates attempt record and returns questions (without answers).
    Resumes the caller's unsubmitted attempt instead (X-Attempt-Id of an earlier start, or the logged-in
    student's own): same attempt_id, resumed=true, saved progress, deadline_at / server_now.
    """
    try:
        student_id = await get_optional_user_id(request)

        result = contest_service.start_attempt(
            short_code=short_code,
            request=body,
            student_id=student_id,
            resume_attempt_id=request.headers.get("X-Attempt-Id"),
        )

        if not result:
            raise HTTPException(status_code=404, detail="Contest not found or not active")

        return result

    except contest_service.ContestNotStarted as e:
        return _not_started(e)
    except contest_service.ContestExpired:
        raise HTTPException(status_code=410, detail="This contest has ended")
    except contest_service.AlreadySubmitted as e:
        return _already_submitted(e)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Start attempt error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start contest")


# ═══════════════════════════════════════════════════════════
# POST /contests/:id/submit — Submit answers
# ═══════════════════════════════════════════════════════════

@router.post("/{contest_id}/submit", response_model=SubmitContestResponse)
async def submit_contest(
    contest_id: str,
    body: SubmitContestRequest,
    request: Request,
):
    """
    Student submits their answers.
    Returns score and optionally correct answers based on contest settings.
    """
    try:
        # attempt_id comes from the body or header
        attempt_id = request.headers.get("X-Attempt-Id")
        if not attempt_id:
            raise HTTPException(
                status_code=400,
                detail="X-Attempt-Id header is required"
            )

        result = contest_service.submit_attempt(
            contest_id=contest_id,
            attempt_id=attempt_id,
            request=body,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Submit contest error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit contest")


# ═══════════════════════════════════════════════════════════
# PUT /contests/:id/attempts/:attempt_id/progress — Save progress
# ═══════════════════════════════════════════════════════════

@router.put("/{contest_id}/attempts/{attempt_id}/progress")
async def save_attempt_progress(
    contest_id: str,
    attempt_id: str,
    body: ProgressRequest,
    request: Request,
):
    """
    Saves answers / position / warnings so a refresh resumes the attempt. Same rules as submit: X-Attempt-Id
    must be this attempt. warning_count never goes down; the max_warnings-th warning, or a save after
    deadline + 120 s, finalizes the attempt (finalized=true + the submit response).
    """
    header = request.headers.get("X-Attempt-Id")
    if not header:
        raise HTTPException(status_code=400, detail="X-Attempt-Id header is required")
    if header != attempt_id:
        raise HTTPException(status_code=403, detail="This is not your attempt")
    try:
        user_id = await get_optional_user_id(request)
        return contest_service.save_progress(contest_id, attempt_id, body, user_id)
    except contest_service.AlreadySubmitted as e:
        return _already_submitted(e)
    except contest_service.AttemptForbidden:
        raise HTTPException(status_code=403, detail="This is not your attempt")
    except ValueError:
        raise HTTPException(status_code=404, detail="Attempt not found")
    except Exception as e:
        logger.error(f"Save progress error: {type(e).__name__}")
        raise HTTPException(status_code=500, detail="Could not save progress")


# ═══════════════════════════════════════════════════════════
# Teacher-only: attempts list, allow-retry, publish results
# ═══════════════════════════════════════════════════════════

NOT_TEACHER = "Only the teacher who created this contest can do this"


@router.get("/{contest_id}/attempts")
async def list_contest_attempts(contest_id: str, request: Request):
    teacher_id = await get_required_user_id(request)
    try:
        result = contest_service.list_attempts(contest_id, teacher_id)
    except contest_service.NotTeacher:
        raise HTTPException(status_code=403, detail=NOT_TEACHER)
    if result is None:
        raise HTTPException(status_code=404, detail="Contest not found")
    return result


@router.post("/{contest_id}/attempts/{attempt_id}/allow-retry")
async def allow_attempt_retry(contest_id: str, attempt_id: str, request: Request):
    teacher_id = await get_required_user_id(request)
    try:
        result = contest_service.allow_retry(contest_id, attempt_id, teacher_id)
    except contest_service.NotTeacher:
        raise HTTPException(status_code=403, detail=NOT_TEACHER)
    except contest_service.AttemptRunning:
        return JSONResponse(status_code=409, content={
            "code": "ATTEMPT_IN_PROGRESS", "attempt_id": attempt_id,
            "detail": "This attempt is still running. Allow a retry after it ends.",
        })
    if result is None:
        raise HTTPException(status_code=404, detail="Attempt not found")
    return result


@router.post("/{contest_id}/publish-results")
async def publish_contest_results(contest_id: str, request: Request):
    teacher_id = await get_required_user_id(request)
    try:
        result = contest_service.publish_results(contest_id, teacher_id)
    except contest_service.NotTeacher:
        raise HTTPException(status_code=403, detail=NOT_TEACHER)
    if result is None:
        raise HTTPException(status_code=404, detail="Contest not found")
    return result


# ═══════════════════════════════════════════════════════════
# GET /contests/:id/leaderboard — Teacher views results
# ═══════════════════════════════════════════════════════════

@router.get("/{contest_id}/leaderboard", response_model=ContestLeaderboardResponse)
async def get_contest_leaderboard(
    contest_id: str,
    request: Request,
):
    """
    Ranked results for students and the teacher (rank, correct / wrong / left, is_me, my_rank). Only the contest's
    teacher gets emails, warning counts, submitted_via, late and unfinished attempts; with answer_mode
    teacher_release students get no rows until results are published. X-Attempt-Id marks an anonymous student's row.
    """
    try:
        viewer_id = await get_optional_user_id(request)

        result = contest_service.get_leaderboard(
            contest_id=contest_id,
            viewer_id=viewer_id,
            attempt_id=request.headers.get("X-Attempt-Id"),
        )

        if not result:
            raise HTTPException(status_code=404, detail="Contest not found")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Leaderboard error: {e}")
        raise HTTPException(status_code=500, detail="Failed to load leaderboard")


# ═══════════════════════════════════════════════════════════
# GET /contests/my — List teacher's contests
# ═══════════════════════════════════════════════════════════

@router.get("/my")
async def list_my_contests(request: Request):
    """List all contests created by the authenticated teacher"""
    try:
        teacher_id = await get_required_user_id(request)

        contests = contest_service.list_teacher_contests(teacher_id)
        return {"contests": contests}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List contests error: {e}")
        raise HTTPException(status_code=500, detail="Failed to list contests")