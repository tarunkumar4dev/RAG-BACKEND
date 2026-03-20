# app/api/v1/endpoints/contest.py
# ──────────────────────────────────────────────────────────
# Contest API Endpoints
#
# POST   /api/v1/contests              → Create contest (teacher)
# GET    /api/v1/contests/:code/info   → Get contest info (student landing)
# POST   /api/v1/contests/:code/start  → Start attempt (student)
# POST   /api/v1/contests/:id/submit   → Submit answers (student)
# GET    /api/v1/contests/:id/leaderboard → View results (teacher)
# GET    /api/v1/contests/my            → List teacher's contests
# ──────────────────────────────────────────────────────────

import logging
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
    ContestLeaderboardResponse,
)
from app.services import contest_service

# If you have auth dependency, import it:
# from app.core.auth import get_current_user_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/contests", tags=["Contests"])


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
        # TODO: Replace with your Supabase JWT verification
        from app.core.auth import verify_token  # your auth utility
        token = auth_header.split(" ")[1]
        user = await verify_token(token)
        return user.get("sub") or user.get("id")
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
    """
    try:
        student_id = await get_optional_user_id(request)

        result = contest_service.start_attempt(
            short_code=short_code,
            request=body,
            student_id=student_id,
        )

        if not result:
            raise HTTPException(status_code=404, detail="Contest not found or not active")

        return result

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
# GET /contests/:id/leaderboard — Teacher views results
# ═══════════════════════════════════════════════════════════

@router.get("/{contest_id}/leaderboard", response_model=ContestLeaderboardResponse)
async def get_contest_leaderboard(
    contest_id: str,
    request: Request,
):
    """
    Teacher views all attempts, scores, and leaderboard for a contest.
    """
    try:
        teacher_id = await get_optional_user_id(request)

        result = contest_service.get_leaderboard(
            contest_id=contest_id,
            teacher_id=teacher_id,
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