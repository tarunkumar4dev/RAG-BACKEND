"""
Internal routes of the checker job queue (no /api/v1 prefix: the Supabase vault holds the exact URL
https://rag-backend-kappa-three.vercel.app/internal/jobs/run). Students never call these.

  POST /internal/jobs/run   kick from Postgres (pg_net on new jobs, pg_cron sweep every minute).
                            X-Job-Secret must equal env JOB_SECRET (constant-time compare), else 401.
                            Answers 202; the work runs after the response inside the same invocation
                            (240 s budget of Vercel's 300 s).
  POST /internal/regrade    teacher (Supabase JWT, same audience as publish_results): new check jobs for selected
                            attempts after the answer key changed; old marks go to the audit table.
"""

import hmac
import logging
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.core.log_mask import PIIMaskFilter
from app.routers.smart_assignment_router import _Caller, require_caller
from app.services import job_worker, regrade as regrade_service

logger = logging.getLogger(__name__)
logger.addFilter(PIIMaskFilter())
router = APIRouter(prefix="/internal", tags=["Internal"])

JOB_SECRET_HEADER = "X-Job-Secret"


def _worker() -> job_worker.JobWorker:
    return job_worker.default_worker()


def _store():
    from app.core.db_pool import get_db_connection
    from app.services.job_store import JobStore
    return JobStore(get_db_connection)


def _secret_ok(request: Request) -> bool:
    expected = os.getenv("JOB_SECRET") or ""
    given = request.headers.get(JOB_SECRET_HEADER) or ""
    if not expected:
        logger.error("[jobs] JOB_SECRET is not set: every kick is refused")
        return False
    return hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8"))


def _run() -> None:
    try:
        stats = _worker().run()
        logger.info(f"[jobs] run finished: {stats}")
    except Exception:
        logger.error("[jobs] run crashed; the sweep kicks again within a minute", exc_info=True)


@router.post("/jobs/run", status_code=202)
async def run_jobs(request: Request, background: BackgroundTasks):
    if not _secret_ok(request):
        raise HTTPException(status_code=401, detail="Unauthorized")
    background.add_task(_run)
    return {"accepted": True}


class RegradeRequest(BaseModel):
    assignment_id: str
    submission_ids: list[str] = Field(min_length=1, max_length=200)


def _regrade(c: _Caller, body: RegradeRequest) -> dict:
    allowed = c.db.rpc("a4_can_grade_assignment", {"p_assignment_id": body.assignment_id}).execute().data
    if allowed is not True:
        raise HTTPException(status_code=403, detail="You can't re-grade this assignment.")

    def unpublish(ids: list, reason: str) -> dict:     # the teacher's own JWT: audited as them
        return c.db.rpc("unpublish_results", {"p_submission_ids": ids, "p_reason": reason}).execute().data or {}

    return regrade_service.regrade(_store(), body.assignment_id, body.submission_ids, c.user.id, unpublish)


@router.post("/regrade")
async def regrade(body: RegradeRequest, c: _Caller = Depends(require_caller)):
    try:
        uuid.UUID(body.assignment_id)
        for s in body.submission_ids:
            uuid.UUID(s)
    except ValueError:
        raise HTTPException(status_code=404, detail="Assignment not found.")
    try:
        return await run_in_threadpool(_regrade, c, body)
    except HTTPException:
        raise
    except Exception:
        logger.error("[jobs] regrade failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Re-grade could not be started. Please try again.")
