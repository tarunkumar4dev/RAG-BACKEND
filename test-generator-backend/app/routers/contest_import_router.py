"""
Contest import — "PDF → contest questions" (teacher reviews them, then the frontend calls POST /contests).

  POST /api/v1/contests/import-pdf            {"path": "<uid>/contest-imports/<uuid>.pdf"} → 202 {"job_id"}
        The frontend uploads the PDF straight to the private 'answer-keys' bucket first (Vercel caps request
        bodies at ~4.5 MB, so the file never goes through this API).
  GET  /api/v1/contests/import-pdf/{job_id}   {status, progress, error, retryable, result} — owner only (404 else)

The work runs as a background task of the same Vercel invocation (vercel_runtime sends its "end" only after the
ASGI app returns, and Starlette runs background tasks after the response body is sent), inside WORK_BUDGET_S of
the 300 s limit. If the invocation dies anyway, the row stays 'processing' and GET reports it failed (retryable)
after STALE_AFTER_S. The uploaded PDF is deleted from Storage when the job ends, success or failure.
POST with a path that already has a job (unique owner + path) returns that job_id — no second job, no second
charge on the daily limit.

Rows: contest_import_jobs (24_contest_import.sql), written with the service key, always filtered by owner.
"""

import logging
import os
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.database import get_supabase_admin
from app.routers.copy_checker_router import _download_own
from app.routers.smart_assignment_router import _Caller, _cleanup, day_start_ist, require_caller, with_retry
from app.schemas.contest_import import ImportJobOut, ImportRequest
from app.services import contest_import_service as importer
from app.services.smart_assignment_service import CallCapExceeded, GeminiBusy

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/contests/import-pdf", tags=["Contests"])

BUCKET = "answer-keys"
TABLE = "contest_import_jobs"
DAILY_LIMIT = int(os.getenv("CONTEST_IMPORT_DAILY_LIMIT", "10"))
WORK_BUDGET_S = float(os.getenv("CONTEST_IMPORT_BUDGET_S", "270"))
STALE_AFTER_S = 360

BUSY_MSG = "AI is busy, please retry in a minute"
STALE_MSG = "The import took too long and was stopped. Please try again."
MISSING_MSG = "The uploaded PDF was not found. Please upload it again."
UNREADABLE_MSG = "Could not read the questions in this PDF. Check that it is a clear question paper and try again."
FAILED_MSG = "Import failed. Please try again."
SLOW_MSG = "The AI took too long to read this paper. Please try again in a few minutes."

_PATH_RE = re.compile(r"^(?P<uid>[0-9a-f-]{36})/contest-imports/[0-9a-f-]{36}\.pdf$")


def _valid_path(uid: str, path: str) -> bool:
    m = _PATH_RE.match(path or "")
    return bool(m) and m.group("uid") == str(uid)


def _table():
    return get_supabase_admin().table(TABLE)


def _imports_today(uid: str) -> int:
    """Jobs since IST midnight; a failure the teacher can retry (AI busy, stopped) does not use up the limit."""
    r = with_retry(lambda: _table().select("id", count="exact").eq("owner", uid)
                   .gte("created_at", day_start_ist().isoformat())
                   .or_("status.neq.failed,retryable.eq.false").execute())
    return r.count or 0


def _job_for_path(uid: str, path: str):
    r = with_retry(lambda: _table().select("id").eq("owner", uid).eq("path", path).limit(1).execute())
    return str(r.data[0]["id"]) if r.data else None


def _set(job_id: str, **fields) -> None:
    try:
        with_retry(lambda: _table().update(fields).eq("id", job_id).execute())
    except Exception:
        logger.warning(f"[import] {job_id}: could not save {sorted(fields)}", exc_info=True)


def _delete_upload(path: str) -> None:
    try:
        with_retry(lambda: get_supabase_admin().storage.from_(BUCKET).remove([path]))
    except Exception as e:
        logger.warning(f"[import] could not delete the uploaded PDF: {type(e).__name__}")


def run_import_job(job_id: str, owner: str, path: str) -> None:
    """The whole job; never raises (the row says what happened)."""
    start = time.monotonic()
    deadline = start + WORK_BUDGET_S
    local = None
    _set(job_id, status="processing", progress=5)
    try:
        try:
            local = _download_own(SimpleNamespace(user=SimpleNamespace(id=owner)), path, "cimport_", bucket=BUCKET)
        except HTTPException as e:
            raise importer.ImportRejected(str(e.detail))
        except Exception as e:
            logger.warning(f"[import] {job_id}: download failed: {type(e).__name__}")
            raise importer.ImportRejected(MISSING_MSG)
        pages = importer.check_pdf(local)
        _set(job_id, progress=10)
        result = importer.import_paper(local, pages, deadline, progress=lambda p: _set(job_id, progress=p))
        _set(job_id, status="done", progress=100, result=result, error=None, retryable=False)
        logger.info(f"[import] {job_id}: done in {time.monotonic() - start:.1f}s, "
                    f"{len(result['questions'])} questions, {len(result['warnings'])} warnings")
    except importer.ImportRejected as e:
        _set(job_id, status="failed", error=str(e), retryable=False)
    except GeminiBusy as e:
        logger.warning(f"[import] {job_id}: Gemini busy ({e.code})")
        _set(job_id, status="failed", error=BUSY_MSG, retryable=True)
    except (ValueError, CallCapExceeded) as e:
        logger.warning(f"[import] {job_id}: unreadable paper: {type(e).__name__}: {str(e)[:200]}")
        _set(job_id, status="failed", error=UNREADABLE_MSG, retryable=True)
    except Exception as e:
        if "timeout" in type(e).__name__.lower():          # httpx.ReadTimeout, TimeoutError: the model was too slow
            logger.warning(f"[import] {job_id}: Gemini too slow: {type(e).__name__}")
            _set(job_id, status="failed", error=SLOW_MSG, retryable=True)
        else:
            logger.error(f"[import] {job_id}: failed", exc_info=True)
            _set(job_id, status="failed", error=FAILED_MSG, retryable=True)
    finally:
        _cleanup(local)
        _delete_upload(path)


@router.post("", status_code=202)
async def start_import(body: ImportRequest, background: BackgroundTasks, c: _Caller = Depends(require_caller)):
    uid = str(c.user.id)
    if not _valid_path(uid, body.path):
        raise HTTPException(status_code=400, detail="Invalid file path. Upload the PDF to <your id>/contest-imports/.")
    existing = await run_in_threadpool(_job_for_path, uid, body.path)
    if existing:
        return {"job_id": existing}
    if await run_in_threadpool(_imports_today, uid) >= DAILY_LIMIT:
        raise HTTPException(status_code=429, detail=f"You can import {DAILY_LIMIT} papers a day. Try again tomorrow.")
    try:
        r = await run_in_threadpool(lambda: with_retry(lambda: _table().insert(
            {"owner": uid, "path": body.path, "status": "queued", "progress": 0}).execute()))
    except Exception as e:
        if "23505" not in str(e):           # not the unique (owner, path) index: a real error
            raise
        return {"job_id": await run_in_threadpool(_job_for_path, uid, body.path)}   # a parallel POST won the race
    job_id = str(r.data[0]["id"])
    background.add_task(run_import_job, job_id, uid, body.path)
    logger.info(f"[import] {job_id}: queued")
    return {"job_id": job_id}


def _stale(row: dict) -> bool:
    if row.get("status") not in ("queued", "processing") or not row.get("updated_at"):
        return False
    updated = datetime.fromisoformat(str(row["updated_at"]).replace("Z", "+00:00"))
    return datetime.now(timezone.utc) - updated > timedelta(seconds=STALE_AFTER_S)


@router.get("/{job_id}", response_model=ImportJobOut)
async def get_import(job_id: str, c: _Caller = Depends(require_caller)):
    try:
        uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Import not found.")
    uid = str(c.user.id)
    r = await run_in_threadpool(lambda: with_retry(lambda: _table().select("*").eq("id", job_id).eq("owner", uid)
                                                   .limit(1).execute()))
    if not r.data:
        raise HTTPException(status_code=404, detail="Import not found.")
    row = r.data[0]
    if _stale(row):
        # the invocation was killed: nothing will finish this job
        row.update(status="failed", error=STALE_MSG, retryable=True)
        await run_in_threadpool(lambda: with_retry(lambda: _table().update(
            {"status": "failed", "error": STALE_MSG, "retryable": True})
            .eq("id", job_id).in_("status", ["queued", "processing"]).execute()))
    return ImportJobOut(status=row["status"], progress=row.get("progress") or 0, error=row.get("error"),
                        retryable=bool(row.get("retryable")),
                        result=row.get("result") if row["status"] == "done" else None)
