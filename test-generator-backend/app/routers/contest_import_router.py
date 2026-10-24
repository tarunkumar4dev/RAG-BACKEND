"""
Contest import — "PDF → contest questions" (teacher reviews them, then the frontend calls POST /contests).

  POST /api/v1/contests/import-pdf            {"path": "<uid>/contest-imports/<uuid>.pdf"} → 202 {"job_id"}
        The frontend uploads the PDF straight to the private 'answer-keys' bucket first (Vercel caps request
        bodies at ~4.5 MB, so the file never goes through this API).
  GET  /api/v1/contests/import-pdf/{job_id}   {status, progress, error, retryable, result} — owner only (404 else)

The work runs as a background task of the same Vercel invocation (vercel_runtime sends its "end" only after the
ASGI app returns, and Starlette runs background tasks after the response body is sent), inside WORK_BUDGET_S of
the 300 s limit. If the invocation dies anyway, the row stays 'processing' and GET reports it failed (retryable)
after STALE_AFTER_S.

One row per uploaded file (unique owner + path). POST with a path that already has a job:
  queued / processing / done, or failed with retryable=false  → that job_id, nothing runs again
  failed with retryable=true (AI busy, too slow, stopped)     → the same row is reset (queued, attempts + 1) and
                                                                runs again; same job_id
The uploaded PDF is deleted when a run ends done or failed with retryable=false; a retryable failure keeps it so
the retry can read it. Best effort, after each new job: the caller's own {uid}/contest-imports/ files older than
24 h are deleted (left behind by retryable failures nobody retried, or killed invocations).

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
OLD_UPLOAD_S = 24 * 3600
CLEANUP_UNTIL_S = 285       # skip the old-upload cleanup when the invocation is this close to Vercel's 300 s

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
    r = with_retry(lambda: _table().select("id, status, retryable, attempts").eq("owner", uid).eq("path", path)
                   .limit(1).execute())
    return r.data[0] if r.data else None


def _reset_for_retry(job: dict) -> bool:
    """failed + retryable → queued again. Conditional, so two parallel re-POSTs start one run, not two."""
    r = with_retry(lambda: _table().update({
        "status": "queued", "progress": 0, "error": None, "retryable": False, "result": None,
        "attempts": (job.get("attempts") or 1) + 1,
    }).eq("id", job["id"]).eq("status", "failed").eq("retryable", True).execute())
    return bool(r.data)


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


def _delete_old_uploads(uid: str, keep: str, started: float) -> None:
    """Best effort: the caller's own {uid}/contest-imports/*.pdf older than OLD_UPLOAD_S. Never another prefix."""
    if time.monotonic() - started > CLEANUP_UNTIL_S:
        return
    folder = f"{uid}/contest-imports"
    try:
        items = with_retry(lambda: get_supabase_admin().storage.from_(BUCKET).list(folder, {"limit": 1000})) or []
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=OLD_UPLOAD_S)
        old = []
        for o in items:
            path = f"{folder}/{o.get('name')}"
            created = o.get("created_at")
            if not created or path == keep or not _valid_path(uid, path):
                continue
            if datetime.fromisoformat(str(created).replace("Z", "+00:00")) < cutoff:
                old.append(path)
        if old:
            with_retry(lambda: get_supabase_admin().storage.from_(BUCKET).remove(old))
            logger.info(f"[import] deleted {len(old)} old upload(s) of one user")
    except Exception as e:
        logger.warning(f"[import] old-upload cleanup skipped: {type(e).__name__}")


def run_import_job(job_id: str, owner: str, path: str) -> None:
    """The whole job; never raises (the row says what happened)."""
    start = time.monotonic()
    deadline = start + WORK_BUDGET_S
    local = None
    final = False           # done, or failed with retryable=false: the upload is no longer needed
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
        final = True
        logger.info(f"[import] {job_id}: done in {time.monotonic() - start:.1f}s, "
                    f"{len(result['questions'])} questions, {len(result['warnings'])} warnings")
    except importer.ImportRejected as e:
        _set(job_id, status="failed", error=str(e), retryable=False)
        final = True
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
        if final:
            _delete_upload(path)


@router.post("", status_code=202)
async def start_import(body: ImportRequest, background: BackgroundTasks, c: _Caller = Depends(require_caller)):
    uid = str(c.user.id)
    if not _valid_path(uid, body.path):
        raise HTTPException(status_code=400, detail="Invalid file path. Upload the PDF to <your id>/contest-imports/.")
    started = time.monotonic()
    existing = await run_in_threadpool(_job_for_path, uid, body.path)
    if existing and not (existing["status"] == "failed" and existing.get("retryable")):
        return {"job_id": str(existing["id"])}
    if await run_in_threadpool(_imports_today, uid) >= DAILY_LIMIT:
        raise HTTPException(status_code=429, detail=f"You can import {DAILY_LIMIT} papers a day. Try again tomorrow.")
    if existing:
        job_id = str(existing["id"])
        if await run_in_threadpool(_reset_for_retry, existing):
            background.add_task(run_import_job, job_id, uid, body.path)
            logger.info(f"[import] {job_id}: retry {(existing.get('attempts') or 1) + 1} queued")
        return {"job_id": job_id}            # not reset = a parallel re-POST already restarted it
    try:
        r = await run_in_threadpool(lambda: with_retry(lambda: _table().insert(
            {"owner": uid, "path": body.path, "status": "queued", "progress": 0}).execute()))
    except Exception as e:
        if "23505" not in str(e):           # not the unique (owner, path) index: a real error
            raise
        winner = await run_in_threadpool(_job_for_path, uid, body.path)    # a parallel POST won the race
        return {"job_id": str(winner["id"])}
    job_id = str(r.data[0]["id"])
    background.add_task(run_import_job, job_id, uid, body.path)
    background.add_task(_delete_old_uploads, uid, body.path, started)
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
