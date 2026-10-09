"""
Smart Assignment Checker Router — v2.0

Endpoints (all require a Supabase access token; the owner is always the JWT user):
  POST /smart-assignment-checker/check    Upload 3 files → grading result (saved to history)
  GET  /smart-assignment-checker/history  The caller's past checks
  GET  /smart-assignment-checker/result/{id}
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from starlette.concurrency import run_in_threadpool
import json
import logging
import uuid
import tempfile
import os
import time

from app.core.auth import AuthUser, require_user
from app.core.database import get_supabase_admin
from app.services.smart_assignment_service import check_assignment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/smart-assignment-checker", tags=["Smart Assignment Checker"])

MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}
# Every check is 2 Gemini calls on up to 60 MB of files → cap per teacher per day (UTC)
DAILY_LIMIT = int(os.getenv("SMART_CHECKER_DAILY_LIMIT", "100"))
TABLE = "smart_assignment_checks"


# ── Helpers ────────────────────────────────────────────────────────

def _validate_file(file: UploadFile, label: str) -> str:
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail=f"No {label} file provided.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"{label}: unsupported file type '{ext}'.")
    return ext


async def _save_temp(file: UploadFile, ext: str, prefix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix=prefix)
    try:
        total = 0
        while chunk := await file.read(1024 * 256):
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(status_code=413, detail=f"File too large. Max 20 MB.")
            tmp.write(chunk)
        tmp.close()
        return tmp.name
    except HTTPException:
        raise
    except Exception as e:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


def _cleanup(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.unlink(p)
        except Exception:
            pass


def _checks_today(user_id: str) -> int:
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    r = (get_supabase_admin().table(TABLE).select("id", count="exact")
         .eq("teacher_id", user_id).gte("created_at", start).limit(1).execute())
    return r.count or 0


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Check
# ═══════════════════════════════════════════════════════════════════════

@router.post("/check")
async def check(
    question_paper: UploadFile = File(..., description="Question paper PDF/image"),
    answer_key: UploadFile = File(..., description="Answer key PDF/image"),
    student_sheet: UploadFile = File(..., description="Student's handwritten answer sheet"),
    strictness: str = Form("medium", description="easy | medium | hard | extreme"),
    class_grade: str = Form("", description="Class/Grade (e.g. Class 10)"),
    subject: str = Form("", description="Subject"),
    user: AuthUser = Depends(require_user),
):
    """
    Upload question paper + answer key + student sheet → graded result.
    Student identity (name, class, roll no) must be on the sheet.
    """
    qp_ext = _validate_file(question_paper, "Question paper")
    ak_ext = _validate_file(answer_key, "Answer key")
    ss_ext = _validate_file(student_sheet, "Student sheet")

    try:
        used = await run_in_threadpool(_checks_today, user.id)
    except Exception as e:
        # The limit protects cost, but a history-table outage must not block teachers
        logger.warning(f"Daily-limit check failed (allowing): {e}")
        used = 0
    if used >= DAILY_LIMIT:
        raise HTTPException(status_code=429,
                            detail=f"Daily limit reached ({DAILY_LIMIT} sheets). Try again tomorrow.")

    qp_path = ak_path = ss_path = None
    try:
        qp_path = await _save_temp(question_paper, qp_ext, "qpaper_")
        ak_path = await _save_temp(answer_key, ak_ext, "akey_")
        ss_path = await _save_temp(student_sheet, ss_ext, "ssheet_")

        logger.info(f"Checking assignment: user={user.id[:8]}, strictness={strictness}, "
                    f"subject={subject}, class={class_grade}")

        start = time.time()
        # Blocking (Gemini calls, ~30 s) → worker thread, so the server keeps serving other requests
        result = await run_in_threadpool(
            check_assignment,
            question_paper_path=qp_path,
            answer_key_path=ak_path,
            student_sheet_path=ss_path,
            strictness=strictness,
        )
        elapsed = round(time.time() - start, 2)

        check_id = str(uuid.uuid4())
        await run_in_threadpool(
            _save_to_db,
            check_id=check_id,
            teacher_id=user.id,
            result=result,
            class_grade=class_grade,
            subject=subject,
            qp_filename=question_paper.filename,
            ss_filename=student_sheet.filename,
        )

        return {
            "ok": True,
            "check_id": check_id,
            "grading_time_seconds": elapsed,
            "result": result,
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Smart assignment check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Grading failed. Please try again.")
    finally:
        _cleanup(qp_path, ak_path, ss_path)


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Get Result
# ═══════════════════════════════════════════════════════════════════════

@router.get("/result/{check_id}")
async def get_result(check_id: str, user: AuthUser = Depends(require_user)):
    try:
        uuid.UUID(check_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Result not found.")
    try:
        r = await run_in_threadpool(
            lambda: get_supabase_admin().table(TABLE).select("*")
            .eq("id", check_id).eq("teacher_id", user.id).maybe_single().execute()
        )
    except Exception as e:
        logger.error(f"Fetch result failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch result.")
    if not r or not r.data:
        raise HTTPException(status_code=404, detail="Result not found.")
    return {"ok": True, "data": r.data}


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: History
# ═══════════════════════════════════════════════════════════════════════

@router.get("/history")
async def get_history(limit: int = 20, offset: int = 0, user: AuthUser = Depends(require_user)):
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    try:
        r = await run_in_threadpool(
            lambda: get_supabase_admin().table(TABLE)
            .select("id, student_name, student_class, student_roll, subject, class_grade, "
                    "total_obtained, max_marks, percentage, is_valid, strictness, needs_review_count, created_at")
            .eq("teacher_id", user.id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return {"ok": True, "checks": r.data or [], "count": len(r.data or [])}
    except Exception as e:
        logger.error(f"Fetch history failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history.")


# ═══════════════════════════════════════════════════════════════════════
# DB HELPER
# ═══════════════════════════════════════════════════════════════════════

def _save_to_db(
    check_id: str, teacher_id: str, result: dict,
    class_grade: str, subject: str, qp_filename: str, ss_filename: str,
):
    try:
        student_info = result.get("student_info", {})
        get_supabase_admin().table(TABLE).insert({
            "id": check_id,
            "teacher_id": teacher_id,
            "student_name": student_info.get("name") or None,
            "student_class": student_info.get("class") or None,
            "student_roll": student_info.get("roll_no") or None,
            "subject": (subject or "")[:100] or None,
            "class_grade": (class_grade or "")[:50] or None,
            "qp_filename": (qp_filename or "")[:255],
            "ss_filename": (ss_filename or "")[:255],
            "is_valid": result.get("is_valid", False),
            "total_obtained": result.get("total_marks_awarded", 0),
            "max_marks": result.get("max_marks", 0),
            "percentage": result.get("percentage", 0),
            "needs_review_count": result.get("needs_review_count", 0),
            "strictness": result.get("_meta", {}).get("strictness", "medium"),
            "overall_remarks": result.get("overall_remarks", ""),
            "result_json": result,
        }).execute()
        logger.info(f"Saved smart assignment check {check_id}")
    except Exception as e:
        logger.warning(f"Could not save (non-fatal): {e}")
