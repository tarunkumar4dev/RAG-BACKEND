"""
Smart Assignment Checker Router — v1.0

Endpoints:
  POST /smart-assignment-checker/check    Upload 3 files → grading result
  GET  /smart-assignment-checker/history  Past checks for a teacher
  GET  /smart-assignment-checker/result/{id}
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import json
import logging
import uuid
import tempfile
import os
import time

from app.core.database import get_supabase
from app.services.smart_assignment_service import check_assignment

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/smart-assignment-checker", tags=["Smart Assignment Checker"])

MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


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


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Check
# ═══════════════════════════════════════════════════════════════════════

@router.post("/check")
async def check(
    question_paper: UploadFile = File(..., description="Question paper PDF/image"),
    answer_key: UploadFile = File(..., description="Answer key PDF/image"),
    student_sheet: UploadFile = File(..., description="Student's handwritten answer sheet"),
    strictness: str = Form("medium", description="easy | medium | hard | extreme"),
    teacher_id: str = Form("", description="Teacher UUID"),
    class_grade: str = Form("", description="Class/Grade (e.g. Class 10)"),
    subject: str = Form("", description="Subject"),
):
    """
    Upload question paper + answer key + student sheet → graded result.
    Student identity (name, class, roll no) must be on the sheet.
    """
    qp_ext = _validate_file(question_paper, "Question paper")
    ak_ext = _validate_file(answer_key, "Answer key")
    ss_ext = _validate_file(student_sheet, "Student sheet")

    qp_path = ak_path = ss_path = None

    try:
        # Save all three files
        qp_path = await _save_temp(question_paper, qp_ext, "qpaper_")
        ak_path = await _save_temp(answer_key, ak_ext, "akey_")
        ss_path = await _save_temp(student_sheet, ss_ext, "ssheet_")

        logger.info(f"Checking assignment: strictness={strictness}, subject={subject}, class={class_grade}")

        start = time.time()
        result = check_assignment(
            question_paper_path=qp_path,
            answer_key_path=ak_path,
            student_sheet_path=ss_path,
            strictness=strictness,
        )
        elapsed = round(time.time() - start, 2)

        check_id = str(uuid.uuid4())
        if teacher_id:
            _save_to_db(
                check_id=check_id,
                teacher_id=teacher_id,
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
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")
    finally:
        _cleanup(qp_path, ak_path, ss_path)


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Get Result
# ═══════════════════════════════════════════════════════════════════════

@router.get("/result/{check_id}")
async def get_result(check_id: str, teacher_id: str):
    supabase = get_supabase()
    try:
        r = (
            supabase.table("smart_assignment_checks")
            .select("*")
            .eq("id", check_id)
            .eq("teacher_id", teacher_id)
            .maybeSingle()
            .execute()
        )
        if not r.data:
            raise HTTPException(status_code=404, detail="Result not found.")
        return {"ok": True, "data": r.data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch result.")


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: History
# ═══════════════════════════════════════════════════════════════════════

@router.get("/history")
async def get_history(teacher_id: str, limit: int = 20, offset: int = 0):
    supabase = get_supabase()
    try:
        r = (
            supabase.table("smart_assignment_checks")
            .select("id, student_name, student_class, student_roll, subject, class_grade, "
                    "total_obtained, max_marks, percentage, is_valid, strictness, created_at")
            .eq("teacher_id", teacher_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return {"ok": True, "checks": r.data or [], "count": len(r.data or [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to fetch history.")


# ═══════════════════════════════════════════════════════════════════════
# DB HELPER
# ═══════════════════════════════════════════════════════════════════════

def _save_to_db(
    check_id: str, teacher_id: str, result: dict,
    class_grade: str, subject: str, qp_filename: str, ss_filename: str,
):
    try:
        supabase = get_supabase()
        student_info = result.get("student_info", {})
        supabase.table("smart_assignment_checks").insert({
            "id": check_id,
            "teacher_id": teacher_id,
            "student_name": student_info.get("name") or None,
            "student_class": student_info.get("class") or None,
            "student_roll": student_info.get("roll_no") or None,
            "subject": subject or None,
            "class_grade": class_grade or None,
            "qp_filename": qp_filename,
            "ss_filename": ss_filename,
            "is_valid": result.get("is_valid", False),
            "total_obtained": result.get("total_marks_awarded", 0),
            "max_marks": result.get("max_marks", 0),
            "percentage": result.get("percentage", 0),
            "strictness": result.get("_meta", {}).get("strictness", "medium"),
            "overall_remarks": result.get("overall_remarks", ""),
            "result_json": json.dumps(result, ensure_ascii=False),
        }).execute()
        logger.info(f"Saved smart assignment check {check_id}")
    except Exception as e:
        logger.warning(f"Could not save (non-fatal): {e}")