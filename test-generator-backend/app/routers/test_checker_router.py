"""
Test Checker API Router — v1.1

v1.1 changes:
  - /check now accepts answer_key_file (PDF/image) instead of JSON string
  - New /extract-key endpoint: upload answer key PDF → get structured JSON back
  - answer_key JSON field kept as optional fallback (if teacher already has JSON)

Endpoints:
  POST /check              Upload answer sheet + answer key (both PDF/image) → grading
  POST /check-with-test    Upload answer sheet, pull answer key from saved test_id
  POST /extract-key        Upload answer key PDF → get extracted JSON (preview before grading)
  GET  /results/{check_id} Retrieve a saved grading result
  GET  /history            List grading history for a teacher
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
import json
import logging
import uuid
import tempfile
import os
import time

from app.core.database import get_supabase
from app.services.test_checker_service import (
    grade_answer_sheet,
    extract_answer_key_from_pdf,
    build_answer_key_from_test,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/test-checker", tags=["Test Checker"])

# Max upload size: 20 MB (Gemini supports up to 20 MB inline)
MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}


# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _validate_file(file: UploadFile) -> str:
    """Validate uploaded file type. Returns the extension."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    return ext


async def _save_temp_file(file: UploadFile, ext: str) -> str:
    """
    Stream uploaded file to a temp path. Returns the path.
    Caller is responsible for cleanup.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix="checker_")
    try:
        total = 0
        while chunk := await file.read(1024 * 256):  # 256 KB chunks
            total += len(chunk)
            if total > MAX_FILE_SIZE:
                tmp.close()
                os.unlink(tmp.name)
                raise HTTPException(status_code=413, detail="File too large. Maximum 20 MB.")
            tmp.write(chunk)
        tmp.close()
        return tmp.name
    except HTTPException:
        raise
    except Exception as e:
        tmp.close()
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")


def _parse_answer_key_json(raw: str) -> dict:
    """Parse and validate the answer key JSON string (fallback path)."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid answer key JSON.")

    if isinstance(data, list):
        data = {"questions": data, "total_marks": sum(q.get("marks", 1) for q in data)}

    if "questions" not in data or not data["questions"]:
        raise HTTPException(status_code=400, detail="Answer key must contain a 'questions' list.")

    for idx, q in enumerate(data["questions"]):
        if "q_no" not in q and "question_number" not in q:
            q["q_no"] = idx + 1

    return data


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Extract answer key from PDF (standalone preview)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/extract-key")
async def extract_key(
    file: UploadFile = File(..., description="Answer key PDF or image"),
):
    """
    Upload an answer key PDF/image → get structured JSON back.
    Useful for teacher to preview/verify the extracted key before grading.
    """
    ext = _validate_file(file)
    tmp_path = None

    try:
        tmp_path = await _save_temp_file(file, ext)
        logger.info(f"Extracting answer key from: {file.filename}")

        start = time.time()
        key_data = extract_answer_key_from_pdf(tmp_path)
        elapsed = round(time.time() - start, 2)

        return {
            "ok": True,
            "extraction_time_seconds": elapsed,
            "answer_key": key_data,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Extract key error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Check — both files uploaded (answer sheet + answer key)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/check")
async def check_answer_sheet(
    file: UploadFile = File(..., description="Student answer sheet (PDF or image)"),
    answer_key_file: Optional[UploadFile] = File(None, description="Answer key PDF/image (Gemini extracts it)"),
    answer_key_json: Optional[str] = Form(None, description="Answer key as JSON string (fallback if no file)"),
    strictness: str = Form("medium", description="easy | medium | hard | extreme"),
    teacher_id: str = Form("", description="Teacher UUID for saving results"),
    student_name: str = Form("", description="Student name (optional, for records)"),
):
    """
    Upload student answer sheet + answer key (PDF or JSON) → get graded result.

    Priority:
      1. answer_key_file (PDF/image) → extracted via Gemini automatically
      2. answer_key_json (JSON string) → used directly as fallback
    """
    ext = _validate_file(file)

    # ── Resolve answer key ─────────────────────────────────────────
    key_data = None
    key_tmp_path = None

    if answer_key_file and answer_key_file.filename:
        # Path 1: Answer key is a PDF/image — extract via Gemini
        key_ext = _validate_file(answer_key_file)
        try:
            key_tmp_path = await _save_temp_file(answer_key_file, key_ext)
            logger.info(f"Extracting answer key from uploaded file: {answer_key_file.filename}")
            key_data = extract_answer_key_from_pdf(key_tmp_path)
            logger.info(f"Answer key extracted: {key_data.get('total_questions', '?')} questions, "
                        f"{key_data.get('total_marks', '?')} marks")
        except Exception as e:
            logger.error(f"Answer key extraction failed: {e}")
            raise HTTPException(
                status_code=422,
                detail=f"Could not read the answer key file: {str(e)}. "
                       "Make sure it's a clear, readable PDF or image.",
            )
        finally:
            if key_tmp_path and os.path.exists(key_tmp_path):
                os.unlink(key_tmp_path)

    elif answer_key_json:
        # Path 2: JSON string fallback
        key_data = _parse_answer_key_json(answer_key_json)

    else:
        raise HTTPException(
            status_code=400,
            detail="Please upload an answer key file (PDF/image) or provide answer key JSON.",
        )

    # ── Grade the answer sheet ─────────────────────────────────────
    sheet_tmp_path = None
    try:
        sheet_tmp_path = await _save_temp_file(file, ext)
        logger.info(f"Grading: file={file.filename}, strictness={strictness}, "
                     f"questions={len(key_data.get('questions', []))}")

        start = time.time()
        result = grade_answer_sheet(sheet_tmp_path, key_data, strictness)
        elapsed = round(time.time() - start, 2)

        # Save to DB if teacher_id provided
        check_id = str(uuid.uuid4())
        if teacher_id:
            _save_result_to_db(check_id, teacher_id, student_name, file.filename, result, key_data)

        return {
            "ok": True,
            "check_id": check_id,
            "grading_time_seconds": elapsed,
            "answer_key_summary": {
                "total_questions": key_data.get("total_questions", len(key_data.get("questions", []))),
                "total_marks": key_data.get("total_marks", 0),
                "extraction_notes": key_data.get("extraction_notes", ""),
            },
            "result": result,
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Check error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")
    finally:
        if sheet_tmp_path and os.path.exists(sheet_tmp_path):
            os.unlink(sheet_tmp_path)


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Check with saved test_id as answer key source
# ═══════════════════════════════════════════════════════════════════════

@router.post("/check-with-test")
async def check_with_saved_test(
    file: UploadFile = File(..., description="Student answer sheet (PDF or image)"),
    test_id: str = Form(..., description="Saved test UUID to use as answer key"),
    strictness: str = Form("medium"),
    teacher_id: str = Form(""),
    student_name: str = Form(""),
):
    """
    Upload answer sheet + reference a saved test_id — answer key is auto-pulled
    from the test's questions in the DB.
    """
    ext = _validate_file(file)
    supabase = get_supabase()

    try:
        test_result = supabase.table("tests").select("*").eq("id", test_id).maybeSingle().execute()
        if not test_result.data:
            raise HTTPException(status_code=404, detail=f"Test {test_id} not found.")

        questions_result = (
            supabase.table("questions")
            .select("*")
            .eq("test_id", test_id)
            .order("position")
            .execute()
        )
        if not questions_result.data:
            raise HTTPException(status_code=404, detail="No questions found for this test.")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"DB fetch error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch test data.")

    key_data = build_answer_key_from_test(test_result.data, questions_result.data)
    tmp_path = None

    try:
        tmp_path = await _save_temp_file(file, ext)
        logger.info(f"Grading against test {test_id}: {len(key_data['questions'])} questions")

        start = time.time()
        result = grade_answer_sheet(tmp_path, key_data, strictness)
        elapsed = round(time.time() - start, 2)

        check_id = str(uuid.uuid4())
        if teacher_id:
            _save_result_to_db(
                check_id, teacher_id, student_name, file.filename,
                result, key_data, test_id=test_id,
            )

        return {
            "ok": True,
            "check_id": check_id,
            "test_id": test_id,
            "exam_title": test_result.data.get("exam_title", ""),
            "grading_time_seconds": elapsed,
            "result": result,
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Check-with-test error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Grading failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Get saved result
# ═══════════════════════════════════════════════════════════════════════

@router.get("/results/{check_id}")
async def get_result(check_id: str, teacher_id: str):
    supabase = get_supabase()
    try:
        result = (
            supabase.table("answer_checks")
            .select("*")
            .eq("id", check_id)
            .eq("teacher_id", teacher_id)
            .maybeSingle()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Result not found.")
        return {"ok": True, "data": result.data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get result error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch result.")


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Grading history
# ═══════════════════════════════════════════════════════════════════════

@router.get("/history")
async def get_history(teacher_id: str, limit: int = 20, offset: int = 0):
    supabase = get_supabase()
    try:
        result = (
            supabase.table("answer_checks")
            .select("id, student_name, file_name, total_obtained, total_possible, percentage, strictness, created_at")
            .eq("teacher_id", teacher_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return {"ok": True, "checks": result.data or [], "count": len(result.data or [])}
    except Exception as e:
        logger.error(f"History error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch history.")


# ═══════════════════════════════════════════════════════════════════════
# DB PERSISTENCE HELPER
# ═══════════════════════════════════════════════════════════════════════

def _save_result_to_db(
    check_id: str,
    teacher_id: str,
    student_name: str,
    file_name: str,
    result: dict,
    answer_key: dict,
    test_id: Optional[str] = None,
):
    """Save grading result to answer_checks table. Non-fatal on failure."""
    try:
        supabase = get_supabase()
        supabase.table("answer_checks").insert({
            "id": check_id,
            "teacher_id": teacher_id,
            "test_id": test_id,
            "student_name": student_name or None,
            "file_name": file_name,
            "total_obtained": result.get("total_marks_obtained", 0),
            "total_possible": result.get("total_marks_possible", 0),
            "percentage": result.get("percentage", 0),
            "strictness": result.get("_meta", {}).get("strictness", "medium"),
            "readability_score": result.get("readability_score"),
            "overall_remarks": result.get("overall_remarks"),
            "result_json": json.dumps(result, ensure_ascii=False),
            "answer_key_json": json.dumps(answer_key, ensure_ascii=False),
        }).execute()
        logger.info(f"Saved check result {check_id}")
    except Exception as e:
        logger.warning(f"Could not save check result (non-fatal): {e}")