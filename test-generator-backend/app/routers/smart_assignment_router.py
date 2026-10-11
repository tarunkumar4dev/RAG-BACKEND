"""
Smart Assignment Checker Router — v2.0

Endpoints (all require a Supabase access token; the owner is always the JWT user):
  POST /smart-assignment-checker/check    Upload 3 files → grading result (saved to history)
  GET  /smart-assignment-checker/history  The caller's past checks
  GET  /smart-assignment-checker/result/{id}

Institute assignments (portal submissions). Access is decided by Supabase RLS: every read/write of
assignments / submissions / assignment_ai_keys goes through a client bound to the caller's JWT.
The service key is only used to download files after that check passes.
  POST /smart-assignment-checker/assignments/{assignment_id}/rubric
        Build (or reuse) the marking scheme from the assignment's question paper + private answer key
  POST /smart-assignment-checker/assignments/{assignment_id}/submissions/{submission_id}/check
        AI-grade one submission → submissions.ai_* (teacher approves later; student sees nothing yet)
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from urllib.parse import unquote, urlparse
import json
import logging
import uuid
import tempfile
import os
import time

from app.core.auth import AuthUser, require_user
from app.core.config import settings
from app.core.database import get_supabase_admin, get_supabase_for_user
from app.services.smart_assignment_service import build_rubric, check_assignment, grade_sheet

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


_TRANSIENT = ("Server disconnected", "ConnectionReset", "Connection reset", "RemoteProtocolError",
              "ReadError", "ReadTimeout", "ConnectError", "timed out")


def with_retry(fn, attempts: int = 3):
    """Supabase calls through a pooled HTTP/2 connection fail with "Server disconnected" when the
    server closed it while idle (e.g. during a 20 s Gemini call). Retry those, nothing else."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:
            text = f"{type(e).__name__}: {e}"
            if i == attempts - 1 or not any(t in text for t in _TRANSIENT):
                raise
            logger.warning(f"Transient Supabase error, retrying ({i + 1}/{attempts - 1}): {text[:120]}")
            time.sleep(0.4 * (i + 1))


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
    assignment_id: str = None, submission_id: str = None, check_sheet_id: str = None,
):
    try:
        student_info = result.get("student_info", {})
        row = {
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
        }
        if assignment_id:
            row["assignment_id"] = assignment_id
            row["submission_id"] = submission_id
        if check_sheet_id:
            row["check_sheet_id"] = check_sheet_id
        get_supabase_admin().table(TABLE).insert(row).execute()
        logger.info(f"Saved smart assignment check {check_id}")
    except Exception as e:
        logger.warning(f"Could not save (non-fatal): {e}")


# ═══════════════════════════════════════════════════════════════════════
# INSTITUTE ASSIGNMENTS — portal submissions
# ═══════════════════════════════════════════════════════════════════════

class _Caller:
    def __init__(self, user: AuthUser, token: str):
        self.user = user
        self.token = token
        self.db = get_supabase_for_user(token)  # RLS = the caller's permissions


async def require_caller(request: Request) -> _Caller:
    user = await require_user(request)
    token = request.headers.get("authorization", "").partition(" ")[2].strip()
    return _Caller(user, token)


def _storage_path(url: str, bucket: str) -> str:
    """Object path from a public URL of THIS project's bucket. Anything else is rejected (no SSRF)."""
    if not url:
        raise HTTPException(status_code=400, detail="File is missing.")
    u = urlparse(url)
    if u.netloc != urlparse(settings.SUPABASE_URL).netloc:
        raise HTTPException(status_code=400, detail="File is not stored in this project.")
    prefix = f"/storage/v1/object/public/{bucket}/"
    if not u.path.startswith(prefix):
        raise HTTPException(status_code=400, detail=f"File is not in the '{bucket}' bucket.")
    path = unquote(u.path[len(prefix):])
    if not path or ".." in path.split("/"):
        raise HTTPException(status_code=400, detail="Invalid file path.")
    return path


def _download_to_temp(bucket: str, path: str, prefix: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}' for AI checking.")
    data = with_retry(lambda: get_supabase_admin().storage.from_(bucket).download(path))
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 20 MB.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix=prefix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _get_assignment(c: _Caller, assignment_id: str) -> dict:
    r = (c.db.table("assignments").select("id, title, file_url, file_name, max_marks, batch_id")
         .eq("id", assignment_id).limit(1).execute())
    if not r.data:
        raise HTTPException(status_code=404, detail="Assignment not found.")
    return r.data[0]


def _get_ai_key(c: _Caller, assignment_id: str) -> dict:
    # RLS on assignment_ai_keys = staff who may grade this assignment
    r = (c.db.table("assignment_ai_keys").select("*").eq("assignment_id", assignment_id).limit(1).execute())
    if not r.data:
        raise HTTPException(status_code=400, detail="Attach an answer key to this assignment first (Edit → Answer key).")
    return r.data[0]


def _ensure_rubric(c: _Caller, assignment_id: str, force: bool = False) -> tuple:
    """Returns (assignment, ai_key_row with a fresh rubric). Builds and caches the rubric when needed."""
    asg = _get_assignment(c, assignment_id)
    key = _get_ai_key(c, assignment_id)
    if not asg.get("file_url"):
        raise HTTPException(status_code=400, detail="Attach the question paper PDF to this assignment first.")
    source = f"{asg['file_url']}|{key['answer_key_path']}"
    if key.get("rubric") and key.get("rubric_source") == source and not force:
        return asg, key

    qp_path = ak_path = None
    try:
        qp_path = _download_to_temp("assignments", _storage_path(asg["file_url"], "assignments"), "qpaper_")
        ak_path = _download_to_temp("answer-keys", key["answer_key_path"], "akey_")
        rubric = build_rubric(qp_path, ak_path)
    finally:
        _cleanup(qp_path, ak_path)

    rubric = {k: v for k, v in rubric.items() if not k.startswith("_")}
    upd = (c.db.table("assignment_ai_keys")
           .update({"rubric": rubric, "rubric_source": source,
                    "rubric_built_at": datetime.now(timezone.utc).isoformat()})
           .eq("assignment_id", assignment_id).execute())
    if not upd.data:
        raise HTTPException(status_code=403, detail="You can't check this assignment.")
    key.update(upd.data[0])
    if not asg.get("max_marks") and rubric.get("total_marks"):
        try:  # only the creator / admin may update the assignment — fine if it fails
            c.db.table("assignments").update({"max_marks": int(round(rubric["total_marks"]))}) \
                .eq("id", assignment_id).execute()
        except Exception as e:
            logger.info(f"max_marks not set on assignment {assignment_id[:8]}: {e}")
    return asg, key


def _rubric_summary(key: dict) -> dict:
    rubric = key.get("rubric") or {}
    return {
        "total_marks": rubric.get("total_marks"),
        "total_questions": rubric.get("total_questions"),
        "marks_mismatch": rubric.get("marks_mismatch"),
        "estimated_questions": [q["q_no"] for q in rubric.get("questions", []) if q.get("scheme_source") != "key"],
        "strictness": key.get("strictness", "medium"),
        "built_at": key.get("rubric_built_at"),
    }


class RubricRequest(BaseModel):
    force: bool = False


@router.post("/assignments/{assignment_id}/rubric")
async def assignment_rubric(assignment_id: str, body: RubricRequest = RubricRequest(),
                            c: _Caller = Depends(require_caller)):
    try:
        uuid.UUID(assignment_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Assignment not found.")
    try:
        _, key = await run_in_threadpool(_ensure_rubric, c, assignment_id, body.force)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Could not read the question paper / answer key: {e}")
    except Exception as e:
        logger.error(f"Rubric build failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not build the marking scheme. Please try again.")
    return {"ok": True, "rubric": _rubric_summary(key)}


def _compact(result: dict) -> dict:
    """What the teacher's review screen needs (no rubric copy, no timings)."""
    keep = ("graded_questions", "total_marks_awarded", "max_marks", "percentage", "overall_remarks",
            "readability_score", "needs_review_count", "student_info")
    return {k: result.get(k) for k in keep}


def _check_submission(c: _Caller, assignment_id: str, submission_id: str) -> dict:
    sr = (c.db.table("submissions")
          .select("id, assignment_id, student_id, file_url, file_name, text_answer, status")
          .eq("id", submission_id).eq("assignment_id", assignment_id).limit(1).execute())
    if not sr.data:
        raise HTTPException(status_code=404, detail="Submission not found.")
    sub = sr.data[0]
    if not sub.get("file_url") and not (sub.get("text_answer") or "").strip():
        raise HTTPException(status_code=400, detail="This submission has no file or text to check.")

    asg, key = _ensure_rubric(c, assignment_id)

    # Claim it (also proves the caller may grade: sub_grade RLS)
    claim = (c.db.table("submissions").update({"ai_status": "checking", "ai_error": None})
             .eq("id", submission_id).execute())
    if not claim.data:
        raise HTTPException(status_code=403, detail="You can't grade this submission.")

    sheet_path = None
    try:
        if sub.get("file_url"):
            sheet_path = _download_to_temp("submissions", _storage_path(sub["file_url"], "submissions"), "ssheet_")
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix="ssheet_", mode="w", encoding="utf-8")
            tmp.write(sub["text_answer"][:20000])
            tmp.close()
            sheet_path = tmp.name

        start = time.time()
        result = grade_sheet(sheet_path, key["rubric"], key.get("strictness") or "medium", require_identity=False)
        result["_meta"] = {"strictness": key.get("strictness") or "medium",
                           "grade_time_s": result.pop("_grade_time_s", round(time.time() - start, 2))}
    except Exception as e:
        with_retry(lambda: c.db.table("submissions").update({"ai_status": "failed", "ai_error": str(e)[:300]})
                   .eq("id", submission_id).execute())
        raise
    finally:
        _cleanup(sheet_path)

    compact = _compact(result)
    with_retry(lambda: c.db.table("submissions").update({
        "ai_status": "done",
        "ai_marks": result.get("total_marks_awarded", 0),
        "ai_max_marks": result.get("max_marks", 0),
        "ai_result": compact,
        "ai_error": None,
        "ai_checked_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", submission_id).execute())

    st = with_retry(lambda: c.db.table("students").select("name, roll_no").eq("id", sub["student_id"]).limit(1).execute())
    if st.data:
        result["student_info"] = {"name": st.data[0].get("name") or "", "class": "",
                                  "roll_no": st.data[0].get("roll_no") or "", "missing_fields": []}
    _save_to_db(str(uuid.uuid4()), c.user.id, result, "", asg.get("title") or "",
                asg.get("file_name") or "", sub.get("file_name") or "text answer",
                assignment_id=assignment_id, submission_id=submission_id)
    return compact


@router.post("/assignments/{assignment_id}/submissions/{submission_id}/check")
async def check_submission(assignment_id: str, submission_id: str, c: _Caller = Depends(require_caller)):
    try:
        uuid.UUID(assignment_id)
        uuid.UUID(submission_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Submission not found.")

    try:
        used = await run_in_threadpool(_checks_today, c.user.id)
    except Exception as e:
        logger.warning(f"Daily-limit check failed (allowing): {e}")
        used = 0
    if used >= DAILY_LIMIT:
        raise HTTPException(status_code=429,
                            detail=f"Daily limit reached ({DAILY_LIMIT} sheets). Try again tomorrow.")

    try:
        result = await run_in_threadpool(_check_submission, c, assignment_id, submission_id)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Submission check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="AI check failed. Please try again.")
    return {"ok": True, "result": result}
