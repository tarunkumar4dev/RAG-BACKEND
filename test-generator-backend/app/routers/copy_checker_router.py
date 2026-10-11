"""
Copy Checker Router — "Check Copies" (teacher photographs students' answer sheets)

Rows live in Supabase (check_sessions / check_sheets, owner-only RLS); the frontend creates them and
uploads page photos to the private 'answer-sheets' bucket. The backend only does the AI work:

  POST /copy-checker/sessions/{session_id}/rubric   marking scheme for the session, built once
        source 'test'   → from the saved test in Test History (questions + answers from the DB)
        source 'upload' → from the uploaded question-paper photos (+ answer key, optional)
  POST /copy-checker/sheets/{sheet_id}/check        AI-grade one student's copy (all its pages)

Every row is read/written through a client bound to the caller's JWT (RLS = owner only). The
service key is used only to download files, and only paths inside the caller's own folder.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
import logging
import os
import tempfile
import time
import uuid

from app.core.database import get_supabase_admin
from app.routers.smart_assignment_router import (
    ALLOWED_EXTENSIONS, DAILY_LIMIT, MAX_FILE_SIZE, _Caller, _checks_today, _cleanup, _save_to_db,
    require_caller,
)
from app.services.smart_assignment_service import build_rubric, grade_sheet, rubric_from_test_fast

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/copy-checker", tags=["Copy Checker"])

BUCKET = "answer-sheets"
MAX_PAGES = int(os.getenv("COPY_CHECKER_MAX_PAGES", "16"))


def _uuid_or_404(value: str, what: str) -> None:
    try:
        uuid.UUID(value)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"{what} not found.")


def _download_own(c: _Caller, path: str, prefix: str) -> str:
    """Download a file from the private bucket — only from the caller's own folder."""
    if not path or not path.startswith(f"{c.user.id}/") or ".." in path.split("/"):
        raise HTTPException(status_code=400, detail="Invalid file path.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")
    data = get_supabase_admin().storage.from_(BUCKET).download(path)
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 20 MB.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix=prefix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _get_session(c: _Caller, session_id: str) -> dict:
    r = c.db.table("check_sessions").select("*").eq("id", session_id).limit(1).execute()
    if not r.data:
        raise HTTPException(status_code=404, detail="Check session not found.")
    return r.data[0]


def _update_session(c: _Caller, session_id: str, fields: dict) -> dict:
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    r = c.db.table("check_sessions").update(fields).eq("id", session_id).execute()
    return r.data[0] if r.data else {}


def _rubric_summary(s: dict) -> dict:
    rubric = s.get("rubric") or {}
    return {
        "status": s.get("rubric_status"),
        "error": s.get("rubric_error"),
        "total_marks": rubric.get("total_marks"),
        "total_questions": rubric.get("total_questions"),
        "marks_mismatch": rubric.get("marks_mismatch"),
        "estimated_questions": [q["q_no"] for q in rubric.get("questions", []) if q.get("scheme_source") != "key"],
    }


def _build_session_rubric(c: _Caller, session_id: str, force: bool) -> dict:
    s = _get_session(c, session_id)
    if s.get("rubric_status") == "ready" and s.get("rubric") and not force:
        return s

    _update_session(c, session_id, {"rubric_status": "building", "rubric_error": None})
    paths = []
    try:
        if s["source"] == "test":
            if not s.get("test_id"):
                raise HTTPException(status_code=400, detail="This session has no test.")
            admin = get_supabase_admin()
            # Only the caller's own tests (same rule as the old checker's /check-with-test)
            t = admin.table("tests").select("*").eq("id", s["test_id"]).eq("teacher_id", c.user.id) \
                .limit(1).execute()
            if not t.data:
                raise HTTPException(status_code=404, detail="Test not found in your Test History.")
            qs = admin.table("questions").select("*").eq("test_id", s["test_id"]).order("position").execute()
            if not qs.data:
                raise HTTPException(status_code=400, detail="This test has no saved questions.")
            rubric = rubric_from_test_fast(t.data[0], qs.data)
        else:
            if not s.get("qp_paths"):
                raise HTTPException(status_code=400, detail="Upload the question paper first.")
            qp = [_download_own(c, p, "qpaper_") for p in s["qp_paths"][:MAX_PAGES]]
            key = [_download_own(c, p, "akey_") for p in (s.get("key_paths") or [])[:MAX_PAGES]]
            paths = qp + key
            rubric = build_rubric(qp, key)
    except HTTPException as e:
        _update_session(c, session_id, {"rubric_status": "failed", "rubric_error": str(e.detail)[:300]})
        raise
    except Exception as e:
        _update_session(c, session_id, {"rubric_status": "failed", "rubric_error": str(e)[:300]})
        raise
    finally:
        _cleanup(*paths)

    rubric = {k: v for k, v in rubric.items() if not k.startswith("_")}
    return _update_session(c, session_id, {
        "rubric": rubric, "rubric_status": "ready", "rubric_error": None,
        "total_marks": rubric.get("total_marks"),
    })


class RubricRequest(BaseModel):
    force: bool = False


@router.post("/sessions/{session_id}/rubric")
async def session_rubric(session_id: str, body: RubricRequest = RubricRequest(),
                         c: _Caller = Depends(require_caller)):
    _uuid_or_404(session_id, "Check session")
    try:
        s = await run_in_threadpool(_build_session_rubric, c, session_id, body.force)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Could not read the question paper: {e}")
    except Exception as e:
        logger.error(f"Copy-check rubric failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not prepare the marking scheme. Please try again.")
    return {"ok": True, "rubric": _rubric_summary(s), "full_rubric": s.get("rubric")}


def _compact(result: dict) -> dict:
    keep = ("graded_questions", "total_marks_awarded", "max_marks", "percentage", "overall_remarks",
            "readability_score", "needs_review_count")
    return {k: result.get(k) for k in keep}


def _check_sheet(c: _Caller, sheet_id: str) -> dict:
    r = c.db.table("check_sheets").select("*").eq("id", sheet_id).limit(1).execute()
    if not r.data:
        raise HTTPException(status_code=404, detail="Student copy not found.")
    sheet = r.data[0]
    pages = sheet.get("pages") or []
    if not pages:
        raise HTTPException(status_code=400, detail="Add at least one page photo first.")
    if len(pages) > MAX_PAGES:
        raise HTTPException(status_code=400, detail=f"Too many pages (max {MAX_PAGES}).")

    session = _get_session(c, sheet["session_id"])
    if session.get("rubric_status") != "ready" or not session.get("rubric"):
        session = _build_session_rubric(c, sheet["session_id"], False)

    c.db.table("check_sheets").update({"status": "checking", "error": None}).eq("id", sheet_id).execute()
    files = []
    try:
        files = [_download_own(c, p, "ssheet_") for p in pages]
        start = time.time()
        result = grade_sheet(files, session["rubric"], session.get("strictness") or "medium",
                             require_identity=False)
        result.pop("_grade_time_s", None)
        result["_meta"] = {"strictness": session.get("strictness") or "medium",
                           "grade_time_s": round(time.time() - start, 2), "pages": len(pages)}
    except Exception as e:
        msg = e.detail if isinstance(e, HTTPException) else str(e)
        c.db.table("check_sheets").update({"status": "failed", "error": str(msg)[:300]}).eq("id", sheet_id).execute()
        raise
    finally:
        _cleanup(*files)

    compact = _compact(result)
    now = datetime.now(timezone.utc).isoformat()
    upd = c.db.table("check_sheets").update({
        "status": "checked", "ai_result": compact,
        "ai_marks": result.get("total_marks_awarded", 0), "max_marks": result.get("max_marks", 0),
        "error": None, "checked_at": now, "updated_at": now,
    }).eq("id", sheet_id).execute()

    result["student_info"] = {"name": sheet.get("student_name") or "", "class": session.get("class_grade") or "",
                              "roll_no": sheet.get("roll_no") or "", "missing_fields": []}
    _save_to_db(str(uuid.uuid4()), c.user.id, result, session.get("class_grade") or "",
                session.get("subject") or session.get("title") or "", session.get("title") or "",
                f"{len(pages)} page(s)", check_sheet_id=sheet_id)
    return upd.data[0] if upd.data else {"id": sheet_id, "status": "checked", "ai_result": compact}


@router.post("/sheets/{sheet_id}/check")
async def check_sheet(sheet_id: str, c: _Caller = Depends(require_caller)):
    _uuid_or_404(sheet_id, "Student copy")
    try:
        used = await run_in_threadpool(_checks_today, c.user.id)
    except Exception as e:
        logger.warning(f"Daily-limit check failed (allowing): {e}")
        used = 0
    if used >= DAILY_LIMIT:
        raise HTTPException(status_code=429, detail=f"Daily limit reached ({DAILY_LIMIT} copies). Try again tomorrow.")
    try:
        sheet = await run_in_threadpool(_check_sheet, c, sheet_id)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Copy check failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="AI check failed. Please try again.")
    return {"ok": True, "sheet": sheet}
