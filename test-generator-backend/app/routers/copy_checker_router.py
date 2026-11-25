"""
Copy Checker Router — "Check Copies" (teacher photographs students' answer sheets)

Rows live in Supabase (check_sessions / check_sheets, owner-only RLS); the frontend creates them and
uploads page photos to the private 'answer-sheets' bucket. The backend only does the AI work:

  POST /copy-checker/sessions/{session_id}/rubric   marking scheme for the session, built once
        source 'test'   → from the saved test in Test History (questions + answers from the DB)
        source 'upload' → from the uploaded question-paper photos (+ answer key, optional)
  POST /copy-checker/sheets/{sheet_id}/check        AI-grade one student's copy (all its pages)
  PATCH /copy-checker/sheets/{sheet_id}/questions/{q_no}/marks   teacher overrides one question's marks
        (the AI's marks stay in ai_result for comparison; every override is logged in
        smart_assignment_checks with result_json.type = "teacher_override")

Every row is read/written through a client bound to the caller's JWT (RLS = owner only). The
service key is used only to download files, and only paths inside the caller's own folder.
"""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import logging
import os
import tempfile
import time
import uuid

from app.core.database import get_supabase_admin
from app.routers.smart_assignment_router import (
    ALLOWED_EXTENSIONS, DAILY_LIMIT, MAX_FILE_SIZE, OVERRIDE_TYPE, TABLE as HISTORY_TABLE, _Caller,
    _checks_today, _cleanup, _save_to_db, require_caller, with_retry,
)
from app.services.answer_checker import preprocess
from app.services.answer_checker.copy_check import grade_copy
from app.core.api_errors import AI_BUSY_MSG, ai_busy
from app.services.answer_checker.usage_cost import (
    add_usage, answering_model, emit_usage_event, has_tokens, rubric_build_record, without_build,
)
from app.services.smart_assignment_service import (
    CHECKER_MODEL, GRADE_CALL_TIMEOUT_S, RUBRIC_CALL_TIMEOUT_S, GeminiBusy, build_rubric, call_settings, is_timeout,
    request_deadline, rubric_from_test_fast, usage_meter,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/copy-checker", tags=["Copy Checker"])

BUCKET = "answer-sheets"
MAX_PAGES = preprocess.MAX_PAGES   # one place for the page cap


# What teachers see when the AI step fails for a reason that isn't theirs to fix (Gemini errors,
# timeouts, parsing). The technical cause goes to the log only.
RUBRIC_FAILED_MSG = "Could not prepare the marking scheme. Please try again."
PAPER_UNREADABLE_MSG = "Could not read the question paper. Check that the photos are clear and try again."
CHECK_FAILED_MSG = "AI check failed. Please try again."
COPY_UNREADABLE_MSG = "Could not read this copy. Check that the photos are clear and try again."
BUSY_MSG = AI_BUSY_MSG


def _uuid_or_404(value: str, what: str) -> None:
    try:
        uuid.UUID(value)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"{what} not found.")


def _download_own(c: _Caller, path: str, prefix: str, bucket: str = BUCKET) -> str:
    """Download a file from a private bucket — only from the caller's own folder."""
    if not path or not path.startswith(f"{c.user.id}/") or ".." in path.split("/"):
        raise HTTPException(status_code=400, detail="Invalid file path.")
    ext = os.path.splitext(path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'.")
    data = with_retry(lambda: get_supabase_admin().storage.from_(bucket).download(path))
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 20 MB.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix=prefix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _get_session(c: _Caller, session_id: str) -> dict:
    r = with_retry(lambda: c.db.table("check_sessions").select("*").eq("id", session_id).limit(1).execute())
    if not r.data:
        raise HTTPException(status_code=404, detail="Check session not found.")
    return r.data[0]


def _update_session(c: _Caller, session_id: str, fields: dict) -> dict:
    fields["updated_at"] = datetime.now(timezone.utc).isoformat()
    r = with_retry(lambda: c.db.table("check_sessions").update(fields).eq("id", session_id).execute())
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


def _db_connect():
    """Direct Postgres connection factory, for writes PostgREST can't do. Replaced in tests."""
    from app.core.db_pool import get_db_connection
    return get_db_connection


def _rubric_store():
    """Direct Postgres for the one write PostgREST cannot do (jsonb_set on a single key). Replaced in tests."""
    from app.core.db_pool import get_db_connection
    from app.services.job_store import JobStore
    return JobStore(get_db_connection)


def _record_failed_build(session_id: str, spent: dict) -> None:
    """A rebuild failed after billed replies: add that attempt to rubric.build, touching nothing else of the
    rubric (see JobStore.record_rubric_build). Never raises: the build's own error is what the caller must see.
    Without a rubric to attach it to (a first build) or without a database, the cost goes to the log only."""
    try:
        if _rubric_store().record_rubric_build(session_id, lambda b: rubric_build_record(spent, b, "sync")):
            return
        why = "the session has no rubric to attach it to"
    except Exception as e:
        why = f"recording it failed ({type(e).__name__})"
    logger.warning(f"[checker] rubric build failed after billed calls, usage not stored: {why}: {add_usage({}, spent)}")


def _build_session_rubric(c: _Caller, session_id: str, force: bool, deadline: float | None = None) -> dict:
    """deadline: time.monotonic() by which this request must be done (request_deadline() at its start); a copy
    check that has to build the rubric first passes its own, so rubric + grade share one ~280 s budget."""
    deadline = deadline or request_deadline()
    s = _get_session(c, session_id)
    if s.get("rubric_status") == "ready" and s.get("rubric") and not force:
        return s

    _update_session(c, session_id, {"rubric_status": "building", "rubric_error": None})
    paths = []
    meter = None          # set when Gemini was asked; its counts are kept even if the build then fails
    prior = s.get("rubric") or {}
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
                raise HTTPException(status_code=400, detail="This test's questions were not saved, so its answers "
                                    "aren't available. Start a new check with “Photo of paper” instead.")
            rubric = rubric_from_test_fast(t.data[0], qs.data)
        else:
            if not s.get("qp_paths"):
                raise HTTPException(status_code=400, detail="Upload the question paper first.")
            qp = [_download_own(c, p, "qpaper_") for p in s["qp_paths"][:MAX_PAGES]]
            key = [_download_own(c, p, "akey_") for p in (s.get("key_paths") or [])[:MAX_PAGES]]
            paths = qp + key
            # the stage timeout, not the global 60 s: a rubric call takes ~75 s on a 30-mark paper
            with usage_meter() as meter, call_settings(timeout_s=RUBRIC_CALL_TIMEOUT_S, deadline=deadline):
                rubric = build_rubric(qp, key)
            logger.info(f"[checker] rubric usage: {meter.as_dict()}")
    except HTTPException as e:
        _update_session(c, session_id, {"rubric_status": "failed", "rubric_error": str(e.detail)[:300]})
        raise
    except Exception as e:
        logger.warning(f"Rubric failed for session {session_id}: {type(e).__name__}: {str(e)[:600]}")
        msg = (BUSY_MSG if isinstance(e, GeminiBusy) or is_timeout(e) else
               PAPER_UNREADABLE_MSG if isinstance(e, ValueError) else RUBRIC_FAILED_MSG)
        spent = meter.as_dict() if meter is not None else None
        if has_tokens(spent):                  # replies came back (billed) but the build still failed
            _record_failed_build(session_id, spent)
        emit_usage_event(_db_connect(), kind="rubric", route="sync", ok=False, usage=spent,
                         session_id=session_id, error_code=type(e).__name__)
        _update_session(c, session_id, {"rubric_status": "failed", "rubric_error": msg})
        raise
    finally:
        _cleanup(*paths)

    rubric = {k: v for k, v in rubric.items() if not k.startswith("_")}
    if meter is not None:                      # what building it cost, all attempts of this session so far
        rubric["build"] = rubric_build_record(meter.as_dict(), prior.get("build"), "sync")
        emit_usage_event(_db_connect(), kind="rubric", route="sync", ok=True,
                         usage=meter.as_dict(), session_id=session_id)
    elif prior.get("build"):                   # rebuilt from the saved test (no Gemini call): keep the history
        rubric["build"] = prior["build"]
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
    except GeminiBusy:
        raise ai_busy()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=PAPER_UNREADABLE_MSG)
    except Exception as e:
        logger.error(f"Copy-check rubric failed: {type(e).__name__}: {str(e)[:600]}")
        if is_timeout(e):
            raise ai_busy()
        raise HTTPException(status_code=500, detail=RUBRIC_FAILED_MSG)
    return {"ok": True, "rubric": _rubric_summary(s), "full_rubric": without_build(s.get("rubric"))}   # no cost data


def _compact(result: dict) -> dict:
    keep = ("graded_questions", "total_marks_awarded", "max_marks", "percentage", "overall_remarks",
            "readability_score", "needs_review_count", "norm_pages", "sheet_flags", "usage")
    return {k: result.get(k) for k in keep if k in result}


def _save_norm_pages(c: _Caller, session_id: str, sheet_id: str, ordered: list) -> list:
    """Final (upright, student-ordered) pages → answer-sheets/<uid>/<session>/<sheet>/norm/p<i>.jpg.
    The marked copy is drawn on exactly these images, so its positions match the grader's."""
    folder = f"{c.user.id}/{session_id}/{sheet_id}/norm"
    store = get_supabase_admin().storage.from_(BUCKET)
    old = {f"{folder}/{o['name']}" for o in (with_retry(lambda: store.list(folder)) or []) if o.get("name")}
    out = []
    for i, p in enumerate(ordered):
        path = f"{folder}/p{i}.jpg"
        with_retry(lambda: store.upload(path, p.jpeg, {"content-type": "image/jpeg", "upsert": "true"}))
        out.append({"index": i, "source_index": p.source_index, "student_page_no": p.student_page_no,
                    "rotation": p.rotation, "blank": p.blank, "path": path, "w": p.w, "h": p.h})
    stale = sorted(old - {o["path"] for o in out})
    if stale:
        with_retry(lambda: store.remove(stale))
    return out


def _check_sheet(c: _Caller, sheet_id: str) -> dict:
    # One budget for the whole request (Vercel kills it at 300 s): an inline rubric (≤ 180 s) + the grade (150 s)
    # could need 330 s, more with retries. Every Gemini call checks this deadline before it starts.
    deadline = request_deadline()
    r = with_retry(lambda: c.db.table("check_sheets").select("*").eq("id", sheet_id).limit(1).execute())
    if not r.data:
        raise HTTPException(status_code=404, detail="Student copy not found.")
    sheet = r.data[0]
    pages = sheet.get("pages") or []
    if not pages:
        raise HTTPException(status_code=400, detail="Add at least one page photo first.")
    if len(pages) > MAX_PAGES:
        raise HTTPException(status_code=400, detail=f"Too many pages in one copy, max {MAX_PAGES}.")

    session = _get_session(c, sheet["session_id"])
    if session.get("rubric_status") != "ready" or not session.get("rubric"):
        session = _build_session_rubric(c, sheet["session_id"], False, deadline)

    # updated_at = when checking started: the app treats a copy still "checking" after 6 min as failed
    # (the request died, e.g. the 300 s function limit) and offers Retry instead of a locked copy.
    started = datetime.now(timezone.utc).isoformat()
    with_retry(lambda: c.db.table("check_sheets").update({"status": "checking", "error": None, "updated_at": started})
               .eq("id", sheet_id).execute())
    files = []
    usage = None
    try:
        files = [_download_own(c, p, "ssheet_") for p in pages]
        start = time.time()
        with call_settings(timeout_s=GRADE_CALL_TIMEOUT_S, deadline=deadline):
            result, ordered, usage = grade_copy(files, session["rubric"], session.get("strictness") or "medium")
        result["norm_pages"] = _save_norm_pages(c, sheet["session_id"], sheet_id, ordered)
        result["usage"] = usage
        if usage.get("fallback_used"):        # never silent: the teacher sees which model marked the copy
            result["sheet_flags"] = (result.get("sheet_flags") or []) + [{
                "code": "FALLBACK_MODEL", "page": None,
                "message": f"Checked with a backup AI model ({usage.get('model')}). Please check the marks."}]
        result["_meta"] = {"strictness": session.get("strictness") or "medium",
                           "grade_time_s": round(time.time() - start, 2), "pages": usage["pages"]}
        logger.info(f"[checker] sheet usage: {usage}")
    except Exception as e:
        if isinstance(e, HTTPException):
            msg = e.detail
        elif isinstance(e, preprocess.BadFile):
            msg = str(e)
        elif isinstance(e, GeminiBusy) or is_timeout(e):
            msg = BUSY_MSG
        else:
            logger.warning(f"Copy check failed for sheet {sheet_id}: {type(e).__name__}")
            msg = COPY_UNREADABLE_MSG if isinstance(e, ValueError) else CHECK_FAILED_MSG
        spent = getattr(e, "usage", None) or usage          # grade_copy attaches the failed run's counts
        if has_tokens(spent):    # billed replies, no result: nowhere to store it without a usage column
            logger.warning(f"[checker] failed attempt on sheet {str(sheet_id)[:8]} still cost: {add_usage({}, spent)}")
        emit_usage_event(_db_connect(), kind="copy", route="sync", ok=False, usage=spent,
                         session_id=sheet["session_id"], sheet_id=sheet_id,
                         error_code=type(e).__name__)
        with_retry(lambda: c.db.table("check_sheets").update({"status": "failed", "error": str(msg)[:300],
                                                             "updated_at": datetime.now(timezone.utc).isoformat()})
                   .eq("id", sheet_id).execute())
        raise
    finally:
        _cleanup(*files)

    compact = _compact(result)
    now = datetime.now(timezone.utc).isoformat()
    upd = with_retry(lambda: c.db.table("check_sheets").update({
        "status": "checked", "ai_result": compact, "model_version": answering_model(usage, CHECKER_MODEL),
        "ai_marks": result.get("total_marks_awarded", 0), "max_marks": result.get("max_marks", 0),
        "error": None, "checked_at": now, "updated_at": now,
    }).eq("id", sheet_id).execute())
    emit_usage_event(_db_connect(), kind="copy", route="sync", ok=True, usage=usage,
                     session_id=sheet["session_id"], sheet_id=sheet_id)

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
    except preprocess.BadFile as e:
        raise HTTPException(status_code=422, detail=str(e))
    except GeminiBusy:
        raise ai_busy()
    except ValueError:
        raise HTTPException(status_code=422, detail=COPY_UNREADABLE_MSG)
    except Exception as e:
        logger.error(f"Copy check failed: {type(e).__name__}", exc_info=True)
        if is_timeout(e):
            raise ai_busy()
        raise HTTPException(status_code=500, detail=CHECK_FAILED_MSG)
    return {"ok": True, "sheet": sheet}


# ═══════════════════════════════════════════════════════════════════════
# TEACHER OVERRIDE — one question's marks
# ═══════════════════════════════════════════════════════════════════════

class MarksOverride(BaseModel):
    marks_awarded: float = Field(ge=0, le=1000)
    teacher_note: str | None = Field(default=None, max_length=500)


def _effective(gq: dict) -> float:
    """Teacher's marks when overridden, else the AI's."""
    t = gq.get("teacher_marks")
    return float(t) if t is not None else float(gq.get("marks_awarded") or 0)


def _apply_overrides(result: dict, rubric: dict | None) -> float:
    """Re-decides which OR-alternatives count (best `attempt_count` by effective marks, same rule as
    the AI scoring) and returns the sheet total with teacher overrides applied."""
    graded = result.get("graded_questions") or []
    attempt = {str(q.get("q_no")): int(q.get("attempt_count") or 0) or 1 for q in (rubric or {}).get("questions", [])}
    groups: dict = {}
    for gq in graded:
        gq["counted"] = True
        if gq.get("or_group"):
            groups.setdefault(gq["or_group"], []).append(gq)
    for members in groups.values():
        need = attempt.get(str(members[0].get("q_no")), 1)
        for gq in sorted(members, key=_effective, reverse=True)[need:]:
            gq["counted"] = False
    total = sum(_effective(gq) for gq in graded if gq.get("counted"))
    max_marks = float(result.get("max_marks") or 0)
    return round(min(total, max_marks) if max_marks else total, 2)


def _override_marks(c: _Caller, sheet_id: str, q_no: str, body: MarksOverride) -> dict:
    # RLS on check_sheets / check_sessions = owner only: someone else's sheet is simply "not found"
    r = with_retry(lambda: c.db.table("check_sheets").select("*").eq("id", sheet_id).limit(1).execute())
    if not r.data:
        raise HTTPException(status_code=404, detail="Student copy not found.")
    sheet = r.data[0]
    result = sheet.get("ai_result") or {}
    if sheet.get("status") not in ("checked", "approved") or not result.get("graded_questions"):
        raise HTTPException(status_code=409, detail="This copy has no AI result to change yet.")
    gq = next((g for g in result["graded_questions"] if str(g.get("q_no")) == q_no), None)
    if gq is None:
        raise HTTPException(status_code=404, detail=f"Question {q_no} is not in this copy's result.")
    max_q = float(gq.get("max_marks") or 0)
    new = round(float(body.marks_awarded), 2)
    if new > max_q:
        raise HTTPException(status_code=422, detail=f"Q{q_no} is out of {max_q:g} marks.")

    session = _get_session(c, sheet["session_id"])
    before_q = _effective(gq)
    before_total = _apply_overrides(result, session.get("rubric"))
    note = (body.teacher_note or "").strip() or None
    if abs(new - float(gq.get("marks_awarded") or 0)) < 1e-9 and not note:
        for k in ("teacher_marks", "teacher_note", "overridden_at"):   # back to the AI's marks
            gq.pop(k, None)
    else:
        gq["teacher_marks"] = new
        gq["teacher_note"] = note
        gq["overridden_at"] = datetime.now(timezone.utc).isoformat()
    total = _apply_overrides(result, session.get("rubric"))
    result["teacher_total"] = total
    result["overridden_questions"] = sum(1 for g in result["graded_questions"] if g.get("teacher_marks") is not None)

    now = datetime.now(timezone.utc).isoformat()
    fields = {"ai_result": result, "updated_at": now}
    if sheet.get("status") == "approved":
        fields["final_marks"] = total      # already approved → the approved marks follow the change
    upd = with_retry(lambda: c.db.table("check_sheets").update(fields).eq("id", sheet_id).execute())
    if not upd.data:
        raise HTTPException(status_code=404, detail="Student copy not found.")

    max_marks = float(result.get("max_marks") or 0)
    audit = {
        "id": str(uuid.uuid4()), "teacher_id": c.user.id, "check_sheet_id": sheet_id,
        "student_name": sheet.get("student_name") or None, "student_roll": sheet.get("roll_no") or None,
        "student_class": session.get("class_grade") or None, "class_grade": session.get("class_grade") or None,
        "subject": (session.get("subject") or session.get("title") or "")[:100] or None,
        "is_valid": True, "total_obtained": total, "max_marks": max_marks,
        "percentage": round(total / max_marks * 100, 1) if max_marks else 0,
        "needs_review_count": result.get("needs_review_count") or 0,
        "strictness": session.get("strictness") or "medium",
        "overall_remarks": f"Teacher changed Q{q_no}: {before_q:g} → {new:g}",
        "result_json": {"type": OVERRIDE_TYPE, "q_no": q_no, "ai_marks": gq.get("marks_awarded"),
                        "previous_marks": before_q, "new_marks": new, "max_marks": max_q,
                        "teacher_note": note, "sheet_total_before": before_total, "sheet_total_after": total},
    }
    try:
        with_retry(lambda: get_supabase_admin().table(HISTORY_TABLE).insert(audit).execute())
    except Exception as e:  # the change is saved; a missing audit row must be visible in the logs
        logger.error(f"Override audit row not saved for sheet {sheet_id} Q{q_no}: {e}")

    s = upd.data[0]
    return {"id": s["id"], "status": s["status"], "ai_marks": s.get("ai_marks"), "final_marks": s.get("final_marks"),
            "max_marks": s.get("max_marks"), "ai_result": s.get("ai_result")}


@router.patch("/sheets/{sheet_id}/questions/{q_no}/marks")
async def override_question_marks(sheet_id: str, q_no: str, body: MarksOverride,
                                  c: _Caller = Depends(require_caller)):
    _uuid_or_404(sheet_id, "Student copy")
    if not q_no or len(q_no) > 20:
        raise HTTPException(status_code=404, detail="Question not found.")
    try:
        sheet = await run_in_threadpool(_override_marks, c, sheet_id, q_no, body)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Marks override failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not save the marks. Please try again.")
    return {"ok": True, "sheet": sheet}
