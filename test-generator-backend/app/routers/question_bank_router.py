"""
Question Bank → Paper Builder.

  POST   /api/v1/question-banks                              {name, questions[]} → 201 {bank, stats, warnings}
         The questions are the ones a teacher just reviewed in "Contest from PDF" (the existing parser is the only
         extraction path; this endpoint never reads a PDF and never calls Gemini).
  GET    /api/v1/question-banks                              the caller's banks, newest first
  GET    /api/v1/question-banks/{bank_id}/questions          ?qtype= &year= &needs_review=
  PATCH  /api/v1/question-banks/{bank_id}/questions/{qid}    text / options / answer / marks / excluded / reviewed
  DELETE /api/v1/question-banks/{bank_id}/questions/{qid}
  DELETE /api/v1/question-banks/{bank_id}
  POST   /api/v1/question-banks/{bank_id}/build-paper        sections → PDF / DOCX made only from this bank; saved to
                                                             Test History like a generated paper

Who: every route needs a Supabase access token. The owner is ALWAYS the token's user — no request field carries a
user id. question_banks / bank_questions are read and written through a client bound to the caller's JWT, so the
RLS policies of 33_question_bank.sql decide (a foreign bank is simply not there → 404). A write that RLS filters
out is reported, never passed off as success: an insert is counted back, an update / delete must return its row.
Only the Test History rows (tests / questions) use the service key, exactly like the AI generator.
"""

import json
import logging
import os
import uuid
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from starlette.concurrency import run_in_threadpool

from app.core.database import get_supabase_admin
from app.routers.smart_assignment_router import _Caller, require_caller, with_retry
from app.schemas.question_bank import (MAX_PAPER_QUESTIONS, BuildPaperRequest, CreateBankRequest,
                                       PatchQuestionRequest)
from app.services import question_bank_service as svc
from app.services.contest_import_service import norm_class

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/question-banks", tags=["Question Banks"])

BANKS = "question_banks"
QUESTIONS = "bank_questions"
MAX_BANKS = int(os.getenv("QUESTION_BANK_MAX_BANKS", "50"))
BANK_COLUMNS = "id, owner_id, institute_id, name, source_filename, page_count, status, warnings, subject, " \
               "class_grade, board, created_at"
MEDIA = {"pdf": "application/pdf",
         "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

NOT_FOUND = "Question bank not found."
Q_NOT_FOUND = "Question not found."
READ_ONLY = "You can view this question bank but not change it."
NOT_SET_UP = "Question banks are not available on this server yet."
HINDI_PDF = ("This paper has Hindi (or other Indian-script) text, which the PDF export cannot print yet. "
             "Choose Word (DOCX) instead.")


# ── helpers ─────────────────────────────────────────────────────────────────

def _id(value: str, message: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=404, detail=message)


def _one_line(value, limit: int) -> Optional[str]:
    s = " ".join(svc.clean_text(value, limit * 2).split())[:limit].strip()
    return s or None


def _fail(e: Exception, doing: str) -> HTTPException:
    """Database error → a message that is safe to show. The details go to the log only."""
    code = str(getattr(e, "code", "") or "")
    logger.error(f"[qbank] {doing} failed: {type(e).__name__} {code} {str(e)[:200]}")
    if code in ("PGRST205", "42P01"):                 # table missing: 33_question_bank.sql has not been run
        return HTTPException(status_code=503, detail=NOT_SET_UP)
    if code == "42501":                               # RLS said no
        return HTTPException(status_code=403, detail=READ_ONLY)
    return HTTPException(status_code=500, detail=f"Could not {doing}. Please try again.")


def _run(doing: str, fn):
    try:
        return with_retry(fn)
    except HTTPException:
        raise
    except Exception as e:
        raise _fail(e, doing)


def _bank_out(row: dict, uid: str, count: Optional[int] = None) -> dict:
    out = {k: row.get(k) for k in ("id", "name", "source_filename", "page_count", "status", "subject",
                                   "class_grade", "board", "created_at")}
    out["warnings"] = row.get("warnings") or []
    out["can_edit"] = str(row.get("owner_id")) == uid
    if count is not None:
        out["question_count"] = count
    return out


def _question_out(row: dict) -> dict:
    return {k: row.get(k) for k in ("id", "position", "qtype", "question_text", "options", "answer",
                                    "answer_source", "marks", "exam_year", "source_tag", "needs_review",
                                    "excluded")}


def _get_bank(c: _Caller, bank_id: str) -> dict:
    """The bank if the caller may read it (RLS), else 404 — a foreign bank and a missing one look the same."""
    r = _run("open the question bank",
             lambda: c.db.table(BANKS).select(BANK_COLUMNS).eq("id", bank_id).limit(1).execute())
    if not r.data:
        raise HTTPException(status_code=404, detail=NOT_FOUND)
    return r.data[0]


def _get_question(c: _Caller, bank_id: str, question_id: str) -> dict:
    r = _run("open the question", lambda: c.db.table(QUESTIONS).select("*").eq("id", question_id)
             .eq("bank_id", bank_id).limit(1).execute())
    if not r.data:
        raise HTTPException(status_code=404, detail=Q_NOT_FOUND)
    return r.data[0]


def _owned(c: _Caller, bank: dict) -> None:
    if str(bank.get("owner_id")) != str(c.user.id):
        raise HTTPException(status_code=403, detail=READ_ONLY)


def _bank_questions(c: _Caller, bank_id: str, qtype=None, year=None, needs_review=None) -> list:
    def go():
        query = c.db.table(QUESTIONS).select("*").eq("bank_id", bank_id)
        if qtype is not None:
            query = query.eq("qtype", qtype)
        if year is not None:
            query = query.eq("exam_year", year)
        if needs_review is not None:
            query = query.eq("needs_review", needs_review)
        return query.order("position").limit(1000).execute()
    return _run("load the questions", go).data or []


# ── create ──────────────────────────────────────────────────────────────────

def _create(c: _Caller, body: CreateBankRequest, rows: list, stats: dict) -> dict:
    uid = str(c.user.id)
    mine = _run("check your question banks",
                lambda: c.db.table(BANKS).select("id", count="exact").eq("owner_id", uid).limit(1).execute())
    if (mine.count or 0) >= MAX_BANKS:
        raise HTTPException(status_code=409, detail=f"You already have {MAX_BANKS} question banks. "
                                                    f"Delete one you no longer need, then save again.")
    warnings = svc.stats_warnings(stats)
    bank_row = {
        "owner_id": uid,                               # the token's user; RLS rejects anything else
        "name": _one_line(body.name, 120) or "Question bank",
        "source_filename": svc.safe_filename(body.source_filename, "") or None,
        "status": "processing",
        "warnings": warnings,
        "subject": _one_line(body.subject, 50),
        "class_grade": _one_line(body.class_grade, 20),
        "board": _one_line(body.board, 30),
    }
    created = _run("save the question bank", lambda: c.db.table(BANKS).insert(bank_row).execute())
    if not created.data:
        logger.error("[qbank] bank insert returned no row (blocked by RLS?)")
        raise HTTPException(status_code=500, detail="Could not save the question bank. Please try again.")
    bank = created.data[0]
    bank_id = str(bank["id"])

    def undo():
        try:
            with_retry(lambda: c.db.table(BANKS).delete().eq("id", bank_id).execute())
        except Exception as e:
            logger.error(f"[qbank] could not remove the half-saved bank {bank_id}: {type(e).__name__}")

    try:
        with_retry(lambda: c.db.table(QUESTIONS).insert([{**r, "bank_id": bank_id} for r in rows]).execute())
        # RLS has blocked writes silently before: count what is really there instead of trusting the insert
        saved = with_retry(lambda: c.db.table(QUESTIONS).select("id", count="exact").eq("bank_id", bank_id)
                           .limit(1).execute())
        if (saved.count or 0) != len(rows):
            logger.error(f"[qbank] {bank_id}: {saved.count} of {len(rows)} questions persisted — bank removed")
            raise HTTPException(status_code=500, detail="The questions could not be saved, so no bank was "
                                                        "created. Please try again.")
        done = with_retry(lambda: c.db.table(BANKS).update({"status": "ready"}).eq("id", bank_id).execute())
        if done.data:
            bank = done.data[0]
    except HTTPException:
        undo()
        raise
    except Exception as e:
        undo()
        raise _fail(e, "save the questions")
    logger.info(f"[qbank] bank {bank_id}: {len(rows)} questions saved, stats {stats}")
    return {"bank": _bank_out(bank, uid, len(rows)), "stats": stats, "warnings": warnings}


@router.post("", status_code=201)
async def create_bank(body: CreateBankRequest, c: _Caller = Depends(require_caller)):
    rows, stats = svc.normalise_questions([q.model_dump() for q in body.questions], body.source_tag)
    if not rows:
        raise HTTPException(status_code=422, detail="None of these questions has any text.")
    return await run_in_threadpool(_create, c, body, rows, stats)


# ── read ────────────────────────────────────────────────────────────────────

def _list(c: _Caller) -> dict:
    r = _run("load your question banks",
             lambda: c.db.table(BANKS).select(f"{BANK_COLUMNS}, {QUESTIONS}(count)")
             .order("created_at", desc=True).limit(200).execute())
    uid = str(c.user.id)
    banks = []
    for row in r.data or []:
        nested = row.get(QUESTIONS) or []
        count = nested[0].get("count", 0) if nested and isinstance(nested[0], dict) else 0
        banks.append(_bank_out(row, uid, count))
    return {"banks": banks, "max_banks": MAX_BANKS}


@router.get("")
async def list_banks(c: _Caller = Depends(require_caller)):
    return await run_in_threadpool(_list, c)


@router.get("/{bank_id}/questions")
async def list_questions(bank_id: str, qtype: Optional[str] = Query(default=None, max_length=20),
                         year: Optional[int] = Query(default=None, ge=1950, le=2100),
                         needs_review: Optional[bool] = None, c: _Caller = Depends(require_caller)):
    bank_id = _id(bank_id, NOT_FOUND)
    if qtype is not None and qtype not in svc.QTYPES:
        raise HTTPException(status_code=422, detail=f"qtype must be one of: {', '.join(svc.QTYPES)}.")

    def go():
        bank = _get_bank(c, bank_id)
        rows = _bank_questions(c, bank_id, qtype, year, needs_review)
        return {"bank": _bank_out(bank, str(c.user.id)), "questions": [_question_out(r) for r in rows]}
    return await run_in_threadpool(go)


# ── edit / delete ───────────────────────────────────────────────────────────

def _patch(c: _Caller, bank_id: str, question_id: str, body: PatchQuestionRequest) -> dict:
    bank = _get_bank(c, bank_id)
    _owned(c, bank)
    row = _get_question(c, bank_id, question_id)
    sent = body.model_fields_set
    if not sent:
        raise HTTPException(status_code=422, detail="Nothing to change.")
    qtype = row["qtype"]
    text, options = row["question_text"], row.get("options") or []
    answer, source = row.get("answer"), row.get("answer_source")
    changes: dict = {}

    if "question_text" in sent:
        text = svc.clean_text(body.question_text)
        if not text:
            raise HTTPException(status_code=422, detail="The question text cannot be empty.")
        changes["question_text"] = text
    if "options" in sent:
        if qtype in ("short", "long"):
            raise HTTPException(status_code=422, detail="A written question has no options.")
        options = svc._options([o.model_dump() for o in body.options or []])
        if qtype == "mcq" and len(options) < 2:
            raise HTTPException(status_code=422, detail="An MCQ needs at least 2 options.")
        changes["options"] = options
    if "answer" in sent:
        if body.answer is None or not body.answer.strip():
            answer, source = None, None
        else:
            answer, source, problem = svc.normalise_answer(qtype, options, body.answer, "teacher")
            if problem:
                raise HTTPException(status_code=422, detail="The answer must be True or False." if qtype == "true_false"
                                    and not options else "The answer must be one of the question's options.")
    elif "options" in sent and answer and options:
        # the options changed under an existing answer: keep it only if it still points at an option
        if svc.option_label(answer, options) is None:
            answer, source = None, None
    if "answer" in sent or "options" in sent:
        changes["answer"], changes["answer_source"] = answer, source
    if "marks" in sent:
        changes["marks"] = svc._marks(body.marks)
    if "excluded" in sent and body.excluded is not None:
        changes["excluded"] = body.excluded
    if "question_text" in sent or "options" in sent:
        changes["content_hash"] = svc.content_hash(text, options)

    # an edit or an explicit "reviewed" is the teacher's review; an objective question still needs its answer
    if sent - {"excluded", "marks"}:
        changes["needs_review"] = svc.needs_review(qtype, text, options, answer, usable=True)
    if not changes:
        raise HTTPException(status_code=422, detail="Nothing to change.")

    try:
        r = with_retry(lambda: c.db.table(QUESTIONS).update(changes).eq("id", question_id)
                       .eq("bank_id", bank_id).execute())
    except Exception as e:
        if "23505" in f"{getattr(e, 'code', '')} {e}":
            raise HTTPException(status_code=409, detail="This bank already has the same question.")
        raise _fail(e, "save the question")
    if not r.data:                                    # readable but not writable: RLS filtered the update out
        logger.warning(f"[qbank] update of {question_id} changed no row")
        raise HTTPException(status_code=403, detail=READ_ONLY)
    return {"question": _question_out(r.data[0])}


@router.patch("/{bank_id}/questions/{question_id}")
async def patch_question(bank_id: str, question_id: str, body: PatchQuestionRequest,
                         c: _Caller = Depends(require_caller)):
    return await run_in_threadpool(_patch, c, _id(bank_id, NOT_FOUND), _id(question_id, Q_NOT_FOUND), body)


@router.delete("/{bank_id}/questions/{question_id}")
async def delete_question(bank_id: str, question_id: str, c: _Caller = Depends(require_caller)):
    bank_id, question_id = _id(bank_id, NOT_FOUND), _id(question_id, Q_NOT_FOUND)

    def go():
        _owned(c, _get_bank(c, bank_id))
        _get_question(c, bank_id, question_id)
        r = _run("delete the question", lambda: c.db.table(QUESTIONS).delete().eq("id", question_id)
                 .eq("bank_id", bank_id).execute())
        if not r.data:
            raise HTTPException(status_code=403, detail=READ_ONLY)
        return {"deleted": True}
    return await run_in_threadpool(go)


@router.delete("/{bank_id}")
async def delete_bank(bank_id: str, c: _Caller = Depends(require_caller)):
    bank_id = _id(bank_id, NOT_FOUND)

    def go():
        _owned(c, _get_bank(c, bank_id))
        r = _run("delete the question bank", lambda: c.db.table(BANKS).delete().eq("id", bank_id).execute())
        if not r.data:
            raise HTTPException(status_code=403, detail=READ_ONLY)
        return {"deleted": True}                       # bank_questions go with it (on delete cascade)
    return await run_in_threadpool(go)


# ── build paper ─────────────────────────────────────────────────────────────

def _class_number(value) -> str:
    """'Class XII' / '12th' / '12' → '12' (the contest import's own rule); anything else as typed."""
    return (norm_class(_one_line(value, 20)) or "")[:20]


def _save_to_history(uid: str, test_id: str, title: str, board: str, class_grade: str, subject: str,
                     questions: list) -> bool:
    """tests + questions rows, the way /test-generator/generate-frontend and /save write them (service key,
    teacher_id = the JWT user). Never raises: the paper is still delivered, the response says it was not saved."""
    try:
        db = get_supabase_admin()
        r = with_retry(lambda: db.table("tests").insert({
            "id": test_id, "teacher_id": uid, "exam_title": title, "board": board, "class_grade": class_grade,
            "subject": subject, "status": "saved", "total_questions": len(questions),
            "total_marks": sum(q["marks"] for q in questions), "paper_date": date.today().isoformat(),
            "cbse_pattern": False,
        }).execute())
        if not r.data:
            logger.error(f"[qbank] tests insert returned no row for {test_id}")
            return False
        with_retry(lambda: db.table("questions").insert([{
            "id": q["id"], "test_id": test_id, "position": i + 1, "text": q["text"], "options": q["options"],
            "correct_answer": q["correctAnswer"], "explanation": "", "marks": q["marks"],
            "difficulty": "medium", "chapter": "", "topic": None, "format": q["format"],
            "section": q["section"], "is_manual": False,
        } for i, q in enumerate(questions)]).execute())
        return True
    except Exception as e:
        logger.error(f"[qbank] could not save paper {test_id} to Test History: {type(e).__name__} {str(e)[:200]}")
        return False


def _build(c: _Caller, bank_id: str, body: BuildPaperRequest) -> Response:
    uid = str(c.user.id)
    template = body.template.strip().lower()
    if template not in svc.allowed_templates():
        raise HTTPException(status_code=422, detail="Unknown template.")
    if sum(s.count for s in body.sections) > MAX_PAPER_QUESTIONS:
        raise HTTPException(status_code=422, detail=f"A paper can have at most {MAX_PAPER_QUESTIONS} questions.")
    bank = _get_bank(c, bank_id)
    rows = _bank_questions(c, bank_id)
    try:
        picked = svc.select_for_paper(rows, body.sections, year_filter=body.year_filter,
                                      allow_needs_review=body.allow_needs_review, shuffle=body.shuffle)
    except svc.PaperError as e:
        raise HTTPException(status_code=422, detail=str(e))
    questions = svc.paper_questions(picked, body.sections)

    title = _one_line(body.title, 200) or "Question Paper"
    subject = _one_line(body.subject, 50) or bank.get("subject") or "General"
    class_grade = _class_number(body.class_grade or bank.get("class_grade")) or "-"
    board = _one_line(body.board, 30) or bank.get("board") or "CBSE"
    institute = _one_line(body.institute_name, 200)
    if body.format == "pdf" and svc.pdf_cannot_print(questions, title, institute, subject):
        raise HTTPException(status_code=422, detail=HINDI_PDF)
    try:
        content = svc.render_paper(
            questions, title=title, fmt=body.format, template=template,
            include_answer_key=body.include_answer_key, subject=subject, class_grade=class_grade, board=board,
            institute_name=institute, duration=_one_line(body.duration, 20),
            teacher_name=_one_line(body.teacher_name, 100), paper_date=date.today().isoformat())
    except Exception:
        logger.error(f"[qbank] rendering a paper from bank {bank_id} failed", exc_info=True)
        raise HTTPException(status_code=500, detail="Could not create the file. Please try again.")

    test_id = str(uuid.uuid4())
    saved = _save_to_history(uid, test_id, title, board, class_grade, subject, questions)
    found = sum(1 for q in questions if q["answerFound"])
    meta = {"test_id": test_id if saved else None, "saved_to_history": saved, "questions": len(questions),
            "total_marks": sum(q["marks"] for q in questions), "answers_found": found,
            "answers_missing": len(questions) - found, "answer_key_included": body.include_answer_key}
    logger.info(f"[qbank] paper from bank {bank_id}: {meta}")
    return Response(content=content, media_type=MEDIA[body.format], headers={
        "Content-Disposition": f'attachment; filename="{svc.download_name(title)}.{body.format}"',
        "X-Paper-Meta": json.dumps(meta),
        "Cache-Control": "no-store",
    })


@router.post("/{bank_id}/build-paper")
async def build_paper(bank_id: str, body: BuildPaperRequest, c: _Caller = Depends(require_caller)):
    return await run_in_threadpool(_build, c, _id(bank_id, NOT_FOUND), body)
