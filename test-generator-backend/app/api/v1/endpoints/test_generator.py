"""
Test Generator API Endpoints — v2.9

v2.9 changes:
  - ExportRequest now accepts `template` field (default "modern")
  - /export passes template through to generate_pdf() / generate_docx()
  - New GET /templates endpoint — returns available export templates for
    a frontend dropdown (Modern / Classic / Compact / Colorful)
  - Split usage check into check_usage (read-only, before generation) + record_usage (increment, after success)
  - check_usage: fail-closed (error = block), null-UUID blocked (401), only checks count
  - record_usage: called only after successful generation
  - check_and_record_usage kept for backward compat, now calls split internally

v2.8 changes:
  - English pseudo-chapter support (Writing Skills, Grammar bypass RAG)
  - _is_english_pseudo() helper skips NCERT lookup for these chapters
  - Error handler filters pseudo-chapters from error messages

v2.7 changes:
  - /chapters endpoint now returns book, chapter_type, chapter_order
  - Adds `groups` field for grouped dropdown UI (English Class 10)
  - Backward compatible: flat `chapters` list still returned

v2.6 changes:
  - FrontendQuestionResponse: added questionTable field for Statistics questions
  - _transform_backend_to_frontend: extracts question_table similar to answer_table
  - /save: persists question_table along with answer_table (forward compat)
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import List, Optional
import re

from app.models.test_generator import (
    TestGenerationRequest,
    TestGenerationResponse,
    TestFeedbackRequest,
    SaveTestRequest,
    QuizSettings,
    ChapterSection,
    DifficultyLevel,
    QuestionFormat,
    ManualQuestionPayload,
)
from app.services.test_generator_service import generate_test, handle_feedback
from app.services.rag_service import retrieve_context
from app.core.database import get_supabase
from app.core.config import settings
import logging
import uuid
import time

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/test-generator", tags=["Test Generator"])


# ═══════════════════════════════════════════════════════════════════════
# USAGE CHECK HELPERS (v2.9 — split check/increment + fail-closed + no null-UUID bypass)
# ═══════════════════════════════════════════════════════════════════════

def check_usage(user_id: str) -> dict:
    """
    Read-only check: does user have remaining quota?
    Does NOT increment the counter.
    
    Rules:
      - No valid user_id → 401 (block, not unlimited)
      - DB error → 503 (fail-closed, not unlimited)
    """
    if not user_id or user_id == "00000000-0000-0000-0000-000000000000":
        logger.warning("Usage check blocked: no valid user_id provided")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "unauthorized",
                "message": "Login required. Please sign in to generate tests.",
            },
        )

    try:
        supabase = get_supabase()
        result = supabase.rpc("check_usage", {
            "p_user_id": user_id,
        }).execute()

        if not result.data:
            logger.error(f"Usage check returned no data for user {user_id} — blocking (fail-closed)")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_error",
                    "message": "Unable to verify usage. Please try again later.",
                },
            )

        usage = result.data

        if not usage.get("allowed"):
            logger.info(f"Usage limit reached: user={user_id}, used={usage.get('used')}, limit={usage.get('limit')}")
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "limit_reached",
                    "message": f"Monthly limit reached ({usage['used']}/{usage['limit']} tests). Upgrade your plan.",
                    "used": usage.get("used"),
                    "limit": usage.get("limit"),
                    "upgrade_url": "/pricing",
                },
            )

        logger.info(f"Usage OK: user={user_id}, used={usage.get('used')}, remaining={usage.get('remaining')}")
        return usage

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Usage check error — blocking (fail-closed): {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_error",
                "message": "Unable to verify usage. Please try again later.",
            },
        )


def record_usage(user_id: str) -> dict:
    """
    Increment usage counter AFTER successful generation.
    Non-fatal — if this fails, generation still succeeds (user gets paper).
    """
    try:
        supabase = get_supabase()
        result = supabase.rpc("record_usage", {
            "p_user_id": user_id,
            "p_action": "test_generated",
        }).execute()

        if result.data:
            logger.info(f"Usage recorded: user={user_id}, used={result.data.get('used')}")
        return result.data or {}

    except Exception as e:
        logger.error(f"Record usage failed (non-fatal): {e}")
        return {"recorded": False, "error": str(e)}


def check_and_record_usage(user_id: str) -> dict:
    """
    Backward-compatible wrapper: check + increment in one call.
    Used by legacy endpoints. New endpoints should use check_usage + record_usage separately.
    """
    usage = check_usage(user_id)
    
    # If check passed, immediately record
    try:
        supabase = get_supabase()
        result = supabase.rpc("record_usage", {
            "p_user_id": user_id,
            "p_action": "test_generated",
        }).execute()
        if result.data:
            usage.update(result.data)
    except Exception as e:
        logger.error(f"Record usage failed in combined call (non-fatal): {e}")
    
    return usage


# ═══════════════════════════════════════════════════════════════════════
# SUBJECT ALIAS
# ═══════════════════════════════════════════════════════════════════════

SUBJECT_ALIASES = {
    "Maths": "Mathematics",
    "Math": "Mathematics",
    "Pol Science": "Political Science",
    "Accounts": "Accountancy",
    "BST": "Business Studies",
    "Eco": "Economics",
}


def _resolve_subject(subject: str) -> str:
    return SUBJECT_ALIASES.get(subject, subject)


# ═══════════════════════════════════════════════════════════════════════
# ENGLISH PSEUDO-CHAPTER SUPPORT (v2.8)
# ═══════════════════════════════════════════════════════════════════════

ENGLISH_PSEUDO_CHAPTERS = {"writing skills", "grammar"}


def _is_english_pseudo(subject: str, chapter: str) -> bool:
    """Check if chapter is a Writing/Grammar pseudo-chapter for English."""
    if (subject or "").lower() != "english":
        return False
    return chapter.lower().strip() in ENGLISH_PSEUDO_CHAPTERS


# ═══════════════════════════════════════════════════════════════════════
# BOOK / CHAPTER_TYPE LABELS (for /chapters response)
# ═══════════════════════════════════════════════════════════════════════

BOOK_LABELS = {
    ("first_flight", "prose"): "First Flight — Prose",
    ("first_flight", "poem"): "First Flight — Poems",
    ("footprints_without_feet", "prose"): "Footprints Without Feet",
}

BOOK_GROUP_ORDER = {
    ("first_flight", "prose"): 1,
    ("first_flight", "poem"): 2,
    ("footprints_without_feet", "prose"): 3,
}


# ═══════════════════════════════════════════════════════════════════════
# FRONTEND MODELS
# ═══════════════════════════════════════════════════════════════════════

class FrontendChapterRow(BaseModel):
    topic: str
    subtopic: Optional[str] = None
    quantity: int = Field(default=5, ge=1, le=50)
    marks: int = Field(default=1, ge=1, le=10)
    difficulty: str = "Medium"
    format: str = "MCQ"


class FrontendGenerateRequest(BaseModel):
    examTitle: str = "Untitled Test"
    paperDate: Optional[str] = None
    board: str = "CBSE"
    classGrade: str = "Class 10"
    subject: str = "Science"
    simpleData: List[FrontendChapterRow] = []
    mode: str = "Simple"
    enableWatermark: bool = True
    shuffleQuestions: bool = False
    useNCERT: bool = True
    ncertClass: Optional[str] = None
    ncertSubject: Optional[str] = None
    ncertChapters: List[str] = []
    userId: Optional[str] = None
    cbsePattern: bool = True


class FrontendQuestionResponse(BaseModel):
    id: str
    text: str
    options: List[str] = []
    correctAnswer: str
    explanation: str
    marks: int
    difficulty: str
    bloomLevel: Optional[str] = None
    chapter: str
    topic: Optional[str] = None
    format: str
    validationStatus: str
    section: Optional[str] = None
    answerTable: Optional[dict] = None
    questionTable: Optional[dict] = None
    isManual: bool = False
    imageUrl: Optional[str] = None


class FrontendGenerateResponse(BaseModel):
    ok: bool
    testId: str
    examTitle: str
    questions: List[FrontendQuestionResponse]
    totalMarks: int
    totalQuestions: int
    generationTime: float
    status: str = "preview"
    meta: dict = {}


# v2.5: Export accepts paperDate + manual flags
# v2.9: Export accepts template (Modern / Classic / Compact / Colorful)
class ExportRequest(BaseModel):
    examTitle: str = "Test Paper"
    paperDate: Optional[str] = None
    board: str = "CBSE"
    classGrade: str = "Class 10"
    subject: str = "Science"
    questions: list
    includeAnswers: bool = False
    includeExplanations: bool = False
    format: str = "pdf"
    logoBase64: Optional[str] = None
    template: str = "modern"  # v2.9: "modern" | "classic" | "compact" | "colorful"


class FrontendSaveRequest(BaseModel):
    test_id: str
    teacher_id: str
    questions: Optional[List[dict]] = None


class AddManualQuestionRequest(BaseModel):
    teacher_id: str
    question: ManualQuestionPayload


# ── Transform helpers ───────────────────────────────────────────────

DIFFICULTY_MAP = {
    "Easy": "easy", "easy": "easy",
    "Medium": "medium", "medium": "medium",
    "Hard": "hard", "hard": "hard",
    "Mixed": "medium", "mixed": "medium",
    "Very Hard": "very_hard", "very_hard": "very_hard",
}

FORMAT_MAP = {
    "MCQ":            QuestionFormat.MCQ,
    "Short":          QuestionFormat.SHORT_ANSWER,
    "Long":           QuestionFormat.LONG_ANSWER,
    "Essay":          QuestionFormat.LONG_ANSWER,

    "mcq":            QuestionFormat.MCQ,
    "short_answer":   QuestionFormat.SHORT_ANSWER,
    "long_answer":    QuestionFormat.LONG_ANSWER,
    "assertion_reason": QuestionFormat.ASSERTION_REASON,
    "case_based":     QuestionFormat.MCQ,

    "Journal Entry":  QuestionFormat.JOURNAL_ENTRY,
    "journal_entry":  QuestionFormat.JOURNAL_ENTRY,
    "JournalEntry":   QuestionFormat.JOURNAL_ENTRY,
    "Ledger":         QuestionFormat.LEDGER,
    "ledger":         QuestionFormat.LEDGER,
    "Trial Balance":  QuestionFormat.TRIAL_BALANCE,
    "trial_balance":  QuestionFormat.TRIAL_BALANCE,
    "TrialBalance":   QuestionFormat.TRIAL_BALANCE,

    "PDF":            QuestionFormat.MCQ,
    "DOC":            QuestionFormat.MCQ,
}

MARKS_MAP = {
    QuestionFormat.MCQ: 1,
    QuestionFormat.SHORT_ANSWER: 2,
    QuestionFormat.LONG_ANSWER: 5,
    QuestionFormat.ASSERTION_REASON: 1,
    QuestionFormat.JOURNAL_ENTRY: 4,
    QuestionFormat.LEDGER: 6,
    QuestionFormat.TRIAL_BALANCE: 6,
}


def _extract_class_number(class_grade: str) -> str:
    match = re.search(r'\d+', class_grade)
    return match.group() if match else "10"


def _transform_frontend_to_backend(req: FrontendGenerateRequest) -> TestGenerationRequest:
    class_num = _extract_class_number(req.classGrade)
    resolved_subject = _resolve_subject(req.subject)
    chapters = []

    for row in req.simpleData:
        if not row.topic:
            continue
        difficulty_str = DIFFICULTY_MAP.get(row.difficulty, "medium")
        question_format = FORMAT_MAP.get(row.format, QuestionFormat.MCQ)
        marks = row.marks if row.marks and row.marks > 0 else MARKS_MAP.get(question_format, 1)

        logger.info(f"Row: topic={row.topic}, format='{row.format}' -> {question_format.value}, marks={marks}")

        chapter = ChapterSection(
            chapter=row.topic,
            topic=row.subtopic if row.subtopic else None,
            subtopics=[row.subtopic] if row.subtopic else [],
            quantity=row.quantity,
            difficulty=DifficultyLevel(difficulty_str),
            format=question_format,
            marks_per_question=marks,
        )
        chapters.append(chapter)

    if not chapters:
        raise ValueError("At least one chapter with a topic is required")

    total_q = sum(c.quantity for c in chapters)
    if total_q > settings.MAX_QUESTIONS_PER_REQUEST:
        raise ValueError(f"Too many questions ({total_q}). Max {settings.MAX_QUESTIONS_PER_REQUEST}.")

    return TestGenerationRequest(
        exam_title=req.examTitle,
        board=req.board,
        class_grade=class_num,
        subject=resolved_subject,
        chapters=chapters,
        pattern="simple",
        bloom_enabled=True,
        teacher_id=req.userId or "00000000-0000-0000-0000-000000000000",
        iteration=0,
    )


def _serialize_table_field(table_obj):
    """Safely convert a Pydantic table model or dict to a plain dict for frontend."""
    if table_obj is None:
        return None
    try:
        if hasattr(table_obj, 'model_dump'):
            return table_obj.model_dump()
        if hasattr(table_obj, 'dict'):
            return table_obj.dict()
        if isinstance(table_obj, dict):
            return table_obj
    except Exception as e:
        logger.warning(f"Table serialization failed: {e}")
    return None


def _transform_backend_to_frontend(resp, req: FrontendGenerateRequest) -> FrontendGenerateResponse:
    if isinstance(resp, list):
        questions_list = resp
        test_id = str(uuid.uuid4())
        exam_title = req.examTitle
        total_marks = sum(q.marks for q in questions_list)
        total_questions = len(questions_list)
        iteration = 0
        generation_time = 0.0
        status = "preview"
    else:
        questions_list = resp.questions
        test_id = resp.test_id
        exam_title = resp.exam_title
        total_marks = resp.total_marks
        total_questions = resp.total_questions
        iteration = resp.iteration
        generation_time = resp.generation_time_seconds
        status = resp.status

    questions = []
    for q in questions_list:
        section = getattr(q, '_section', None) or getattr(q, 'section', None)
        answer_table_data = _serialize_table_field(getattr(q, 'answer_table', None))
        question_table_data = _serialize_table_field(getattr(q, 'question_table', None))

        questions.append(FrontendQuestionResponse(
            id=q.id,
            text=q.text,
            options=q.options or [],
            correctAnswer=q.correct_answer,
            explanation=q.explanation,
            marks=q.marks,
            difficulty=q.difficulty.value if hasattr(q.difficulty, "value") else q.difficulty,
            bloomLevel=q.bloom_level.value if q.bloom_level and hasattr(q.bloom_level, "value") else q.bloom_level,
            chapter=q.chapter,
            topic=q.topic,
            format=q.format.value if hasattr(q.format, "value") else q.format,
            validationStatus=q.validation_status,
            section=section,
            answerTable=answer_table_data,
            questionTable=question_table_data,
            isManual=getattr(q, 'is_manual', False),
            imageUrl=getattr(q, 'image_url', None),
        ))

    return FrontendGenerateResponse(
        ok=True,
        testId=test_id,
        examTitle=exam_title,
        questions=questions,
        totalMarks=total_marks,
        totalQuestions=total_questions,
        generationTime=generation_time,
        status=status,
        meta={
            "ncertBased": True,
            "ragUsed": True,
            "iteration": iteration,
            "board": req.board,
            "classGrade": req.classGrade,
            "subject": req.subject,
            "cbsePattern": req.cbsePattern,
            "paperDate": req.paperDate,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Generate from Frontend (v2.9 — split check/increment)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/generate-frontend", response_model=FrontendGenerateResponse)
async def generate_from_frontend(req: FrontendGenerateRequest):
    start = time.time()
    logger.info(f"Frontend generate: {req.subject} {req.classGrade}, "
                f"{len(req.simpleData)} chapters, cbsePattern={req.cbsePattern}, paperDate={req.paperDate}")

    try:
        # ── Step 1: Check usage (read-only, BEFORE generation) ──
        # Fail-closed: error here = block (401/403/503)
        # Null UUID: blocked with 401 (not unlimited)
        usage = check_usage(req.userId)

        backend_request = _transform_frontend_to_backend(req)
        total_q = sum(c.quantity for c in backend_request.chapters)
        logger.info(f"Transformed: {len(backend_request.chapters)} chapters, {total_q} questions")

        # Filter out English pseudo-chapters from RAG lookup
        real_chapters = [
            ch.chapter for ch in backend_request.chapters
            if not _is_english_pseudo(backend_request.subject, ch.chapter)
        ]
        topics = [ch.topic for ch in backend_request.chapters if ch.topic]
        if not topics:
            topics = real_chapters

        if real_chapters:
            context_chunks = retrieve_context(
                real_chapters, topics,
                backend_request.subject, backend_request.class_grade,
            )
            logger.info(f"Retrieved {len(context_chunks)} context chunks (skipped pseudo-chapters)")
        else:
            context_chunks = []
            logger.info("All chapters are English pseudo-chapters — skipping RAG entirely")

        resolved_subject = _resolve_subject(req.subject)
        is_accountancy = resolved_subject.lower() in ("accountancy", "accounts", "accounting")

        # ── Step 2: Generate paper ──
        if req.cbsePattern and is_accountancy:
            from app.services.test_generator_service import generate_cbse_accountancy_paper
            backend_response = generate_cbse_accountancy_paper(backend_request, context_chunks)
        else:
            backend_response = generate_test(backend_request, context_chunks, cbse_pattern=req.cbsePattern)

        # ── Step 3: Increment usage (AFTER successful generation) ──
        # Non-fatal: if this fails, paper is still returned
        recorded = record_usage(req.userId)
        if recorded.get("recorded") is False:
            logger.warning(f"Usage not recorded for {req.userId}: {recorded.get('error')}")
        else:
            usage.update(recorded)

        frontend_response = _transform_backend_to_frontend(backend_response, req)

        elapsed = round(time.time() - start, 2)
        frontend_response.generationTime = elapsed

        frontend_response.meta["usage"] = {
            "used": usage.get("used", 0),
            "limit": usage.get("limit", -1),
            "remaining": usage.get("remaining", -1),
        }

        logger.info(f"Done: {frontend_response.totalQuestions} questions in {elapsed}s")
        return frontend_response

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        error_str = str(e)
        if "No NCERT content found" in error_str:
            if 'backend_request' in locals():
                real_names = [
                    ch.chapter for ch in backend_request.chapters
                    if not _is_english_pseudo(backend_request.subject, ch.chapter)
                ]
            else:
                real_names = []
            
            if real_names:
                raise HTTPException(
                    status_code=404,
                    detail=f"Content not available for selected chapters: {', '.join(real_names)}. Try different chapters or contact support."
                )
            else:
                logger.warning("No NCERT content found, but all chapters are pseudo-chapters. Proceeding.")
        logger.error(f"Frontend generate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Export PDF / DOCX (v2.9 — template support added)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/export")
async def export_test(req: ExportRequest):
    try:
        class_num = _extract_class_number(req.classGrade)

        # v2.9: normalize template value (frontend may send "" or None)
        template = (req.template or "modern").strip().lower()

        # Normalize questions: ensure section field exists, mark manual
        normalized_questions = []
        for q in req.questions:
            if not isinstance(q, dict):
                continue
            if 'section' not in q:
                q['section'] = None
            is_manual = bool(
                q.get('isManual')
                or q.get('is_manual')
                or q.get('validationStatus') == 'manual'
                or q.get('validation_status') == 'manual'
            )
            if is_manual:
                q['isManual'] = True
                q['is_manual'] = True
            normalized_questions.append(q)

        manual_count = sum(1 for q in normalized_questions if q.get('is_manual'))
        if manual_count:
            logger.info(f"Export includes {manual_count} manual question(s)")

        logger.info(f"Export: format={req.format}, template={template}, questions={len(normalized_questions)}")

        if req.format.lower() == "docx":
            from app.services.export_service import generate_docx
            file_bytes = generate_docx(
                questions=normalized_questions,
                exam_title=req.examTitle,
                board=req.board,
                class_grade=class_num,
                subject=req.subject,
                include_answers=req.includeAnswers,
                include_explanations=req.includeExplanations,
                logo_base64=req.logoBase64,
                paper_date=req.paperDate,
                template=template,  # v2.9
            )
            filename = f"{req.examTitle.replace(' ', '_')}.docx"
            return Response(
                content=file_bytes,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        else:
            from app.services.export_service import generate_pdf
            file_bytes = generate_pdf(
                questions=normalized_questions,
                exam_title=req.examTitle,
                board=req.board,
                class_grade=class_num,
                subject=req.subject,
                include_answers=req.includeAnswers,
                include_explanations=req.includeExplanations,
                logo_base64=req.logoBase64,
                paper_date=req.paperDate,
                template=template,  # v2.9
            )
            filename = f"{req.examTitle.replace(' ', '_')}.pdf"
            return Response(
                content=file_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Available Export Templates (v2.9 — for frontend dropdown)
# ═══════════════════════════════════════════════════════════════════════

@router.get("/templates")
async def get_templates():
    """
    Returns available PDF/DOCX export templates for a frontend dropdown.
    Each item: { id, label, description }
    """
    try:
        from app.services.export_service import get_available_templates
        return {"ok": True, "templates": get_available_templates()}
    except Exception as e:
        logger.error(f"Templates fetch error: {e}")
        # Safe fallback so the frontend dropdown never breaks
        return {
            "ok": False,
            "templates": [
                {"id": "modern", "label": "Modern", "description": "Clean card-style layout."},
                {"id": "classic", "label": "Classic", "description": "Traditional serif exam-paper look."},
                {"id": "compact", "label": "Compact", "description": "Dense layout, saves paper."},
                {"id": "colorful", "label": "Colorful", "description": "Section-wise accent colors."},
            ],
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Chapters (v2.7 — grouped by book + chapter_type)
# ═══════════════════════════════════════════════════════════════════════

@router.get("/chapters")
async def get_chapters(subject: str = "Science", class_grade: str = "10"):
    subject = _resolve_subject(subject)
    try:
        supabase = get_supabase()
        result = supabase.table("ncert_chunks") \
            .select("chapter, book, chapter_type, chapter_order") \
            .ilike("subject", f"%{subject}%") \
            .eq("class_grade", class_grade) \
            .execute()

        rows = result.data or []

        chapters = sorted({row["chapter"] for row in rows if row.get("chapter")})

        seen_chapters = {}
        for row in rows:
            chapter = row.get("chapter")
            book = row.get("book")
            ctype = row.get("chapter_type")
            order = row.get("chapter_order")

            if not chapter:
                continue
            if not book or not ctype:
                continue

            key = (chapter, book, ctype)
            if key not in seen_chapters:
                seen_chapters[key] = {"name": chapter, "order": order}

        groups_dict = {}
        for (chapter, book, ctype), data in seen_chapters.items():
            gkey = (book, ctype)
            if gkey not in groups_dict:
                groups_dict[gkey] = []
            groups_dict[gkey].append(data)

        groups = []
        for (book, ctype), chs in groups_dict.items():
            chs_sorted = sorted(
                chs,
                key=lambda x: (x["order"] if x["order"] is not None else 9999, x["name"])
            )
            label = BOOK_LABELS.get(
                (book, ctype),
                f"{(book or 'Other').replace('_', ' ').title()} — {(ctype or 'Other').title()}"
            )
            groups.append({
                "book": book,
                "chapter_type": ctype,
                "label": label,
                "_sort_order": BOOK_GROUP_ORDER.get((book, ctype), 99),
                "chapters": chs_sorted,
            })

        groups.sort(key=lambda g: g["_sort_order"])
        for g in groups:
            g.pop("_sort_order", None)

        return {
            "ok": True,
            "subject": subject,
            "classGrade": class_grade,
            "chapters": chapters,
            "groups": groups if groups else None,
            "count": len(chapters),
        }
    except Exception as e:
        logger.error(f"Chapters error: {e}")
        return {
            "ok": False,
            "subject": subject,
            "classGrade": class_grade,
            "chapters": [],
            "groups": None,
            "count": 0,
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Health Detail
# ═══════════════════════════════════════════════════════════════════════

@router.get("/health-detail")
async def health_detail():
    result = {
        "ok": False,
        "services": {"postgresql": False, "supabase": False, "gemini": False, "ncertChunks": 0},
        "version": settings.APP_VERSION,
        "model": settings.GEMINI_MODEL,
    }

    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(settings.DATABASE_URL, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) as count FROM ncert_chunks")
            row = cur.fetchone()
            result["services"]["postgresql"] = True
            result["services"]["ncertChunks"] = row["count"] if row else 0
        conn.close()
    except Exception as e:
        logger.warning(f"PostgreSQL check failed: {e}")

    try:
        sb = get_supabase()
        sb.table("tests").select("id").limit(1).execute()
        result["services"]["supabase"] = True
    except Exception as e:
        logger.warning(f"Supabase check failed: {e}")

    try:
        from google import genai
        client = genai.Client(api_key=settings.GEMINI_API_KEY)
        response = client.models.generate_content(model=settings.GEMINI_MODEL, contents="Reply with just: OK")
        result["services"]["gemini"] = bool(response.text)
    except Exception as e:
        logger.warning(f"Gemini check failed: {e}")

    result["ok"] = all([result["services"]["postgresql"], result["services"]["supabase"], result["services"]["gemini"]])
    return result


# ═══════════════════════════════════════════════════════════════════════
# OTHER ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════

@router.post("/generate", response_model=TestGenerationResponse)
async def generate(request: TestGenerationRequest):
    try:
        chapters = [ch.chapter for ch in request.chapters]
        topics = [ch.topic for ch in request.chapters if ch.topic]
        if not topics:
            topics = chapters
        context_chunks = retrieve_context(chapters, topics, request.subject, request.class_grade)
        return generate_test(request, context_chunks)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generate error: {e}")
        raise HTTPException(status_code=500, detail="Generation failed.")


@router.post("/feedback", response_model=TestGenerationResponse)
async def feedback(request: TestFeedbackRequest):
    try:
        return handle_feedback(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Feedback processing failed.")


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Save Test
# ═══════════════════════════════════════════════════════════════════════

@router.post("/save")
async def save_test(request: FrontendSaveRequest):
    supabase = get_supabase()
    try:
        supabase.table("tests").update({"status": "saved"}).eq("id", request.test_id).execute()

        if request.questions:
            logger.info(f"Saving {len(request.questions)} questions for test {request.test_id}")

            try:
                supabase.table("questions").delete().eq("test_id", request.test_id).execute()
            except Exception as del_err:
                logger.warning(f"Could not delete old questions: {del_err}")

            rows_to_insert = []
            for idx, q in enumerate(request.questions):
                if not isinstance(q, dict):
                    continue

                is_manual = bool(
                    q.get('isManual')
                    or q.get('is_manual')
                    or q.get('validationStatus') == 'manual'
                )

                row = {
                    "id": q.get("id") or str(uuid.uuid4()),
                    "test_id": request.test_id,
                    "position": idx + 1,
                    "text": q.get("text", ""),
                    "options": q.get("options") or [],
                    "correct_answer": q.get("correctAnswer") or q.get("correct_answer", ""),
                    "explanation": q.get("explanation", ""),
                    "marks": q.get("marks", 1),
                    "difficulty": q.get("difficulty", "medium"),
                    "chapter": q.get("chapter", ""),
                    "topic": q.get("topic"),
                    "format": q.get("format", "mcq"),
                    "bloom_level": q.get("bloomLevel") or q.get("bloom_level"),
                    "section": q.get("section"),
                    "is_manual": is_manual,
                    "image_url": q.get("imageUrl") or q.get("image_url"),
                    "answer_table": q.get("answerTable") or q.get("answer_table"),
                    "question_table": q.get("questionTable") or q.get("question_table"),
                }
                rows_to_insert.append(row)

            if rows_to_insert:
                try:
                    supabase.table("questions").insert(rows_to_insert).execute()
                    logger.info(f"Inserted {len(rows_to_insert)} questions")
                except Exception as ins_err:
                    err_str = str(ins_err).lower()
                    if "question_table" in err_str and ("column" in err_str or "schema" in err_str):
                        logger.warning("question_table column missing in DB, retrying without it")
                        for r in rows_to_insert:
                            r.pop("question_table", None)
                        try:
                            supabase.table("questions").insert(rows_to_insert).execute()
                            logger.info(f"Inserted {len(rows_to_insert)} questions (without question_table)")
                        except Exception as retry_err:
                            logger.error(f"Insert retry also failed: {retry_err}")
                    else:
                        logger.error(f"Failed to insert questions: {ins_err}")

        return {"success": True, "test_id": request.test_id, "message": "Test saved."}
    except Exception as e:
        logger.error(f"Save error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Save failed.")


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Add Manual Question
# ═══════════════════════════════════════════════════════════════════════

@router.post("/tests/{test_id}/add-manual-question")
async def add_manual_question(test_id: str, req: AddManualQuestionRequest):
    supabase = get_supabase()
    try:
        gq = req.question.to_generated_question()

        try:
            existing = supabase.table("questions").select("position").eq("test_id", test_id).execute()
            max_pos = max((r.get("position", 0) for r in (existing.data or [])), default=0)
        except Exception:
            max_pos = 0

        row = {
            "id": gq.id,
            "test_id": test_id,
            "position": max_pos + 1,
            "text": gq.text,
            "options": gq.options or [],
            "correct_answer": gq.correct_answer,
            "explanation": gq.explanation,
            "marks": gq.marks,
            "difficulty": gq.difficulty.value,
            "chapter": gq.chapter,
            "topic": gq.topic,
            "format": gq.format.value,
            "bloom_level": None,
            "section": gq.section,
            "is_manual": True,
            "image_url": gq.image_url,
            "answer_table": None,
        }

        try:
            supabase.table("questions").insert(row).execute()
        except Exception as ins_err:
            logger.error(f"Insert manual question failed: {ins_err}")
            raise HTTPException(status_code=500, detail="Failed to save manual question")

        return {
            "ok": True,
            "test_id": test_id,
            "question": {
                "id": gq.id,
                "text": gq.text,
                "options": gq.options,
                "correctAnswer": gq.correct_answer,
                "explanation": gq.explanation,
                "marks": gq.marks,
                "difficulty": gq.difficulty.value,
                "chapter": gq.chapter,
                "format": gq.format.value,
                "section": gq.section,
                "imageUrl": gq.image_url,
                "isManual": True,
                "validationStatus": "manual",
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add manual question error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed: {str(e)}")


@router.post("/quiz/create")
async def create_quiz(settings_req: QuizSettings):
    supabase = get_supabase()
    try:
        quiz_id = str(uuid.uuid4())
        supabase.table("quizzes").insert({
            "id": quiz_id,
            "test_id": settings_req.test_id,
            "teacher_id": settings_req.teacher_id,
            "duration_minutes": settings_req.duration_minutes,
            "max_marks": settings_req.max_marks,
            "passing_marks": settings_req.passing_marks,
            "shuffle_questions": settings_req.shuffle_questions,
            "shuffle_options": settings_req.shuffle_options,
            "camera_required": settings_req.camera_required,
            "tab_switch_limit": settings_req.tab_switch_limit,
            "status": "active",
        }).execute()
        return {"success": True, "quiz_id": quiz_id, "quiz_link": f"/quiz/{quiz_id}"}
    except Exception as e:
        logger.error(f"Quiz create error: {e}")
        raise HTTPException(status_code=500, detail="Quiz creation failed.")


@router.get("/test/{test_id}")
async def get_test(test_id: str, teacher_id: str):
    supabase = get_supabase()
    try:
        test = supabase.table("tests").select("*").eq("id", test_id).eq("teacher_id", teacher_id).single().execute()
        if not test.data:
            raise HTTPException(status_code=404, detail="Test not found")
        questions = supabase.table("questions").select("*").eq("test_id", test_id).order("position").execute()
        return {**test.data, "questions": questions.data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get test error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch test.")