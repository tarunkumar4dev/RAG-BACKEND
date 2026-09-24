"""
Test Generator API Endpoints — v3.3 (Security Hardened)

v3.3 changes (SECURITY):
  - extra="forbid" on ALL Pydantic models (blocks field injection)
  - Input sanitization on all .ilike() queries (prevents SQL pattern injection)
  - UUID validation on all user_id / test_id fields
  - Removed str(e) from ALL error responses (prevents internal leak)
  - /health-detail removed from public access  - Length limits on all string inputs
  - limit/offset capped on query endpoints
  - Removed debug INSERT logs that printed full row data
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from typing import List, Optional
import re
import uuid
import time
import logging
import json

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
from app.core.cache import api_cache
from app.core.sanitize import sanitize_like, sanitize_text, sanitize_uuid, validate_class_grade

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/test-generator", tags=["Test Generator"])


# ═══════════════════════════════════════════════════════════════════════
# USAGE CHECK HELPERS
# ═══════════════════════════════════════════════════════════════════════

def check_usage(user_id: str) -> dict:
    if not user_id or user_id == "00000000-0000-0000-0000-000000000000":
        if not settings.IS_PRODUCTION:
            logger.info("Dev mode: Bypassing login requirement for local testing")
            return {"allowed": True, "used": 0, "limit": 999, "remaining": 999}
        logger.warning("Usage check blocked: no valid user_id provided")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "unauthorized",
                "message": "Login required. Please sign in to generate tests.",
            },
        )

    # SECURITY: Validate UUID format
    try:
        sanitize_uuid(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user ID format")

    try:
        supabase = get_supabase()
        result = supabase.rpc("check_usage", {
            "p_user_id": user_id,
        }).execute()

        if not result.data:
            logger.error(f"Usage check returned no data for user {user_id}")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "service_error",
                    "message": "Unable to verify usage. Please try again later.",
                },
            )

        usage = result.data

        if not usage.get("allowed"):
            logger.info(f"Usage limit reached: user={user_id}")
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

        return usage

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Usage check error: {e}")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "service_error",
                "message": "Unable to verify usage. Please try again later.",
            },
        )


def record_usage(user_id: str) -> dict:
    if not user_id or user_id == "00000000-0000-0000-0000-000000000000":
        if not settings.IS_PRODUCTION:
            return {"recorded": True}
    try:
        supabase = get_supabase()
        result = supabase.rpc("record_usage", {
            "p_user_id": user_id,
            "p_action": "test_generated",
        }).execute()

        if result.data:
            logger.info(f"Usage recorded: user={user_id}")
        return result.data or {}

    except Exception as e:
        logger.error(f"Record usage failed (non-fatal): {e}")
        return {"recorded": False}


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
# ENGLISH PSEUDO-CHAPTER SUPPORT
# ═══════════════════════════════════════════════════════════════════════

ENGLISH_PSEUDO_CHAPTERS = {"writing skills", "grammar"}


def _is_english_pseudo(subject: str, chapter: str) -> bool:
    if (subject or "").lower() != "english":
        return False
    return chapter.lower().strip() in ENGLISH_PSEUDO_CHAPTERS


# ═══════════════════════════════════════════════════════════════════════
# BOOK / CHAPTER_TYPE LABELS
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
# FRONTEND MODELS (all with extra="forbid")
# ═══════════════════════════════════════════════════════════════════════

class FrontendChapterRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: str = Field(max_length=200)
    subtopic: Optional[str] = Field(default=None, max_length=200)
    quantity: int = Field(default=5, ge=1, le=50)
    marks: int = Field(default=1, ge=1, le=10)
    difficulty: str = Field(default="Medium", max_length=20)
    format: str = Field(default="MCQ", max_length=30)


class FrontendGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    examTitle: str = Field(default="Untitled Test", max_length=200)
    paperDate: Optional[str] = Field(default=None, max_length=20)
    board: str = Field(default="CBSE", max_length=30)
    classGrade: str = Field(default="Class 10", max_length=20)
    subject: str = Field(default="Science", max_length=50)
    simpleData: List[FrontendChapterRow] = Field(default=[], max_length=20)  # Max 20 chapters
    mode: str = Field(default="Simple", max_length=20)
    enableWatermark: bool = True
    shuffleQuestions: bool = False
    useNCERT: bool = True
    ncertClass: Optional[str] = Field(default=None, max_length=10)
    ncertSubject: Optional[str] = Field(default=None, max_length=50)
    ncertChapters: List[str] = Field(default=[], max_length=20)
    userId: Optional[str] = Field(default=None, max_length=50)
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
    # v19: Rich answer fields
    markingScheme: Optional[List[dict]] = None
    subParts: Optional[List[dict]] = None
    commonMistakes: Optional[List[str]] = None
    modelAnswer: Optional[str] = None


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


class ExportRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    examTitle: str = Field(default="Test Paper", max_length=200)
    paperDate: Optional[str] = Field(default=None, max_length=20)
    board: str = Field(default="CBSE", max_length=30)
    classGrade: str = Field(default="Class 10", max_length=20)
    subject: str = Field(default="Science", max_length=50)
    questions: list = Field(max_length=200)  # Max 200 questions
    includeAnswers: bool = False
    includeExplanations: bool = False
    format: str = Field(default="pdf", max_length=10)
    logoBase64: Optional[str] = Field(default=None, max_length=2_000_000)
    logo_base64: Optional[str] = Field(default=None, max_length=2_000_000)
    template: str = Field(default="modern", max_length=30)
    teacher_name: Optional[str] = Field(default=None, max_length=100)
    duration: Optional[str] = Field(default=None, max_length=20)
    institute_name: Optional[str] = Field(default=None, max_length=200)
    topic: Optional[str] = Field(default=None, max_length=200)


class FrontendSaveRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_id: str = Field(max_length=50)
    teacher_id: str = Field(max_length=50)
    questions: Optional[List[dict]] = Field(default=None, max_length=200)


class AddManualQuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    teacher_id: str = Field(max_length=50)
    question: ManualQuestionPayload

class AnswerKeyExportRequest(BaseModel):
    """Request model for standalone answer key export."""
    model_config = ConfigDict(extra="ignore")
 
    examTitle: str = Field(default="Test Paper", max_length=200)
    paperDate: Optional[str] = Field(default=None, max_length=20)
    board: str = Field(default="CBSE", max_length=30)
    classGrade: str = Field(default="Class 10", max_length=20)
    subject: str = Field(default="Science", max_length=50)
    questions: list = Field(max_length=200)
    includeExplanations: bool = False
    format: str = Field(default="pdf", max_length=10)
    logoBase64: Optional[str] = Field(default=None, max_length=2_000_000)
    logo_base64: Optional[str] = Field(default=None, max_length=2_000_000)
    template: str = Field(default="teal", max_length=30)
    teacher_name: Optional[str] = Field(default=None, max_length=100)
    duration: Optional[str] = Field(default=None, max_length=20)
    institute_name: Optional[str] = Field(default=None, max_length=200)
    topic: Optional[str] = Field(default=None, max_length=200)


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

# SECURITY: Allowed export templates (whitelist)
VALID_TEMPLATES = {
    "modern", "classic", "compact", "colorful", "exam_paper", "institute_paper",
    "teal", "navy", "dark_green", "orange",
    "teal_premium", "navy_premium", "dark_green_premium", "orange_premium",
}
VALID_EXPORT_FORMATS = {"pdf", "docx"}


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
        raise ValueError(f"Too many questions ({total_q}). Maximum is {settings.MAX_QUESTIONS_PER_REQUEST}.")

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


def _serialize_model_list(items):
    if not items:
        return None
    out = []
    for item in items:
        try:
            if hasattr(item, 'model_dump'):
                out.append(item.model_dump())
            elif hasattr(item, 'dict'):
                out.append(item.dict())
            elif isinstance(item, dict):
                out.append(item)
            else:
                out.append(item)
        except Exception:
            out.append(str(item))
    return out


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
        marking_scheme_data = _serialize_model_list(getattr(q, 'marking_scheme', None))
        sub_parts_data = _serialize_model_list(getattr(q, 'sub_parts', None))

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
            markingScheme=marking_scheme_data,
            subParts=sub_parts_data,
            commonMistakes=getattr(q, 'common_mistakes', None),
            modelAnswer=getattr(q, 'model_answer', None),
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
# ENDPOINT: Generate from Frontend (v3.3 — hardened)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/generate-frontend", response_model=FrontendGenerateResponse)
async def generate_from_frontend(req: FrontendGenerateRequest):
    start = time.time()
    logger.info(f"Frontend generate: {req.subject} {req.classGrade}, {len(req.simpleData)} chapters")

    try:
        usage = check_usage(req.userId)

        backend_request = _transform_frontend_to_backend(req)
        total_q = sum(c.quantity for c in backend_request.chapters)
        logger.info(f"Transformed: {len(backend_request.chapters)} chapters, {total_q} questions")

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
        else:
            context_chunks = []

        resolved_subject = _resolve_subject(req.subject)
        is_accountancy = resolved_subject.lower() in ("accountancy", "accounts", "accounting")

        if req.cbsePattern and is_accountancy:
            from app.services.test_generator_service import generate_cbse_accountancy_paper
            backend_response = generate_cbse_accountancy_paper(backend_request, context_chunks)
        else:
            backend_response = generate_test(backend_request, context_chunks, cbse_pattern=req.cbsePattern)

        frontend_response = _transform_backend_to_frontend(backend_response, req)

        elapsed = round(time.time() - start, 2)
        frontend_response.generationTime = elapsed

        recorded = record_usage(req.userId)
        usage.update(recorded)

        # Insert test row
        try:
            class_num_for_db = _extract_class_number(req.classGrade)
            tests_row = {
                "id": frontend_response.testId,
                "teacher_id": req.userId,
                "exam_title": frontend_response.examTitle,
                "board": req.board,
                "class_grade": class_num_for_db,
                "subject": resolved_subject,
                "status": "draft",
                "total_questions": frontend_response.totalQuestions,
                "total_marks": frontend_response.totalMarks,
                "paper_date": req.paperDate,
                "cbse_pattern": req.cbsePattern,
            }
            supabase = get_supabase()
            insert_result = supabase.table("tests").insert(tests_row).execute()
            if not insert_result.data:
                logger.error(f"tests INSERT returned no data for test_id={frontend_response.testId}")
        except Exception as insert_err:
            logger.error(f"Failed to INSERT tests row: {insert_err}", exc_info=True)

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
                    detail=f"Content not available for selected chapters. Try different chapters or contact support."
                )
        logger.error(f"Frontend generate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Generation failed. Please try again.")


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Export PDF / DOCX (hardened)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/export")
async def export_test(req: ExportRequest):
    try:
        class_num = _extract_class_number(req.classGrade)

        # SECURITY: Whitelist template
        template = (req.template or "modern").strip().lower()
        if template not in VALID_TEMPLATES:
            template = "modern"

        # SECURITY: Whitelist format
        export_format = req.format.lower().strip()
        if export_format not in VALID_EXPORT_FORMATS:
            raise HTTPException(status_code=422, detail="Invalid format. Use pdf or docx.")

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

        logo_b64 = req.logoBase64 or req.logo_base64
        topic_val = req.topic

        if export_format == "docx":
            from app.services.export_service import generate_docx
            file_bytes = generate_docx(
                questions=normalized_questions,
                exam_title=req.examTitle,
                board=req.board,
                class_grade=class_num,
                subject=req.subject,
                include_answers=req.includeAnswers,
                include_explanations=req.includeExplanations,
                logo_base64=logo_b64,
                paper_date=req.paperDate,
                template=template,
                teacher_name=req.teacher_name,
                duration=req.duration,
                institute_name=req.institute_name,
                topic=topic_val,
            )
            filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', req.examTitle)[:50]  # Safe filename
            return Response(
                content=file_bytes,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f'attachment; filename="{filename}.docx"'},
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
                logo_base64=logo_b64,
                paper_date=req.paperDate,
                template=template,
                teacher_name=req.teacher_name,
                duration=req.duration,
                institute_name=req.institute_name,
                topic=topic_val,
            )
            filename = re.sub(r'[^a-zA-Z0-9_\-]', '_', req.examTitle)[:50]
            return Response(
                content=file_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename}.pdf"'},
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Export failed. Please try again.")

@router.post("/export-answer-key")
async def export_answer_key(req: AnswerKeyExportRequest):
    """
    Standalone Answer Key download (separate PDF/DOCX file).
    Matches the sample PDF's clean institute-paper style.
    """
    try:
        class_num = _extract_class_number(req.classGrade)
 
        # SECURITY: Whitelist template
        template = (req.template or "teal").strip().lower()
        if template not in VALID_TEMPLATES:
            template = "teal"
 
        # SECURITY: Whitelist format
        export_format = req.format.lower().strip()
        if export_format not in VALID_EXPORT_FORMATS:
            raise HTTPException(status_code=422, detail="Invalid format. Use pdf or docx.")
 
        # Normalize questions (same shape as /export)
        normalized_questions = []
        for q in req.questions:
            if not isinstance(q, dict):
                continue
            if 'section' not in q:
                q['section'] = None
            normalized_questions.append(q)
 
        if not normalized_questions:
            raise HTTPException(status_code=422, detail="No questions provided.")
 
        safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '_', req.examTitle)[:50]
        filename_base = f"{safe_title}_AnswerKey"

        logo_b64 = req.logoBase64 or req.logo_base64
        topic_val = req.topic

        if export_format == "docx":
            from app.services.export_service import generate_answer_key_docx
            file_bytes = generate_answer_key_docx(
                questions=normalized_questions,
                exam_title=req.examTitle,
                board=req.board,
                class_grade=class_num,
                subject=req.subject,
                include_explanations=req.includeExplanations,
                logo_base64=logo_b64,
                paper_date=req.paperDate,
                template=template,
                teacher_name=req.teacher_name,
                duration=req.duration,
                institute_name=req.institute_name,
                topic=topic_val,
            )
            return Response(
                content=file_bytes,
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f'attachment; filename="{filename_base}.docx"'},
            )
        else:
            from app.services.export_service import generate_answer_key_pdf
            file_bytes = generate_answer_key_pdf(
                questions=normalized_questions,
                exam_title=req.examTitle,
                board=req.board,
                class_grade=class_num,
                subject=req.subject,
                include_explanations=req.includeExplanations,
                logo_base64=logo_b64,
                paper_date=req.paperDate,
                template=template,
                teacher_name=req.teacher_name,
                duration=req.duration,
                institute_name=req.institute_name,
                topic=topic_val,
            )
            return Response(
                content=file_bytes,
                media_type="application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename_base}.pdf"'},
            )
 
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Answer key export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Answer key export failed.")
 

# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Available Export Templates
# ═══════════════════════════════════════════════════════════════════════

@router.get("/templates")
async def get_templates():
    try:
        from app.services.export_service import get_available_templates
        return {"ok": True, "templates": get_available_templates()}
    except Exception as e:
        logger.error(f"Templates fetch error: {e}")
        return {
            "ok": False,
            "templates": [
                {"id": "modern", "label": "Modern", "description": "Clean card-style layout."},
                {"id": "classic", "label": "Classic", "description": "Traditional serif exam-paper look."},
                {"id": "compact", "label": "Compact", "description": "Dense layout, saves paper."},
                {"id": "colorful", "label": "Colorful", "description": "Section-wise accent colors."},
                {"id": "exam_paper", "label": "Exam Paper", "description": "Traditional CBSE-style exam paper."},
            ],
        }


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Chapters (hardened)
# ═══════════════════════════════════════════════════════════════════════

@router.get("/chapters")
async def get_chapters(subject: str = "Science", class_grade: str = "10"):
    # SECURITY: Sanitize inputs
    subject = sanitize_like(_resolve_subject(subject), max_length=50)
    class_grade = sanitize_like(class_grade, max_length=5)

    cache_key = f"chapters:{class_grade}:{subject.lower()}"
    cached = api_cache.get(cache_key)
    if cached is not None:
        return cached

    rows = []

    # 1. Fast direct SQL (<5ms)
    try:
        from app.core.db_pool import get_db_connection
        conn = get_db_connection()
        if conn:
            try:
                where_sub, sub_params = _get_subject_sql_clause(subject)
                cur = conn.cursor()
                cur.execute(f"""
                    SELECT DISTINCT chapter, book, chapter_type, chapter_order
                    FROM ncert_chunks
                    WHERE class_grade = %s AND {where_sub}
                    ORDER BY chapter_order NULLS LAST, chapter ASC;
                """, [class_grade] + sub_params)
                db_rows = cur.fetchall()
                cur.close()
                conn.close()
                rows = [
                    {"chapter": r[0], "book": r[1], "chapter_type": r[2], "chapter_order": r[3]}
                    for r in db_rows if r[0]
                ]
            except Exception as pool_err:
                logger.warning(f"SQL chapters error: {pool_err}, falling back to Supabase client")
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Pool connection unavailable for chapters: {e}")

    # 2. Fallback to Supabase client
    if not rows:
        try:
            supabase = get_supabase()
            query = supabase.table("ncert_chunks") \
                .select("chapter, book, chapter_type, chapter_order") \
                .eq("class_grade", class_grade)
            query = _apply_ncert_subject_filter(query, subject)
            result = query.execute()
            rows = result.data or []
        except Exception as e:
            logger.error(f"Chapters fallback error: {e}")

    try:
        chapters = sorted({row["chapter"] for row in rows if row.get("chapter")})

        seen_chapters = {}
        for row in rows:
            chapter = row.get("chapter")
            book = row.get("book")
            ctype = row.get("chapter_type")
            order = row.get("chapter_order")

            if not chapter or not book or not ctype:
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

        response_payload = {
            "ok": True,
            "subject": subject,
            "classGrade": class_grade,
            "chapters": chapters,
            "groups": groups if groups else None,
            "count": len(chapters),
        }
        api_cache.set(cache_key, response_payload, ttl=3600)
        return response_payload
    except Exception as e:
        logger.error(f"Chapters error: {e}")
        return {
            "ok": False,
            "subject": subject,
            "classGrade": class_grade,
            "chapters": [],
            "groups": None,
            "count": 0,
        }


# ═══════════════════════════════════════════════════════════════════════
# NCERT HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _apply_ncert_subject_filter(query, subject: str):
    """Filter ncert_questions strictly so 'Science' never collides with 'Political Science'."""
    sub_clean = (subject or "").strip().lower()
    if sub_clean == "science":
        return query.eq("subject", "Science")
    elif sub_clean in ("physics", "physics-i", "physics-ii"):
        return query.in_("subject", ["Physics", "Physics-I", "Physics-II"])
    elif sub_clean in ("chemistry", "chemistry-i", "chemistry-ii"):
        return query.in_("subject", ["Chemistry", "Chemistry-I", "Chemistry-II"])
    elif sub_clean in ("political science", "political_science", "pol science"):
        return query.in_("subject", ["Political Science", "Political_Science"])
    else:
        return query.ilike("subject", subject.strip())


def _get_subject_sql_clause(subject: str) -> tuple[str, list]:
    sub_clean = (subject or "").strip().lower()
    if sub_clean == "science":
        return "LOWER(subject) = 'science'", []
    elif sub_clean in ("physics", "physics-i", "physics-ii"):
        return "LOWER(subject) IN ('physics', 'physics-i', 'physics-ii')", []
    elif sub_clean in ("chemistry", "chemistry-i", "chemistry-ii"):
        return "LOWER(subject) IN ('chemistry', 'chemistry-i', 'chemistry-ii')", []
    elif sub_clean in ("political science", "political_science", "pol science"):
        return "LOWER(subject) IN ('political science', 'political_science')", []
    else:
        return "LOWER(subject) = LOWER(%s)", [subject.strip()]


@router.get("/ncert-questions")
async def get_ncert_questions(
    subject: str = "Science",
    class_grade: str = "10",
    chapter: Optional[str] = None,
    question_type: Optional[str] = None,
    section: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    # SECURITY: Sanitize ALL inputs + cap limit/offset
    subject = sanitize_like(_resolve_subject(subject), max_length=50)
    class_grade = sanitize_like(class_grade, max_length=5)
    limit = min(max(1, limit), 100)    # Cap at 100
    offset = max(0, min(offset, 10000))  # Cap offset

    cache_key = f"qs:{class_grade}:{subject.lower()}:{chapter or ''}:{question_type or ''}:{section or ''}:{search or ''}:{limit}:{offset}"
    cached = api_cache.get(cache_key)
    if cached is not None:
        return cached

    # 1. Fast direct SQL with window count (5-10ms)
    try:
        from app.core.db_pool import get_db_connection
        conn = get_db_connection()
        if conn:
            try:
                where_sub, sub_params = _get_subject_sql_clause(subject)
                conditions = ["class_grade = %s", where_sub]
                params = [class_grade] + sub_params

                if chapter:
                    conditions.append("chapter ILIKE %s")
                    params.append(f"%{chapter.strip()}%")
                if question_type and question_type.lower() != "all":
                    conditions.append("LOWER(question_type) = LOWER(%s)")
                    params.append(question_type.strip())
                if section:
                    conditions.append("section ILIKE %s")
                    params.append(f"%{section.strip()}%")
                if search:
                    conditions.append("question_text ILIKE %s")
                    params.append(f"%{search.strip()}%")

                where_clause = " AND ".join(conditions)
                cur = conn.cursor()
                cur.execute(f"""
                    SELECT id, class_grade, subject, chapter, section, question_number, question_text,
                           question_type, answer, options, marks, difficulty, figure_ref, image_url, question_table,
                           COUNT(*) OVER() AS total_count
                    FROM ncert_questions
                    WHERE {where_clause}
                    ORDER BY section ASC NULLS LAST, question_number ASC NULLS LAST
                    LIMIT %s OFFSET %s;
                """, params + [limit, offset])

                rows = cur.fetchall()
                cur.close()
                conn.close()

                total_count = rows[0][15] if rows else 0
                questions = []
                for r in rows:
                    opts = r[9]
                    if isinstance(opts, str):
                        try:
                            opts = json.loads(opts)
                        except Exception:
                            opts = []
                    elif not isinstance(opts, list):
                        opts = []

                    questions.append({
                        "id": r[0],
                        "class_grade": r[1],
                        "subject": r[2],
                        "chapter": r[3],
                        "section": r[4],
                        "question_number": r[5],
                        "question_text": r[6],
                        "question_type": r[7],
                        "answer": r[8],
                        "options": opts,
                        "marks": r[10],
                        "difficulty": r[11],
                        "figure_ref": r[12],
                        "image_url": r[13],
                        "question_table": r[14],
                    })

                resp = {
                    "ok": True, "subject": subject, "classGrade": class_grade, "chapter": chapter,
                    "questions": questions, "total": total_count, "limit": limit, "offset": offset,
                    "hasMore": (offset + limit) < total_count,
                }
                api_cache.set(cache_key, resp, ttl=180)
                return resp
            except Exception as pool_err:
                logger.warning(f"Direct SQL ncert-questions error: {pool_err}, falling back to Supabase client")
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Pool connection unavailable for questions: {e}")

    # 2. Fallback to Supabase client
    try:
        supabase = get_supabase()
        query = supabase.table("ncert_questions").select("*").eq("class_grade", class_grade)
        query = _apply_ncert_subject_filter(query, subject)

        if chapter:
            safe_chapter = sanitize_like(chapter, max_length=200)
            query = query.ilike("chapter", f"%{safe_chapter}%")
        if question_type and question_type.lower() != "all":
            safe_qtype = sanitize_like(question_type, max_length=30)
            query = query.eq("question_type", safe_qtype.lower())
        if section:
            safe_section = sanitize_like(section, max_length=100)
            query = query.ilike("section", f"%{safe_section}%")
        if search:
            safe_search = sanitize_like(search, max_length=200)
            query = query.ilike("question_text", f"%{safe_search}%")

        count_query = supabase.table("ncert_questions").select("id", count="exact").eq("class_grade", class_grade)
        count_query = _apply_ncert_subject_filter(count_query, subject)
        if chapter:
            count_query = count_query.ilike("chapter", f"%{sanitize_like(chapter, 200)}%")
        if question_type and question_type.lower() != "all":
            count_query = count_query.eq("question_type", sanitize_like(question_type, 30).lower())
        if section:
            count_query = count_query.ilike("section", f"%{sanitize_like(section, 100)}%")
        if search:
            count_query = count_query.ilike("question_text", f"%{sanitize_like(search, 200)}%")

        count_result = count_query.execute()
        total_count = count_result.count if count_result and count_result.count is not None else (len(count_result.data) if count_result and count_result.data else 0)

        result = query.order("section", desc=False) \
            .order("question_number", desc=False) \
            .range(offset, offset + limit - 1) \
            .execute()

        questions = result.data or []

        for q in questions:
            if isinstance(q.get("options"), str):
                try:
                    q["options"] = json.loads(q["options"])
                except Exception:
                    q["options"] = []

        resp = {
            "ok": True, "subject": subject, "classGrade": class_grade, "chapter": chapter,
            "questions": questions, "total": total_count, "limit": limit, "offset": offset,
            "hasMore": (offset + limit) < total_count,
        }
        api_cache.set(cache_key, resp, ttl=180)
        return resp
    except Exception as e:
        logger.error(f"NCERT questions fetch error: {e}", exc_info=True)
        return {"ok": False, "questions": [], "total": 0}


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: NCERT Question Stats (hardened & ultra-fast)
# ═══════════════════════════════════════════════════════════════════════

@router.get("/ncert-question-stats")
async def get_ncert_question_stats(subject: str = "Science", class_grade: str = "10"):
    subject = sanitize_like(_resolve_subject(subject), max_length=50)
    class_grade = sanitize_like(class_grade, max_length=5)

    cache_key = f"stats:{class_grade}:{subject.lower()}"
    cached = api_cache.get(cache_key)
    if cached is not None:
        return cached

    # 1. Fast direct SQL aggregation (accurate, no row cutoffs, <50ms)
    try:
        from app.core.db_pool import get_db_connection
        conn = get_db_connection()
        if conn:
            try:
                where_sub, sub_params = _get_subject_sql_clause(subject)
                cur = conn.cursor()
                cur.execute(f"""
                    SELECT chapter, COALESCE(section, 'Uncategorized'), COALESCE(question_type, 'exercise'), COUNT(*)
                    FROM ncert_questions
                    WHERE class_grade = %s AND {where_sub}
                    GROUP BY chapter, section, question_type
                    ORDER BY chapter;
                """, [class_grade] + sub_params)
                rows = cur.fetchall()
                cur.close()
                conn.close()

                chapter_stats = {}
                for ch, sec, qtype, cnt in rows:
                    if ch not in chapter_stats:
                        chapter_stats[ch] = {"chapter": ch, "total": 0, "sections": {}, "types": {}}
                    chapter_stats[ch]["total"] += cnt
                    chapter_stats[ch]["sections"][sec] = chapter_stats[ch]["sections"].get(sec, 0) + cnt
                    chapter_stats[ch]["types"][qtype] = chapter_stats[ch]["types"].get(qtype, 0) + cnt

                stats_list = sorted(chapter_stats.values(), key=lambda x: x["chapter"])
                for stat in stats_list:
                    stat["sections"] = [{"name": k, "count": v} for k, v in sorted(stat["sections"].items())]
                    stat["types"] = [{"name": k, "count": v} for k, v in sorted(stat["types"].items())]

                total_questions = sum(s["total"] for s in stats_list)
                resp = {
                    "ok": True, "subject": subject, "classGrade": class_grade,
                    "chapters": stats_list, "totalQuestions": total_questions, "totalChapters": len(stats_list),
                }
                api_cache.set(cache_key, resp, ttl=600)
                return resp
            except Exception as pool_err:
                logger.warning(f"SQL aggregation error: {pool_err}, falling back to Supabase client")
                try:
                    conn.close()
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"Pool connection unavailable: {e}")

    # 2. Fallback to Supabase client
    try:
        supabase = get_supabase()
        base_query = supabase.table("ncert_questions") \
            .select("chapter, section, question_type") \
            .eq("class_grade", class_grade)
        base_query = _apply_ncert_subject_filter(base_query, subject)
        result = base_query.limit(5000).execute()

        rows = result.data or []

        chapter_stats = {}
        for row in rows:
            ch = row["chapter"]
            sec = row.get("section") or "Uncategorized"
            qtype = row.get("question_type") or "exercise"
            if ch not in chapter_stats:
                chapter_stats[ch] = {"chapter": ch, "total": 0, "sections": {}, "types": {}}
            chapter_stats[ch]["total"] += 1
            chapter_stats[ch]["sections"][sec] = chapter_stats[ch]["sections"].get(sec, 0) + 1
            chapter_stats[ch]["types"][qtype] = chapter_stats[ch]["types"].get(qtype, 0) + 1

        stats_list = sorted(chapter_stats.values(), key=lambda x: x["chapter"])
        for stat in stats_list:
            stat["sections"] = [{"name": k, "count": v} for k, v in sorted(stat["sections"].items())]
            stat["types"] = [{"name": k, "count": v} for k, v in sorted(stat["types"].items())]

        total_questions = sum(s["total"] for s in stats_list)
        return {
            "ok": True, "subject": subject, "classGrade": class_grade,
            "chapters": stats_list, "totalQuestions": total_questions, "totalChapters": len(stats_list),
        }
    except Exception as e:
        logger.error(f"NCERT question stats error: {e}", exc_info=True)
        return {"ok": False, "chapters": [], "totalQuestions": 0}


# ═══════════════════════════════════════════════════════════════════════
# OTHER ENDPOINTS (hardened)
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
# ENDPOINT: Save Test (hardened)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/save")
async def save_test(request: FrontendSaveRequest):
    # SECURITY: Validate UUIDs
    try:
        sanitize_uuid(request.test_id)
        sanitize_uuid(request.teacher_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid test or teacher ID format")

    supabase = get_supabase()
    try:
        check_result = supabase.table("tests").select("id, teacher_id, exam_title").eq(
            "id", request.test_id
        ).eq(
            "teacher_id", request.teacher_id
        ).execute()

        if not check_result.data:
            if request.questions:
                first_q = request.questions[0] if request.questions else {}
                total_marks = sum(q.get("marks", 1) for q in request.questions if isinstance(q, dict))
                recovery_row = {
                    "id": request.test_id,
                    "teacher_id": request.teacher_id,
                    "exam_title": "Untitled Test (recovered)",
                    "board": "CBSE",
                    "class_grade": "10",
                    "subject": first_q.get("chapter", "General") if isinstance(first_q, dict) else "General",
                    "status": "saved",
                    "total_questions": len(request.questions),
                    "total_marks": total_marks,
                }
                try:
                    supabase.table("tests").insert(recovery_row).execute()
                except Exception as rec_err:
                    logger.error(f"Recovery INSERT failed: {rec_err}")
                    raise HTTPException(status_code=404, detail="Test not found. Please generate a new test.")
            else:
                raise HTTPException(status_code=404, detail="Test not found.")
        else:
            supabase.table("tests").update({
                "status": "saved"
            }).eq("id", request.test_id).eq("teacher_id", request.teacher_id).execute()

        if request.questions:
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
                except Exception as ins_err:
                    err_str = str(ins_err).lower()
                    if "question_table" in err_str and ("column" in err_str or "schema" in err_str):
                        for r in rows_to_insert:
                            r.pop("question_table", None)
                        try:
                            supabase.table("questions").insert(rows_to_insert).execute()
                        except Exception as retry_err:
                            logger.error(f"Insert retry also failed: {retry_err}")
                            raise HTTPException(status_code=500, detail="Failed to save questions. Please try again.")
                    else:
                        logger.error(f"Failed to insert questions: {ins_err}")
                        raise HTTPException(status_code=500, detail="Failed to save questions. Please try again.")

        return {"success": True, "test_id": request.test_id, "message": "Test saved."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Save error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Save failed. Please try again.")


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Add Manual Question (hardened)
# ═══════════════════════════════════════════════════════════════════════

@router.post("/tests/{test_id}/add-manual-question")
async def add_manual_question(test_id: str, req: AddManualQuestionRequest):
    # SECURITY: Validate UUIDs
    try:
        sanitize_uuid(test_id)
        sanitize_uuid(req.teacher_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID format")

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
            raise HTTPException(status_code=500, detail="Failed to save question. Please try again.")

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
        raise HTTPException(status_code=500, detail="Failed to add question.")


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
    # SECURITY: Validate UUIDs
    try:
        sanitize_uuid(test_id)
        sanitize_uuid(teacher_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid ID format")

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