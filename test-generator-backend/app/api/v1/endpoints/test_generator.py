"""
Test Generator API Endpoints — SYNC (Production)

v2.4 changes:
  - Fixed: FORMAT_MAP now includes all format variants (Short, Long, short_answer, long_answer)
  - Fixed: Accountancy import path → test_generator_service (not generation_service)
  - All v2.3 features retained (usage limits, cbsePattern, sections, answerTable)
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
# USAGE CHECK HELPER
# ═══════════════════════════════════════════════════════════════════════

def check_and_record_usage(user_id: str) -> dict:
    if not user_id or user_id == "00000000-0000-0000-0000-000000000000":
        logger.warning("Usage check skipped: no valid user_id")
        return {"allowed": True, "used": 0, "limit": -1, "remaining": -1}

    try:
        supabase = get_supabase()
        result = supabase.rpc("increment_usage", {
            "p_user_id": user_id,
            "p_action": "test_generated",
        }).execute()

        if not result.data:
            logger.warning(f"Usage check returned no data for user {user_id}")
            return {"allowed": True, "used": 0, "limit": -1, "remaining": -1}

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
        logger.error(f"Usage check error (non-blocking): {e}")
        return {"allowed": True, "used": 0, "limit": -1, "remaining": -1}


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
    examTitle: str = "Test Paper"
    board: str = "CBSE"
    classGrade: str = "Class 10"
    subject: str = "Science"
    questions: list
    includeAnswers: bool = False
    includeExplanations: bool = False
    format: str = "pdf"
    logoBase64: Optional[str] = None


# ── Transform helpers ───────────────────────────────────────────────

DIFFICULTY_MAP = {
    "Easy": "easy", "easy": "easy",
    "Medium": "medium", "medium": "medium",
    "Hard": "hard", "hard": "hard",
    "Mixed": "medium", "mixed": "medium",
    "Very Hard": "very_hard", "very_hard": "very_hard",
}

# ═══════════════════════════════════════════════════════════════════════
# FIX v2.4: Complete FORMAT_MAP with ALL format variants
# BUG WAS: "short_answer", "long_answer", "Short", "Long" were MISSING
#          → everything defaulted to MCQ
# ═══════════════════════════════════════════════════════════════════════
FORMAT_MAP = {
    # ── Frontend display values (TestRowEditor buttons) ──
    "MCQ":            QuestionFormat.MCQ,
    "Short":          QuestionFormat.SHORT_ANSWER,
    "Long":           QuestionFormat.LONG_ANSWER,
    "Essay":          QuestionFormat.LONG_ANSWER,

    # ── Backend snake_case values (from useTestGenerator FORMAT_MAP conversion) ──
    "mcq":            QuestionFormat.MCQ,
    "short_answer":   QuestionFormat.SHORT_ANSWER,
    "long_answer":    QuestionFormat.LONG_ANSWER,
    "assertion_reason": QuestionFormat.ASSERTION_REASON,
    "case_based":     QuestionFormat.MCQ,

    # ── Accountancy formats ──
    "Journal Entry":  QuestionFormat.JOURNAL_ENTRY,
    "journal_entry":  QuestionFormat.JOURNAL_ENTRY,
    "JournalEntry":   QuestionFormat.JOURNAL_ENTRY,
    "Ledger":         QuestionFormat.LEDGER,
    "ledger":         QuestionFormat.LEDGER,
    "Trial Balance":  QuestionFormat.TRIAL_BALANCE,
    "trial_balance":  QuestionFormat.TRIAL_BALANCE,
    "TrialBalance":   QuestionFormat.TRIAL_BALANCE,

    # ── Legacy/fallback ──
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
        section = getattr(q, '_section', None)

        answer_table_data = None
        if hasattr(q, 'answer_table') and q.answer_table is not None:
            try:
                if hasattr(q.answer_table, 'model_dump'):
                    answer_table_data = q.answer_table.model_dump()
                elif hasattr(q.answer_table, 'dict'):
                    answer_table_data = q.answer_table.dict()
                elif isinstance(q.answer_table, dict):
                    answer_table_data = q.answer_table
            except Exception:
                answer_table_data = None

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
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Generate from Frontend
# ═══════════════════════════════════════════════════════════════════════

@router.post("/generate-frontend", response_model=FrontendGenerateResponse)
async def generate_from_frontend(req: FrontendGenerateRequest):
    start = time.time()
    logger.info(f"Frontend generate: {req.subject} {req.classGrade}, "
                f"{len(req.simpleData)} chapters, cbsePattern={req.cbsePattern}")

    try:
        usage = check_and_record_usage(req.userId)

        backend_request = _transform_frontend_to_backend(req)
        total_q = sum(c.quantity for c in backend_request.chapters)
        logger.info(f"Transformed: {len(backend_request.chapters)} chapters, {total_q} questions")

        chapters = [ch.chapter for ch in backend_request.chapters]
        topics = [ch.topic for ch in backend_request.chapters if ch.topic]
        if not topics:
            topics = chapters
        context_chunks = retrieve_context(chapters, topics, backend_request.subject, backend_request.class_grade)
        logger.info(f"Retrieved {len(context_chunks)} context chunks")

        # ═══════════════════════════════════════════════════════════
        # FIX v2.4: Import from test_generator_service (NOT generation_service)
        # ═══════════════════════════════════════════════════════════
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
        logger.error(f"Frontend generate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Export PDF / DOCX
# ═══════════════════════════════════════════════════════════════════════

@router.post("/export")
async def export_test(req: ExportRequest):
    try:
        class_num = _extract_class_number(req.classGrade)

        for q in req.questions:
            if isinstance(q, dict) and 'section' not in q:
                q['section'] = None

        if req.format.lower() == "docx":
            from app.services.export_service import generate_docx
            file_bytes = generate_docx(
                questions=req.questions,
                exam_title=req.examTitle,
                board=req.board,
                class_grade=class_num,
                subject=req.subject,
                include_answers=req.includeAnswers,
                include_explanations=req.includeExplanations,
                logo_base64=req.logoBase64,
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
                questions=req.questions,
                exam_title=req.examTitle,
                board=req.board,
                class_grade=class_num,
                subject=req.subject,
                include_answers=req.includeAnswers,
                include_explanations=req.includeExplanations,
                logo_base64=req.logoBase64,
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
# ENDPOINT: Chapters
# ═══════════════════════════════════════════════════════════════════════

@router.get("/chapters")
async def get_chapters(subject: str = "Science", class_grade: str = "10"):
    subject = _resolve_subject(subject)
    try:
        supabase = get_supabase()
        result = supabase.table("ncert_chunks") \
            .select("chapter") \
            .ilike("subject", subject) \
            .eq("class_grade", class_grade) \
            .execute()

        # Extract unique chapters and sort
        chapters = sorted(set(row["chapter"] for row in (result.data or [])))
        return {"ok": True, "subject": subject, "classGrade": class_grade, "chapters": chapters, "count": len(chapters)}
    except Exception as e:
        logger.error(f"Chapters error: {e}")
        return {"ok": False, "subject": subject, "classGrade": class_grade, "chapters": [], "count": 0, "error": str(e)}


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
# OTHER ENDPOINTS (unchanged)
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


@router.post("/save")
async def save_test(request: SaveTestRequest):
    supabase = get_supabase()
    try:
        supabase.table("tests").update({"status": "saved"}).eq("id", request.test_id).execute()
        return {"success": True, "test_id": request.test_id, "message": "Test saved."}
    except Exception as e:
        logger.error(f"Save error: {e}")
        raise HTTPException(status_code=500, detail="Save failed.")


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





        