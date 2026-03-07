"""
Test Generator API Endpoints — COMPLETE FILE

ALL endpoints:
  POST /generate-frontend → React form data → pipeline → questions (camelCase)
  POST /generate          → Direct backend-format request
  POST /feedback          → Teacher feedback + regenerate
  POST /save              → Save final test
  POST /quiz/create       → Create student quiz
  POST /export            → Download PDF or DOCX
  GET  /test/{test_id}    → Get saved test
  GET  /chapters          → NCERT chapters for dropdowns
  GET  /health-detail     → Detailed health check
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
from app.core.database import get_supabase
from app.core.config import settings
import logging
import uuid
import time

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/test-generator", tags=["Test Generator"])


# ═══════════════════════════════════════════════════════════════════════
# FRONTEND BRIDGE — Models & Helpers
# ═══════════════════════════════════════════════════════════════════════

class FrontendChapterRow(BaseModel):
    topic: str
    subtopic: Optional[str] = None
    quantity: int = Field(default=5, ge=1, le=50)
    difficulty: str = "Medium"
    format: str = "PDF"


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


def _extract_class_number(class_grade: str) -> str:
    match = re.search(r'\d+', class_grade)
    return match.group() if match else "10"


def _transform_frontend_to_backend(req: FrontendGenerateRequest) -> TestGenerationRequest:
    class_num = _extract_class_number(req.classGrade)
    chapters = []
    for row in req.simpleData:
        if not row.topic:
            continue
        difficulty_str = DIFFICULTY_MAP.get(row.difficulty, "medium")
        chapter = ChapterSection(
            chapter=row.topic,
            topic=row.subtopic if row.subtopic else None,
            subtopics=[row.subtopic] if row.subtopic else [],
            quantity=row.quantity,
            difficulty=DifficultyLevel(difficulty_str),
            format=QuestionFormat.MCQ,
            marks_per_question=1,
        )
        chapters.append(chapter)

    if not chapters:
        raise ValueError("At least one chapter with a topic is required")

    total_q = sum(c.quantity for c in chapters)
    if total_q > settings.MAX_QUESTIONS_PER_REQUEST:
        raise ValueError(f"Too many questions ({total_q}). Maximum {settings.MAX_QUESTIONS_PER_REQUEST} per test.")

    return TestGenerationRequest(
        exam_title=req.examTitle,
        board=req.board,
        class_grade=class_num,
        subject=req.subject,
        chapters=chapters,
        pattern="simple",
        bloom_enabled=True,
        teacher_id=req.userId or "00000000-0000-0000-0000-000000000000",
        iteration=0,
    )


def _transform_backend_to_frontend(resp: TestGenerationResponse) -> FrontendGenerateResponse:
    questions = []
    for q in resp.questions:
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
        ))

    return FrontendGenerateResponse(
        ok=True,
        testId=resp.test_id,
        examTitle=resp.exam_title,
        questions=questions,
        totalMarks=resp.total_marks,
        totalQuestions=resp.total_questions,
        generationTime=resp.generation_time_seconds,
        status=resp.status,
        meta={"ncertBased": True, "ragUsed": True, "iteration": resp.iteration},
    )


# ═══════════════════════════════════════════════════════════════════════
# ENDPOINT: Generate from Frontend
# ═══════════════════════════════════════════════════════════════════════

@router.post("/generate-frontend", response_model=FrontendGenerateResponse)
async def generate_from_frontend(req: FrontendGenerateRequest):
    start = time.time()
    logger.info(f"Frontend generate: {req.subject} {req.classGrade}, {len(req.simpleData)} chapters")

    try:
        backend_request = _transform_frontend_to_backend(req)
        total_q = sum(c.quantity for c in backend_request.chapters)
        logger.info(f"Transformed: {len(backend_request.chapters)} chapters, {total_q} questions")

        backend_response = generate_test(backend_request)
        frontend_response = _transform_backend_to_frontend(backend_response)

        elapsed = round(time.time() - start, 2)
        frontend_response.generationTime = elapsed
        logger.info(f"Done: {frontend_response.totalQuestions} questions in {elapsed}s")
        return frontend_response

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
    """Download PDF or DOCX — student copy, answer key, or teacher copy."""
    try:
        class_num = _extract_class_number(req.classGrade)

        if req.format == "docx":
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
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
        conn = psycopg2.connect(settings.DATABASE_URL, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT chapter FROM ncert_chunks WHERE subject = %s AND class_grade = %s ORDER BY chapter",
                (subject, class_grade),
            )
            rows = cur.fetchall()
        conn.close()
        chapters = [row["chapter"] for row in rows]
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
# EXISTING ENDPOINTS — UNTOUCHED
# ═══════════════════════════════════════════════════════════════════════

@router.post("/generate", response_model=TestGenerationResponse)
async def generate(request: TestGenerationRequest):
    try:
        return generate_test(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Generate error: {e}")
        raise HTTPException(status_code=500, detail="Generation failed. Please try again.")


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
        return {"success": True, "quiz_id": quiz_id, "quiz_link": f"/quiz/{quiz_id}", "message": "Quiz created."}
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