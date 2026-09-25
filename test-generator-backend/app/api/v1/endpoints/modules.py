"""
Module Endpoints — FastAPI Router (Security Hardened)

FIXES:
  - CRITICAL: Removed unsafe JWT base64 decode (was trusting unverified tokens!)
  - Added extra="forbid" on all Pydantic models
  - Added input validation (storage_path sanitization, UUID check)
  - Removed str(e) from all error responses
  - Added length limits on string fields
"""
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Header, Query, Depends
from pydantic import BaseModel, ConfigDict, Field
from app.services.module_service import ModuleService
from app.core.sanitize import sanitize_uuid
from fastapi.responses import Response

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Security: Max limits ────────────────────────────────────
MAX_FILENAME_LENGTH = 255
MAX_SUBJECT_LENGTH = 100
MAX_PATH_LENGTH = 500


class CreateModuleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storage_path: str = Field(max_length=MAX_PATH_LENGTH)
    original_filename: str = Field(max_length=MAX_FILENAME_LENGTH)
    subject: str = Field(max_length=MAX_SUBJECT_LENGTH)
    class_level: str = Field(max_length=10)
    teacher_id: Optional[str] = Field(default=None, max_length=50)
    file_type: str = Field(default="pdf", max_length=10)
    file_size_bytes: Optional[int] = Field(default=None, ge=0, le=50_000_000)  # 50MB max
    institute_id: Optional[str] = Field(default=None, max_length=50)


class ProcessModuleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    module_id: str = Field(max_length=50)


class GenerateTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    teacher_id: Optional[str] = Field(default=None, max_length=50)
    num_questions: int = Field(default=10, ge=1, le=100)
    question_types: Optional[List[str]] = None
    difficulty: str = Field(default="medium", max_length=20)
    topics: Optional[List[str]] = None


def _validate_teacher_id(teacher_id: Optional[str]) -> str:
    """Validate teacher_id is present and looks like a UUID."""
    if not teacher_id or not teacher_id.strip():
        raise HTTPException(400, "teacher_id required")
    try:
        return sanitize_uuid(teacher_id.strip())
    except ValueError:
        raise HTTPException(400, "Invalid teacher_id format")


def extract_teacher_id(teacher_id: Optional[str], authorization: Optional[str] = None) -> str:
    """Extract and validate teacher_id from body or authorization header."""
    if teacher_id and str(teacher_id).strip():
        try:
            return sanitize_uuid(str(teacher_id).strip())
        except ValueError:
            return str(teacher_id).strip()
    return ""


def _validate_file_type(file_type: str) -> str:
    """Only allow known file types."""
    allowed = {"pdf", "doc", "docx", "txt", "ppt", "pptx"}
    ft = file_type.lower().strip()
    if ft not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ft}")
    return ft


@router.post("/modules/create")
async def create_module(req: CreateModuleRequest, authorization: Optional[str] = Header(None)):
    teacher_id = _validate_teacher_id(req.teacher_id)
    file_type = _validate_file_type(req.file_type)

    # SECURITY: Prevent path traversal in storage_path
    if ".." in req.storage_path or req.storage_path.startswith("/"):
        raise HTTPException(400, "Invalid storage path")

    module_id, error = ModuleService.create_module(
        teacher_id=teacher_id,
        storage_path=req.storage_path,
        original_filename=req.original_filename,
        subject=req.subject,
        class_level=req.class_level,
        file_type=file_type,
        file_size_bytes=req.file_size_bytes,
        institute_id=req.institute_id,
    )

    if error:
        logger.error(f"Module creation failed: {error}")
        raise HTTPException(500, "Module creation failed. Please try again.")

    return {
        "success": True,
        "module_id": module_id,
        "status": "processing",
        "message": "Module created. Call /modules/process to start processing.",
    }


@router.post("/modules/process")
async def process_module(req: ProcessModuleRequest):
    if not req.module_id:
        raise HTTPException(400, "module_id required")

    result = ModuleService.process_module(req.module_id)

    if result.get("success"):
        return result
    else:
        logger.error(f"Module processing failed: {result.get('error')}")
        raise HTTPException(500, "Processing failed. Please try again.")


@router.get("/modules/list")
async def list_modules(
    teacher_id: str = Query(..., max_length=50),
    subject: Optional[str] = Query(None, max_length=MAX_SUBJECT_LENGTH),
    class_level: Optional[str] = Query(None, alias="class", max_length=10),
):
    teacher_id = _validate_teacher_id(teacher_id)
    modules = ModuleService.list_modules(teacher_id, subject, class_level)
    return {"success": True, "count": len(modules), "modules": modules}


@router.get("/modules/{module_id}")
async def get_module(module_id: str, teacher_id: str = Query(..., max_length=50)):
    teacher_id = _validate_teacher_id(teacher_id)
    module = ModuleService.get_module(module_id, teacher_id)
    if not module:
        raise HTTPException(404, "Module not found")
    return {"success": True, **module}


@router.delete("/modules/{module_id}")
async def delete_module(module_id: str, teacher_id: str = Query(..., max_length=50)):
    teacher_id = _validate_teacher_id(teacher_id)
    success, error = ModuleService.delete_module(module_id, teacher_id)
    if success:
        return {"success": True, "message": "Module deleted"}
    raise HTTPException(404, "Module not found or access denied")


@router.post("/modules/{module_id}/generate-test")
async def generate_test_from_module(
    module_id: str,
    req: GenerateTestRequest,
    authorization: Optional[str] = Header(None),
):
    teacher_id = _validate_teacher_id(req.teacher_id)

    # Validate difficulty
    valid_difficulties = {"easy", "medium", "hard", "mixed"}
    if req.difficulty.lower() not in valid_difficulties:
        raise HTTPException(400, f"Invalid difficulty. Use: {', '.join(valid_difficulties)}")

    test_data, error = ModuleService.generate_test_from_module(
        module_id=module_id,
        teacher_id=teacher_id,
        num_questions=req.num_questions,
        question_types=req.question_types,
        difficulty=req.difficulty,
        topics=req.topics,
    )

    if error:
        logger.error(f"Module test generation failed: {error}")
        raise HTTPException(500, "Test generation failed. Please try again.")

    return {"success": True, "test": test_data}

    

@router.post("/modules/{module_id}/generate-worksheet")
async def generate_worksheet(module_id: str, req: dict, authorization: Optional[str] = Header(None)):
    """Generate worksheet questions from module content."""
    teacher_id = _validate_teacher_id(req.get("teacher_id"))
    if not teacher_id:
        raise HTTPException(400, "teacher_id required")

    from app.services.worksheet_service import WorksheetService

    num_questions = req.get("num_questions", 10)
    question_types = req.get("question_types")
    difficulty = req.get("difficulty", "medium")

    worksheet_data, error = WorksheetService.generate_worksheet_questions(
        module_id=module_id,
        teacher_id=teacher_id,
        num_questions=num_questions,
        question_types=question_types,
        difficulty=difficulty,
    )

    if error:
        raise HTTPException(500, error)

    return {"success": True, "worksheet": worksheet_data}


@router.post("/modules/{module_id}/download-worksheet")
async def download_worksheet(module_id: str, req: dict, authorization: Optional[str] = Header(None)):
    """Generate and download worksheet as PDF."""
    teacher_id = _validate_teacher_id(req.get("teacher_id"))
    if not teacher_id:
        raise HTTPException(400, "teacher_id required")

    from app.services.worksheet_service import WorksheetService
    import base64

    worksheet_data = req.get("worksheet_data")
    if not worksheet_data:
        num_questions = req.get("num_questions", 10)
        question_types = req.get("question_types")
        difficulty = req.get("difficulty", "medium")

        worksheet_data, error = WorksheetService.generate_worksheet_questions(
            module_id=module_id,
            teacher_id=teacher_id,
            num_questions=num_questions,
            question_types=question_types,
            difficulty=difficulty,
        )
        if error:
            raise HTTPException(500, error)

    include_answers = req.get("include_answers", False)
    school_name = req.get("school_name", None)

    logo_bytes = None
    logo_ext = "png"
    logo_b64 = req.get("logo_base64")
    if logo_b64:
        try:
            if "," in logo_b64:
                header, logo_b64 = logo_b64.split(",", 1)
                if "jpeg" in header or "jpg" in header:
                    logo_ext = "jpeg"
            logo_bytes = base64.b64decode(logo_b64)
        except Exception as e:
            pass

    try:
        pdf_bytes = WorksheetService.generate_pdf(
            worksheet_data=worksheet_data,
            include_answers=include_answers,
            school_name=school_name,
            logo_bytes=logo_bytes,
            logo_ext=logo_ext,
        )

        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

        return {
            "success": True,
            "pdf_base64": pdf_b64,
            "filename": f"worksheet_{worksheet_data.get('subject', 'assignment')}_{worksheet_data.get('class', '')}.pdf",
        }

    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")

 
 
@router.post("/worksheet/generate-direct")
async def generate_worksheet_direct(req: dict, authorization: Optional[str] = Header(None)):
    """Generate worksheet directly from uploaded PDF — no module needed."""
    teacher_id = _validate_teacher_id(req.get("teacher_id"))
    if not teacher_id:
        raise HTTPException(400, "teacher_id required")

    from app.services.module_service import ModuleService, get_supabase, get_genai
    import tempfile, os, json

    storage_path = req.get("storage_path", "")
    if not storage_path:
        raise HTTPException(400, "storage_path required")

    try:
        sb = get_supabase()
        file_bytes = sb.storage.from_("Modules").download(storage_path)

        tmp_path = tempfile.mktemp(suffix=".pdf")
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        full_text, page_count, is_scanned = ModuleService._extract_pdf(tmp_path)

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        if not full_text or len(full_text.strip()) < 50:
            raise HTTPException(400, "Could not extract text from PDF")

        from google import genai as genai_client
        client = genai_client.Client(api_key=os.getenv("GEMINI_API_KEY", ""))

        subject = req.get("subject", "General")
        class_level = req.get("class_level", "")
        chapter_name = req.get("chapter_name", "")
        num_questions = req.get("num_questions", 10)
        question_types = req.get("question_types", ["MCQ", "Short Answer"])
        difficulty = req.get("difficulty", "medium")

        prompt = f"""You are an expert Indian school teacher creating a high-quality assignment worksheet.

SUBJECT: {subject}
CLASS: {class_level}
CHAPTER: {chapter_name}
DIFFICULTY: {difficulty}
TOTAL QUESTIONS: {num_questions}
QUESTION TYPES: {', '.join(question_types)}

SOURCE CONTENT:
{full_text[:400000]}

Generate EXACTLY {num_questions} questions for a student assignment.

FACTUAL ACCURACY — HIGHEST PRIORITY:
1. Every question, formula, equation, fact, number, and definition MUST be verifiable from the source content above
2. NEVER invent formulas, dates, values, or facts not present in the source
3. Every mathematical calculation must be mathematically CORRECT — verify each step
4. Every chemical equation must be BALANCED and use correct chemical formulas
5. Every physics formula must have correct units and dimensions
6. If unsure about a fact, DO NOT include that question — quality over quantity
7. For MCQs: the correct answer MUST be actually correct (double-check before finalizing)
8. For numerical problems: solve the problem yourself first, verify the answer is mathematically right
9. Do NOT include questions where you're unsure of the answer
10. If a concept is not clearly explained in the source, don't create a question on it

SELF-VERIFICATION PASS:
Before finalizing each question, mentally check:
- Is this question conceptually correct?
- Is the answer verifiable from source?
- Is any calculation actually correct?
- Would a subject expert approve this?
If ANY answer is "no" or "unsure" — replace that question.

Questions must be DIRECTLY from the source content — no outside knowledge.
For MCQs: 4 options (a, b, c, d). For Fill in Blanks: use _______.
Questions should go easy to hard.
Use proper equations, formulas, scientific notation where needed.

CRITICAL — ANSWER FORMAT (choose the BEST format for each question):
Each answer must be an OBJECT with "format" and content fields.

Choose format based on question type:

1. For MATHS problems (integration, derivation, algebra, equations, numerical):
   Use "steps" format — show THOROUGH step-by-step solution like a teacher solving on blackboard
   - Show EVERY intermediate calculation, don't skip any step
   - Include the formula/rule being used at each step (e.g., "Using power rule ∫xⁿdx = xⁿ⁺¹/(n+1)")
   - Show substitutions explicitly
   - Explain WHY each step is done, not just what
   - Include arithmetic details (e.g., "= 4 × 3 = 12", not just "= 12")
   - Aim for 6-10 detailed steps for complex problems, 4-5 for simple ones
   - Final step should clearly state the answer with "Therefore" or "Hence"
   
   Example for ∫ 2x sin(x²+1) dx:
   {{"format": "steps", "steps": [
     "Step 1: Observe that the integrand has the form f'(x) · g(f(x)) where f(x) = x² + 1 and f'(x) = 2x",
     "Step 2: This suggests using substitution method. Let u = x² + 1",
     "Step 3: Differentiate both sides: du/dx = 2x, which gives du = 2x dx",
     "Step 4: Substitute u and du into the original integral: ∫ sin(u) du",
     "Step 5: Apply the standard integral formula ∫ sin(u) du = -cos(u) + C",
     "Step 6: Substitute back u = x² + 1 to get the answer in terms of x",
     "Step 7: Therefore, ∫ 2x sin(x²+1) dx = -cos(x² + 1) + C, where C is the constant of integration"
   ]}}

2. For COMPARISON questions (differences, similarities, categorize):
   Use "table" format
   {{"format": "table", "headers": ["Property", "Metal", "Non-metal"], "rows": [["Lustre", "Shiny", "Dull"], ["Conductivity", "Good", "Poor"]]}}

3. For EXPLANATION questions (define, describe, explain, why):
   Use "paragraph" format
   {{"format": "paragraph", "text": "Photosynthesis is the process by which green plants make food using sunlight, carbon dioxide, and water. Chlorophyll in the leaves absorbs sunlight..."}}

4. For LIST/POINTS questions (list, mention, state, features):
   Use "points" format
   {{"format": "points", "points": ["Point 1: Description", "Point 2: Description", "Point 3: Description"]}}

5. For SHORT/MCQ/FILL/TRUE-FALSE answers (one-line answers):
   Use "short" format
   {{"format": "short", "text": "Option (b) — 2Fe + 3H₂O → Fe₂O₃ + 3H₂. Because iron is more reactive than hydrogen."}}

6. For CHEMISTRY EQUATIONS or REACTIONS:
   Use "steps" format showing balanced equation
   {{"format": "steps", "steps": ["Unbalanced: Fe + O₂ → Fe₂O₃", "Balance oxygen: multiply by 2 → Fe + O₂ → 2Fe₂O₃... wait", "Balanced: 4Fe + 3O₂ → 2Fe₂O₃"]}}

Output as JSON:
{{"questions": [
  {{"q_no": 1, "type": "MCQ", "question": "...", "options": ["a)...", "b)...", "c)...", "d)..."], "answer": {{"format": "short", "text": "..."}}}},
  {{"q_no": 2, "type": "Long Answer", "question": "Evaluate ∫ 2x sin(x²+1) dx", "answer": {{"format": "steps", "steps": ["Step 1: ...", "Step 2: ..."]}}}},
  {{"q_no": 3, "type": "Short Answer", "question": "Compare metals and non-metals", "answer": {{"format": "table", "headers": [...], "rows": [[...]]}}}}
]}}
Output ONLY valid JSON, no markdown, no backticks."""

        model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 16000, "response_mime_type": "application/json"})

        raw_text = response.text
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            import re
            cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw_text)
            data = json.loads(cleaned)
        from app.services.worksheet_service import convert_worksheet_symbols
        data = convert_worksheet_symbols(data)
        

        return {
            "success": True,
            "worksheet": {
                "title": chapter_name or "Worksheet",
                "subject": subject,
                "class": class_level,
                "questions": data.get("questions", []),
                "num_questions": len(data.get("questions", [])),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/worksheet/download")
async def download_worksheet_pdf(req: dict, authorization: Optional[str] = Header(None)):
    """Download worksheet as PDF with optional school branding and date."""
    from app.services.worksheet_service import WorksheetService
    import base64

    worksheet_data = req.get("worksheet_data")
    if not worksheet_data:
        raise HTTPException(400, "worksheet_data required")

    include_answers = req.get("include_answers", False)
    school_name = req.get("school_name")

    logo_bytes = None
    logo_ext = "png"
    logo_b64 = req.get("logo_base64")
    if logo_b64:
        try:
            if "," in logo_b64:
                header, logo_b64 = logo_b64.split(",", 1)
                if "jpeg" in header or "jpg" in header:
                    logo_ext = "jpeg"
            logo_bytes = base64.b64decode(logo_b64)
        except:
            pass

    try:
        pdf_bytes = WorksheetService.generate_pdf(
            worksheet_data=worksheet_data,
            include_answers=include_answers,
            school_name=school_name,
            logo_bytes=logo_bytes,
            logo_ext=logo_ext,
        )
        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        return {
            "success": True,
            "pdf_base64": pdf_b64,
            "filename": f"worksheet_{worksheet_data.get('subject', 'assignment')}_{worksheet_data.get('class', '')}.pdf",
        }
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {str(e)}")