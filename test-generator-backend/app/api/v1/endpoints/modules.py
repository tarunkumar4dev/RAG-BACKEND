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