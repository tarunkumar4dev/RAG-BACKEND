"""
Module Endpoints — FastAPI Router
"""
import json
import base64
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Header, Query
from pydantic import BaseModel
from app.services.module_service import ModuleService

logger = logging.getLogger(__name__)
router = APIRouter()


class CreateModuleRequest(BaseModel):
    storage_path: str
    original_filename: str
    subject: str
    class_level: str
    teacher_id: Optional[str] = None
    file_type: str = "pdf"
    file_size_bytes: Optional[int] = None
    institute_id: Optional[str] = None


class ProcessModuleRequest(BaseModel):
    module_id: str


class GenerateTestRequest(BaseModel):
    teacher_id: Optional[str] = None
    num_questions: int = 10
    question_types: Optional[List[str]] = None
    difficulty: str = "medium"
    topics: Optional[List[str]] = None


def extract_teacher_id(body_teacher_id: Optional[str], authorization: Optional[str] = None) -> Optional[str]:
    if body_teacher_id:
        return body_teacher_id.strip()
    if authorization and authorization.startswith("Bearer "):
        try:
            token = authorization.replace("Bearer ", "")
            payload = token.split(".")[1]
            payload += "=" * (4 - len(payload) % 4)
            decoded = json.loads(base64.b64decode(payload))
            return decoded.get("sub", "")
        except Exception:
            pass
    return None


@router.post("/modules/create")
async def create_module(req: CreateModuleRequest, authorization: Optional[str] = Header(None)):
    teacher_id = extract_teacher_id(req.teacher_id, authorization)
    if not teacher_id:
        raise HTTPException(400, "teacher_id required")

    module_id, error = ModuleService.create_module(
        teacher_id=teacher_id,
        storage_path=req.storage_path,
        original_filename=req.original_filename,
        subject=req.subject,
        class_level=req.class_level,
        file_type=req.file_type,
        file_size_bytes=req.file_size_bytes,
        institute_id=req.institute_id,
    )

    if error:
        raise HTTPException(500, error)

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
        raise HTTPException(500, result.get("error", "Processing failed"))


@router.get("/modules/list")
async def list_modules(
    teacher_id: str = Query(...),
    subject: Optional[str] = Query(None),
    class_level: Optional[str] = Query(None, alias="class"),
):
    modules = ModuleService.list_modules(teacher_id, subject, class_level)
    return {"success": True, "count": len(modules), "modules": modules}


@router.get("/modules/{module_id}")
async def get_module(module_id: str, teacher_id: str = Query(...)):
    module = ModuleService.get_module(module_id, teacher_id)
    if not module:
        raise HTTPException(404, "Module not found")
    return {"success": True, **module}


@router.delete("/modules/{module_id}")
async def delete_module(module_id: str, teacher_id: str = Query(...)):
    success, error = ModuleService.delete_module(module_id, teacher_id)
    if success:
        return {"success": True, "message": "Module deleted"}
    raise HTTPException(404, error)


@router.post("/modules/{module_id}/generate-test")
async def generate_test_from_module(module_id: str, req: GenerateTestRequest, authorization: Optional[str] = Header(None)):
    teacher_id = extract_teacher_id(req.teacher_id, authorization)
    if not teacher_id:
        raise HTTPException(400, "teacher_id required")

    test_data, error = ModuleService.generate_test_from_module(
        module_id=module_id,
        teacher_id=teacher_id,
        num_questions=req.num_questions,
        question_types=req.question_types,
        difficulty=req.difficulty,
        topics=req.topics,
    )

    if error:
        raise HTTPException(500, error)

    return {"success": True, "test": test_data}