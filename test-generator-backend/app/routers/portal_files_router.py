"""
app/routers/portal_files_router.py — student portal: open private files with an access code.

The `assignments` and `submissions` buckets are private (product migrations 13 + 18). The portal has no
Supabase session, so it sends its access code here; Postgres decides which files that code may open
(portal_assignment_files, service_role only — own batch / lab batch, own submission only) and this
endpoint returns 10-minute signed URLs. Nothing else about the student is returned.

  POST /api/v1/portal/files   { "code": "<access code>", "assignment_id": "<uuid>" }
       → { "question_url": str | null, "submission_url": str | null, "expires_in": 600 }

Abuse: per-IP limit in main.py HEAVY_ENDPOINTS (access codes are guessable by brute force otherwise).
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from app.core.database import get_supabase_admin

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portal", tags=["Student portal"])

SIGNED_URL_TTL_SECONDS = 600
_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


class PortalFilesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str = Field(min_length=4, max_length=32)
    assignment_id: str = Field(min_length=36, max_length=36)


class PortalFilesResponse(BaseModel):
    question_url: Optional[str]
    submission_url: Optional[str]
    expires_in: int


def _sign(bucket: str, path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    try:
        res = get_supabase_admin().storage.from_(bucket).create_signed_url(path, SIGNED_URL_TTL_SECONDS)
    except Exception as e:  # missing object, storage outage
        logger.warning("portal sign failed for %s: %s", bucket, type(e).__name__)
        return None
    if isinstance(res, dict):
        return res.get("signedURL") or res.get("signedUrl") or None
    return None


def _lookup(code: str, assignment_id: str) -> dict:
    res = get_supabase_admin().rpc(
        "portal_assignment_files", {"p_code": code.strip().upper(), "p_assignment_id": assignment_id}
    ).execute()
    return res.data if isinstance(res.data, dict) else {}


@router.post("/files", response_model=PortalFilesResponse)
async def portal_files(body: PortalFilesRequest):
    if not _UUID.match(body.assignment_id):
        raise HTTPException(status_code=422, detail="Invalid assignment.")
    try:
        found = await run_in_threadpool(_lookup, body.code, body.assignment_id)
    except Exception as e:
        logger.error("portal_assignment_files failed: %s", type(e).__name__)
        raise HTTPException(status_code=503, detail="Files are unavailable right now. Please try again.")

    if not found.get("ok"):
        # Same answer for a wrong code and a wrong assignment: don't help guessing codes.
        raise HTTPException(status_code=404, detail="File not found.")

    question_url = await run_in_threadpool(_sign, "assignments", found.get("question_path"))
    submission_url = await run_in_threadpool(_sign, "submissions", found.get("submission_path"))
    return PortalFilesResponse(question_url=question_url, submission_url=submission_url,
                               expires_in=SIGNED_URL_TTL_SECONDS)
