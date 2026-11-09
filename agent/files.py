"""Generated files (read-only exports). Filled in by the export tools (Phase 5)."""
from __future__ import annotations

from fastapi import HTTPException


def serve_file(ctx, file_id: str):
    raise HTTPException(404, "not found")
