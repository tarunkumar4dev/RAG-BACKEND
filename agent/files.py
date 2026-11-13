"""Generated file store: temp directory, 1-hour expiry, owner-only download.

Files are .xlsx exports (defaulters, proctor reports). They live on this process only, never in Supabase.
Each file is keyed by a random id + owner (user_id, institute_id). GET /agent/files/{id} streams it if the
caller is the owner and the file hasn't expired; otherwise 404 (foreign) or 410 (expired).
"""
from __future__ import annotations

import io
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import Response

from .schemas import SessionCtx, now_ist

EXPIRY_S = 3600  # 1 hour
MAX_FILES = 500  # per-process safety cap

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# Store dir: agent/.devdata/files/ (gitignored), or AGENT_FILES_DIR
_DIR: Optional[Path] = None


def _store_dir() -> Path:
    global _DIR
    if _DIR is None:
        d = os.getenv("AGENT_FILES_DIR") or str(Path(__file__).resolve().parent / ".devdata" / "files")
        _DIR = Path(d)
        _DIR.mkdir(parents=True, exist_ok=True)
    return _DIR


@dataclass
class FileEntry:
    file_id: str
    user_id: str
    institute_id: str
    kind: str
    name: str
    path: Path
    created_at: float
    expires_at: float


_lock = threading.Lock()
_files: dict[str, FileEntry] = {}


def store_file(ctx: SessionCtx, workbook, name: str, kind: str) -> str:
    """Save a workbook to the file store and return the file_id."""
    fid = uuid.uuid4().hex[:16]
    now = time.time()
    path = _store_dir() / f"{fid}.xlsx"
    workbook.save(str(path))
    entry = FileEntry(
        file_id=fid, user_id=str(ctx.user_id), institute_id=str(ctx.institute_id),
        kind=kind, name=name, path=path, created_at=now, expires_at=now + EXPIRY_S,
    )
    with _lock:
        _files[fid] = entry
        # Purge expired
        expired = [k for k, v in _files.items() if v.expires_at < now]
        for k in expired:
            _cleanup(k)
        # Safety cap
        while len(_files) > MAX_FILES:
            oldest = min(_files, key=lambda k: _files[k].created_at)
            _cleanup(oldest)
    return fid


def _cleanup(fid: str) -> None:
    entry = _files.pop(fid, None)
    if entry and entry.path.exists():
        try:
            entry.path.unlink()
        except OSError:
            pass


def file_info(file_id: str) -> Optional[dict]:
    """Get file info (for building the `file` event). None if not found."""
    with _lock:
        entry = _files.get(file_id)
    if entry is None:
        return None
    return {
        "file_id": entry.file_id,
        "kind": entry.kind,
        "name": entry.name,
        "expires_at": datetime.fromtimestamp(entry.expires_at, tz=timezone.utc).isoformat(timespec="seconds"),
        "download_path": f"/agent/files/{entry.file_id}",
    }


def serve_file(ctx: SessionCtx, file_id: str) -> Response:
    """Serve a file to its owner. 404 for foreign, 410 for expired."""
    with _lock:
        entry = _files.get(file_id)
    if entry is None:
        raise HTTPException(404, "not found")
    if entry.user_id != str(ctx.user_id) or entry.institute_id != str(ctx.institute_id):
        raise HTTPException(404, "not found")  # foreign = not found
    if entry.expires_at < time.time():
        raise HTTPException(410, "file expired; ask the agent again")
    if not entry.path.exists():
        raise HTTPException(410, "file expired; ask the agent again")
    data = entry.path.read_bytes()
    return Response(
        content=data, media_type=XLSX_MIME,
        headers={"Content-Disposition": f'attachment; filename="{entry.name}"'},
    )


def reset() -> None:
    """For tests."""
    with _lock:
        for fid in list(_files):
            _cleanup(fid)
