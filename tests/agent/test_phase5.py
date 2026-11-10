"""Phase 5 tests: export tools, file store, file events, download endpoint."""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent import files as file_mod
from agent.schemas import SessionCtx


def _ctx(**kw) -> SessionCtx:
    defaults = dict(
        user_id=uuid.UUID("00000000-0000-0000-0000-000000000001"),
        institute_id=uuid.UUID("00000000-0000-0000-0000-000000000099"),
        role="admin",
    )
    defaults.update(kw)
    return SessionCtx(**defaults)


@pytest.fixture(autouse=True)
def _clean():
    file_mod.reset()
    yield
    file_mod.reset()


# ── file store ────────────────────────────────────────────────────────────────

class TestFileStore:
    def test_store_and_info(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_mod, "_DIR", tmp_path)
        ctx = _ctx()
        wb = MagicMock()
        fid = file_mod.store_file(ctx, wb, "test.xlsx", "defaulters_export")
        assert len(fid) == 16
        wb.save.assert_called_once()
        info = file_mod.file_info(fid)
        assert info is not None
        assert info["file_id"] == fid
        assert info["kind"] == "defaulters_export"
        assert info["name"] == "test.xlsx"
        assert info["download_path"] == f"/agent/files/{fid}"

    def test_serve_owner_only(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_mod, "_DIR", tmp_path)
        ctx = _ctx()
        wb = MagicMock()
        (tmp_path / "dummy.xlsx").write_bytes(b"PK\x03\x04fake")
        wb.save.side_effect = lambda p: Path(p).write_bytes(b"PK\x03\x04fake")
        fid = file_mod.store_file(ctx, wb, "test.xlsx", "defaulters_export")
        resp = file_mod.serve_file(ctx, fid)
        assert resp.status_code == 200

        other = _ctx(user_id=uuid.UUID("00000000-0000-0000-0000-000000000002"))
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            file_mod.serve_file(other, fid)
        assert exc_info.value.status_code == 404

    def test_serve_expired(self, tmp_path, monkeypatch):
        monkeypatch.setattr(file_mod, "_DIR", tmp_path)
        ctx = _ctx()
        wb = MagicMock()
        wb.save.side_effect = lambda p: Path(p).write_bytes(b"PK\x03\x04fake")
        fid = file_mod.store_file(ctx, wb, "test.xlsx", "defaulters_export")
        # Force the entry to be expired
        with file_mod._lock:
            file_mod._files[fid].expires_at = time.time() - 10
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            file_mod.serve_file(ctx, fid)
        assert exc_info.value.status_code == 410

    def test_info_not_found(self):
        assert file_mod.file_info("nonexistent") is None


# ── registry ──────────────────────────────────────────────────────────────────

class TestExportRegistry:
    def test_export_tools_registered(self):
        from agent.tools.registry import TOOLS, INSIGHT_TOOLS
        assert "defaulters_export" in TOOLS
        assert "proctor_month_report" in TOOLS
        assert TOOLS["defaulters_export"].kind == "read"
        assert TOOLS["proctor_month_report"].kind == "read"
        assert "defaulters_export" in INSIGHT_TOOLS
        assert "proctor_month_report" in INSIGHT_TOOLS

    def test_export_tools_have_fn(self):
        from agent.tools.registry import TOOLS
        assert TOOLS["defaulters_export"].fn is not None
        assert TOOLS["proctor_month_report"].fn is not None

    def test_export_tools_in_schemas(self):
        from agent.tools.registry import tool_schemas
        schemas = tool_schemas()
        names = {s["name"] for s in schemas}
        assert "defaulters_export" in names
        assert "proctor_month_report" in names


# ── args validation ───────────────────────────────────────────────────────────

class TestExportArgs:
    def test_defaulters_args_valid(self):
        from agent.tools.exports import DefaultersExportArgs
        a = DefaultersExportArgs(date_from="2026-10-01", date_to="2026-10-31")
        assert a.threshold == 75
        assert a.section_id is None

    def test_defaulters_args_threshold_bounds(self):
        from agent.tools.exports import DefaultersExportArgs
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            DefaultersExportArgs(date_from="2026-10-01", date_to="2026-10-31", threshold=101)

    def test_proctor_args_valid(self):
        from agent.tools.exports import ProctorMonthReportArgs
        a = ProctorMonthReportArgs(section_id=uuid.uuid4(), month="2026-10")
        assert a.month == "2026-10"

    def test_proctor_args_bad_month(self):
        from agent.tools.exports import ProctorMonthReportArgs
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ProctorMonthReportArgs(section_id=uuid.uuid4(), month="October")


# ── UI events (file event from _ui_events) ───────────────────────────────────

class TestFileUIEvent:
    def test_file_event_emitted(self, tmp_path, monkeypatch):
        import threading
        monkeypatch.setattr(file_mod, "_DIR", tmp_path)
        ctx = _ctx()
        wb = MagicMock()
        wb.save.side_effect = lambda p: Path(p).write_bytes(b"PK\x03\x04fake")
        fid = file_mod.store_file(ctx, wb, "defaulters_oct.xlsx", "defaulters_export")

        from agent.core import Agent
        agent = Agent.__new__(Agent)
        agent.ctx = ctx
        agent.seq = 0
        agent._seq_lock = threading.Lock()
        agent.session_id = "test"
        agent.request_id = "req-test"
        agent.trace = MagicMock()
        res = {"status": "ok", "file_id": fid, "name": "defaulters_oct.xlsx", "rows": 5, "kind": "defaulters_export"}
        events = agent._ui_events("defaulters_export", res)
        assert len(events) == 1
        assert events[0]["type"] == "file"
        assert events[0]["file_id"] == fid
        assert events[0]["rows"] == 5
        assert "/agent/files/" in events[0]["download_path"]

    def test_no_file_event_without_file_id(self):
        import threading
        from agent.core import Agent
        ctx = _ctx()
        agent = Agent.__new__(Agent)
        agent.ctx = ctx
        agent.seq = 0
        agent._seq_lock = threading.Lock()
        agent.session_id = "test"
        agent.trace = MagicMock()
        res = {"status": "ok", "rows": 3}
        events = agent._ui_events("defaulters_export", res)
        assert events == []


# ── routing: export queries stay on fast ──────────────────────────────────────

class TestExportRouting:
    def test_defaulters_hindi_fast(self):
        from agent.routing import choose
        route, reason = choose("75% se kam wale ki list Excel mein do", [])
        assert route == "fast"

    def test_proctor_report_fast(self):
        from agent.routing import choose
        route, reason = choose("proctor report October 2026 section wise", [])
        assert route == "fast"

    def test_export_not_escalated(self):
        from agent.routing import escalate
        from agent.tools.registry import TOOLS
        read_tools = [n for n, t in TOOLS.items() if t.kind == "read"]
        assert not escalate(["defaulters_export"], read_tools)
        assert not escalate(["proctor_month_report"], read_tools)
