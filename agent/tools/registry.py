"""Tool registry. Every tool is fn(ctx, args_model). ctx (SessionCtx) is NEVER in a schema."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel

from . import insights, paper


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: type[BaseModel]
    kind: Literal["read", "draft", "approval"]
    fn: Callable | None = None


TOOLS: dict[str, ToolSpec] = {t.name: t for t in [
    ToolSpec("resolve_entity", "Resolve a teacher/student/section name to an id within the caller's scope. "
             "Honorifics are stripped. Ambiguous -> clarify chips; not found -> say so honestly.",
             insights.ResolveEntityArgs, "read", insights.resolve_entity),
    ToolSpec("attendance_stats", "Attendance numbers for a teacher/section/student over a date range (IST).",
             insights.AttendanceStatsArgs, "read", insights.attendance_stats),
    ToolSpec("low_attendance_list", "Students below an attendance % threshold in a date range.",
             insights.LowAttendanceArgs, "read", insights.low_attendance_list),
    ToolSpec("marks_stats", "Test marks summary for a section or a student.",
             insights.MarksStatsArgs, "read", insights.marks_stats),
    ToolSpec("blueprint_defaults", "CBSE-style blueprint (sections, marks) for a test paper. Show it to the user.",
             paper.BlueprintArgs, "read"),
    ToolSpec("generate_questions", "Generate the whole paper for the current blueprint in one batch (NCERT "
             "context), then run the verification gate and regenerate failed questions (max 3 rounds).",
             paper.DraftArgs, "draft"),
    ToolSpec("verify_paper", "Re-run the verification gate on a draft.", paper.DraftArgs, "draft"),
    ToolSpec("regenerate_failed", "Regenerate only the given failed question ids, then verify again.",
             paper.RegenArgs, "draft"),
    ToolSpec("save_draft", "Save the draft (not visible to students).", paper.DraftArgs, "draft"),
    ToolSpec("finalize_test", "Finalize a verified draft. Requires human approval.", paper.DraftArgs, "approval"),
    ToolSpec("assign_to_section", "Assign a finalized test to a section. Requires human approval.",
             paper.AssignArgs, "approval"),
    ToolSpec("export_pdf", "Export PDF of an approved test (only after approval).", paper.DraftArgs, "draft"),
]}


def tool_schemas() -> list[dict]:
    out = []
    for t in TOOLS.values():
        s = t.args.model_json_schema()
        s.pop("title", None)
        out.append({"name": t.name, "description": t.description, "parameters": s})
    return out
