"""Tool registry. Every tool is fn(ctx, args_model). ctx (SessionCtx) is NEVER in a schema."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

from pydantic import BaseModel

from . import exports, insights, paper
from .. import routes


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
    ToolSpec("attendance_trend", "Attendance % over time (day/week/month buckets) for a teacher/section/student. "
             "Shown as a chart.", insights.AttendanceTrendArgs, "read", insights.attendance_trend),
    ToolSpec("tests_created_by_teacher", "How many test papers a teacher created in a date range (IST), with "
             "recent titles. teacher_id omitted = me (teacher) or my whole scope (admin/HOD).",
             insights.TestsCreatedArgs, "read", insights.tests_created_by_teacher),
    ToolSpec("my_scope", "The caller's own batches/sections/subjects with student counts and the caller's role "
             "in each. Use for 'meri class/classes', 'mere students', 'my batches', 'what do I teach'.",
             insights.MyScopeArgs, "read", insights.my_scope),
    ToolSpec("section_comparison", "Rank every batch/section in the caller's scope by attendance % or average "
             "marks % over a date range (IST); returns rows plus `lowest`/`highest` names (ties included). Use for "
             "'which class has the lowest/highest attendance', 'compare classes'. Shown as a table + bar chart.",
             insights.SectionComparisonArgs, "read", insights.section_comparison),
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
    ToolSpec("defaulters_export", "Export an Excel (.xlsx) list of students below an attendance threshold over a "
             "date range. Use for '75% se kam wale ki list Excel mein do', 'defaulters list download'.",
             exports.DefaultersExportArgs, "read", exports.defaulters_export),
    ToolSpec("proctor_month_report", "Export a subject-wise attendance matrix (.xlsx) for a section in a given "
             "month. Use for 'proctor report October', 'section attendance Excel'.",
             exports.ProctorMonthReportArgs, "read", exports.proctor_month_report),
    ToolSpec("open_feature", routes.TOOL_DESCRIPTION, routes.OpenFeatureArgs, "read", routes.open_feature),
]}


INSIGHT_TOOLS = ("resolve_entity", "attendance_stats", "low_attendance_list", "attendance_trend", "marks_stats",
                 "tests_created_by_teacher", "my_scope", "section_comparison",
                 "defaulters_export", "proctor_month_report")
WRITE_TOOLS = ("save_draft", "finalize_test", "assign_to_section", "export_pdf")  # disabled in real mode


def active_tools(real: bool | None = None) -> dict[str, ToolSpec]:
    """Seed tools by default; with AGENT_REAL_DB_MODE=1 the insight tools run on the real schema (read-only)."""
    from dataclasses import replace
    from ..mode import real_mode
    if not (real_mode() if real is None else real):
        return TOOLS
    from ..real import schema_adapter as real_tools
    return {n: (replace(t, fn=getattr(real_tools, n)) if n in INSIGHT_TOOLS and hasattr(real_tools, n) else t)
            for n, t in TOOLS.items()}


def tool_schemas(tools: dict[str, ToolSpec] | None = None) -> list[dict]:
    out = []
    for t in (tools or TOOLS).values():
        s = t.args.model_json_schema()
        s.pop("title", None)
        out.append({"name": t.name, "description": t.description, "parameters": s})
    return out
