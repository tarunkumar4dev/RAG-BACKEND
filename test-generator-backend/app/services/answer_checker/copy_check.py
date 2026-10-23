"""
One student's copy, end to end, without any database or storage access (G1 pipeline):
Stage A preprocess → current grader (one call, also reports page numbers / rotation) → pages turned upright and
put in the student's order → answer positions remapped onto the final pages.

The router saves the returned pages to Storage; tests and scripts call this directly with GEMINI_MOCK=1.
"""

import os
import tempfile

from app.services import smart_assignment_service as svc
from app.services.answer_checker import preprocess

# Long edge per attempt when the pages don't fit Gemini's inline request budget
LONG_EDGES = (preprocess.MAX_LONG_EDGE, 1600, 1280)


def _encoded_size(pages: list) -> int:
    return sum(svc.b64_size(len(p.jpeg)) for p in pages)


def fit_pages(source_paths: list) -> list:
    """Stage A with the inline-size guard: render at 2000 px; if the pages that will be sent are over the
    budget, re-render at 1600, then 1280; still over → BadFile (no Gemini call)."""
    full = preprocess.normalize_sources(source_paths, max_pages=preprocess.MAX_PAGES, long_edge=LONG_EDGES[0])
    for edge in LONG_EDGES:
        norm = full if edge == LONG_EDGES[0] else preprocess.downscale(full, edge)
        sent = [p for p in norm if not p.blank]
        size = _encoded_size(sent)
        if size <= svc.INLINE_FILES_BUDGET:
            return norm
    per_page = size / max(1, len(sent))
    fits = max(1, int(svc.INLINE_FILES_BUDGET // per_page))
    raise preprocess.BadFile(f"Too many pages in one copy, max {fits}. Split it into two copies.")


def grade_copy(source_paths: list, rubric: dict, strictness: str = "medium") -> tuple:
    """Returns (result, ordered_pages, usage). result has graded_questions with page_index/y_start on
    ordered_pages, plus sheet_flags. Raises preprocess.BadFile for unreadable uploads (no Gemini call)."""
    norm = fit_pages(source_paths)
    graded_index = [i for i, p in enumerate(norm) if not p.blank]
    if not graded_index:
        raise preprocess.BadFile("Every page of this copy is blank.")
    temps = []
    try:
        for i in graded_index:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", prefix="norm_")
            tmp.write(norm[i].jpeg)
            tmp.close()
            temps.append(tmp.name)
        with svc.usage_meter() as meter:
            result = svc.grade_sheet(temps, rubric, strictness, require_identity=False, page_meta=True)
    finally:
        for t in temps:
            try:
                os.unlink(t)
            except OSError:
                pass
    ordered, old_to_new, sheet_flags = preprocess.finalize_pages(norm, graded_index, result.pop("_page_meta", []))
    preprocess.remap_positions(result, old_to_new)
    result["sheet_flags"] = sheet_flags
    result.pop("_grade_time_s", None)
    return result, ordered, {**meter.as_dict(), "pages": len(norm), "pages_sent": len(graded_index),
                             "long_edge": max(max(p.w, p.h) for p in norm),
                             "inline_bytes": _encoded_size([norm[i] for i in graded_index])}
