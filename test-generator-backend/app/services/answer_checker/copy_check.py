"""
One student's copy, end to end, without any database or storage access (G1 pipeline):
Stage A preprocess → current grader (one call, also reports page numbers / rotation) → pages turned upright and
put in the student's order → answer positions remapped onto the final pages.

The router saves the returned pages to Storage; tests and scripts call this directly with GEMINI_MOCK=1.
"""

import os
import tempfile

from app.services.answer_checker import preprocess
from app.services.smart_assignment_service import grade_sheet, usage_meter


def grade_copy(source_paths: list, rubric: dict, strictness: str = "medium") -> tuple:
    """Returns (result, ordered_pages, usage). result has graded_questions with page_index/y_start on
    ordered_pages, plus sheet_flags. Raises preprocess.BadFile for unreadable uploads (no Gemini call)."""
    norm = preprocess.normalize_sources(source_paths)
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
        with usage_meter() as meter:
            result = grade_sheet(temps, rubric, strictness, require_identity=False, page_meta=True)
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
    return result, ordered, {**meter.as_dict(), "pages": len(norm), "pages_sent": len(graded_index)}
