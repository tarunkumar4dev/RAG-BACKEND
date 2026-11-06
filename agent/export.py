"""PDF export adapter -> existing exporter app.services.export_service.generate_pdf (reportlab)."""
from __future__ import annotations

import os
import sys

_FMT = {"mcq": "mcq", "assertion_reason": "assertion_reason", "very_short": "short_answer",
        "short_answer": "short_answer", "long_answer": "long_answer", "case_study": "case_study",
        "numeric": "short_answer"}


def to_exporter_questions(paper: dict) -> list[dict]:
    out = []
    for q in paper["questions"]:
        text = q["text"]
        if q.get("sub_questions"):
            text += "\n" + "\n".join(f"({chr(105 + i) if False else i + 1}) {p['text']} [{p.get('marks', '')}]"
                                     for i, p in enumerate(q["sub_questions"]))
        out.append({"id": q["id"], "text": text, "options": q.get("options"), "correct_answer": str(q.get("answer") or ""),
                    "explanation": q.get("solution") or "", "marks": int(q["marks"]) if float(q["marks"]).is_integer()
                    else q["marks"], "difficulty": "medium", "chapter": q.get("chapter", ""),
                    "format": _FMT.get(q["qtype"], "short_answer"), "section": q.get("section")})
    return out


def export_pdf(draft: dict, include_answers: bool = False) -> bytes:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "test-generator-backend"))
    from app.services.export_service import generate_pdf  # type: ignore  (reuse existing exporter)
    bp = draft["blueprint"]
    return generate_pdf(to_exporter_questions(draft["paper"]), exam_title=f"{bp['chapters'][0]} — Test",
                        class_grade=str(bp["class_grade"]), subject=bp["subject"], include_answers=include_answers,
                        duration=f"{bp['duration_min']} min", topic=bp["chapters"][0])
