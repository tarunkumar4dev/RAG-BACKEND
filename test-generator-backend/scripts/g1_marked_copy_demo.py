"""
G1 proof: the messy fixture through the REAL G1 code path with recorded (mock) Gemini replies — zero API calls.

  python scripts/g1_marked_copy_demo.py [out_dir]

Writes <out_dir>/norm/p<i>.jpg (upright, student order), ai_result.json (what check_sheets.ai_result holds) and
rubric.json, and prints marks vs expected_evaluation.json + flags + Gemini call count.
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "answer_checker"
os.environ["GEMINI_MOCK"] = "1"
os.environ.setdefault("GEMINI_MOCK_DIR", str(FIX / "recordings" / "g1_mock"))
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")   # config needs it; nothing is called
os.environ.setdefault("GEMINI_API_KEY", "mock")
sys.path.insert(0, str(ROOT))

from app.routers.copy_checker_router import _compact  # noqa: E402
from app.services.answer_checker.copy_check import grade_copy  # noqa: E402
from app.services.smart_assignment_service import build_rubric, usage_meter  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "output" / "g1"
    (out / "norm").mkdir(parents=True, exist_ok=True)

    with usage_meter() as rubric_meter:
        rubric = build_rubric(str(FIX / "question_paper.pdf"), str(FIX / "answer_key.pdf"))
    rubric = {k: v for k, v in rubric.items() if not k.startswith("_")}
    result, ordered, usage = grade_copy([str(FIX / "answer_sheet_messy.pdf")], rubric)

    norm_pages = []
    for i, p in enumerate(ordered):
        (out / "norm" / f"p{i}.jpg").write_bytes(p.jpeg)
        norm_pages.append({"index": i, "source_index": p.source_index, "student_page_no": p.student_page_no,
                           "rotation": p.rotation, "blank": p.blank, "path": f"norm/p{i}.jpg", "w": p.w, "h": p.h})
    result["norm_pages"] = norm_pages
    result["usage"] = usage
    ai_result = _compact(result)
    (out / "ai_result.json").write_text(json.dumps(ai_result, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "rubric.json").write_text(json.dumps(rubric, indent=1, ensure_ascii=False), encoding="utf-8")

    expected = {str(q["q"]): q["expected"] for q in json.loads((FIX / "expected_evaluation.json").read_text("utf-8"))["questions"]}
    print(f"{'Q':6s} {'got':>5s} {'exp':>5s}  page  y")
    for gq in ai_result["graded_questions"]:
        if gq.get("counted") is False:
            print(f"{gq['q_no']:6s} {'—':>5s} {'':>5s}  (OR alternative, not counted)")
            continue
        exp = expected.get(gq["q_no"])
        mark = "" if exp is None or abs(gq["marks_awarded"] - exp) <= 0.5 else "  <-- off"
        print(f"{gq['q_no']:6s} {gq['marks_awarded']:5g} {exp if exp is not None else '-':>5}  "
              f"{gq.get('page_index')!s:4s}  {gq.get('y_start')}{mark}")
    print(f"TOTAL  {ai_result['total_marks_awarded']:g} / {ai_result['max_marks']:g}   (expected 35)")
    print("pages (student order):", [(p["student_page_no"], p["source_index"], p["rotation"]) for p in norm_pages])
    for f in ai_result["sheet_flags"]:
        print("FLAG", f["code"], "-", f["message"])
    print("Gemini calls: rubric", rubric_meter.calls, "+ sheet", usage["gemini_calls"], "(mocked, ₹0)")
    print("written to", out)


if __name__ == "__main__":
    main()
