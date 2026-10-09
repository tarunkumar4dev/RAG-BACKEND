"""
Accuracy harness for the Smart Assignment Checker.

Each case is a folder under scripts/eval_cases/smart_checker/ with:
    question_paper.(pdf|png|jpg)   answer_key.(…)   student_sheet.(…)
    expected.json  {"strictness": "medium", "expected_valid": true,
                    "marks": {"1": 1, "4": 2.5, ...},   ← the teacher's marks per question
                    "total": 9.5}

Runs the real pipeline (same code as POST /smart-assignment-checker/check, no DB writes) and reports,
per case: identity verdict, per-question error vs the teacher, total error, time.
Summary: mean absolute error per question, % of sheets whose total is within ±1 of the teacher.

Run from test-generator-backend/:
    python scripts/eval_smart_checker.py
    python scripts/eval_smart_checker.py --cases science10_rahul --runs 3     # consistency check
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(ROOT, ".env"))

from app.services.smart_assignment_service import check_assignment  # noqa: E402

CASES_DIR = os.path.join(ROOT, "scripts", "eval_cases", "smart_checker")
OUT_DIR = os.path.join(ROOT, "scripts", "eval_runs")


def _find(case_dir: str, stem: str) -> str:
    hits = [p for p in glob.glob(os.path.join(case_dir, stem + ".*")) if not p.endswith(".json")]
    if not hits:
        raise FileNotFoundError(f"{case_dir}: missing {stem}.*")
    return hits[0]


def run_case(name: str) -> dict:
    case_dir = os.path.join(CASES_DIR, name)
    exp = json.load(open(os.path.join(case_dir, "expected.json"), encoding="utf-8"))
    start = time.time()
    try:
        res = check_assignment(_find(case_dir, "question_paper"), _find(case_dir, "answer_key"),
                               _find(case_dir, "student_sheet"), exp.get("strictness", "medium"))
    except Exception as e:
        return {"case": name, "error": str(e)[:300], "time_s": round(time.time() - start, 1)}

    row = {"case": name, "time_s": res["_meta"]["total_time_s"],
           "valid_ok": res["is_valid"] == exp.get("expected_valid", True),
           "is_valid": res["is_valid"], "needs_review": res.get("needs_review_count", 0)}
    if res["is_valid"] and exp.get("marks"):
        got = {q["q_no"]: q["marks_awarded"] for q in res["graded_questions"] if q.get("counted", True)}
        row["per_q"] = {q: {"teacher": m, "ai": got.get(q), "err": round(abs((got.get(q) or 0) - m), 2)}
                        for q, m in exp["marks"].items()}
        row["total_teacher"] = exp["total"]
        row["total_ai"] = res["total_marks_awarded"]
        row["total_err"] = round(abs(res["total_marks_awarded"] - exp["total"]), 2)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", help="comma-separated case folder names (default: all)")
    ap.add_argument("--runs", type=int, default=1, help="repeat each case (consistency)")
    args = ap.parse_args()

    names = args.cases.split(",") if args.cases else sorted(
        d for d in os.listdir(CASES_DIR) if os.path.isdir(os.path.join(CASES_DIR, d)))
    rows = []
    for name in names:
        for i in range(args.runs):
            row = run_case(name)
            row["run"] = i + 1
            rows.append(row)
            if "error" in row:
                print(f"✗ {name} #{i + 1}: ERROR {row['error']}")
                continue
            line = f"{'✓' if row['valid_ok'] else '✗'} {name} #{i + 1}: valid={row['is_valid']} {row['time_s']}s"
            if "total_ai" in row:
                line += f" | total AI {row['total_ai']:g} vs teacher {row['total_teacher']:g} (err {row['total_err']:g})"
                line += " | " + " ".join(f"Q{q}:{v['ai']}/{v['teacher']}" for q, v in row["per_q"].items())
            line += f" | review flags {row['needs_review']}"
            print(line)

    graded = [r for r in rows if "total_err" in r]
    q_errs = [v["err"] for r in graded for v in r["per_q"].values()]
    summary = {
        "sheets": len(rows),
        "errors": sum(1 for r in rows if "error" in r),
        "identity_correct": f"{sum(1 for r in rows if r.get('valid_ok'))}/{len(rows)}",
        "mae_per_question": round(sum(q_errs) / len(q_errs), 3) if q_errs else None,
        "total_within_1": f"{sum(1 for r in graded if r['total_err'] <= 1)}/{len(graded)}",
        "avg_time_s": round(sum(r["time_s"] for r in rows) / len(rows), 1) if rows else None,
    }
    print("\nSUMMARY", json.dumps(summary, indent=1))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"smart_checker_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("saved", out)


if __name__ == "__main__":
    main()
