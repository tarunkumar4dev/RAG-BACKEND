"""
Per-question AI marks of one assignment, for the eval harness and compare_runs.py:

  python scripts/export_ai_marks.py --assignment-id <uuid> --out C:/path/outside/repo/prod.csv
  python scripts/export_ai_marks.py --assignment-id <uuid> --run-label shadow-2026-10-27 --out …/shadow.csv

--run-label prod (default) = the marks students / teachers see today (check_sheets.ai_result); any other label =
a shadow re-grade (scripts/shadow_regrade.py, table check_shadow_results).

Columns: roll,name,question,marks,max_marks + submission_id,model,counted,teacher_marks. `marks` is always the
AI's own mark (marks_awarded); a teacher's override is in teacher_marks. Questions with sub-parts get one row for
the question (sum, as graded) — the same unit the teacher sheet uses. Read-only.

The file holds student names: keep it out of the repo.
"""

import argparse
import csv
import sys

from checker_runs import PROD, connect, run_results

COLUMNS = ["roll", "name", "question", "marks", "max_marks", "submission_id", "model", "counted", "teacher_marks"]


def export_rows(results: list) -> list:
    out = []
    for r in results:
        for gq in (r["ai_result"] or {}).get("graded_questions") or []:
            out.append({"roll": r["roll"] or "", "name": r["name"] or "", "question": gq.get("q_no"),
                        "marks": gq.get("marks_awarded"), "max_marks": gq.get("max_marks"),
                        "submission_id": r["submission_id"], "model": r["model"] or "",
                        "counted": "false" if gq.get("counted") is False else "true",
                        "teacher_marks": "" if gq.get("teacher_marks") is None else gq["teacher_marks"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assignment-id", required=True)
    ap.add_argument("--run-label", default=PROD)
    ap.add_argument("--out", required=True, help="CSV path (outside the repo: it holds student names)")
    args = ap.parse_args()

    conn = connect()
    try:
        results = run_results(conn, args.assignment_id, args.run_label)
    finally:
        conn.close()
    if not results:
        sys.exit(f"no copies for run '{args.run_label}' of assignment {args.assignment_id}")
    data = export_rows(results)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(data)
    print(f"{len(results)} copies, {len(data)} rows → {args.out}")


if __name__ == "__main__":
    main()
