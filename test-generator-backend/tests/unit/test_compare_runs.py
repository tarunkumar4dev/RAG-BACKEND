"""scripts/compare_runs.py + export_ai_marks.export_rows: run A vs run B per copy / question, split by original model."""

import csv
import sys
import types
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import compare_runs  # noqa: E402


def _csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["roll", "name", "question", "marks", "max_marks", "submission_id", "model",
                                          "counted", "teacher_marks"])
        w.writeheader()
        for r in rows:
            w.writerow({"max_marks": 3, "counted": "true", "teacher_marks": "", **r})


def test_diff_per_copy_and_split_by_original_model(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    _csv(a, [{"roll": "1", "name": "A", "question": "1", "marks": 2, "submission_id": "s1", "model": "gemini-3.8-flash"},
             {"roll": "1", "name": "A", "question": "2", "marks": 1, "submission_id": "s1", "model": "gemini-3.8-flash"},
             {"roll": "2", "name": "B", "question": "1", "marks": 3, "submission_id": "s2", "model": "gemini-3-flash-preview"},
             {"roll": "2", "name": "B", "question": "2", "marks": 3, "submission_id": "s2", "model": "gemini-3-flash-preview"},
             {"roll": "2", "name": "B", "question": "3", "marks": 2, "submission_id": "s2", "model": "gemini-3-flash-preview",
              "counted": "false"}])
    _csv(b, [{"roll": "1", "name": "A", "question": "1", "marks": 2, "submission_id": "s1", "model": "gemini-3.8-flash"},
             {"roll": "1", "name": "A", "question": "2", "marks": 1, "submission_id": "s1", "model": "gemini-3.8-flash"},
             {"roll": "2", "name": "B", "question": "1", "marks": 1, "submission_id": "s2", "model": "gemini-3.8-flash"},
             {"roll": "2", "name": "B", "question": "2", "marks": 2.5, "submission_id": "s2", "model": "gemini-3.8-flash"},
             {"roll": "3", "name": "C", "question": "1", "marks": 1, "submission_id": "s3", "model": "gemini-3.8-flash"}])
    res = compare_runs.compare(compare_runs.load(str(a)), compare_runs.load(str(b)), "gemini-3.8-flash")
    by = {c["submission_id"]: c for c in res["copies"]}
    assert by["s1"]["total_diff"] == 0 and by["s1"]["changed"] == 0 and by["s1"]["group"] == "primary"
    assert by["s2"]["total_a"] == 6 and by["s2"]["total_b"] == 3.5 and by["s2"]["total_diff"] == -2.5
    assert by["s2"]["diffs"] == {"1": -2, "2": -0.5} and by["s2"]["group"] == "fallback"    # uncounted Q3 ignored
    s = res["summary"]
    assert s["fallback"]["copies"] == 1 and s["fallback"]["mean_abs_total_diff"] == 2.5
    assert s["primary"]["copies_changed"] == 0 and s["all"]["copies"] == 2
    assert res["only_b"] == ["s3"]
    text = compare_runs.report(res, "a.csv", "b.csv")
    assert "| fallback | 1 | 2.5 | -2.5 |" in text and "Q1 -2, Q2 -0.5" in text


def test_export_rows_uses_the_ai_marks_and_keeps_overrides_apart(monkeypatch):
    monkeypatch.setitem(sys.modules, "checker_runs", types.SimpleNamespace(PROD="prod", connect=None, run_results=None))
    import importlib
    export = importlib.import_module("export_ai_marks")
    rows = export.export_rows([{"submission_id": "s1", "roll": "7", "name": "Asha", "model": "gemini-3.8-flash",
                                "ai_result": {"graded_questions": [
                                    {"q_no": "1", "marks_awarded": 1.5, "max_marks": 2, "teacher_marks": 2},
                                    {"q_no": "2", "marks_awarded": 0, "max_marks": 3, "counted": False}]}}])
    assert rows[0] == {"roll": "7", "name": "Asha", "question": "1", "marks": 1.5, "max_marks": 2, "submission_id": "s1",
                       "model": "gemini-3.8-flash", "counted": "true", "teacher_marks": 2}
    assert rows[1]["counted"] == "false" and rows[1]["teacher_marks"] == ""
