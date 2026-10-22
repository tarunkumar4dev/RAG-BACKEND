"""
G1 LIVE check — real Gemini calls (costs money). Run only when the owner approves. Hard cap: 5 calls per run.

PowerShell:
  $env:CHECKER_GEMINI_API_KEY = "<key>"          # never in a file
  $env:CHECKER_GEMINI_MODEL   = "<same model as production>"
  python scripts/g1_live_check.py [--real <dir with <id>.pdf + <id>.json>] [--baseline <old result json>]

1. Fixture: RUBRIC (1 call) + GRADE (1 call) on tests/fixtures/answer_checker/answer_sheet_messy.pdf.
   Recordings → tests/fixtures/answer_checker/recordings/g1_live/ (synthetic sheet, safe to commit).
2. Each real sheet in --real (pdf + json holding {"rubric", "ai_result", "strictness"} read from the DB):
   GRADE only (1 call each, the stored rubric is reused), compared with its old result.
   Recordings stay in that folder (minors' handwriting — never commit them).
Prints page numbers / rotation / order, answer positions on the corrected pages, marks, flags, calls, tokens.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures" / "answer_checker"
sys.path.insert(0, str(ROOT))

if os.getenv("GEMINI_MOCK") == "1":
    sys.exit("GEMINI_MOCK=1 is set — this script is the LIVE check. Unset it.")
if not (os.getenv("CHECKER_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")):
    sys.exit("Set $env:CHECKER_GEMINI_API_KEY in this terminal first.")
if not os.getenv("CHECKER_GEMINI_MODEL"):
    sys.exit("Set $env:CHECKER_GEMINI_MODEL to the model production uses (no default on purpose).")
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")   # config needs it; nothing is called
os.environ.setdefault("GEMINI_API_KEY", os.environ.get("CHECKER_GEMINI_API_KEY", ""))

from app.services import smart_assignment_service as svc  # noqa: E402
from app.services.answer_checker.copy_check import grade_copy  # noqa: E402

TOTAL_CAP = 5
# Where each answer starts on the CORRECTED fixture pages (page index, y 0–1000), measured by hand
EXPECTED_POS = {"1": (0, 344), "2": (0, 367), "3": (0, 398), "4": (0, 428), "5": (2, 126), "7": (0, 493),
                "8": (1, 375), "9": (1, 459), "10": (2, 333), "11": (0, 138), "12": (3, 132), "13": (1, 136),
                "14": (2, 187), "15": (3, 228), "16": (4, 149), "17": (5, 141)}
Y_TOLERANCE = 60

_record_dir: Path = FIX / "recordings" / "g1_live"
_tag = "CALL"
_seen: dict = {}
LEDGER = {"http_calls": 0, "input_tokens": 0, "output_tokens": 0, "models": []}


class _CappedModels:
    """Wraps client.models: EVERY HTTP generate call (fallbacks and retries included) is counted against
    TOTAL_CAP before it is made, and its raw reply is saved as <TAG>_<n>.json (reusable with GEMINI_MOCK=1)."""

    def __init__(self, models):
        self._models = models

    def generate_content(self, model, contents, config):
        if LEDGER["http_calls"] >= TOTAL_CAP:
            raise RuntimeError(f"live check stopped: {TOTAL_CAP} Gemini calls used")
        LEDGER["http_calls"] += 1
        LEDGER["models"].append(model)
        response = self._models.generate_content(model=model, contents=contents, config=config)
        um = getattr(response, "usage_metadata", None)
        inp, out = getattr(um, "prompt_token_count", 0) or 0, getattr(um, "candidates_token_count", 0) or 0
        LEDGER["input_tokens"] += inp
        LEDGER["output_tokens"] += out
        tag = _current_tag()
        n = _seen[tag] = _seen.get(tag, 0) + 1
        try:
            data = json.loads(response.text)
        except ValueError:
            data = {"_unparsed_text": response.text}
        if isinstance(data, dict):
            data["_usage"] = {"input": inp, "output": out, "model": model}
        _record_dir.mkdir(parents=True, exist_ok=True)
        (_record_dir / f"{tag}_{n}.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        return response


class _CappedClient:
    def __init__(self, client):
        self.models = _CappedModels(client.models)
        self.files = client.files


_real_get_client = svc._get_client
svc._get_client = lambda: _CappedClient(_real_get_client())
_real_generate = svc._generate


def _tagging_generate(client, contents, tag):
    global _tag
    _tag = tag
    return _real_generate(client, contents, tag)


def _current_tag():
    return _tag


svc._generate = _tagging_generate


def _pages_report(ordered):
    return [{"student_page_no": p.student_page_no, "source_index": p.source_index, "rotation": p.rotation,
             "blank": p.blank} for p in ordered]


def _positions(result):
    return {q["q_no"]: (q.get("page_index"), q.get("y_start")) for q in result["graded_questions"] if q["counted"]}


def check_fixture(meter, baseline: dict) -> dict:
    global _record_dir
    _record_dir = FIX / "recordings" / "g1_live"
    expected = json.loads((FIX / "expected_evaluation.json").read_text("utf-8"))
    t = time.time()
    rubric = svc.build_rubric(str(FIX / "question_paper.pdf"), str(FIX / "answer_key.pdf"))
    rubric = {k: v for k, v in rubric.items() if not k.startswith("_")}
    result, ordered, usage = grade_copy([str(FIX / "answer_sheet_messy.pdf")], rubric)
    secs = round(time.time() - t, 1)

    old = {q["q_no"]: q.get("teacher_marks", q["marks_awarded"])
           for q in (baseline.get("graded_questions") or []) if q.get("counted", True)}
    got = {q["q_no"]: q["marks_awarded"] for q in result["graded_questions"] if q["counted"]}
    pos = _positions(result)
    rows = []
    for q in expected["questions"]:
        k = str(q["q"])
        p_exp = EXPECTED_POS.get(k)
        p_got = pos.get(k, (None, None))
        pos_ok = None if p_exp is None else (p_got[0] == p_exp[0] and p_got[1] is not None
                                            and abs(p_got[1] - p_exp[1]) <= Y_TOLERANCE)
        rows.append({"q": k, "expected": q["expected"], "got": got.get(k), "prod_old": old.get(k),
                     "pos_expected": p_exp, "pos_got": p_got, "pos_ok": pos_ok})
    out = {"total": result["total_marks_awarded"], "expected_total": expected["expected_total"],
           "prod_old_total": baseline.get("total_marks_awarded"), "pages": _pages_report(ordered),
           "flags": result["sheet_flags"], "rows": rows, "usage": usage, "seconds": secs,
           "model": svc._preferred_model}
    (ROOT / "tests" / "output" / "g1_live").mkdir(parents=True, exist_ok=True)
    norm_dir = ROOT / "tests" / "output" / "g1_live" / "norm"
    norm_dir.mkdir(exist_ok=True)
    result["norm_pages"] = []
    for i, p in enumerate(ordered):
        (norm_dir / f"p{i}.jpg").write_bytes(p.jpeg)
        result["norm_pages"].append({"index": i, "source_index": p.source_index, "student_page_no": p.student_page_no,
                                     "rotation": p.rotation, "blank": p.blank, "path": f"norm/p{i}.jpg",
                                     "w": p.w, "h": p.h})
    (ROOT / "tests" / "output" / "g1_live" / "ai_result.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def check_real(meter, real_dir: Path) -> list:
    global _record_dir
    reports = []
    for meta_path in sorted(real_dir.glob("*.json")):
        pdf = real_dir / f"{meta_path.stem}.pdf"
        if not pdf.exists():
            continue
        if LEDGER["http_calls"] >= TOTAL_CAP:
            reports.append({"sheet": meta_path.stem, "skipped": "call cap reached"})
            continue
        meta = json.loads(meta_path.read_text("utf-8"))
        _record_dir = real_dir / "recordings" / meta_path.stem
        _seen.clear()                     # each recording folder starts at <TAG>_1.json
        t = time.time()
        result, ordered, usage = grade_copy([str(pdf)], meta["rubric"], meta.get("strictness") or "medium")
        old = {q["q_no"]: q["marks_awarded"] for q in (meta.get("ai_result") or {}).get("graded_questions") or []
               if q.get("counted", True)}
        got = {q["q_no"]: q["marks_awarded"] for q in result["graded_questions"] if q["counted"]}
        reports.append({"sheet": meta_path.stem, "total": result["total_marks_awarded"],
                        "old_total": (meta.get("ai_result") or {}).get("total_marks_awarded"),
                        "per_question": {k: (old.get(k), got.get(k)) for k in sorted(set(old) | set(got), key=str)},
                        "pages": _pages_report(ordered), "positions": _positions(result),
                        "flags": result["sheet_flags"], "usage": usage, "seconds": round(time.time() - t, 1)})
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", type=Path, help="folder with <sheet>.pdf + <sheet>.json (rubric + old ai_result)")
    ap.add_argument("--baseline", type=Path, help="old production ai_result JSON for the fixture (optional)")
    ap.add_argument("--report", type=Path, default=ROOT / "tests" / "output" / "g1_live" / "report.json")
    args = ap.parse_args()
    baseline = {}
    if args.baseline:
        b = json.loads(args.baseline.read_text("utf-8"))
        baseline = b.get("ai_result", b)

    meter = None
    report = {"model_env": os.environ["CHECKER_GEMINI_MODEL"], "fixture": check_fixture(meter, baseline)}
    if args.real:
        report["real"] = check_real(meter, args.real)
    report["total_calls"] = LEDGER

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    f = report["fixture"]
    print(f"model={f['model']}  fixture total={f['total']} (expected {f['expected_total']}, old prod {f['prod_old_total']})")
    print("pages (student order):", [(p["student_page_no"], p["source_index"], p["rotation"]) for p in f["pages"]])
    print(f"{'Q':4s} {'exp':>4s} {'got':>4s} {'old':>4s}  pos expected → got")
    for r in f["rows"]:
        print(f"{r['q']:4s} {r['expected']:>4} {r['got']!s:>4} {r['prod_old']!s:>4}  {r['pos_expected']} → {r['pos_got']}"
              f"{'' if r['pos_ok'] in (True, None) else '  <-- position off'}")
    for fl in f["flags"]:
        print("FLAG", fl["code"], "-", fl["message"])
    for r in report.get("real", []):
        print("REAL", r.get("sheet"), r.get("skipped") or f"total {r['total']} (old {r['old_total']})")
    print("calls/tokens:", report["total_calls"])
    print("report:", args.report)


if __name__ == "__main__":
    main()
