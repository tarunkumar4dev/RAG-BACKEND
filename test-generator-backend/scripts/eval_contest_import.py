"""
Live check of "PDF → contest questions" against a golden paper (real Gemini calls, ~2 calls per run).

  python scripts/eval_contest_import.py                         # the Economics fixture vs its golden file
  python scripts/eval_contest_import.py --pdf X.pdf             # any paper: prints the result, no golden check
  python scripts/eval_contest_import.py --record DIR            # also save Gemini's replies as GEMINI_MOCK recordings
                                                                # (IMPORT-PARSE_1.json, IMPORT-SOLVE_1..n.json) for the
                                                                # offline golden test

Exit code 1 when the golden check fails. golden_problems() is shared with tests/unit/test_contest_import_golden.py.
"""

import argparse
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))

FIXTURES = BACKEND / "tests" / "fixtures"
GOLDEN = FIXTURES / "contest_import" / "golden_economics.json"


def _dashes(s) -> str:
    return " ".join(re.sub(r"[–—‒-]", "—", str(s or "")).split())


def golden_problems(result: dict, golden: dict) -> list:
    """Everything in `result` that differs from the golden expectations (empty list = pass)."""
    bad = []
    qs = result["questions"]
    if len(qs) != golden["questions"]:
        bad.append(f"{len(qs)} questions, expected {golden['questions']}")
    if result["answer_key_count"] != golden["key_count"] or not result["answer_key_found"]:
        bad.append(f"key: found={result['answer_key_found']} count={result['answer_key_count']}, "
                   f"expected {golden['key_count']}")
    for field in ("subject", "class_grade", "duration_minutes", "total_marks", "board"):
        if result[field] != golden[field]:
            bad.append(f"{field} = {result[field]!r}, expected {golden[field]!r}")
    if _dashes(result["title"]) != _dashes(golden["title"]):
        bad.append(f"title = {result['title']!r}, expected {golden['title']!r}")
    key = golden["key"].replace(" ", "").upper()
    level = {n: d for d, (a, b) in golden["difficulty"].items() for n in range(a, b + 1)}
    for i, q in enumerate(qs, start=1):
        if q["source_number"] != i:
            bad.append(f"position {i}: source_number {q['source_number']}")
        if q["question_type"] != "MCQ" or len(q["options"]) != 4 or [o["label"] for o in q["options"]] != list("ABCD"):
            bad.append(f"Q{i}: {q['question_type']} with options {[o['label'] for o in q['options']]}")
        if i <= len(key) and (q["correct_answer"] != key[i - 1] or q["answer_source"] != "key"):
            bad.append(f"Q{i}: answer {q['correct_answer']} ({q['answer_source']}), expected {key[i - 1]}")
        if level.get(i) and q["difficulty"] != level[i]:
            bad.append(f"Q{i}: difficulty {q['difficulty']}, expected {level[i]}")
        if not q["question_text"] or re.match(rf"^\s*(Q\.?\s*)?{i}\s*[.)]", q["question_text"]):
            bad.append(f"Q{i}: question text {q['question_text'][:40]!r}")
        if any(re.match(r"^\s*\(?[a-dA-D]\)", o["text"]) for o in q["options"]):
            bad.append(f"Q{i}: option prefix left in {[o['text'][:15] for o in q['options']]}")
    extra = [w for w in result["warnings"] if w["code"] not in golden["allowed_warnings"]]
    if extra:
        bad.append(f"unexpected warnings: {[(w['question'], w['code']) for w in extra]}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=Path, default=None)
    ap.add_argument("--record", type=Path, default=None)
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv(BACKEND / ".env", override=False)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.stdout.reconfigure(encoding="utf-8")

    from app.services import contest_import_service as ci
    from app.services import contest_import_gemini as gemini

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    pdf = args.pdf or FIXTURES / golden["pdf"]

    if args.record:
        args.record.mkdir(parents=True, exist_ok=True)
        real = gemini.generate_json
        lock, seen = threading.Lock(), {}

        def recording(parts, tag, schema, **kw):
            data = real(parts, tag, schema, **kw)
            with lock:
                n = seen[tag] = seen.get(tag, 0) + 1
            (args.record / f"{tag}_{n}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                                         encoding="utf-8")
            return data
        gemini.generate_json = recording

    start = time.monotonic()
    pages = ci.check_pdf(str(pdf))
    result = ci.import_paper(str(pdf), pages, start + 270, progress=lambda p: print(f"  progress {p}%"))
    took = time.monotonic() - start

    for q in result["questions"]:
        opts = " | ".join(f"{o['label']}) {o['text']}" for o in q["options"])
        print(f"Q{q['source_number']:>2} [{q['difficulty']:<6}] {q['correct_answer']} ({q['answer_source']}) "
              f"{q['question_text'][:70]!r}\n      {opts[:160]}")
    print(json.dumps({k: v for k, v in result.items() if k not in ("questions",)}, ensure_ascii=False, indent=1))
    print(f"\n{len(result['questions'])} questions in {took:.1f}s")
    print(f"RUN: seconds={took:.1f} key_check_skipped={any(w['code'] == 'KEY_CHECK_SKIPPED' for w in result['warnings'])}"
          f" warnings={[(w['question'], w['code']) for w in result['warnings']]}")

    if args.pdf:
        return 0
    bad = golden_problems(result, golden)
    print("GOLDEN: PASS" if not bad else "GOLDEN: FAIL\n  " + "\n  ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
