"""
Contest-from-PDF regression snapshot: runs the existing parser (contest_import_service.import_paper) on both
Question Bank fixtures with their GEMINI_MOCK recordings (no network) and prints / writes the full result.

    python scripts/qb_contest_flow_snapshot.py            # compare with tests/fixtures/question_bank/contest_flow_baseline/
    python scripts/qb_contest_flow_snapshot.py --write    # (re)write the baseline — only on a commit where the
                                                          # contest flow is known good (first written on origin/main
                                                          # 7cba4eb, before any Question Bank code existed)

tests/unit/test_question_bank_contest_regression.py runs the same comparison.
"""

import json
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ.setdefault("GEMINI_MOCK", "1")
os.environ.setdefault("GEMINI_MOCK_DIR", str(BACKEND / "tests" / "fixtures" / "contest_import" / "recordings"))
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("GEMINI_API_KEY", "mock")

FIXTURES = BACKEND / "tests" / "fixtures"
BASELINE = FIXTURES / "question_bank" / "contest_flow_baseline"
CASES = {
    # name: (pdf, recordings dir)
    "economics": (FIXTURES / "Class12_Economics_ForeignExchangeRate_50MCQs_EducationBeast.pdf",
                  FIXTURES / "contest_import" / "recordings"),
    "business_studies": (FIXTURES / "question_bank" / "mixed_business_studies.pdf",
                         FIXTURES / "question_bank" / "recordings_business_studies"),
}


def run_case(name: str) -> dict:
    """The contest flow's parse result for one fixture, exactly as the import job stores it."""
    from app.services import contest_import_service as ci
    from app.services import smart_assignment_service as svc
    pdf, recordings = CASES[name]
    old_dir, old_seen, old_mock = svc.GEMINI_MOCK_DIR, svc._mock_seen, svc.GEMINI_MOCK
    svc.GEMINI_MOCK_DIR, svc._mock_seen, svc.GEMINI_MOCK = str(recordings), {}, True
    try:
        return ci.import_paper(str(pdf), ci.check_pdf(str(pdf)), time.monotonic() + 270)
    finally:
        svc.GEMINI_MOCK_DIR, svc._mock_seen, svc.GEMINI_MOCK = old_dir, old_seen, old_mock


def canonical(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True)


def main() -> int:
    write = "--write" in sys.argv
    sys.stdout.reconfigure(encoding="utf-8")
    bad = 0
    for name in CASES:
        got = canonical(run_case(name))
        path = BASELINE / f"{name}.json"
        if write:
            BASELINE.mkdir(parents=True, exist_ok=True)
            path.write_text(got, encoding="utf-8")
            print(f"{name}: baseline written ({len(json.loads(got)['questions'])} questions)")
            continue
        same = path.exists() and path.read_text(encoding="utf-8") == got
        bad += not same
        print(f"{name}: {'IDENTICAL' if same else 'DIFFERENT'} ({len(json.loads(got)['questions'])} questions)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
