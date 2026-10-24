"""Golden paper (Class XII Economics, 50 MCQs + answer key grid and table) on the Gemini replies recorded by
scripts/eval_contest_import.py --record (24 Oct 2026, REST path, gemini-3-flash-preview). Re-record after a prompt / schema
change; the live run is `python scripts/eval_contest_import.py`."""

import json
import sys
import time

import pytest

from app.services import contest_import_service as ci
from app.services import smart_assignment_service as svc
from tests.conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from eval_contest_import import GOLDEN, FIXTURES, golden_problems  # noqa: E402

GOLD = json.loads(GOLDEN.read_text(encoding="utf-8"))
PDF = FIXTURES / GOLD["pdf"]
pytestmark = pytest.mark.skipif(not PDF.exists(), reason=f"fixture {PDF.name} not in tests/fixtures/")


@pytest.fixture
def recordings(monkeypatch):
    monkeypatch.setattr(svc, "GEMINI_MOCK_DIR", str(GOLDEN.parent / "recordings"))
    monkeypatch.setattr(svc, "_mock_seen", {})


def test_golden_economics_paper(recordings):
    r = ci.import_paper(str(PDF), ci.check_pdf(str(PDF)), time.monotonic() + 270)
    assert golden_problems(r, GOLD) == []
    assert r["institute"] == "EDUCATION BEAST"
    q43 = r["questions"][42]
    assert "₹" in q43["question_text"] and q43["options"][1]["text"] == "₹42,000 (up from ₹40,000)"
    q45 = r["questions"][44]                                  # A/R with all four options on one paragraph
    assert q45["question_text"].startswith("Assertion (A):") and "Reason (R):" in q45["question_text"]
    assert [o["text"] for o in q45["options"]][3].startswith("A is false, but R is true")
    q47 = r["questions"][46]
    assert "I. Devaluation" in q47["question_text"] and "II. Depreciation" in q47["question_text"]
    assert all(q["explanation"] is None for q in r["questions"])   # the paper prints none; none invented


def test_golden_key_mismatch_is_flagged_not_fixed(recordings, tmp_path, monkeypatch):
    """The same paper with one wrong key entry (Q12 a → c): the key is kept, KEY_MISMATCH_SUSPECT on Q12 only."""
    src = GOLDEN.parent / "recordings"
    for f in src.glob("*.json"):
        (tmp_path / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    parse = json.loads((tmp_path / "IMPORT-PARSE_1.json").read_text(encoding="utf-8"))
    for part in ("table", "list"):
        for row in parse["answer_key"][part]:
            if row["number"] == 12:
                row["letter"] = "(c)" if part == "table" else "c"
                row["text"] = ""
    (tmp_path / "IMPORT-PARSE_1.json").write_text(json.dumps(parse, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(svc, "GEMINI_MOCK_DIR", str(tmp_path))
    r = ci.import_paper(str(PDF), ci.check_pdf(str(PDF)), time.monotonic() + 270)
    assert r["questions"][11]["correct_answer"] == "C" and r["questions"][11]["answer_source"] == "key"
    assert [(w["question"], w["code"]) for w in r["warnings"]] == [(12, "KEY_MISMATCH_SUSPECT")]
