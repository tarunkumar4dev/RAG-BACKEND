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


SEED = ROOT / "scripts" / "contest_seed" / "contest_payload.json"


def _ws(s) -> str:
    return " ".join(str(s or "").split())


@pytest.mark.skipif(not SEED.exists(), reason="scripts/contest_seed/contest_payload.json not added yet")
def test_golden_matches_the_seeded_contest_payload(recordings):
    """The importer's reading of the paper = the hand-made contest payload (whitespace-normalised)."""
    seed = json.loads(SEED.read_text(encoding="utf-8-sig"))["questions"]
    r = ci.import_paper(str(PDF), ci.check_pdf(str(PDF)), time.monotonic() + 270)
    assert len(r["questions"]) == len(seed)
    diffs = []
    for i, (got, want) in enumerate(zip(r["questions"], seed), start=1):
        want_opts = [(o["label"], _ws(o["text"])) if isinstance(o, dict) else ("ABCDEF"[k], _ws(o))
                     for k, o in enumerate(want.get("options") or [])]
        if _ws(got["question_text"]) != _ws(want["question_text"]):
            diffs.append((i, "question_text", got["question_text"][:60], want["question_text"][:60]))
        if [(o["label"], _ws(o["text"])) for o in got["options"]] != want_opts:
            diffs.append((i, "options"))
        if got["correct_answer"] != want.get("correct_answer"):
            diffs.append((i, "correct_answer", got["correct_answer"], want.get("correct_answer")))
    # Q24: the recorded reply reads "supply curve FOR foreign exchange" where the paper (and the payload) say "OF"
    # — a model slip seen in 2 of 4 live runs on 24 Oct; no validator can catch a plausible one-word swap, which is
    # why the teacher reviews every imported paper. Any other difference fails.
    known = {24: ("question_text",)}
    assert [d for d in diffs if d[1:2] != known.get(d[0], ())[:1]] == []
