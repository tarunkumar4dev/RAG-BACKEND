"""Real anonymised copies (outside the repo, $REAL_COPIES_DIR/anonymised). Skipped when the variable is not set.
G3 adds the accuracy checks (checker marks vs teacher marks) on top of this structure check."""

import json
import os
from pathlib import Path

import pytest

BASE = os.getenv("REAL_COPIES_DIR")
pytestmark = pytest.mark.skipif(not BASE, reason="REAL_COPIES_DIR not set (real copies live outside the repo)")


def _copies():
    root = Path(BASE or ".") / "anonymised"
    return sorted(p for p in root.glob("copy_*") if p.is_dir()) if BASE else []


@pytest.mark.parametrize("copy", _copies(), ids=lambda p: p.name)
def test_copy_structure(copy):
    assert sorted((copy / "pages").glob("p*.jpg")), "no pages"
    assert (copy / "anonymise.json").exists(), "not produced by scripts/anonymise_copies.py"
    assert (copy / "rubric.json").exists() or (copy / "test.json").exists()
    exp = json.loads((copy / "expected.json").read_text("utf-8"))
    assert abs(sum(q["marks"] for q in exp["questions"]) - exp["total"]) <= 0.01
    assert all(0 <= q["marks"] <= q["max"] for q in exp["questions"])
