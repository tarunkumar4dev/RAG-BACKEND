"""Option labels against the fixtures shared with the frontend. Run: cd test-generator-backend && python -m pytest tests/unit"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.services.ncert_options import decode_options, normalize_option_label, strip_option_labels  # noqa: E402

FIXTURES = json.loads((ROOT / "scripts/fixtures/ncert_option_label_fixtures.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", FIXTURES["normalize_option_label"], ids=lambda c: c["note"])
def test_normalize_option_label(case):
    assert normalize_option_label(case["input"]) == case["expected"]


@pytest.mark.parametrize("case", FIXTURES["strip_option_labels"], ids=lambda c: c["note"])
def test_strip_option_labels(case):
    assert strip_option_labels(case["input"]) == case["expected"]


@pytest.mark.parametrize("raw, expected", [
    (["a", "b"], ["a", "b"]),
    ('["a", "b"]', ["a", "b"]),
    (json.dumps(json.dumps(["a"])), ["a"]),
    ("not json", []),
    (None, []),
    ({"a": 1}, []),
])
def test_decode_options(raw, expected):
    assert decode_options(raw) == expected
