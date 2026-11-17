"""Task 9: the paper header's CBSE session comes from the date (April-March), not a hardcoded "2025-26"."""
from datetime import date
from pathlib import Path

import pytest

from app.services import export_service
from app.services.export_service import _previous_session, cbse_session


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv("CBSE_SESSION", raising=False)


@pytest.mark.parametrize("today, expected", [
    (date(2026, 11, 17), "2026-27"),
    (date(2027, 2, 15), "2026-27"),
    (date(2027, 4, 10), "2027-28"),
    (date(2027, 3, 31), "2026-27"),
    (date(2027, 4, 1), "2027-28"),
    (date(2099, 6, 1), "2099-00"),
])
def test_session_from_date(today, expected):
    assert cbse_session(today) == expected


def test_env_override(monkeypatch):
    monkeypatch.setenv("CBSE_SESSION", " 2025-26 ")
    assert cbse_session(date(2026, 11, 17)) == "2025-26"


def test_previous_session_for_footer_note():
    assert _previous_session("2026-27") == "2025-26"
    assert _previous_session("custom") == ""


def test_no_hardcoded_session_left_in_exports():
    src = Path(export_service.__file__).read_text(encoding="utf-8")
    assert "2025-26" not in src and "2024-25" not in src
