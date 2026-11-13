"""Contest submit sends X-Attempt-Id from the browser: the CORS preflight must allow it (it returned 400 since
11 Sep 2026, so no contest could be submitted). Run: cd test-generator-backend && python -m pytest tests/unit"""

import os
import sys
from pathlib import Path

os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")   # config needs it; nothing is called
os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("CORS_ORIGINS", "https://www.a4ai.in")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.main import app  # noqa: E402

SUBMIT = "/api/v1/contests/00000000-0000-0000-0000-000000000000/submit"

# Settings are evaluated when app.core.config is first imported, so the
# setdefault above only takes effect if this module happens to import config
# first. Reading the origin back from the app's own configuration keeps these
# tests order-independent — what they actually assert is that X-Attempt-Id is
# an allowed request header, not which origin is configured.
ALLOWED_ORIGIN = (settings.cors_origin_list or ["https://www.a4ai.in"])[0]


def _preflight(headers: str):
    return TestClient(app).options(SUBMIT, headers={
        "Origin": ALLOWED_ORIGIN,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": headers,
    })


def test_contest_submit_preflight_allows_attempt_id():
    r = _preflight("content-type,x-attempt-id,authorization")
    assert r.status_code == 200
    assert "x-attempt-id" in r.headers["access-control-allow-headers"].lower()


def test_unknown_headers_still_rejected():
    assert _preflight("content-type,x-something-else").status_code == 400
