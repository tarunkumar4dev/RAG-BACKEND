"""Test setup: zero Gemini calls (GEMINI_MOCK=1 with the hand-written G1 recordings), dummy config."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "answer_checker"

os.environ["GEMINI_MOCK"] = "1"
os.environ["GEMINI_MOCK_DIR"] = str(FIXTURES / "recordings" / "g1_mock")
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("GEMINI_API_KEY", "mock")
sys.path.insert(0, str(ROOT))
