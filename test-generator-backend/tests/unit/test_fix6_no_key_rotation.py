"""Gate 2 fix #6: module_service makes ONE Gemini call with ONE key; a 429 is raised, never rotated around,
and no key material reaches the log."""
import logging
from types import SimpleNamespace

import pytest

from app.services import module_service as ms
from audit_support import no_network  # noqa: F401


@pytest.fixture
def genai(monkeypatch):
    created, calls = [], []

    class Client:
        def __init__(self, api_key):
            created.append(api_key)
            self.models = SimpleNamespace(generate_content=lambda **kw: (calls.append(kw) or (_ for _ in ()).throw(
                Exception("429 RESOURCE_EXHAUSTED"))))
            self.files = SimpleNamespace(upload=lambda **kw: "file")
    monkeypatch.setattr(ms, "get_genai", lambda: SimpleNamespace(Client=Client))
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSECRETKEY0123456789")
    monkeypatch.setenv("GEMINI_API_KEY_1", "AIzaOTHERKEY0123456789")
    monkeypatch.setenv("GEMINI_API", "AIzaTHIRDKEY0123456789")
    return created, calls


def test_summary_call_uses_one_key_and_raises_on_429(genai, caplog, no_network):  # noqa: F811
    created, calls = genai
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception, match="429"):
            ms._call_gemini_with_fallback("hello")
    assert created == ["AIzaSECRETKEY0123456789"] and len(calls) == 1
    assert "AIza" not in caplog.text


def test_ocr_uses_one_key_and_raises_on_429(genai, caplog, no_network):  # noqa: F811
    created, calls = genai
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(Exception, match="429"):
            ms.ModuleService._extract_with_gemini(b"%PDF-")
    assert len(created) == 1 and len(calls) == 1
    assert "AIza" not in caplog.text


def test_no_key_rotation_helper_left():
    assert not hasattr(ms, "_get_gemini_api_keys")
