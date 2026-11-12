"""Shared helpers for the Gate-1 cost audit tests (docs/COST_MAP.md). No network, no paid calls.

`no_network` blocks every outbound HTTP path the services use (httpx, requests), so a test that forgets a mock
fails loudly instead of reaching a provider. The local .env can set LLM_PROVIDER=deepseek with a real key;
`force_gemini` turns that off for the test.
"""
from __future__ import annotations

import itertools
import json
import uuid
from types import SimpleNamespace

import pytest


class NetworkBlocked(RuntimeError):
    pass


def _blocked(*_a, **_k):
    raise NetworkBlocked("outbound HTTP is blocked in audit tests")


@pytest.fixture
def no_network(monkeypatch):
    import httpx
    import requests
    monkeypatch.setattr(httpx, "post", _blocked)
    monkeypatch.setattr(httpx, "get", _blocked)
    monkeypatch.setattr(httpx.Client, "send", _blocked)
    monkeypatch.setattr(httpx.AsyncClient, "send", _blocked)
    monkeypatch.setattr(requests.Session, "request", _blocked)


@pytest.fixture
def force_gemini(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(settings, "CHECKER_PROVIDER", "gemini")


def approx_tokens(text: str) -> int:
    """chars / 4. Not a tokenizer (none is available offline); good enough to catch a prompt that grows 20%."""
    return (len(text) + 3) // 4


class FakeGenModels:
    """client.models stand-in for the test generator. `script(n)` -> str reply or an Exception to raise."""

    def __init__(self, script):
        self.script = script
        self.calls: list[tuple[str, str]] = []   # (model, prompt)

    def generate_content(self, model, contents, config=None):
        self.calls.append((model, contents if isinstance(contents, str) else json.dumps(str(contents))))
        item = self.script(len(self.calls))
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item, usage_metadata=SimpleNamespace(
            prompt_token_count=approx_tokens(self.calls[-1][1]), candidates_token_count=100, thoughts_token_count=0))


def fake_parse_batch_factory():
    """Replaces _parse_batch: returns BATCH_SIZE valid questions of the chapter's own format, so the happy path
    is measured without depending on the parser's acceptance rules (those are not what the cost guard watches)."""
    from app.core.config import settings
    from app.models.test_generator import GeneratedQuestion, QuestionFormat
    seq = itertools.count(1)

    def parse(raw, chapter, request, section_key=None):
        if raw == "UNUSABLE":
            return []
        out = []
        fmt = chapter.format if isinstance(chapter.format, QuestionFormat) else QuestionFormat(str(chapter.format))
        for _ in range(settings.BATCH_SIZE):
            i = next(seq)
            gq = GeneratedQuestion(
                id=str(uuid.uuid4()), text=f"Audit question {i} about {chapter.chapter}: explain the idea.",
                options=["A) one", "B) two", "C) three", "D) four"] if fmt.value in ("mcq", "assertion_reason") else None,
                correct_answer="A) one", explanation="A long enough explanation for the audit question.",
                marks=chapter.marks_per_question, difficulty=chapter.difficulty, chapter=chapter.chapter,
                format=fmt)
            gq._section = section_key
            gq._is_or = False
            gq._or_of = None
            out.append(gq)
        return out
    return parse
