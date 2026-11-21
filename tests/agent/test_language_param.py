"""ChatRequest.language (optional, backward compatible): validated, default hinglish, one line at the END of the
system prompt (the cached prefix stays shared), voice and text turns alike."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.core import LANGUAGE_LINES, _base_prompt, system_prompt
from agent.schemas import ChatRequest
from agent.voice.api import VoiceChatRequest


def test_default_is_hinglish_and_old_bodies_still_validate():
    assert ChatRequest(message="hi").language == "hinglish"
    assert VoiceChatRequest(message="hi").language == "hinglish"


@pytest.mark.parametrize("lang", ["en", "hi", "hinglish"])
def test_each_language_is_accepted(lang):
    assert ChatRequest(message="x", language=lang).language == lang
    assert VoiceChatRequest(message="x", language=lang).language == lang


@pytest.mark.parametrize("bad", ["fr", "hi-en", "", "EN"])
def test_unknown_language_is_422(bad):
    with pytest.raises(ValidationError):
        ChatRequest(message="x", language=bad)


def test_extra_fields_still_forbidden():
    with pytest.raises(ValidationError):
        ChatRequest(message="x", lang="en")


@pytest.mark.parametrize("voice", [False, True])
def test_language_line_is_appended_last_and_prefix_is_shared(admin_ctx, voice):
    prompts = {lang: system_prompt(admin_ctx, voice, lang) for lang in LANGUAGE_LINES}
    base = _base_prompt(admin_ctx)
    for lang, p in prompts.items():
        assert p.startswith(base)
        assert LANGUAGE_LINES[lang] in p
        assert "test-paper questions" in p
    assert "Devanagari" in LANGUAGE_LINES["hi"] and "never Devanagari" in LANGUAGE_LINES["hinglish"]
    assert system_prompt(admin_ctx, voice) == prompts["hinglish"]
