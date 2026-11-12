"""Gate-1 audit §5: model ids (docs/COST_MAP.md §5). Static scan, no network.

Scans Python string literals (single-line; docstrings and comments are ignored) in test-generator-backend/app and
agent/ for provider model ids.
- test_no_new_hardcoded_model_ids: every id must sit in a file listed in KNOWN (today's inventory). A new hardcoded
  id anywhere else fails the build -> put it in config instead.
- test_all_model_ids_come_from_one_config (xfail strict): the target — ids only in the config modules.
- test_deprecated_ids_in_use: ids with a provider shutdown date are listed here with the source, so the list is
  reviewed whenever it changes.
"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCAN = ("test-generator-backend/app", "agent")
PAT = re.compile(r"(gemini-(?:[0-9][\w.\-]*|flash-latest)|claude-(?:sonnet|haiku|opus|fable)-[\w\-]+|"
                 r"deepseek[/\-][\w\-]+|gpt-[0-9][\w.\-]*|text-embedding-[\w\-]+|saaras:v\d|bulbul:v[\w\-]+)")

CONFIG_FILES = {"test-generator-backend/app/core/config.py", "agent/config.py", "agent/voice/config.py"}

# file -> ids hardcoded there on 12 Nov 2026 (Gate-1 inventory). Shrink this, never grow it.
KNOWN = {
    "test-generator-backend/app/core/config.py": {"deepseek-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash"},
    "test-generator-backend/app/api/v1/endpoints/chat.py": {"deepseek/deepseek-chat", "gemini-2.5-flash"},
    "test-generator-backend/app/api/v1/endpoints/modules.py": {"gemini-3.5-flash-lite"},
    "test-generator-backend/app/services/community_quiz_generator.py": {"gemini-flash-latest"},
    "test-generator-backend/app/services/contest_import_gemini.py": {"gemini-3-flash-preview", "gemini-3.8-flash"},
    "test-generator-backend/app/services/contest_import_service.py": {"gemini-3-flash-preview", "gemini-3.8-flash"},
    "test-generator-backend/app/services/module_service.py": {"gemini-3.5-flash-lite"},
    "test-generator-backend/app/services/smart_assignment_service.py": {"gemini-3.6-flash", "gemini-3.8-flash"},
    # gemini-3.6-flash here is a key of the cost table _PRICE_PER_M (Gate 2 #7), not a model selection
    "test-generator-backend/app/services/test_generator_service.py": {"deepseek-flash", "gemini-3.5-flash-lite",
                                                                      "gemini-3.6-flash"},
    "test-generator-backend/app/services/worksheet_service.py": {"gemini-3.5-flash-lite"},
    "agent/config.py": {"claude-haiku-5-5", "claude-sonnet-5-5", "gemini-2.5-flash"},
    "agent/voice/config.py": {"bulbul:v2", "bulbul:v3", "bulbul:v4-flash", "saaras:v3", "saaras:v4"},
    "agent/voice/api.py": {"saaras:v4"},
    "agent/voice/providers.py": {"saaras:v3", "saaras:v4"},
    "agent/voice/repair.py": {"saaras:v4"},
}

# Provider shutdown dates (checked 12 Nov 2026):
#   aistudio.google.com/docs/deprecations — gemini-2.5-flash: shutdown 1 Dec 2026, requests then route to
#   gemini-3.5-flash-lite; text-embedding-004: shut down 14 Jan 2026. gemini-3-flash-preview: preview, no date.
#   platform.claude.com/docs/en/about-claude/model-deprecations — claude-sonnet-5-5 / claude-haiku-5-5 active
#   (not sooner than 28 Sep / 7 Oct 2027).
DEPRECATED = {"gemini-2.5-flash": "2026-12-01", "text-embedding-004": "2026-01-14"}
PREVIEW = {"gemini-3-flash-preview"}


def _ids_by_file() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for base in SCAN:
        for p in sorted((REPO / base).rglob("*.py")):
            if "_archived" in p.name or "__pycache__" in p.parts or ".devdata" in p.parts:
                continue
            src = p.read_text(encoding="utf-8")
            for tok in tokenize.generate_tokens(io.StringIO(src).readline):
                if tok.type != tokenize.STRING:
                    continue
                body = tok.string.lstrip("rbuRBUfF")
                if body.startswith(('"""', "'''")):
                    continue   # docstrings / prompt templates
                for m in PAT.findall(tok.string):
                    out.setdefault(p.relative_to(REPO).as_posix(), set()).add(m)
    return out


def test_no_new_hardcoded_model_ids():
    found = _ids_by_file()
    # the config modules are where ids (and price tables keyed by id) belong: always allowed
    new = {f: ids - KNOWN.get(f, set()) for f, ids in found.items()
           if f not in CONFIG_FILES and ids - KNOWN.get(f, set())}
    assert not new, f"model ids hardcoded outside config (move them to config): {new}"


@pytest.mark.xfail(strict=True, reason="AUDIT: ids are hardcoded in 12 non-config files; GEMINI_MODEL is read "
                                       "with its own default in 4 places")
def test_all_model_ids_come_from_one_config():
    found = _ids_by_file()
    assert set(found) <= CONFIG_FILES, sorted(set(found) - CONFIG_FILES)


def test_deprecated_and_preview_ids_in_use():
    in_use = set().union(*_ids_by_file().values())
    assert in_use & set(DEPRECATED) == {"gemini-2.5-flash"}
    assert in_use & PREVIEW == {"gemini-3-flash-preview"}


def test_gemini_model_env_read_with_own_default_in_several_places():
    """GEMINI_MODEL is read in four modules, each with its own default — changing config.py alone does not move
    worksheets / modules. Pinned so the count only goes down."""
    hits = []
    for p in (REPO / "test-generator-backend/app").rglob("*.py"):
        if "_archived" in p.name:
            continue
        if re.search(r"getenv\(\s*[\"']GEMINI_MODEL[\"']", p.read_text(encoding="utf-8")):
            hits.append(p.relative_to(REPO).as_posix())
    assert len(hits) <= 4, hits
