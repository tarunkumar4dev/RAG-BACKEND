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

# file -> ids hardcoded there (Gate-1 inventory 12 Nov 2026; 21 Nov 2026: every model *selection* moved to
# config.py, what is left is price data keyed by model id). Shrink this, never grow it.
KNOWN = {
    # est_cost_usd: the flash-lite list price is only applied when that model answered
    "test-generator-backend/app/api/v1/endpoints/modules.py": {"gemini-3.5-flash-lite"},
    # keys of the cost table _PRICE_PER_M (Gate 2 #7), not a model selection
    "test-generator-backend/app/services/test_generator_service.py": {"deepseek-flash", "gemini-3.5-flash-lite",
                                                                      "gemini-3.6-flash"},
    "test-generator-backend/app/services/worksheet_service.py": {"gemini-3.5-flash-lite"},
    "agent/voice/api.py": {"saaras:v4"},
    "agent/voice/providers.py": {"saaras:v4"},
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


def test_inventory_has_no_stale_entries():
    """KNOWN is exact: an id removed from a file must be removed here too, so the list only ever shrinks."""
    found = _ids_by_file()
    stale = {f: ids - found.get(f, set()) for f, ids in KNOWN.items() if ids - found.get(f, set())}
    assert not stale, f"no longer hardcoded there, remove from KNOWN: {stale}"


def test_checker_model_ids_only_in_config():
    """The copy checker's primary and fallbacks come from config.py (settings.CHECKER_GEMINI_*), nowhere else."""
    checker_ids = {"gemini-3.8-flash", "gemini-3.6-flash", "gemini-3-flash-preview"}
    hits = {f: ids & checker_ids for f, ids in _ids_by_file().items()
            if f not in CONFIG_FILES and ids & checker_ids and f not in KNOWN}
    assert not hits, hits


@pytest.mark.xfail(strict=True, reason="AUDIT: price tables keyed by model id still sit in 3 non-config files "
                                       "(and 2 agent/voice files)")
def test_all_model_ids_come_from_one_config():
    found = _ids_by_file()
    assert set(found) <= CONFIG_FILES, sorted(set(found) - CONFIG_FILES)


def test_deprecated_and_preview_ids_in_use():
    in_use = set().union(*_ids_by_file().values())
    assert in_use & set(DEPRECATED) == {"gemini-2.5-flash"}
    assert in_use & PREVIEW == {"gemini-3-flash-preview"}


def test_gemini_model_env_read_with_own_default_in_several_places():
    """GEMINI_MODEL is read only in config.py since 21 Nov 2026 (was four modules, each with its own default)."""
    hits = []
    for p in (REPO / "test-generator-backend/app").rglob("*.py"):
        if "_archived" in p.name:
            continue
        if re.search(r"getenv\(\s*[\"']GEMINI_MODEL[\"']", p.read_text(encoding="utf-8")):
            hits.append(p.relative_to(REPO).as_posix())
    assert hits == ["test-generator-backend/app/core/config.py"], hits
