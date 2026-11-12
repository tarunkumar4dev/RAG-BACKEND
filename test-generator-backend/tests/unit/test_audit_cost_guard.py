"""Gate-1 audit §6: cost guard (docs/COST_MAP.md §6). Mocked providers, no network, no paid calls.

Per operation it records (a) provider calls on the happy path and (b) static prompt size in approx tokens
(chars/4 of every text prompt sent; images are not counted). Inputs are fixed and tiny, so (b) measures our own
prompt templates, not the teacher's content. Fails when either grows > 20 % over tests/cost_guard_baseline.json —
update the baseline in the same commit as the prompt/call change (and say why in the commit message):

    COST_GUARD_WRITE=1 python -m pytest tests/unit/test_audit_cost_guard.py

The agent's operations (voice query, agent question) are guarded by tests/agent/test_audit_cost_guard.py against
the same file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services import smart_assignment_service as svc
from app.services import test_generator_service as g
from app.models.test_generator import ChapterSection, TestGenerationRequest

from audit_support import FakeGenModels, approx_tokens, fake_parse_batch_factory, force_gemini, no_network  # noqa: F401

BASELINE = Path(__file__).resolve().parents[3] / "tests" / "cost_guard_baseline.json"
TOLERANCE = 1.20
MEASURED: dict[str, dict] = {}

CHUNKS = [{"chapter": "Light", "content": "Light reflects from a plane mirror. " * 40},
          {"chapter": "Electricity", "content": "Current flows through a conductor. " * 40}]


def _req(*chapters):
    return TestGenerationRequest(exam_title="Guard", subject="Science", class_grade="10", teacher_id="u",
                                 chapters=list(chapters))


def _ch(name, fmt, marks):
    return ChapterSection(chapter=name, quantity=5, difficulty="medium", format=fmt, marks_per_question=marks)


@pytest.fixture
def gen(monkeypatch, no_network, force_gemini):  # noqa: F811
    from app.core.config import settings
    monkeypatch.setattr(settings, "GEMINI_GEN_MODEL", "primary")
    monkeypatch.setattr(settings, "GEMINI_FALLBACK_MODEL", "fallback")
    monkeypatch.setattr(g, "_parse_batch", fake_parse_batch_factory())
    models = FakeGenModels(lambda n: json.dumps({"questions": []}))
    client = SimpleNamespace(models=models)
    monkeypatch.setattr(g, "_get_gemini_client", lambda: client)
    return models


def _record(op, prompts):
    MEASURED[op] = {"calls": len(prompts), "prompt_tokens": sum(approx_tokens(p) for p in prompts)}
    return MEASURED[op]


def test_measure_test_paper_custom(gen):
    g.generate_questions(_req(_ch("Light", "mcq", 1), _ch("Electricity", "short_answer", 3)), CHUNKS)
    _record("test_paper_custom_2ch", [p for _, p in gen.calls])


def test_measure_test_paper_cbse(gen):
    g.generate_questions(_req(_ch("Light", "mcq", 1), _ch("Electricity", "mcq", 1)), CHUNKS, cbse_pattern=True)
    _record("test_paper_cbse_2ch", [p for _, p in gen.calls])


def test_measure_worksheet_direct(monkeypatch, no_network):  # noqa: F811
    import asyncio
    import google.genai as genai_mod
    from app.api.v1.endpoints import modules as ep
    from app.services import module_service as ms
    prompts = []

    class Client:
        def __init__(self, **_):
            self.models = SimpleNamespace(generate_content=lambda model, contents, config: (
                prompts.append(contents) or SimpleNamespace(text=json.dumps({"questions": []}))))
    monkeypatch.setattr(genai_mod, "Client", Client)
    monkeypatch.setattr(ms, "get_supabase", lambda: SimpleNamespace(storage=SimpleNamespace(
        from_=lambda b: SimpleNamespace(download=lambda p: b"%PDF-fake"))))
    monkeypatch.setattr(ms.ModuleService, "_extract_pdf", staticmethod(lambda path: ("Photosynthesis. " * 60, 1, False)))
    from app.core.auth import AuthUser
    uid = "11111111-2222-3333-4444-555555555555"
    monkeypatch.setattr(ep, "check_usage", lambda u: {"allowed": True})
    monkeypatch.setattr(ep, "record_usage", lambda u: {})
    ep._worksheet_limiter._hits.clear()
    user = AuthUser(id=uid, email=None, is_anonymous=False, role="authenticated")
    out = asyncio.run(ep.generate_worksheet_direct({"storage_path": f"{uid}/worksheets/w.pdf", "subject": "Science",
                                                    "class_level": "10", "chapter_name": "Life processes",
                                                    "num_questions": 10}, user))
    assert out["success"]
    _record("worksheet_direct", prompts)


def test_measure_copy_check(monkeypatch):
    """Rubric (once per assignment) + GRADE for one copy, on the G1 fixture with recorded (mock) replies."""
    from app.services.answer_checker.copy_check import grade_copy
    from tests.conftest import FIXTURES
    prompts = []
    real = svc._generate

    def spy(client, contents, tag):
        prompts.append("".join(c for c in contents if isinstance(c, str)))
        return real(client, contents, tag)
    monkeypatch.setattr(svc, "_generate", spy)
    svc._mock_seen.clear()
    rubric = svc.build_rubric(str(FIXTURES / "question_paper.pdf"), str(FIXTURES / "answer_key.pdf"))
    grade_copy([str(FIXTURES / "answer_sheet_messy.pdf")], rubric)
    _record("copy_check_rubric_plus_one_copy", prompts)


def test_zz_compare_with_baseline():
    """Runs last in this module (name order). Compares everything measured above."""
    assert MEASURED, "no operation was measured"
    base = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    if os.getenv("COST_GUARD_WRITE") == "1":
        base.update(MEASURED)
        BASELINE.write_text(json.dumps(dict(sorted(base.items())), indent=2) + "\n", encoding="utf-8")
        pytest.skip(f"baseline written: {sorted(MEASURED)}")
    problems = []
    for op, now in MEASURED.items():
        if op not in base:
            problems.append(f"{op}: not in baseline (measured {now}); run with COST_GUARD_WRITE=1")
            continue
        for k in ("calls", "prompt_tokens"):
            if now[k] > base[op][k] * TOLERANCE:
                problems.append(f"{op}.{k}: {now[k]} > {base[op][k]} x {TOLERANCE}")
    assert not problems, "; ".join(problems)
