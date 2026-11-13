"""Phase 0 acceptance test: the harness runs in CI and each metric fires.

Scores five hand-written synthetic papers (one clean, four each triggering one
specific metric) and asserts the expected per-metric status. Then runs
`runner.run_all()` across the real fixtures to confirm they load.

No LLM calls, no network. Deterministic.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.benchmark import metrics as M
from tests.benchmark import runner
from tests.benchmark.schema import BenchmarkConfig, ExpectedNumericItem, Paper

SYNTH = Path(__file__).parent / "synthetic"


def _load_synth_paper(name: str) -> Paper:
    data = json.loads((SYNTH / name).read_text(encoding="utf-8"))
    return Paper.from_backend_response(data)


# Hand-written config used to score the synthetic papers (matches clean.json).
def _synth_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        id="synth",
        description="synthetic",
        request={},
        expected_total_questions=3,
        expected_total_marks=3,
        expected_numeric_items=[
            ExpectedNumericItem(qid="ioc_1", description="5e5 * 10%", expected_answer=50000.0,
                                tolerance=1.0, key_source="author-computed"),
            ExpectedNumericItem(qid="ioc_2", description="2e5 * 6%", expected_answer=12000.0,
                                tolerance=1.0, key_source="author-computed"),
            ExpectedNumericItem(qid="ioc_3", description="8e5 * 12%", expected_answer=96000.0,
                                tolerance=1.0, key_source="author-computed"),
        ],
    )


# ── Each metric fires on the right synthetic paper ────────────────────────

def test_clean_paper_passes_every_metric():
    paper = _load_synth_paper("clean.json")
    cfg = _synth_config()
    report = runner.score(cfg, paper, pdf_path=None)
    # pdf skipped (we did not supply a PDF); every other metric passes.
    statuses = {r.metric: r.status for r in report.results}
    assert statuses["exactly_one_correct_option"] == "pass", statuses
    assert statuses["numeric_correctness"] == "pass", statuses
    assert statuses["semantic_duplicate_count"] == "pass", statuses
    assert statuses["count_marks_reconcile"] == "pass", statuses
    assert statuses["pdf_render_integrity"] == "skipped", statuses


def test_wrong_numeric_fails_numeric_metric():
    paper = _load_synth_paper("wrong_numeric.json")
    result = M.numeric_correctness(paper, _synth_config())
    assert result.status == "fail"
    failed_qids = {f["qid"] for f in result.failures}
    assert "ioc_1" in failed_qids
    # The other two should still pass.
    assert {"ioc_2", "ioc_3"} & failed_qids == set()


def test_no_match_option_fails_one_correct_metric():
    paper = _load_synth_paper("no_match_option.json")
    result = M.exactly_one_correct_option(paper, _synth_config())
    assert result.status == "fail"
    assert any(f["qid"] == "ioc_1" and f["reason"] == "no_match" for f in result.failures)


def test_duplicate_pair_fails_dedup():
    paper = _load_synth_paper("duplicate_pair.json")
    result = M.semantic_duplicate_count(paper, _synth_config())
    assert result.status == "fail"
    # The production gate removes hard duplicates (exact repeat / lexical near-identity).
    kinds = {f["kind"] for f in result.failures}
    assert kinds & {"exact", "lexical"}, result.failures


def test_count_wrong_fails_reconcile():
    paper = _load_synth_paper("count_wrong.json")
    result = M.count_marks_reconcile(paper, _synth_config())
    assert result.status == "fail"
    kinds = {f["kind"] for f in result.failures}
    assert "question_count" in kinds


# ── Pending-SME fixture is reported, not errored ─────────────────────────

def test_target_fixture_reports_needs_solver():
    """The Fundamentals of Partnership 50-MCQ Hard fixture has `pending_sme`
    numeric items only. It must surface as `needs_solver`, not silently pass."""
    cfg = next(c for c in runner.list_configs()
               if c.id == "acc12_fundamentals_50mcq_hard")
    # Score with an empty paper — numeric_correctness should still report
    # needs_solver because the config declares pending items.
    empty = Paper(questions=[], total_marks=0, subject="Accountancy",
                  class_grade="12", chapters=["Fundamentals of Partnership"])
    result = M.numeric_correctness(empty, cfg)
    assert result.status == "needs_solver"
    assert result.stats["pending"] >= 1


# ── Full harness smoke test ──────────────────────────────────────────────

def test_run_all_loads_every_fixture():
    reports = runner.run_all()
    ids = {r.config_id for r in reports}
    assert "acc12_fundamentals_50mcq_hard" in ids
    assert "acc12_fundamentals_rotation_50" in ids
    assert "acc12_interest_on_capital_mini" in ids
    # >= 15 fixtures total (the per-family coverage + minis + target configs).
    assert len(reports) >= 15, sorted(ids)


# ── Phase 1B acceptance gate ─────────────────────────────────────────────
# Phase 1B asserts: numeric (no wrong answers), exactly-one-correct, and A/B/C/D
# balance. Semantic dedup + PDF integrity are reported but become blocking gates
# in Phase 2 / Phase 3 respectively (the lexical dedup proxy over-flags the fixed
# question scaffolding; Phase 2 replaces it with embeddings + blueprint rotation).

PHASE1B_GATED = {"exactly_one_correct_option", "numeric_correctness",
                 "option_position_balance", "count_marks_reconcile"}


def test_phase1b_acceptance_on_every_golden_fixture():
    reports = runner.run_all()
    scored = [r for r in reports
              if any(m.metric != "harness" for m in r.results)]
    assert scored, "no fixture has a golden paper"
    for r in scored:
        for m in r.results:
            if m.metric not in PHASE1B_GATED:
                continue
            # numeric may be needs_solver (pending SME); never 'fail'.
            if m.metric == "numeric_correctness":
                assert m.status in ("pass", "needs_solver", "skipped"), \
                    f"{r.config_id}: {m.metric} = {m.status}\n{m.failures}"
            else:
                assert m.status in ("pass", "skipped"), \
                    f"{r.config_id}: {m.metric} = {m.status}\n{m.failures[:5]}"


def test_phase1b_zero_wrong_numeric_on_hand_verified():
    """The author-computed mini fixtures carry independent hand literals; their
    numeric_correctness MUST pass (0 wrong numeric answers)."""
    reports = {r.config_id: r for r in runner.run_all()}
    for cid in ("acc12_interest_on_capital_mini", "acc12_pl_appropriation_mini",
                "acc12_interest_on_drawings_mini"):
        r = reports[cid]
        num = next(m for m in r.results if m.metric == "numeric_correctness")
        assert num.status == "pass", f"{cid}: {num.summary}\n{num.failures}"


def test_rotation_paper_balance_within_one():
    r = next(r for r in runner.run_all() if r.config_id == "acc12_fundamentals_rotation_50")
    bal = next(m for m in r.results if m.metric == "option_position_balance")
    assert bal.status == "pass", bal.summary


def test_rotation_paper_has_no_duplicates():
    """Phase 2B: the realistic 50-MCQ Fundamentals paper must pass the dedup gate
    (the 'same question 3 times' complaint)."""
    r = next(r for r in runner.run_all() if r.config_id == "acc12_fundamentals_rotation_50")
    dup = next(m for m in r.results if m.metric == "semantic_duplicate_count")
    assert dup.status == "pass", dup.summary


def test_rotation_paper_coverage_within_bounds():
    """Phase 2B: the blueprint-planned paper satisfies the coverage gate."""
    r = next(r for r in runner.run_all() if r.config_id == "acc12_fundamentals_rotation_50")
    cov = next(m for m in r.results if m.metric == "blueprint_coverage")
    assert cov.status == "pass", cov.summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
