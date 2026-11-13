"""Benchmark runner: load a fixture + paper (and optional PDF), score five metrics.

Offline only in Phase 0: feed it a saved paper JSON. A later phase can wire
`run_live(config)` to call generate_test through the real backend.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import metrics as M
from .schema import BenchmarkConfig, ExpectedNumericItem, Paper

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@dataclass
class BenchmarkReport:
    config_id: str
    results: list[M.MetricResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r.ok or r.status == "needs_solver" for r in self.results)

    @property
    def hard_fail(self) -> bool:
        return any(r.status == "fail" for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "ok": self.ok,
            "hard_fail": self.hard_fail,
            "metrics": [asdict(r) for r in self.results],
        }


# ── Loading ──────────────────────────────────────────────────────────────

def _load_config(path: Path) -> BenchmarkConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    nums = [ExpectedNumericItem(**item) for item in data.get("expected_numeric_items", [])]
    return BenchmarkConfig(
        id=data["id"],
        description=data.get("description", ""),
        request=data["request"],
        expected_total_questions=data["expected_total_questions"],
        expected_total_marks=data["expected_total_marks"],
        expected_numeric_items=nums,
        golden_paper_file=data.get("golden_paper_file"),
        pdf_file=data.get("pdf_file"),
        key_source=data.get("key_source", "author-computed"),
        notes=data.get("notes", ""),
    )


def list_configs() -> list[BenchmarkConfig]:
    # Golden papers (`*_golden.json`) share the fixtures dir but are not configs.
    paths = [p for p in FIXTURES_DIR.glob("*.json")
             if not p.name.endswith("_golden.json") and p.name != "schema.json"]
    return sorted([_load_config(p) for p in paths], key=lambda c: c.id)


def _load_paper(config: BenchmarkConfig) -> Paper | None:
    if not config.golden_paper_file:
        return None
    path = Path(config.resolve_path(config.golden_paper_file))
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Paper.from_backend_response(data)


# ── Scoring ──────────────────────────────────────────────────────────────

def score(config: BenchmarkConfig, paper: Paper, pdf_path: str | None = None) -> BenchmarkReport:
    results = [
        M.exactly_one_correct_option(paper, config),
        M.numeric_correctness(paper, config),
        M.option_position_balance(paper, config),
        M.semantic_duplicate_count(paper, config),
        M.blueprint_coverage(paper, config),
        M.count_marks_reconcile(paper, config),
    ]
    if pdf_path:
        results.append(M.pdf_render_integrity(paper, pdf_path))
    else:
        results.append(M.MetricResult(
            "pdf_render_integrity", "skipped",
            "no pdf_file declared on fixture",
        ))
    return BenchmarkReport(config_id=config.id, results=results)


def run_all() -> list[BenchmarkReport]:
    """Score every fixture that ships with a golden paper. Fixtures without a
    golden paper are reported as `skipped` (they just verify the fixture loads).
    """
    reports: list[BenchmarkReport] = []
    for cfg in list_configs():
        paper = _load_paper(cfg)
        if paper is None:
            # Fixture is in flight (no paper generated yet). Record one "skipped"
            # report so the harness output is still a complete table.
            reports.append(BenchmarkReport(
                config_id=cfg.id,
                results=[M.MetricResult(
                    "harness", "skipped",
                    f"fixture {cfg.id} has no golden_paper_file yet; "
                    f"key_source={cfg.key_source}",
                )],
            ))
            continue
        pdf_path = None
        if cfg.pdf_file:
            p = Path(cfg.resolve_path(cfg.pdf_file))
            pdf_path = str(p) if p.exists() else None
        reports.append(score(cfg, paper, pdf_path))
    return reports


def format_report(reports: list[BenchmarkReport]) -> str:
    lines = []
    for r in reports:
        lines.append(f"[{r.config_id}] ok={r.ok} hard_fail={r.hard_fail}")
        for m in r.results:
            lines.append(f"  - {m.metric:30s} {m.status:14s} {m.summary}")
            for f in m.failures[:5]:
                lines.append(f"      · {f}")
            if len(m.failures) > 5:
                lines.append(f"      · ... ({len(m.failures) - 5} more)")
    return "\n".join(lines)
