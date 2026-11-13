# Benchmark harness — Phase 0

Correctness floor for every change to the test-generator. If a PR regresses any
metric on these fixtures, CI fails.

## What it checks (five metrics)

| metric | needs solver? | Phase 0 status |
|---|---|---|
| `exactly_one_correct_option` | no | wired |
| `numeric_correctness` | yes (Phase 1A) | wired; returns `needs_solver` for `pending_sme` fixtures |
| `semantic_duplicate_count` | no | wired (lexical + numeric-pattern; Phase 2 adds embeddings) |
| `count_marks_reconcile` | no | wired |
| `pdf_render_integrity` | no (PyMuPDF already in requirements) | wired |

Statuses: `pass` · `fail` · `needs_solver` · `skipped` · `error`.

## Run it

```bash
cd test-generator-backend
pytest tests/benchmark -v
```

Or score a saved paper manually:

```python
from tests.benchmark import runner
reports = runner.run_all()
print(runner.format_report(reports))
```

## Add a fixture

1. Drop `tests/benchmark/fixtures/NN_<short_id>.json` matching `schema.BenchmarkConfig`.
2. Set `key_source`:
   - `author-computed` — fixture author can compute every expected answer by hand
     (arithmetic simple enough to verify by inspection).
   - `sme-verified` — an Accountancy SME has signed off on every expected answer.
   - `pending_sme` — expected answers not yet known; `numeric_correctness` will
     report `needs_solver` until the solver (Phase 1A) and SME sign-off land.
3. Once a generated paper is saved as `fixtures/<id>_golden.json` and linked via
   `golden_paper_file`, the runner starts scoring it. Optionally attach a
   pre-rendered PDF via `pdf_file` to enable `pdf_render_integrity`.

## Fixture inventory (Phase 0 seed)

| id | key_source | reason |
|---|---|---|
| `acc12_fundamentals_50mcq_hard` | `pending_sme` | the paper that shipped D1–D6; items TBD once Phase 1A solver lands |
| `acc12_interest_on_capital_mini` | `author-computed` | 3 single-step items, trivial by inspection |
| `acc12_pl_appropriation_mini` | `author-computed` | 3 fixed-ratio splits |
| `acc12_interest_on_drawings_mini` | `author-computed` | 3 average-period cases (6.5 / 5.5 / 6 months) |

Phase 1A target: ≥15 fixtures across the numerical families listed in the audit
(interest on capital, interest on drawings monthly/quarterly/half-yearly, average
period, product method, commission before/after, P&L appropriation, guarantee +
deficiency, past adjustments, opening/closing capital, inadequate profit).

## Why synthetic papers

`synthetic/*.json` is a deliberate test input, not a service output. It proves
each metric *detects* its failure class (D1, D3, count mismatch, etc.) without
needing an LLM call. The real-fixture reports start passing once Phase 1 lands
and we save a generated paper per config.

## Design notes

- `Paper` is a tiny normalised view. The runner never imports the service's
  pydantic models — fixtures can be scored post-hoc against any exported paper.
- The runner never calls an LLM. A later phase may add `run_live(config)` that
  invokes `generate_test` directly, behind an `A4_BENCH_LIVE=1` env flag; Phase 0
  is offline-only.
- `semantic_duplicate_count` uses `SequenceMatcher` + a numeric-pattern key, not
  embeddings. Deliberate: the metric is a pre-commit gate, not the production
  dedup. Phase 2 swaps in embeddings for the production path and may raise this
  threshold.
- `pdf_render_integrity` uses PyMuPDF (`fitz`). The repo already pins it, so no
  new dep.
