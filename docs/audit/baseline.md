# Checker audit — Phase 0.5 accuracy baseline

Date: 2026-10-26. Status: **provisional. The gate the brief wants cannot be set yet; reasons below.**

## What exists

| Set | Copies | Real handwriting? | Teacher marks | Last run |
|---|---|---|---|---|
| `Downloads/golden_set.json` (Trigo, Class 10, 40 marks) | 3 (Rohan, Priya, Aman) | No — generated sheets | per question (Section A summed) + notes on 2 expected review flags | `scripts/eval_runs/checker_v3_20261017-155740.json` |
| `scripts/eval_cases/smart_checker/science10_rahul` | 1 | No ("typed in a handwriting font") | per question | `smart_checker_20261010_013520.json` |
| Live: 23 Class XII Economics copies | 23 | Yes | **None.** All 23 were published at the AI's marks; no overrides | — |

The "30 copies" set and the 10–12 mark gap from the brief aren't in any file I can reach. [EVIDENCE: search of Downloads, Desktop, both repos]

## Measured baseline (17 Oct run, v3 single-call grader, gemini-3.8-flash, medium)

Computed from `checker_v3_20261017-155740.json`; nothing re-run today.

| Metric | Value |
|---|---|
| Question cells | 36 (3 copies × 12 keys) |
| Exact match | 34 / 36 |
| Within ±1 | 36 / 36 |
| MAE per cell | 0.042 marks; signed bias +0.042 (both misses were AI too lenient) |
| Paper-total MAE | 0.5 / 40 (Rohan +0.5, Priya 0, Aman +1.0) |
| Misses | Rohan Q19 2 → 2.5; Aman Q18 2 → 3 |
| Expected review flags raised | **0 / 2** (Aman Q12 written under label "Q.13"; Q16 unlabelled — neither flagged) |
| Cost / time | ₹3.15–4.68 per copy, 58–87 s wall |
| Partial-upload / arithmetic errors | n/a (none in set; arithmetic is done in code) |
| Inter-rater floor | not measurable (one marker, no second pass) |

## Why this can't serve as the Phase 3 gate

1. **Too small.** 3 copies cannot support a dev / calibration / held-out split (brief §8). Any split gives a held-out set of 1 copy.
2. **Not representative.** These are clean generated sheets. Live copies are phone photos of real handwriting, 1–9 pages, Class XII Economics.
3. **Stale.** Since 17 Oct the graded path changed: G1 preprocess (Stage A pages, rotation, page meta in the same call), size-driven downscaling, per-call timeouts (commits 5415b4b … 711a501). The prompt now also asks for page metadata. Memory of 17 Oct: the page_index prompt change alone shifted borderline verdicts by ±0.5.
4. **Production uses a different model mix.** 10 of 23 live copies were graded by `gemini-3-flash-preview` (fallback). The golden run used 3.8-flash only.

## Proposed target (an assumption until real data exists)

`[ASSUMPTION]` Per-paper total MAE ≤ 1.5 marks on a 30-mark paper on the held-out set, no copy off by > 3, with the inter-rater floor reported next to it. This is to be replaced by "≤ IRR floor + 0.5" once a second teacher pass exists.

## What's needed to make the gate real (cheapest first)

1. **Teacher marks for the 23 live Economics copies.** Assumption: the teacher has the physical copies or the marked PDFs. Per-question marks in `golden_teacher_template.csv` format, with the MAX row filled from the real rubric. This gives a real-handwriting set, and its AI marks are already stored.
2. A blind second pass on 5 of them (same or another teacher) → IRR floor.
3. Re-run both existing sets on today's `main` with the model **pinned** (fallback disabled) to separate the model effect from the pipeline effect. About ₹15 and 5 min on the 3 Trigo copies.
4. Only then split dev / calibration / held-out by student, and record `eval/run0/` with provenance (git sha, rubric hash, model, prompt hash).
