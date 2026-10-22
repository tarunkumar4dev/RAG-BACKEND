# G4.5 — quality pass applied to production, Class 10 Mathematics only (22 Oct 2026)

- Migration 19 applied to `public.ncert_questions` (5.3 s wall incl. table rewrite for `quality_rank`); the md5 of
  every original column was unchanged since the G4.2 backup right before it.
- Input check: a fresh dump of the 10,793 production rows was identical to the dry-run input; the 52 emitted
  statements were identical to the dry run (only the table names differ).
- BEGIN / EXPLAIN ANALYZE / ROLLBACK per statement: 1,053 ms total database time, slowest statement 46 ms; nothing kept.
- Applied in 52 transactions (one statement each — the Management API rejects bigger bodies).

| | dry run (copy) | production |
|---|---:|---:|
| clean | 470 | 470 |
| fixed | 93 | 93 |
| needs_review | 1,094 | 1,094 |
| duplicate | 9,136 | 9,136 |
| junk | 0 | 0 |
| visible | 1,657 | 1,657 |
| fix log rows | 2,775 | 2,775 |
| duplicates pointing at a hidden row | 0 | 0 |
| rows audited outside Class 10 Maths | — | 0 |

## G5 criterion (query: `g5_check_class10_maths.sql`)

"Every G1a-clean row is either visible, or its duplicate_of points to a visible row":
2,339 G1a-clean rows = 1,446 visible + 893 hidden duplicates with a visible canonical row · **violations: 0**.

## needs_review by reason (1,094 rows; a row can have several)

near_duplicate 917 · garbled_math 148 · short_text 135 · mcq_no_answer 70 · skeleton_operator_conflict 35 ·
diagram_type_no_image 20 · raw_latex 3 · mcq_type_no_options 1. One reason only: 891 rows (near_duplicate 739).

## Rule 6 (granularity) — evidence, not applied

See `rule6_samples_class10_maths.md`: 40 questions are visible as split sub-part rows next to a stem-only or merged
row. Splits contain the stem in all rows for 2, in none for 9, mixed for 29 — two extraction runs split the same
question differently (one repeats the stem in every sub-part, the other does not). These rows stay needs_review.
