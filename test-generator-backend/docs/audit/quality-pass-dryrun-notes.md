## Verified on the copy table (not on production)

The emitted SQL (52 statements, one transaction each — one big file is rejected by the Management API with 413)
was run against `a4_backup_20261022.ncert_q_c10m_copy` (the 10,793 Class 10 Maths rows + migration 19). Read back
from the copy: clean 470 · fixed 93 · needs_review 1,094 · duplicate 9,136 · junk 0 — **identical to this report**;
1,657 visible rows; 2,775 rows in the fix log (symbol map: 1,430 texts, 613 answers, 29 option lists;
figure_ref: 702); **0 duplicates point at a hidden row**; every row has a `v2` content_hash. Symbol-font glyphs
left after the map: 15 rows (bracket pieces / unknown glyphs), the 7 visible ones are `needs_review`.
Wall time 219 s is Management-API round trips (≈ 4 s per call), not database time — G4.5 measures real timing
with `--apply --rollback` (one transaction, rolled back).

## Choices made beyond the brief (please confirm)

1. **Operator guard on rule 5.** A same-skeleton group merges only if every ungarbled copy has the same operator
   sequence (+ − = < > ≤ ≥ × ÷), and the skeleton has ≥ 20 characters. "x + y = 5" and "x − y = 5" have the same
   skeleton but are different questions. Held back here: 11 groups, 35 rows → needs_review.
2. **"Most math symbols" counts ₹ and `^` too**, so `₹ 2000` beats the backtick copy and `n^2` beats `n2`.
3. **Not done: "balance a single `$`"** — every `$` in the table is a currency sign; balancing would turn prose into math.
4. **`options` keep their current encoding** (a JSON string inside jsonb) when the symbol map changes them —
   fixing the double encoding is a separate change (root cause: the extractor calls `json.dumps()` before
   supabase-py encodes again).
5. **needs_review is visible** (only duplicate / junk are hidden), as G4.6 says.

## Sanity check: what is still too high, and why

Examples are close to the book (1.0–1.6 × in 12 of 14 chapters). Exercise rows are 2–7 × the book's numbered
questions in Arithmetic Progressions, Circles, Introduction to Trigonometry, Pair of Linear Equations and Quadratic
Equations. Checked by hand in Introduction to Trigonometry: the **same NCERT question is stored at three
granularities** — the whole question with all sub-parts in one row ("Evaluate the following : (i) … (ii) …"), the
stem alone ("Evaluate the following :"), and each sub-part as its own row ("1. (i)", "1. (ii)") — plus notation
variants (`sin^3 θ` / `sin³ θ` / `sin theta`, with and without "Prove the identity:") that stay below the 0.9
similarity line. None of these is a deterministic duplicate.

**Decision needed before G4.5 (proposed rule 6, not implemented):** per chapter + question number, if the split
sub-part rows exist, hide the stem-only row (junk) and mark the merged whole-question row `duplicate` of the
first sub-part — or the other way round (keep merged, hide the splits). Teachers pick sub-parts in the builder, so
keeping the splits is my recommendation.

## G5 criterion

893 rows that G1a called clean are now hidden (861 same skeleton, 32 exact hash). All 893 are duplicates whose
canonical copy stays visible. G1a had no skeleton/near-duplicate rule, so the G5 line should read "no G1a-clean
*question* disappears — its canonical copy is visible", not "no G1a-clean row is hidden".
