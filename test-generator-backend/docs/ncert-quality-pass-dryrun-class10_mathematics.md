# NCERT quality pass — dry run, Class 10 Mathematics (G4.4)

- **Repo / branch:** RAG-BACKEND `fix/ncert-builder-quality` · `scripts/ncert_quality_pass.py`
- **Input:** JSON dump · 10,793 rows · hash `v2` · seed 20261022
- **Target of the emitted SQL:** `a4_backup_20261022.ncert_q_c10m_copy` (+ `a4_backup_20261022.ncert_q_c10m_copy_fixes`) — production rows untouched.

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

## Statuses

| status | rows | % | visible to teachers |
|---|---:|---:|---|
| `clean` | 470 | 4.4% | yes |
| `fixed` | 93 | 0.9% | yes |
| `needs_review` | 1,094 | 10.1% | yes |
| `duplicate` | 9,136 | 84.6% | no |
| `junk` | 0 | 0.0% | no |

Duplicates: 3,972 exact content_hash · 5,164 same skeleton (rule 5). Skeleton groups held back by the operator guard: 11 (35 rows → needs_review).

needs_review reasons (a row can have several): `near_duplicate` 917, `garbled_math` 148, `short_text` 135, `mcq_no_answer` 70, `skeleton_operator_conflict` 35, `diagram_type_no_image` 20, `raw_latex` 3, `mcq_type_no_options` 1.

Deterministic fixes: 2,247 rows changed — `symbol_font_map` 2,072, `figure_ref_from_image` 702, `symbol_font_map+trim_whitespace` 1. Every change is in `docs/audit/quality_pass_dryrun_fixes.csv` (and `ncert_question_fixes` when applied). Not done: *balance a single `$`* — every `$` in the table is a currency sign.

## Visible questions per chapter

| chapter | before | after | Δ | clean | fixed | needs_review | of which near-dup copies |
|---|---:|---:|---:|---:|---:|---:|---:|
| Areas Related to Circles | 676 | 20 | -656 | 6 | 7 | 7 | 2 |
| Arithmetic Progressions | 2,068 | 243 | -1,825 | 103 | 1 | 139 | 57 |
| Circles | 866 | 93 | -773 | 29 | 5 | 59 | 32 |
| Coordinate Geometry | 918 | 91 | -827 | 24 | 4 | 63 | 37 |
| Introduction to Trigonometry | 1,381 | 318 | -1,063 | 60 | 29 | 229 | 136 |
| Pair of Linear Equations in Two Variables | 895 | 227 | -668 | 50 | 2 | 175 | 113 |
| Polynomials | 271 | 40 | -231 | 8 | 0 | 32 | 22 |
| Probability | 733 | 78 | -655 | 51 | 6 | 21 | 8 |
| Quadratic Equations | 639 | 166 | -473 | 23 | 0 | 143 | 97 |
| Real Numbers | 373 | 63 | -310 | 21 | 1 | 41 | 30 |
| Some Applications of Trigonometry | 428 | 32 | -396 | 27 | 4 | 1 | 0 |
| Statistics | 593 | 101 | -492 | 26 | 0 | 75 | 44 |
| Surface Areas and Volumes | 359 | 48 | -311 | 10 | 4 | 34 | 17 |
| Triangles | 593 | 137 | -456 | 32 | 30 | 75 | 32 |
| **total** | **10,793** | **1,657** | **-9,136** | | | | **627** |

### Sanity check against the NCERT book

Like for like: visible rows **with near-duplicate copies collapsed** (one row per near-dup cluster), split into solved examples (`Example N`) and exercise-type rows (exercise / mcq / fillblank / diagram that is not an example). `NCERT` = counted in the chapter PDF: distinct `Example N :` headings and numbered questions under `EXERCISE x.y` (sub-parts (i), (ii) count once — the table often stores them as separate rows, so exercise ratios above 1 are expected). In-text questions and activities are not in either column. ⚠️ = ratio outside 0.75–2.0. `Ex N >1` = Example numbers still shown more than once after collapsing.

| chapter | examples visible / NCERT | exercise rows visible / NCERT questions | Ex N >1 |
|---|---:|---:|---:|
| Areas Related to Circles | 2 / 2 = 1.00 | 15 / 14 = 1.07 | 0 |
| Arithmetic Progressions | 26 / 16 = 1.62 | 125 / 49 = 2.55 ⚠️ | 3 |
| Circles | 5 / 3 = 1.67 | 40 / 17 = 2.35 ⚠️ | 2 |
| Coordinate Geometry | 10 / 10 = 1.00 | 27 / 20 = 1.35 | 0 |
| Introduction to Trigonometry | 16 / 12 = 1.33 | 132 / 19 = 6.95 ⚠️ | 2 |
| Pair of Linear Equations in Two Variables | 11 / 10 = 1.10 | 86 / 12 = 7.17 ⚠️ | 0 |
| Polynomials | 4 / 4 = 1.00 | 5 / 3 = 1.67 | 0 |
| Probability | 16 / 11 = 1.45 | 29 / 19 = 1.53 | 2 |
| Quadratic Equations | 19 / 9 = 2.11 ⚠️ | 47 / 13 = 3.62 ⚠️ | 4 |
| Real Numbers | 7 / 7 = 1.00 | 15 / 10 = 1.50 | 0 |
| Some Applications of Trigonometry | 8 / 7 = 1.14 | 17 / 15 = 1.13 | 0 |
| Statistics | 12 / 8 = 1.50 | 28 / 22 = 1.27 | 4 |
| Surface Areas and Volumes | 7 / 7 = 1.00 | 18 / 9 = 2.00 | 0 |
| Triangles | 8 / 8 = 1.00 | 57 / 29 = 1.97 | 0 |

## G5 pre-check — rows that were clean in G1a and are now hidden

**893** G1a-clean rows become hidden (same_skeleton 861, exact_hash 32). Every one is a `duplicate` whose canonical row stays visible: **yes**. G1a had no near-duplicate / skeleton rule, so its "clean" counted these copies as separate questions; the G5 criterion should read "no G1a-clean *question* disappears" (its canonical copy is visible).

## 20 sample clusters — which copy was kept and why

**1. Quadratic Equations** — same_skeleton · 3 rows · kept **#45885** (2 math symbols, lowest id)

- #45885 (kept, 2 sym): Check whether the following is a quadratic equation : ( x + 2)³ = 2x (x² – 1)
- #69360 (dup of #45885, 2 sym): Check whether the following is a quadratic equation: (x + 2)³ = 2x(x² - 1)
- #76420 (dup of #45885, 2 sym): Check whether the following is a quadratic equation: (x + 2)³ = 2x(x² – 1)

**2. Polynomials** — same_skeleton · 4 rows · kept **#45317** (0 math symbols, lowest id)

- #45317 (kept, 0 sym): Find the zeroes of the quadratic polynomial 4u2 + 8u and verify the relationship between the zeroes and the coefficients.
- #75868 (dup of #45317, 0 sym): Find the zeroes of the quadratic polynomial 4 u2 + 8u and verify the relationship between the zeroes and the coefficients.

**3. Introduction to Trigonometry** — same_skeleton · 3 rows · kept **#52495** (1 math symbols, lowest id)

- #52495 (kept, 1 sym): Prove the following identity, where the angles involved are acute angles for which the expressions are defined: (1 + sec A) / sec A = sin² A / (1 – c…
- #70104 (dup of #52495, 1 sym): Prove the following identity, where the angles involved are acute angles for which the expressions are defined: (1 + sec A) / sec A = sin² A / (1 – c…
- #77101 (dup of #52495, 1 sym): Prove the following identity, where the angles involved are acute angles for which the expressions are defined. (1 + sec A) / sec A = sin² A / (1 – c…

**4. Arithmetic Progressions** — same_skeleton · 11 rows · kept **#47100** (1 math symbols, vs 0 in the next best copy)

- #47100 (kept, 1 sym): If the sum of the first n terms of an AP is 4 n – n^2, what is the first term (that is S1)? What is the sum of first two terms? What is the second te…
- #1931 (dup of #47100, 0 sym): If the sum of the first n terms of an AP is 4 n – n2, what is the first term (that is S1)? What is the sum of first two terms? What is the second ter…
- #2403 (dup of #47100, 0 sym): If the sum of the first n terms of an AP is 4 n – n2, what is the first term (that is S1)? What is the sum of first two terms? What is the second ter…

**5. Pair of Linear Equations in Two Variables** — same_skeleton · 4 rows · kept **#4104** (0 math symbols, lowest id)

- #4104 (kept, 0 sym): Form the pair of linear equations in the following problems, and find their solutions(if they exist) by the elimination method : If we add 1 to the n…
- #4146 (dup of #4104, 0 sym): Form the pair of linear equations in the following problems, and find their solutions(if they exist) by the elimination method : If we add 1 to the n…
- #7446 (dup of #4104, 0 sym): Form the pair of linear equations in the following problems, and find their solutions(if they exist) by the elimination method : If we add 1 to the n…

**6. Pair of Linear Equations in Two Variables** — same_skeleton · 10 rows · kept **#4088** (4 math symbols, lowest id)

- #4088 (kept, 4 sym): Solve the following pair of linear equations by the substitution method. √2 x + √3 y = 0 √3 x – √8 y = 0
- #4130 (dup of #4088, 4 sym): Solve the following pair of linear equations by the substitution method: √2 x + √3 y = 0, √3 x – √8 y = 0
- #5505 (dup of #4088, 4 sym): Solve the following pair of linear equations by the substitution method: √2x + √3y = 0, √3x - √8y = 0
- … 6 more distinct texts

**7. Pair of Linear Equations in Two Variables** — same_skeleton · 4 rows · kept **#4094** (0 math symbols, lowest id)

- #4094 (kept, 0 sym): Form the pair of linear equations for the following problems and find their solution by substitution method. The taxi charges in a city consist of a …
- #4136 (dup of #4094, 0 sym): Form the pair of linear equations for the following problems and find their solution by substitution method: The taxi charges in a city consist of a …
- #7436 (dup of #4094, 0 sym): Form the pair of linear equations for the following problems and find their solution by substitution method: The taxi charges in a city consist of a …
- … 1 more distinct texts

**8. Areas Related to Circles** — same_skeleton · 44 rows · kept **#1349** (3 math symbols, lowest id)

- #1349 (kept, 3 sym): A chord of a circle of radius 15 cm subtends an angle of 60° at the centre. Find the areas of the corresponding minor and major segments of the circl…
- #25174 (dup of #1349, 3 sym): A chord of a circle of radius 15 cm subtends an angle of 60° at the centre. Find the areas of the corresponding minor and majorsegments of the circle…
- #1363 (dup of #1349, 2 sym): A chord of a circle of radius 15 cm subtends an angle of 60° at the centre. Find the areas of the corresponding minor and majorsegments of the circle…
- … 1 more distinct texts

**9. Introduction to Trigonometry** — same_skeleton · 5 rows · kept **#2574** (1 math symbols, lowest id)

- #2574 (kept, 1 sym): State whether the following are true or false. Justify your answer. cot A is not defined for A = 0°.
- #10695 (dup of #2574, 1 sym): State whether the following are true or false. Justify your answer : cot A is not defined for A = 0°.

**10. Arithmetic Progressions** — same_skeleton · 5 rows · kept **#3846** (0 math symbols, lowest id)

- #3846 (kept, 0 sym): A ladder has rungs 25 cm apart (see Fig. 5.7). The rungs decrease uniformly in length from 45 cm at the bottom to 25 cm at the top. If the top and th…
- #6409 (dup of #3846, 0 sym): A ladder has rungs 25 cm apart.(see Fig. 5.7). The rungs decrease uniformly in length from 45 cm at the bottom to 25 cm at the top. If the top and th…
- #47112 (dup of #3846, 0 sym): A ladder has rungs 25 cm apart (see Fig. 5.7). The rungs decrease uniformly in length from 45 cm at the bottom to 25 cm at the top. If the top and th…

**11. Pair of Linear Equations in Two Variables** — same_skeleton · 18 rows · kept **#46275** (1 math symbols, lowest id)

- #46275 (kept, 1 sym, answer): The ratio of incomes of two persons is 9 : 7 and the ratio of their expenditures is 4 : 3. If each of them manages to save ₹ 2000 per month, find the…
- #2676 (dup of #46275, 0 sym, answer): The ratio of incomes of two persons is 9 : 7 and the ratio of their expenditures is 4 : 3. If each of them manages to save ` 2000 per month, find the…
- #12987 (dup of #46275, 0 sym, answer): Example 8 : The ratio of incomes of two persons is 9 : 7 and the ratio of their expenditures is 4 : 3. If each of them manages to save ` 2000 per mon…

**12. Pair of Linear Equations in Two Variables** — same_skeleton · 3 rows · kept **#2668** (0 math symbols, lowest id)

- #2668 (kept, 0 sym): Solve the following pair of linear equations by the substitution method. (vi) 3x/2 – 5y/3 = -2 x/3 + y/2 = 13/6
- #7390 (dup of #2668, 0 sym): Solve the following pair of linear equations by the substitution method. (vi) 3x/2 – 5y/3 = -2 x/3 + y/2 = 13/6
- #69691 (dup of #2668, 0 sym): Solve the following pair of linear equations by the substitution method: (vi) (3x/2) – (5y/3) = –2, (x/3) + (y/2) = 13/6

**13. Pair of Linear Equations in Two Variables** — exact_hash · 2 rows · kept **#22365** (0 math symbols, lowest id)

- #22365 (kept, 0 sym): 2 x + y – 6 = 0, 4 x – 2y – 4 = 0

**14. Coordinate Geometry** — exact_hash · 27 rows · kept **#1538** (0 math symbols, lowest id)

- #1538 (kept, 0 sym, answer): Find a point on the y-axis which is equidistant from the points A(6, 5) and B(– 4, 3).

**15. Introduction to Trigonometry** — exact_hash · 6 rows · kept **#2647** (0 math symbols, lowest id)

- #2647 (kept, 0 sym): 22 21 tan A 1 tan A 1 – cot A 1 + cot A    + −=        = tan2 A
- #9569 (dup of #2647, 0 sym): (x)22 21 tan A 1 tan A 1 – cot A 1 + cot A    + −=        = tan2 A

**16. Introduction to Trigonometry** — exact_hash · 24 rows · kept **#2544** (0 math symbols, lowest id)

- #2544 (kept, 0 sym, answer): In a right triangle ABC, right-angled at B, if tan A = 1, then verify that 2 sin A cos A = 1.

**17. Surface Areas and Volumes** — exact_hash · 2 rows · kept **#4536** (0 math symbols, lowest id)

- #4536 (kept, 0 sym): 4.A pen stand made of wood is in the shape of acuboid with four conical depressions to hold pens.The dimensions of the cuboid are 15 cm by 10 cm by3.…

**18. Coordinate Geometry** — exact_hash · 5 rows · kept **#1550** (0 math symbols, lowest id)

- #1550 (kept, 0 sym): 7.Find the point on the x-axis which is equidistant from (2, –5) and (–2, 9).

**19. Introduction to Trigonometry** — exact_hash · 3 rows · kept **#16111** (3 math symbols, lowest id)

- #16111 (kept, 3 sym): Now Δ ABD ≅Δ ACD (Why?)

**20. Real Numbers** — exact_hash · 14 rows · kept **#2910** (0 math symbols, lowest id)

- #2910 (kept, 0 sym): Express each number as a product of its prime factors: (i) 140 (ii) 156 (iii) 3825 (iv) 5005 (v) 7429
- #2929 (dup of #2910, 0 sym): Express each number as a product of its prime factors: (i) 140 (ii) 156 (iii) 3825 (iv) 5005 (v) 7429
- #4343 (dup of #2910, 0 sym): Express each number as a product of its prime factors: (i) 140 (ii) 156 (iii) 3825 (iv) 5005 (v) 7429
