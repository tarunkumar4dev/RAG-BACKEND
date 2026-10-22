# NCERT audit — Class 10 Mathematics (G1a)

- **Repo / branch:** RAG-BACKEND `fix/ncert-builder-quality` (from `main` 1551f72) — not committed
- **Run:** 2026-10-21 19:56 UTC · DB `now()` at dump: 2026-10-21 19:46:15+00:00
- **Source:** JSON dump of `supabase db query --linked` (SELECT only, no writes)
- **Rows:** Class 10 Mathematics 10,793 · full table 91,906
- **Newest row:** Class 10 Maths 2026-09-19 11:42:08.461631+00 · whole table 2026-09-19 11:42:08.878441+00
- **Caveat — rows inserted in the 24 h before the run:** 0 rows — none, the data did not change around the audit
- **Normalization:** pinned G0 definition (`app/services/ncert_text_hash.py`, 10/10 fixtures pass) — NFC, lowercase, strip `$ \( \) \[ \]`, collapse whitespace, strip leading `Q1.` / `1)` / `(i)` / `(a)`, strip trailing punctuation. `Example 1` is **not** stripped (it is part of the text).
- **Not in G1a (→ G1b):** near-duplicates (> 0.9), KaTeX parse, truncated/merged rows, image HEAD checks, garbled-math cost estimate.

## Key findings

1. **76% of Class 10 Maths rows are exact copies inside their own chapter** (8,181 of 10,793). Only 2,612 distinct
   questions. Every chapter is affected (Areas Related to Circles: 676 rows → 51 questions).
2. **"Example 1" × 9–10 has two causes, both confirmed:**
   - *Retry duplication* — every chapter stores its own Example 1 11–38 times (identical copies, re-extracted
     variants with/without the `Example 1 :` prefix, the same example split into `(i)`/`(ii)` rows in one run and
     merged in another, cut stems like "Represent the following situations mathematically:").
   - *Backend chapter filter* — `/ncert-questions` uses `chapter ILIKE '%<name>%'`, so picking **Circles** also
     returns **Areas Related to Circles** (UI shows 1,542 rows / 62 `Example 1` rows; the sidebar count says 866).
   - The library card shows only `question_number` ("Example 1"), never the chapter, so copies look identical.
3. **Label-only rows: 0 in Class 10 Maths** (331 in the whole table). The 226 `short_text` rows are mostly real but
   context-less fragments (sub-parts split from their stem, in-text "(Why?)") → `needs_review`, not auto-junk.
4. **MCQ:** 439 rows with options have no answer (4.1%); no answer points outside its options. Options keep their
   `(A)` labels on some copies and not on others.
5. **Diagrams:** 346 rows are `question_type='diagram'` with no image — several are not diagram questions at all
   (#3407 a proof, #1473 an AP word problem) → the type itself is wrong on some rows. 25 rows mention a figure
   without being typed `diagram` and have no image.
6. **Garbled math is visible in the samples** (#3596 `2 21t a n 4 5 1t a n 4 5`, `3 2, 1 2` for 3/2, 1/2). These rows
   are not flagged by G1a — G1b estimates them.
7. **The pinned normalize() keeps `Example 1 :`**, so a copy with the prefix and one without are different hashes →
   they surface as near-duplicates in G1b, not as exact duplicates. Stripping `Example N :` would turn them into
   exact duplicates — your call before G4.

## Pre-flight checklist

| item | result |
|---|---|
| `feat/ncert-quality` (BE) deployed to production? | **No.** Not merged to `main` (2 commits ahead). Production `/ncert-question-stats` returns 10,793 for Class 10 Maths (unfiltered), and `/ncert-questions` still returns `(A) 12 cm` labels, which that branch strips. Vercel dashboard itself not checked (no Vercel CLI here) — this is evidence from live behaviour. |
| Migration 18 on staging? | There is **no staging project** (brief: production only). On production `quality_flag` / `quality_status` columns: **0** → 18 is not applied. |
| `psycopg2` + `supabase-py`? | Both importable (psycopg2 2.9.11, Python 3.12.10). No `DATABASE_URL` / service key in this shell (keys stay in your terminal), so this run read a JSON dump made with `supabase db query --linked` (SELECT only). With `DATABASE_URL` set, the script reads the table itself in a READ ONLY session. It does not use the service_role REST key (REST pages 1,000 rows). |
| Class 10 Maths count / newest row | 10,793 rows, `max(created_at)` 2026-09-19 11:42:08 UTC. Audit DB `now()` 2026-10-21 19:46:15 UTC. Inserted in the last 24 h: **0** (whole table). |
| Exporter field names | Table below. |
| Untracked files stashed? | **Not needed, nothing stashed.** Work happens in separate worktrees on `fix/ncert-builder-quality` (`Downloads/worktrees/be-ncert-builder-quality`, `fe-ncert-builder-quality`), both from `main`. Your checkouts (`feat/checker-v4`, `feat/loadtest`) and their untracked files are untouched. |
| Normalization | Your G1a note (no NFC, only `Q\d+\.`) differs from the pinned G0 definition. **The pinned one is used** (G0: "use these exactly"). |

## Export field names (what the builder sends → what each exporter reads)

| data | builder payload (`TestBuilderPage.tsx` `buildQuestionsPayload`) | backend `/export` (`export_service.py`) | browser fallback (`testPdfExporter.ts`, only when the backend fails) |
|---|---|---|---|
| question text | `text` | `text` | `text` (also `question_text`, `questionText`) |
| options | `options` (labels like `(A)` kept) | `options`, prints its own `A`… → **"A (A) 60°"** | `options`, prints its own label → same doubling |
| answer | `correctAnswer` (answer key only) | `correctAnswer` / `correct_answer` | `correct_answer` |
| explanation | always `""` → "With Explanations" has no effect | `explanation` | `explanation` (`""`) |
| image | `image_url` + `imageUrl` | `imageUrl` / `image_url`, rendered | passed but **not rendered** in the question paper |
| table | `question_table` | `questionTable` / `question_table`, rendered | **not passed, not rendered** |
| section | `section` | groups by `section` | not passed |
| math | KaTeX only for `$…$` / `$$…$$` | `_process_latex`: strips `$…$`, `\frac`→`(a/b)`, symbols→Unicode, `^`→superscript; no `\(` `\[` | `cleanTextForPdf`: strips `$$`/`$`, `\frac`→`a/b`, `\sqrt`→`√()`; no superscript |

Class 10 Maths has no LaTeX at all, so all three print the same plain text, and garbled math stays garbled in all three.

## Summary

| | rows | % of Class 10 Maths |
|---|---:|---:|
| total rows | 10,793 | 100% |
| distinct questions (chapter + hash) | 2,612 | 24.2% |
| **clean rows** | **2,339** | **21.7%** |
| rows that would be hidden (duplicate / label / short) | 8,317 | |

A row can be in several categories, so the category counts do not add up to the total. **clean** = no flag except `cross_chapter`. A NULL answer on a non-MCQ is not a flag (NCERT exercises have no answers in the text) — see *Answers*.

## Categories

| category | count | % of Class 10 Maths | 5 sample rows (id: text) |
|---|---:|---:|---|
| `exact_duplicate` | 8,181 | 75.8% | #1358 (dup of #1344): Find the area of a sector of a circle with radius 6 cm if angle of the sector is 60°.<br>#5226 (dup of #1396): Find the number of terms in each of the following APs : (i) 7, 13, 19, . . . , 205 (ii) 18, 15 1/2, 13, . . . , – 47<br>#9479 (dup of #1539): Find the distance between the following pairs of points : (i) (2, 3), (4, 1) (ii) (– 5, 7), (– 1, 3) (iii) ( a, b), (– a, – b)<br>#12979 (dup of #2659): Solve the following pair of equations by substitution method: 7x – 15 y =2 (1) x + 2y =3 (2)<br>#45463 (dup of #2961): A straight highway leads to the foot of a tower. A man standing at the top of the tower observes a car at an angle of depression of 30°, which is app… |
| `label_only` | 0 | 0.0% | — |
| `short_text` | 226 | 2.1% | #2604: Why is AB not – 2√2 k?<br>#16111: Now  ABD  ACD (Why?)<br>#22397: x + y = 5 and 2 x – 3y = 4<br>#25026: 2 x – 3y = 8 ; 4x – 6y = 9<br>#25696: 2, 4, 8, 16, . . . |
| `mcq_no_answer` | 439 | 4.1% | #1357: Tick the correct answer in the following : Area of a sector of angle p (in degrees) of a circle with radius R is<br>#3596: 2 21t a n 4 5 1t a n 4 5 <br>#6432: Choose the correct choice in the following and justify : 30th term of the AP: 10, 7, 4, . . . , is<br>#10167: Choose the correct choice in the following and justify : 30th term of the AP: 10, 7, 4, . . . , is<br>#12862: sin 2A = 2 sin A is true when A = |
| `mcq_answer_not_in_options` | 0 | 0.0% | — |
| `mcq_type_no_options` | 1 | 0.0% | #25721: Choose the correct choice in the following and justify : |
| `diagram_text_no_image` | 25 | 0.2% | #16086: Suppose the students of a school are visiting Qutub Minar. Now, if a student is looking at the top of the Minar, a right triangle can be imagined to …<br>#23059: Give two different examples of pair of (i) similar figures. (ii) non-similar figures.<br>#45979: Give two different examples of pair of (i) similar figures. (ii) non-similar figures.<br>#52431: Suppose a girl is sitting on the balcony of her house located on the bank of a river. She is looking down at a flower pot placed on a stair of a temp…<br>#76329: Shanta runs an industry in a shed which is in the shape of a cuboid surmounted by a half cylinder (see Fig.12.12). If the base of the shed is of dime… |
| `diagram_type_no_image` | 346 | 3.2% | #1473: 15. A contract on construction job specifies a penalty for delay of completion beyond acertain date as follows: ` 200 for the first day, ` 250 for th…<br>#3407: Prove that the tangents drawn at the ends of a diameter of a circle are parallel.<br>#5290: Draw a circle on a paper. T ake a point P inside it. Can you draw a tangent to the circle through this point? ... Next take a point P on the circle a…<br>#7787: Example 5 : Shanta runs an industry in a shed which is in the shape of a cuboid surmounted by a half cylinder (see Fig.12.12). If the base of the she…<br>#11858: Draw a circle and two lines parallel to a given line such that one is a tangent and the other , a secant to the circle. |
| `cross_chapter` | 0 | 0.0% | — |

## Per chapter

`UI rows` = what the builder shows when the chapter is picked today: the backend filters `chapter ILIKE '%<name>%'`, so a chapter whose name is inside another one also gets that chapter's rows.

| chapter | rows | distinct | exact dup | label | short | MCQ problem | diagram | clean | clean % | UI rows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Areas Related to Circles | 676 | 51 | 625 | 0 | 0 | 44 | 0 | 48 | 7.1% | 676 |
| Arithmetic Progressions | 2,068 | 351 | 1,717 | 0 | 50 | 66 | 11 | 291 | 14.1% | 2,068 |
| Circles | 866 | 142 | 724 | 0 | 1 | 144 | 116 | 120 | 13.9% | 1,542 ⚠️ |
| Coordinate Geometry | 918 | 120 | 798 | 0 | 0 | 0 | 21 | 119 | 13.0% | 918 |
| Introduction to Trigonometry | 1,381 | 445 | 936 | 0 | 67 | 151 | 21 | 372 | 26.9% | 1,381 |
| Pair of Linear Equations in Two Variables | 895 | 393 | 502 | 0 | 40 | 0 | 25 | 361 | 40.3% | 895 |
| Polynomials | 271 | 63 | 208 | 0 | 1 | 0 | 0 | 62 | 22.9% | 271 |
| Probability | 733 | 135 | 598 | 0 | 3 | 14 | 118 | 116 | 15.8% | 733 |
| Quadratic Equations | 639 | 259 | 380 | 0 | 46 | 0 | 0 | 237 | 37.1% | 639 |
| Real Numbers | 373 | 91 | 282 | 0 | 1 | 0 | 0 | 90 | 24.1% | 373 |
| Some Applications of Trigonometry | 428 | 55 | 373 | 0 | 0 | 0 | 14 | 53 | 12.4% | 428 |
| Statistics | 593 | 208 | 385 | 0 | 0 | 0 | 0 | 208 | 35.1% | 593 |
| Surface Areas and Volumes | 359 | 94 | 265 | 0 | 4 | 0 | 10 | 89 | 24.8% | 359 |
| Triangles | 593 | 205 | 388 | 0 | 13 | 21 | 35 | 173 | 29.2% | 593 |

## "Example N" analysis

Rows whose `question_number` or text starts with `Example N`. The verdict compares the `Example 1` texts of one chapter (without the `Example 1 :` prefix) with the longest one: ≥ 0.9 similar = re-extracted variant; contained in it = split part / cut stem; *other* = neither — a different question labelled Example 1 (Circles: a concentric-circles proof) or a variant too different for this check (→ G1b near-duplicates).

| chapter | rows labelled Example | distinct N | `Example 1` rows | `Example 1` distinct texts | `Example 1` rows in UI | verdict |
|---|---:|---:|---:|---:|---:|---|
| Areas Related to Circles | 52 | 2 | 26 | 2 | 26 | retry duplication — 2 texts: 2 re-extracted variants |
| Arithmetic Progressions | 484 | 16 | 38 | 5 | 38 | retry duplication — 5 texts: 5 re-extracted variants |
| Circles | 107 | 3 | 36 | 3 | 62 ⚠️ | retry duplication — 3 texts: 1 re-extracted variants, 1 other ('prove that in two concentric circles, th') |
| Coordinate Geometry | 267 | 10 | 28 | 1 | 28 | retry duplication — identical copies |
| Introduction to Trigonometry | 286 | 12 | 24 | 3 | 24 | retry duplication — 3 texts: 3 re-extracted variants |
| Pair of Linear Equations in Two Variables | 194 | 10 | 20 | 4 | 20 | retry duplication — 4 texts: 4 re-extracted variants |
| Polynomials | 91 | 5 | 19 | 1 | 19 | retry duplication — identical copies |
| Probability | 223 | 13 | 18 | 3 | 18 | retry duplication — 3 texts: 1 re-extracted variants, 1 other ('(why ?)') |
| Quadratic Equations | 197 | 9 | 27 | 12 | 27 | retry duplication — 12 texts: 3 re-extracted variants, 3 split part / cut stem, 6 other ('represent the following situation mathem'; 'represent the following situations mathe') |
| Real Numbers | 125 | 7 | 18 | 2 | 18 | retry duplication — 2 texts: 2 re-extracted variants |
| Some Applications of Trigonometry | 120 | 7 | 17 | 2 | 17 | retry duplication — 2 texts: 2 re-extracted variants |
| Statistics | 130 | 8 | 17 | 7 | 17 | retry duplication — 7 texts: 2 re-extracted variants, 4 other ('the marks obtained by 30 students of cla'; 'the marks obtained by 30 students of cla') |
| Surface Areas and Volumes | 97 | 7 | 14 | 5 | 14 | retry duplication — 5 texts: 5 re-extracted variants |
| Triangles | 83 | 8 | 11 | 5 | 11 | retry duplication — 5 texts: 4 re-extracted variants |

Example — all `Example 1` rows of **Arithmetic Progressions** (id: text):

- #1374 `7bf0daa3fd` For the AP : 3 2, 1 2, – 1 2, – 32 , . . ., write the first term a and the common difference d.
- #1423 `7bf0daa3fd` For the AP : 3 2, 1 2, – 1 2, – 32 , . . ., write the first term a and the common difference d.
- #1842 `7bf0daa3fd` For the AP : 3 2, 1 2, – 1 2, – 32 , . . ., write the first term a and the common difference d.
- #1885 `7bf0daa3fd` For the AP : 3 2, 1 2, – 1 2, – 32 , . . ., write the first term a and the common difference d.
- #2310 `1a68580c3b` For the AP : 3/2, 1/2, – 1/2, – 3/2, . . ., write the first term a and the common difference d.
- #2357 `7bf0daa3fd` For the AP : 3 2, 1 2, – 1 2, – 32 , . . ., write the first term a and the common difference d.
- #3283 `b4f249e09b` For the AP : 3/2, 1/2, – 1/2, – 3/2 , . . ., write the first term a and the common difference d.
- #3331 `b4f249e09b` For the AP : 3/2, 1/2, – 1/2, – 3/2 , . . ., write the first term a and the common difference d.
- #3740 `1a68580c3b` For the AP : 3/2, 1/2, – 1/2, – 3/2, . . ., write the first term a and the common difference d.
- #3788 `7bf0daa3fd` For the AP : 3 2, 1 2, – 1 2, – 32 , . . ., write the first term a and the common difference d.
- #5205 `7bf0daa3fd` For the AP : 3 2, 1 2, – 1 2, – 32 , . . ., write the first term a and the common difference d.
- #5251 `7bf0daa3fd` For the AP : 3 2, 1 2, – 1 2, – 32 , . . ., write the first term a and the common difference d.

## Answers

| question_type | rows | answer NULL/empty | with options | MCQ answer missing | MCQ answer not in options |
|---|---:|---:|---:|---:|---:|
| exercise | 6,268 | 6,258 | 336 | 336 | 0 |
| example | 2,158 | 42 | 0 | 0 | 0 |
| diagram | 1,275 | 915 | 35 | 35 | 0 |
| intext | 810 | 430 | 0 | 0 | 0 |
| mcq | 108 | 61 | 107 | 60 | 0 |
| activity | 97 | 76 | 0 | 0 | 0 |
| fillblank | 56 | 38 | 8 | 8 | 0 |
| hots | 21 | 21 | 0 | 0 | 0 |

`options` jsonb type over the whole table: string: 91,906 (a JSON string holding the list — the script decodes it twice).

## Full database — counts only

| category | all classes | Class 6 | Class 7 | Class 9 | Class 10 | Class 11 | Class 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rows | 91,906 | 4,679 | 3,824 | 12,756 | 25,963 | 28,225 | 16,459 |
| distinct (subject+chapter+hash) | 38,167 | 2,697 | 2,669 | 7,579 | 5,749 | 10,370 | 9,103 |
| `exact_duplicate` | 53,739 | 1,982 | 1,155 | 5,177 | 20,214 | 17,855 | 7,356 |
| `label_only` | 331 | 18 | 18 | 39 | 49 | 174 | 33 |
| `short_text` | 1,552 | 155 | 108 | 206 | 317 | 462 | 304 |
| `mcq_no_answer` | 3,671 | 39 | 32 | 626 | 1,261 | 1,126 | 587 |
| `mcq_answer_not_in_options` | 29 | 10 | 4 | 4 | 1 | 10 | 0 |
| `mcq_type_no_options` | 27 | 1 | 5 | 4 | 1 | 16 | 0 |
| `diagram_text_no_image` | 2,777 | 403 | 222 | 817 | 336 | 711 | 288 |
| `diagram_type_no_image` | 874 | 0 | 0 | 0 | 853 | 21 | 0 |
| `cross_chapter` | 302 | 0 | 0 | 4 | 286 | 12 | 0 |
| **clean** | **34,148 (37.2%)** | 2,286 (48.9%) | 2,383 (62.3%) | 6,592 (51.7%) | 5,154 (19.9%) | 9,377 (33.2%) | 8,356 (50.8%) |

# G1b

- **Hash:** `v2` (`Example N :` / `Q N.` label stripped before hashing; empty after the strip = junk, never a duplicate; prefixes are stripped in any order until none is left). 14/14 fixtures pass; the same file is copied to the frontend worktree (`src/lib/ncert/ncert_normalize_fixtures.json`).
- **Same input as G1a** (dump at DB `now()` 2026-10-21 19:46:15+00:00); random samples use seed 20261022.
- Read-only: SELECT dumps, HTTP HEAD on 50 image URLs, local PDFs. No DB writes, no AI calls.

## Clarifications (from the G1a review)

- **MCQ in G1a = a row whose `options` decode to a non-empty list, whatever its `question_type`.** Class 10 Maths:
  486 such rows (exercise 336, mcq 107, diagram 35, fillblank 8) → 439 without an answer, 47 with one.
  By `question_type = 'mcq'`: 108 rows (G0's number) → 107 have options, 61 have no answer (60 with options +
  the 1 `mcq` row without options = `mcq_type_no_options`).
- **"answer not in options = 0" is after double-decoding and label stripping.** `parse_options` runs `json.loads`
  until it gets a list (all 91,906 rows are a JSON string inside jsonb). `answer_matches` strips `(A)` / `A.` / `a)`
  from the answer and from every option before comparing, and also accepts a bare letter (`B`, `(D)`) by position.
  All 47 answered rows matched; 8 were printed and checked by hand in G1a.
- **Duplicate key = class_grade + subject + chapter + content_hash.** `section` is not in the key. `subject` is the raw
  string, so `Physics` and `Physics-I` (or `Political Science` / `Political_Science`) are never merged.

## Changes made in G1b (please review)

1. **v2 prefix order.** Your regex runs inside the existing numbering loop (strip until nothing changes), not once
   before it. Run once, `(i) Example 2 : Find x` kept its label while `Find x` didn't → a different hash.
   Fixture 14 covers it; 14/14 pass in the backend; the frontend copy is identical.
2. **G1a bug: 188 of the 331 `label_only` rows are real text**, not labels. Their text equals their section heading
   ("What is a mineral?", "Why do we need political parties?" — question-style NCERT headings). They are now
   `heading_only` → `needs_review`, never junk. Only 143 rows are true labels/empty. Class 10 Maths: 0 either way.
   Before this fix the full-DB v2 delta showed −90 duplicates / ±0 clean, because G1a let a heading row be the
   "canonical" copy of its group. With it, every class except Class 10 Maths keeps exactly its G1a clean count.
3. **`question_blocks/q_<id>.png` images are not wrong-chapter images.** 127 Class 11/12 Chemistry rows already have
   whole-question images on production — `add_question_block_images.py` from the old `feat/ncert-quality` work was
   applied to the live table even though migration 18 was not. G4's backup must treat those `image_url`s as current.

## Key findings

1. **Near-duplicates are the bigger half of the problem.** Class 10 Maths: 10,793 rows → 2,582 after exact dedupe →
   **1,027 clusters**. 1,448 near-duplicate pairs differ only by a math symbol (one copy has ∠/π/√/²/°, the other
   lost it). If each cluster's canonical copy is the one that kept the symbol, most garbled copies just drop out.
2. **Garbled math: 570 of 2,582 Class 10 Maths questions (22.1%)** after exact dedupe. After near-dup collapse only
   **69 questions have no clean sibling** (layout tier) — those are the only ones that need re-extraction
   (full DB: 893).
3. **The garble comes from the PDF text layer, not from Gemini.** `ncert_chunks.content` already contains the
   Symbol-font Private-Use-Area glyphs (U+F0D0 = ∠, U+F071 = θ, U+F070 = π, U+F044 = Δ …) and the flattened
   fractions. All 10,793 Class 10 Maths rows come from only 59 chunks. Re-running the current text pipeline would
   reproduce the garble — a real fix needs the page image (PDF pages are local, 217 for Class 10 Maths).
4. **`symbol_only` tier (246 Class 10 Maths questions, 68 left after near-dup collapse in the whole DB)** could be
   fixed with a fixed Adobe-Symbol→Unicode table, no AI. That is not in the allowed deterministic fixes — needs
   your OK.
5. **`figure_ref` is wrong on 752 rows, the image is right.** In every one of the 752 rows the question text names
   the same figure as the image file (`see Fig. 11.10` + `fig_11_10.png`), but `figure_ref` kept one digit after
   the dot (`Fig. 11.1`). The builder badge prints `figure_ref`, so it shows the wrong figure number. 182 are
   trailing-zero cuts (11.10 → 11.1), 570 are second-digit cuts (10.13 → 10.1). Setting `figure_ref` from the image
   file name is deterministic, but also not in the allowed list — your call.
6. **LaTeX:** no subject stores `$…$` math. The 8 `$` rows are currency (4 are wrongly split into KaTeX by the
   builder). 533 rows (Class 11/12 Maths, a few Physics) carry LaTeX **without delimiters**, so the builder shows
   `\sqrt{2}x^2` as raw text while the PDF export converts it. All 524 rows parsed by KaTeX parse cleanly — showing
   them properly is a display fix, not a data fix.
7. **Images load:** 50/50 random URLs → 200 `image/png`; the trailing `?` is harmless. 1,833 rows point to only 268
   distinct image files.
8. **Cross-chapter (302 rows, flag only):** almost all are Class 10 English questions filed under two chapters of
   different books (a story and a poem), i.e. wrong chapter attribution, not genuine repeats.
9. **`02_extract_ncert_questions.py` on `main` has 19 merge-conflict marker lines** — it cannot run as committed.
   Any re-extraction needs that fixed first.
10. **Cost is small either way:** under $1 for Class 10 Maths and under ~$9 for every local PDF at 2027 prices (table
    in §7). The real cost is review time, not tokens.

## Garbled-math precision — manual check of 30 random deduped Class 10 Maths questions

`docs/audit/garbled_precision_sample.csv` (seed 20261023). Verdict = is the text a teacher sees actually broken?

| id | signals | verdict | why |
|---|---|---|---|
| 4540 | symbol_font_glyph | ✅ | π missing ("and  = 3.14") |
| 4561 | symbol_font_glyph | ✅ | Δ missing, AD/AB flattened to "AD AB" |
| 4290 | flattened_power | ✅ | "100 x2" = 100x² |
| 45981 | symbol_font_glyph | ✅ | Δ missing |
| 10692 | symbol_font_glyph | ✅ | θ missing ("sin  increases as  increases") |
| 4505 | symbol + fraction | ✅ | π missing, "22 7" = 22/7, "m 3" = m³ |
| 10759 | symbol + spaced_function | ✅ | "1s i n A…" |
| 1504 | symbol_font_glyph | ✅ | X′Y′ primes and ∠AOB lost |
| 12959 | flattened_fraction | ✅ | "632st+=" |
| 2638 | symbol + spaced_function | ✅ | "(cosec  – cot )2 = 1c o s 1c o s" |
| 12715 | symbol_font_glyph | ✅ | ∠POQ, ∠PTQ lost |
| 4303 | flattened_power | ✅ | "2 x2", "4 3x" = 4√3x |
| 7262 | symbol_font_glyph | ✅ | Δ lost |
| 10089 | symbol_font_glyph | ✅ | π and √3 lost |
| 5870 | symbol + fraction | ✅ | "5 72, , 3" |
| 4546 | symbol_font_glyph | ✅ | Δ lost |
| 10694 | symbol_font_glyph | ✅ | θ lost |
| 2551 | symbol_font_glyph | ✅ | ∠ lost |
| 12920 | symbol + spaced_function | ✅ | bracket pieces, "t a n s e c" |
| 1434 | symbol + fraction | ✅ | "1 2" = 1/2 |
| 3022 | flattened_fraction | ❌ | frequency table on two lines — numbers intact and readable |
| 16059 | symbol_font_glyph | ✅ | "21a n d 323 3xy" |
| 10776 | flattened_fraction | ✅ | "11 2 2,ab ab" |
| 3417 | symbol_font_glyph | ✅ | A′B′ primes lost (minor) |
| 5451 | symbol_font_glyph | ✅ | θ lost |
| 2568 | symbol + spaced + fraction | ✅ | "22t a n3 0 1t a n 3 0" |
| 4634 | symbol_font_glyph | ✅ | Δ and fractions lost |
| 4434 | flattened_fraction | ✅ | table numbers corrupted ("31 0 1 18 3" for 3, 10, 11, 8, 3) |
| 12984 | symbol + fraction | ✅ | "632st", "23 0xy" |
| 4547 | symbol_font_glyph | ✅ | Δ lost |

**Precision 29/30 = 97%.** The weak signal is `flattened_fraction` on frequency tables (3/4 here): a flattened table
with intact numbers is a formatting problem, not garbled math. Recall was not measured (would need a hand-labelled
random sample of *unflagged* rows; `p` written for θ in Symbol font, e.g. "angle p (in degrees)", is not caught).

## Environment

rapidfuzz 3.13.0, numpy 2.1.3 and PyMuPDF 1.28.2 were installed in a throw-away venv in the session scratchpad —
not in your global Python and not added to `requirements.txt`. KaTeX = the frontend's own `node_modules/katex`
(via `scripts/check_katex_segments.cjs` in the FE worktree).

## 1. Exact duplicates — v1 (G1a) vs v2

v1 is recomputed in this run with the G1a rules and reproduces the G1a numbers above.

| | Class 10 Maths v1 | v2 | Δ | Full DB v1 | v2 | Δ |
|---|---:|---:|---:|---:|---:|---:|
| exact_duplicate rows | 8,181 | 8,211 | +30 | 53,739 | 53,756 | +17 |
| distinct questions | 2,612 | 2,582 | -30 | 37,956 | 38,007 | +51 |
| label_only (junk) | 0 | 0 | +0 | 331 | 143 | -188 |
| heading_only (needs_review, new) | 0 | 0 | +0 | 0 | 188 | +188 |
| clean (G1a definition) | 2,339 | 2,312 | -27 | 34,148 | 34,121 | -27 |

`distinct questions` leaves junk (label_only) rows out in both columns; G1a's full-DB figure counted them.

## 2. Same hash in different chapters (flag only, never merged)

Class 10 Maths: **0** rows · full DB: **302** rows (74 hashes). Junk and short rows excluded.

| class · subject | chapters (row ids) | text |
|---|---|---|
| 10 · English | For Anne Gregory: #520<br>The Sermon at Benares: #861 | You have per haps concluded that people ar e not objects to be valued for their qualities or riches rather th… |
| 10 · English | Glimpses of India: #606<br>The Trees: #893 | Why do you think the poet does not mention “the departure of the forest from the house” in her letters? (Coul… |
| 10 · English | Stories About Flying: #771<br>The Ball Poem: #778 | What does the poet say the boy is learning from the loss of the ball? Try to explain this in your own words. |
| 10 · English | How to Tell Wild Animals: #614<br>Stories About Flying: #763 | Do you know what a ‘bearhug’ is? It’s a friendly and strong hug — such as bears are thought to give, as they … |
| 10 · English | Stories About Flying: #768<br>The Ball Poem: #775 | “… staring down/All his young days into the harbour where/His ball went …” Do you think the boy has had the b… |
| 10 · English | A Letter to God: #423<br>Dust of Snow: #488 | What is a “dust of snow”? What does the poet say has changed his mood? How has the poet’s mood changed? |
| 10 · Science | Acids, Bases and Salts: #13908, #22596, #25927, #45658 …<br>Life Processes: #17122 | What is the colour of the solution? |
| 10 · English | Glimpses of India: #604<br>The Trees: #891 | How does the poet describe the moon: (a) at the beginning of the third stanza, and (b) at its end? What cause… |
| 10 · English | Stories About Flying: #770<br>The Ball Poem: #777 | Do you think the boy has lost anything earlier? Pick out the words that suggest the answer . |
| 10 · English | How to Tell Wild Animals: #611<br>Stories About Flying: #760 | Does ‘dyin ’ really rhyme with ‘lion’? Can you say it in such a way that it does? |

## 3. Garbled math (new category → `needs_review`)

Class 10 Maths: **2,057 rows (19.1%)**, **570 of 2,582 questions after exact dedupe (22.1%)**.

| signal | what it catches | rows | questions (deduped) |
|---|---|---:|---:|
| `symbol_font_glyph` | Symbol-font glyph stored as U+F0xx (∠ θ π Δ ° × ≤ …, bracket pieces) | 1,430 | 348 |
| `spaced_function` | `t a n 4 5` — function name split into letters | 133 | 52 |
| `digit_letter_soup` | `2R180p` — flattened fraction | 2 | 1 |
| `missing_numerator` | `= /360` — numerator symbol lost | 0 | 0 |
| `flattened_power` | `x2 +`, `PQ2 =` — superscript lost (Maths only) | 253 | 83 |
| `flattened_fraction` | `3 2, 1 2` for 3/2, 1/2 (Maths only) | 581 | 201 |

Tier (deduped questions): **symbol_only 246** — every bad character is a single Symbol glyph, a fixed table could map it back · **layout 324** — fractions/powers/brackets flattened, needs re-extraction from the PDF page.

10 samples (deduped questions):

- #10692 [symbol_font_glyph] State whether the following are true or false. Justify your answer : The value of sin  increases as  increases.
- #4625 [symbol_font_glyph] 9.ABCD is a trapezium in which AB \|\| DC and itsdiagonals intersect each other at the point O. Show that AO CO BO DO
- #3620 [symbol_font_glyph] Prove the following identity, where the angles involved are acute angles for which the expressions are defined: 22 21 tan A 1 tan A 1 – cot…
- #2557 [symbol_font_glyph] In  ABC, right-angled at B, AB = 5 cm and  ACB = 30° (see Fig. 8.19). Determine the lengths of the sides BC and AC.
- #4506 [symbol_font_glyph] A juice seller was serving his customers using glasses as shown in Fig. 12.13. The inner diameter of the cylindrical glass was 5 cm, but th…
- #4449 [flattened_fraction] The following frequency distribution gives the monthly consumption of electricity of 68 consumers of a locality. Find the median, mean and …
- #4456 [flattened_fraction] 7.The distribution below gives the weights of 30 students of a class. Find the median weight of the students. Weight (in kg) 40 - 45 45 - 5…
- #1347 [symbol_font_glyph] A chord of a circle of radius 10 cm subtends a right angle at the centre. Find the area of the corresponding : (i) minor segment (ii) major…
- #52762 [symbol_font_glyph] If AD and PM are medians of triangles ABC and PQR, respectively where  ABC ~  PQR, prove that AB / PQ = AD / PM.
- #3001 [flattened_fraction] Example 5 : A survey conducted on 20 households in a locality by a group of students resulted in the following frequency table for the numb…

Precision: see the manual 30-row check in the G1b notes above (`docs/audit/garbled_precision_sample.csv`).

## 4. Near-duplicates (rapidfuzz `fuzz.ratio` > 90, same chapter, after exact dedupe)

Class 10 Maths: **5,698 pairs**; **1,555 rows** sit in a cluster with a better copy (→ `needs_review`, never auto-merged) · full DB: 19,141 pairs, 10,010 rows.

Class 10 Maths questions: 10,793 rows → 2,582 after exact dedupe → **1,027 if every near-duplicate cluster were one question**.

1,448 of the pairs differ in a math symbol (one copy has ∠/π/√/²/° …, the other lost it) — the canonical copy should be the one with the symbol.

| chapter | near-dup pairs | rows flagged |
|---|---:|---:|
| Areas Related to Circles | 110 | 33 |
| Arithmetic Progressions | 502 | 165 |
| Circles | 199 | 78 |
| Coordinate Geometry | 129 | 62 |
| Introduction to Trigonometry | 637 | 264 |
| Pair of Linear Equations in Two Variables | 1,581 | 277 |
| Polynomials | 190 | 45 |
| Probability | 163 | 59 |
| Quadratic Equations | 1,317 | 191 |
| Real Numbers | 147 | 59 |
| Some Applications of Trigonometry | 28 | 23 |
| Statistics | 397 | 142 |
| Surface Areas and Volumes | 119 | 60 |
| Triangles | 179 | 97 |

Pairs: `docs/audit/near_duplicate_pairs.csv`.

## 5. LaTeX

Subjects with any `$` or `\command` (all others: 0):

| class · subject | rows | with `$` | with `\command` |
|---|---:|---:|---:|
| 9 · Economics | 534 | 2 | 0 |
| 9 · Political Science | 804 | 3 | 0 |
| 10 · Mathematics | 10,793 | 0 | 4 |
| 11 · Mathematics | 5,326 | 0 | 134 |
| 11 · Physics | 2,550 | 0 | 2 |
| 11 · Physics-II | 341 | 0 | 3 |
| 12 · Economics | 651 | 3 | 0 |
| 12 · Mathematics | 3,372 | 0 | 390 |

- **`$` rows (8):** every one is a currency dollar. The builder splits them like math: 4 rows get a `$…$` part rendered by KaTeX (prose shown as italic math), 0 of those fail to parse.
- **`\command` without `$` (533 rows):** the builder only renders `$…$`, so these show as raw `\sqrt{2}x^2` text. The PDF export converts them (`_process_latex`), so builder and PDF disagree.
- **KaTeX parse** (subjects with ≥ 20 such rows: 11 Mathematics, 12 Mathematics): 524 rows parsed, **0 fail**.

## 6. Images — HEAD check of 50 random URLs (of 268 distinct)

Status codes: **200** × 50. Trailing `?`: 50 of 50 URLs end with `?`; without it: 200 × 50 (the `?` is an empty query string and changes nothing).

Content types: image/png × 50.


Image vs row (whole table, no network): `figure_ref` number ≠ image file number: **752** rows (Class 10 Maths 702); image stored under another chapter's folder: **0** (Class 10 Maths 0).
- #1353 figure_number: figure_ref `Fig. 11.1` · `['Areas_Related_to_Circles', 'fig_11_10.png?']` · An umbrella has 8 ribs which are equally spaced (see Fig. 11.10). Assuming umbr…
- #1356 figure_number: figure_ref `Fig. 11.1` · `['Areas_Related_to_Circles', 'fig_11_11.png?']` · A round table cover has six equal designs as shown in Fig. 11.11. If the radius…
- #1367 figure_number: figure_ref `Fig. 11.1` · `['Areas_Related_to_Circles', 'fig_11_10.png?']` · An umbrella has 8 ribs which are equally spaced (see Fig. 11.10). Assuming umbr…
- #1370 figure_number: figure_ref `Fig. 11.1` · `['Areas_Related_to_Circles', 'fig_11_11.png?']` · A round table cover has six equal designs asshown in Fig. 11.11. If the radius …
- #1493 figure_number: figure_ref `Fig. 10.1` · `['Circles', 'fig_10_10.png?']` · PQ is a chord of length 8 cm of a circle of radius 5 cm. The tangents at P and …

## 7. Re-extraction cost estimate (nothing executed)

- **Models in the code:** `test-generator-backend/.env` sets `GEMINI_MODEL=gemini-3.5-flash-lite`; `02_extract_ncert_questions.py` defaults to `gemini-3.6-flash` when the env var is missing. Both priced.
- **Prices** (ai.google.dev/gemini-api/docs/pricing, read 22 Oct 2026, per 1M tokens, output includes thinking): `gemini-3.5-flash-lite` $0.300 in / $2.500 out; `gemini-3.6-flash (to 31 Dec 2026)` $0.750 in / $3.750 out; `gemini-3.6-flash (from 1 Jan 2027)` $1.500 in / $7.500 out.
- **Tokens:** PDF page at media_resolution `medium` = 560 + its native text (avg 492 tokens/page here) — Google's table does not say "per page"; that is our reading. Extraction prompt ≈ 1,300 tokens (5,188 chars). Question tokens = chars/4 of text + answer + options × 1.3 JSON overhead (no `count_tokens` call). Thinking billed as output: shown as none and as +100 %.
- **Why the PDF and not the chunks:** the garble is already in `ncert_chunks.content` (the PDF text layer), so re-running the current text pipeline reproduces it. The fix needs the page image.

| scenario | size | input tokens | output tokens | `gemini-3.5-flash-lite` (no thinking / +100 %) | `gemini-3.6-flash (to 31 Dec 2026)` (no thinking / +100 %) | `gemini-3.6-flash (from 1 Jan 2027)` (no thinking / +100 %) |
|---|---|---:|---:|---:|---:|---:|
| A. Class 10 Maths — re-extract the pages with Symbol glyphs | 148 of 217 pages | 325,990 | 49,613 | $0.22 / $0.35 | $0.43 / $0.62 | $0.86 / $1.23 |
| B. Class 10 Maths — repair each layout-garbled question (1 page image + row) | 69 questions | 103,344 | 4,118 | $0.04 / $0.05 | $0.09 / $0.11 | $0.19 / $0.22 |
| C. Every local NCERT PDF (22 books) — re-extract pages with Symbol glyphs | 794 of 3,937 pages | 1,867,345 | 378,857 | $1.51 / $2.45 | $2.82 / $4.24 | $5.64 / $8.48 |
| D. Full DB — repair each layout-garbled question | 896 questions | 1,336,409 | 46,252 | $0.52 / $0.63 | $1.18 / $1.35 | $2.35 / $2.70 |
| E. symbol_only tier — deterministic Symbol→Unicode table (needs your OK as a new allowed fix) | 68 questions (full DB) | 0 | 0 | $0.00 / $0.00 | $0.00 / $0.00 | $0.00 / $0.00 |

The local PDF folder covers Class10_English, Class10_Geography, Class10_History, Class10_Maths, Class10_Political_Science, Class10_Science, Class11_Chemistry, Class11_Mathematics, Class11_Physics - I, Class11_Physics - II, Class12_Chemistry - I, Class12_Chemistry - II, Class12_Flamingo, Class12_Mathematics - I, Class12_Mathematics - II, Class12_Physics - I, Class12_Physics - II, Class12_Vistas, Class9_Maths, Class9_Maths_Ganita_Manjari, Class9_Science, Class9_Science_Exploration — not every subject in the DB (no Accountancy, Economics, History … PDFs here), so C is a lower bound for the whole table. Retries (the extraction script retries up to 4× + empty-result retries) are not included.

## 8. Full database — new categories (counts only)

| category | all classes | Class 6 | Class 7 | Class 9 | Class 10 | Class 11 | Class 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact_duplicate (v2) | 53,756 (58.5%) | 1,982 | 1,155 | 5,176 | 20,244 | 17,843 | 7,356 |
| label_only (v2 junk) | 143 (0.2%) | 0 | 0 | 2 | 0 | 141 | 0 |
| heading_only (needs_review) | 188 (0.2%) | 18 | 18 | 37 | 49 | 33 | 33 |
| cross_chapter | 302 (0.3%) | 0 | 0 | 4 | 286 | 12 | 0 |
| garbled_math (any) | 4,055 (4.4%) | 36 | 111 | 235 | 2,121 | 895 | 657 |
| garbled — symbol_only | 1,211 (1.3%) | 1 | 1 | 0 | 1,190 | 10 | 9 |
| garbled — layout | 2,844 (3.1%) | 35 | 110 | 235 | 931 | 885 | 648 |
| near_duplicate | 10,010 (10.9%) | 473 | 592 | 1,355 | 2,109 | 3,107 | 2,374 |
| raw LaTeX shown as text in builder | 533 (0.6%) | 0 | 0 | 0 | 4 | 139 | 390 |
| currency `$` parsed as math | 8 (0.0%) | 0 | 0 | 5 | 0 | 0 | 3 |
| figure_ref ≠ image file | 752 (0.8%) | 0 | 0 | 0 | 752 | 0 | 0 |
| image in another chapter's folder | 0 (0.0%) | 0 | 0 | 0 | 0 | 0 | 0 |
| **clean (G1a definition, v2 hash)** | 34,121 (37.1%) | 2,286 | 2,383 | 6,592 | 5,127 | 9,377 | 8,356 |
| **clean strict (also no garble / near-dup / raw LaTeX)** | 23,803 (25.9%) | 1,883 | 1,802 | 5,311 | 3,088 | 6,187 | 5,532 |

Class 10 Maths: clean (G1a def, v2) **2,312 (21.4%)** · clean strict **768 (7.1%)**.
