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
