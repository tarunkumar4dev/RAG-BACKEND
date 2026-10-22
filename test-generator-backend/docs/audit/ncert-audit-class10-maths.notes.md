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
