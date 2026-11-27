# Exact-duplicate cleanup of `ncert_questions` — revised plan v2 (NOT executed)

Supersedes §9 steps 2–4 of `docs/audits/ncert-ingestion-audit.md` (branch `audit/ncert-ingestion`).
SQL lives in `scripts/cleanup/ncert_dedupe_exact/`; nothing there has been run against Supabase.

v2 (28 Nov) adds:
- step **1b**, which normalises chapter and subject names before the dedup key is computed;
- the **English poem rule** in step 2;
- the JSON/array reference check.

Every step was tested end to end on a throwaway local Postgres: backup → normalise → map → apply → unique index →
rollback. The fixture covers:
- a practice-mark collision and a remap;
- a G4.5 pointer to a doomed row;
- fix-log rows and an all-hidden group;
- Physics vs Physics-I copies, an explicit chapter alias, a History section-title rename;
- an English prose/poem pair with a student mark on the prose copy.

A full-row rollback (subject/chapter included) is byte-identical.

## Dry run — live DB, 28 Nov 2026 (read-only)

| Metric | v1 raw key (`16d`) | **v2 normalised key (`16g`)** |
|---|---|---|
| rows that would be deleted | 53,378 | **54,902** |
| … extra from subject/chapter normalisation | — | 1,459 |
| … extra from the English poem rule | — | 65 |
| rows kept | 38,528 | 37,004 |
| chapter strings | 504 | 425 after normalisation |
| rows whose subject/chapter string is rewritten (step 1b) | — | 4,967 |
| `ncert_practice_marks` rows / students | 0 / 0 | **0 / 0** |
| marks to remap / chapter copies rewritten | 0 | **0 / 0** |
| groups keeping a hidden copy over a visible one | 0 | 0 |
| `ncert_question_fixes` rows on doomed rows | 2,206 | 2,206 |
| unique-key collisions left after cleanup | — | 0 |

**Student records: none affected today.** The session role bypasses RLS, the table has had 0 inserts since
stats were reset, and no frontend code on `main` calls `record_practice_mark` yet. The remap logic stays in, so
**re-run `16g` the same day as step 3.**

### Does "fix pointers" cover ids inside JSON/JSONB/arrays? (`scripts/audit/20_json_question_id_refs.sql`, `20b`)

**Checked:** 60 columns in 31 tables, i.e. every json/jsonb column and every integer/text array column in schemas
`public` and `practice`. Among them:
- `tests.request_payload` (223 rows) and the saved-paper `questions.*` columns (1,786 rows);
- `contest_attempts.answers/progress/warning_log`, `contest_questions.options`, `contest_import_jobs.result`;
- `bank_questions.options`, `community_quiz*`, `check_*`, `submissions.*`, `assignments.allowed_mimes`,
  `assignment_ai_keys.rubric`, `smart_assignment_checks.result_json`;
- `modules.*`, `payments.metadata`, `practice*.questions.*`.

**Result: 0 references to `ncert_questions` ids.** Details:
- The only id-like keys are `questionId` in `contest_attempts.answers` (1,399 values) and `.progress` (1,200).
  Every value is a `contest_questions` **UUID**, verified by matching them back to `contest_questions.id`.
- The 6 integer hits in `practice_questions.options` are numeric answer options, not ids (no id key).
- No drafts table exists.

So the id references to fix are the FKs (`ncert_practice_marks`, `ncert_question_fixes`, `duplicate_of`) plus
browser localStorage (covered by redirects).

## Normalisation (step 1b) — what changes

| Rule | Detail | Evidence |
|---|---|---|
| Subject family | `Physics-I/II` → Physics, `Chemistry-I/II` → Chemistry, `Political_Science` → Political Science | |
| Same chapter, different spelling | Case- and punctuation-insensitive match inside class + family, e.g. "ELECTRIC CHARGES AND FIELDS" = "Electric Charges and Fields" | 69 chapters |
| Explicit renames, Class 12 | "ALTERNATING CURRENT", "CURRENT AND ELECTRICITY", "ELECTROMAGNETIC INDUCTION", "ELECTROMAGNETIC WAVE", "ELECTROSTATIC POTENTIAL", "RAY OPTICS", "SEMICONDUCTORS" → the full Physics names; "Chemical Coordination" → Coordination Compounds | 41–71% shared exact texts (`19a`); Semiconductors by name |
| Explicit renames, Class 10 History | "LIVELIHOODS, ECONOMIES AND SOCIETIES" → **The Making of a Global World**; "EVERYDAY LIFE, CULTURE AND POLITICS" → **Print Culture and the Modern World** | Content check (`19b`): 448 global-economy hits / 0 industrialisation; 37 of 43 rows about print |
| English poem rule (step 2) | A text in both a prose chapter and its paired poem keeps only the poem copy | 9 pairs; 83–100% of each poem's texts also sit in its prose chapter (`19a`). The prose PDFs include the poem that follows them in First Flight |

Kept spelling: the explicit name, else the most-used raw spelling of the target. The real same-name clash Class 11
"Thermodynamics" (Physics vs Chemistry) stays separate because the families differ.

Side effects:
- `ncert_chunks` is **not** renamed. RAG and `/chapters` keep their own names until a chapters table exists.
- The builder shows the new names. Saved papers hold ids, so they are unaffected.

## Canonical row per group (priority order)
1. Most student practice marks.
2. Visible today (not G4.5 `duplicate`/`junk`).
3. Lowest `quality_rank`.
4. Has `image_url`, then 5. answer, 6. options, 7. `question_table`.
8. Lowest id.

On collisions the newest mark wins. Marks that move to another chapter (poem rule) take that chapter's names.

## Steps — each needs GO (Tarun + Nitin)

**0. Preconditions.**
- Migrations 41 and 42 applied.
- No extraction/ingest running.
- Re-run `16g`.
- Before practice ships, add an index on `ncert_practice_marks(question_id)`.

**1. Backup** (`01_backup.sql`).
- Copies `ncert_questions`, `ncert_practice_marks` and `ncert_question_fixes`, plus fingerprints.
- Fails if the backup schema already exists.
- Also take an offline `pg_dump` of the same three tables.

**1b. Normalise names** (`01b_normalize_chapters.sql`).
- Persists `a4_backup_ncert_dedupe.chapter_alias`.
- Rewrites subject/chapter on 4,967 rows, plus the copies in `ncert_practice_marks`.
- Asserts the row count is unchanged and no subject variants remain.
- Check the builder's chapter list per class+subject afterwards.

**2. Map** (`02_build_map.sql`).
- Same-chapter groups on the normalised names, then the English poem rule.
- Copies that pointed at a prose row are re-pointed to the poem row.
- Export the map to CSV and spot-check before GO.

**3. Apply** (`03_apply.sql`). One transaction:
- stale-map guard;
- marks: collapse collisions, remap, refresh chapter copies;
- `duplicate_of` repoint;
- explicit fix-log removal;
- `ncert_question_redirects` (SELECT-only);
- delete;
- asserts (row count, zero cascade into marks, no exact duplicate left).

**4. Unique index** (`04_unique_index.sql`, alone, after step 3 is verified):
`(class_grade, subject, chapter, md5(lower(btrim(regexp_replace(question_text, '\s+', ' ', 'g')))))`.
All four functions are IMMUTABLE (`16f`).

**Rollback** (tested):
- Run `99a` (alone, if step 4 ran), then `99b`.
- 99b re-inserts deleted rows with their original ids and restores subject/chapter (undoes 1b), marks, fix log
  and `duplicate_of`. It also drops the redirects.

## Not covered (next passes, separate GO)
- **Near-duplicates:** spacing/punctuation-only pairs can be auto-merged by the same mechanism. Digit/word
  differences need review.
  - Most subject-variant books were extracted from different PDFs, so normalisation mostly renames them rather
    than exposing exact copies (only 1,459 extra exact copies). The rest are near-duplicates.
- **Non-chapters:** "Answers" / "Index" (303 rows) → `junk`, a separate step.
- **Code follow-ups:**
  - `/ncert-questions/by-ids` follows `ncert_question_redirects`.
  - Ingest uses `ON CONFLICT DO NOTHING`.
  - G4.5 must expect unique violations.
  - Remove `_apply_ncert_subject_filter`'s variant lists once 1b is live.
