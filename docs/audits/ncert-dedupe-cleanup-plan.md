# Exact-duplicate cleanup of `ncert_questions` — revised plan v3 (NOT executed)

Supersedes §9 steps 2–4 of `docs/audits/ncert-ingestion-audit.md` (branch `audit/ncert-ingestion`).
SQL lives in `scripts/cleanup/ncert_dedupe_exact/`; nothing there has been run against Supabase.

v2 (28 Nov) added:
- step **1b**, which normalises chapter and subject names before the dedup key is computed;
- the **English poem rule** in step 2;
- the JSON/array reference check.

v3 (28 Nov, later) adds:
- step **1c**: the Class 11 Physics back-matter ("Answers", "Index", 303 rows) is **hidden, not deleted, and not a
  chapter** (see *Back-matter* below);
- step 4's unique index is **partial** (it leaves out the hidden back-matter);
- evidence that Class 12 "Chemical Coordination" really is **Coordination Compounds**, so the merge stays.

Every step was tested end to end on a throwaway local Postgres: backup → normalise → map → apply → unique index →
rollback. The fixture covers:
- a practice-mark collision and a remap;
- a G4.5 pointer to a doomed row;
- fix-log rows and an all-hidden group;
- Physics vs Physics-I copies, an explicit chapter alias, a History section-title rename;
- an English prose/poem pair with a student mark on the prose copy;
- back-matter Answers/ANSWERS/Index rows, including an exact copy across Physics-I and Physics-II (kept, hidden);
- Economics "Index Numbers", which must stay untouched;
- a "Chemical Coordination" / "Coordination Compounds" pair;
- the live CHECK constraints on `quality_status` / `duplicate_of`.

A full-row rollback (subject, chapter, `quality_status` and `quality_flags` included) is byte-identical.

## Dry run — live DB, 28 Nov 2026 (read-only, `scripts/audit/16g`, v3 rules)

| Metric | v1 raw key | v2 | **v3** |
|---|---|---|---|
| rows that would be deleted | 53,378 | 54,902 | **54,882** |
| … same-chapter exact copies (normalised key) | — | 54,837 | 54,817 |
| … English poem rule | — | 65 | 65 |
| back-matter rows hidden by 1c (not deleted) | — | — | **303** (Answers 262, Index 41) |
| … exact copies among them, now kept hidden (v2 deleted them) | — | — | 20 |
| rows kept: visible / hidden back-matter | 38,528 | 37,004 | **36,721 / 303** |
| chapter strings | 504 | 425 | 425 |
| chapters teachers can pick | — | 425 | **423** |
| rows whose subject/chapter string is rewritten (1b) | — | 4,967 | 4,967 |
| `ncert_practice_marks` rows on deleted / hidden rows | 0 | 0 | **0 / 0** |
| groups keeping a hidden copy over a visible one | 0 | 0 | 0 |
| `ncert_question_fixes` rows on deleted rows | 2,206 | 2,206 | 2,206 |
| collisions left inside the unique index | — | 0 | 0 |

v2 → v3: the only change in deletions is the 20 exact copies inside "Index" (54,902 − 20 = 54,882). They are now
hidden instead. 91,906 rows = 54,882 deleted + 36,721 visible + 303 hidden.

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

## Back-matter (step 1c): hidden, not deleted, not a chapter
Class 11 Physics Part I/II have no chapter called "Answers" or "Index": these are the books' back pages.
Evidence: `scripts/audit/22a`–`22e`.

| String | Rows | What it is | Samples (10 each, `22b`) |
|---|---|---|---|
| Physics-I "Answers" + Physics-II "ANSWERS" | 215 + 47 = 262 | Exercise-answer pages. Every row has an answer, and many stems are just a number ("Question 2.10", "Question 3.19"). 262 distinct texts. | 30537 true/false motion statements · 30541 "x-t plot shows the trajectory?" · 30553 displacement after turns · 30559 "Which of the following statements is true?" · 30561 pebble force · 51074 "Question 2.10" · 51101 "Question 3.19" · 51127 "Question 4.23" · 51147 "Question 5.21" · 30935 "Calculate the height." |
| Physics-I "Index" + Physics-II "Index" | 30 + 11 = 41 | Activity/in-text prompts the extractor read from the back pages, cut off mid-sentence. None has an answer. Only 21 distinct texts (20 exact copies). | 20766 pressure/temperature change · 29827 "other areas where such an approach is used?" · 29828 system describable by components · 29831 "which will diffuse more" · 48634 gas diffusion · 72024, 72027, 79089 pressure/temperature · 72025, 79087 system description |

None of the 303 texts appears in a real Class 11 Physics chapter (`22e`), so hiding them removes no content.
Step 1c:
- **Hidden:** `quality_status = 'junk'`, which the CHECK allows. Every teacher-facing read already filters it out
  (`_NCERT_VISIBLE_SQL`), so "Answers" and "Index" drop out of the builder's chapter list (`/ncert-question-stats`
  counts visible rows only).
- **Tagged:** `quality_flags.hidden_reason = 'back_matter'`, which tells these rows apart from G4.5 junk.
- **Not deleted:** steps 2–3 skip tagged rows, and the unique index leaves them out (its `WHERE` clause).
- **Traceable:** the chapter strings stay ("Answers", "Index" after 1b), and the original status/flags are saved in
  `a4_backup_ncert_dedupe.back_matter`.

Out of scope: `/chapters` lists from `ncert_chunks`, which this plan doesn't touch. It already drops all-caps names;
whether chunks hold "Answers"/"Index" is a separate check.

## Class 12 Chemistry "Chemical Coordination" = Coordination Compounds (`scripts/audit/23a`–`23d`)
There is no NCERT chapter by that name. The candidates were Class 11 Biology "Chemical Coordination and Integration"
and Class 12 Chemistry "Coordination Compounds". **It is Coordination Compounds:**
- All 46 rows (Chemistry-I, 36 distinct texts) are coordination chemistry. 43 match complex/ligand/isomer/VBT/IUPAC
  terms (`23c`). The other 3 are Werner's theory (51501), complex formulas (51504) and a copper-complex experiment
  (75377). 0 rows mention hormones, glands or other endocrine terms.
- Their sections are Unit 5 of Chemistry Part I: "5.1 Werner's Theory", "5.3 Nomenclature of Coordination
  Compounds", "5.5.2 Valence Bond Theory".
- 25 of the 36 texts are also in Class 12 Chemistry "Coordination Compounds". 0 are in Class 11 Biology (which has
  its own 72 rows under the correct name).
- 15 samples (`23b`): 32738 ions from Co(NH3)6Cl2 · 32739 highest magnetic moment · 32740 most stable complex ·
  58994, 58996, 81996, 81998 IUPAC names · 58995 formulas · 59046 [MnBr4]2− geometry · 75374 geometrical isomers ·
  75376 draw isomers · 75378 VBT bonding · 75379 oxidation state/d-orbital occupation · 81999 chirality ·
  82426 role of coordination compounds in biology/medicine.

So the `explicit_merge` → Coordination Compounds rule is unchanged. The CSV row now carries this evidence in its
`note` column. `ncert_chunks` also holds 26 chunks under "Chemistry-I / Chemical Coordination"; that is a RAG
labelling issue for the chunks pass, not this plan.

## Normalisation (step 1b) — what changes

| Rule | Detail | Evidence |
|---|---|---|
| Subject family | `Physics-I/II` → Physics, `Chemistry-I/II` → Chemistry, `Political_Science` → Political Science | |
| Same chapter, different spelling | Case- and punctuation-insensitive match inside class + family, e.g. "ELECTRIC CHARGES AND FIELDS" = "Electric Charges and Fields" | 69 chapters |
| Explicit renames, Class 12 | "ALTERNATING CURRENT", "CURRENT AND ELECTRICITY", "ELECTROMAGNETIC INDUCTION", "ELECTROMAGNETIC WAVE", "ELECTROSTATIC POTENTIAL", "RAY OPTICS", "SEMICONDUCTORS" → the full Physics names; "Chemical Coordination" → Coordination Compounds | 41–71% shared exact texts (`19a`); Semiconductors by name; Chemical Coordination: content check above (`23a`–`23d`) |
| Explicit renames, Class 10 History | "LIVELIHOODS, ECONOMIES AND SOCIETIES" → **The Making of a Global World**; "EVERYDAY LIFE, CULTURE AND POLITICS" → **Print Culture and the Modern World** | Content check (`19b`): 448 global-economy hits / 0 industrialisation; 37 of 43 rows about print |
| English poem rule (step 2) | A text in both a prose chapter and its paired poem keeps only the poem copy | 9 pairs; 83–100% of each poem's texts also sit in its prose chapter (`19a`). The prose PDFs include the poem that follows them in First Flight |

**Full mapping for review:** `docs/audits/samples/chapter_merge_map.csv`, from `scripts/audit/21_chapter_merge_map.sql`.
It has 513 rows: every old (class, subject, chapter) → new, plus the 9 prose → poem pairs with the rows that move.
Rules: `unchanged` 421, `subject_family` 60, `spelling` 9, `explicit_merge` 8, `explicit_rename` 2,
`back_matter_hidden` 4, `english_poem_pair` 9. A `note` column gives the evidence for back-matter, Chemical
Coordination and the poem pairs.

**Why 425 strings and 423 chapters, against the audit's 431 (`21b_chapter_count_reconcile.sql`):**
431 − 8 + 2 = 425 chapter strings, and 431 − 8 = **423 chapters teachers can pick**.
- **−8 merges:** the audit's 431 only merged spelling variants. Step 1b additionally merges 8 differently-named
  Class 12 chapters into an existing chapter: Physics ALTERNATING CURRENT, CURRENT AND ELECTRICITY,
  ELECTROMAGNETIC INDUCTION, ELECTROMAGNETIC WAVE, ELECTROSTATIC POTENTIAL, RAY OPTICS, SEMICONDUCTORS, and
  Chemistry "Chemical Coordination".
- **+2 strings, 0 chapters:** the audit's 431 excluded the non-chapters. 1b keeps Class 11 Physics "Answers" and
  "Index" as strings on the rows, and 1c hides them, so they are not chapters.
- **0 from the History renames:** they get new names but merge nothing.

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

**1c. Hide back-matter** (`01c_hide_back_matter.sql`).
- Saves the 303 rows' status/flags in `a4_backup_ncert_dedupe.back_matter`.
- Sets `quality_status = 'junk'`, adds `quality_flags.hidden_reason = 'back_matter'`, and clears `duplicate_of`
  (required by the CHECK; all 303 are NULL today).
- Asserts: the count matches the step-1 backup, none is still visible, the row count is unchanged.
- Check: Class 11 Physics in the builder has no "Answers"/"Index" chapter.

**2. Map** (`02_build_map.sql`). Refuses to run before 1c.
- Same-chapter groups on the normalised names, then the English poem rule. Back-matter rows are skipped, and a
  check fails if any is in the map.
- Copies that pointed at a prose row are re-pointed to the poem row.
- Export the map to CSV and spot-check before GO.

**3. Apply** (`03_apply.sql`). One transaction:
- stale-map guard;
- marks: collapse collisions, remap, refresh chapter copies;
- `duplicate_of` repoint;
- explicit fix-log removal;
- `ncert_question_redirects` (SELECT-only);
- delete;
- asserts (row count, zero cascade into marks, no exact duplicate left outside the back-matter, all 303
  back-matter rows still present and hidden).

**4. Unique index** (`04_unique_index.sql`, alone, after step 3 is verified):
`(class_grade, subject, chapter, md5(lower(btrim(regexp_replace(question_text, '\s+', ' ', 'g')))))`
`WHERE (quality_flags->>'hidden_reason') IS DISTINCT FROM 'back_matter'`.
- All four functions are IMMUTABLE (`16f`).
- It is partial because the hidden "Index" rows keep their 20 exact copies.
- An ingest using `ON CONFLICT` must repeat the `WHERE` clause.

**Rollback** (tested):
- Run `99a` (alone, if step 4 ran), then `99b`.
- 99b re-inserts deleted rows with their original ids. It restores subject/chapter (undoes 1b),
  `quality_status`/`quality_flags` (undoes 1c), marks, the fix log and `duplicate_of`, and drops the redirects.

## Not covered (next passes, separate GO)
- **Near-duplicates:** spacing/punctuation-only pairs can be auto-merged by the same mechanism. Digit/word
  differences need review.
  - Most subject-variant books were extracted from different PDFs, so normalisation mostly renames them rather
    than exposing exact copies (only 1,459 extra exact copies). The rest are near-duplicates.
- **Code follow-ups:**
  - `/ncert-questions/by-ids` follows `ncert_question_redirects`.
  - Ingest uses `ON CONFLICT DO NOTHING`.
  - G4.5 must expect unique violations.
  - Remove `_apply_ncert_subject_filter`'s variant lists once 1b is live.
  - **`ncert_quality_pass.py` rewrites `quality_status` and `quality_flags` wholesale.** If it ever runs on Class 11
    Physics, it would un-hide the back-matter and drop `hidden_reason`. Its update would then also hit the unique
    index on the "Index" copies. Before any such run, make it keep rows tagged `hidden_reason = back_matter` as
    `junk` (or skip them).
  - Relabel the 26 `ncert_chunks` under Class 12 "Chemistry-I / Chemical Coordination" as Coordination Compounds
    (chunks pass).
