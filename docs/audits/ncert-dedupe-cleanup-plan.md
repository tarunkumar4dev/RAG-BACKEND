# Exact-duplicate cleanup of `ncert_questions` — revised plan (NOT executed)

Supersedes §9 steps 2–4 of `docs/audits/ncert-ingestion-audit.md` (branch `audit/ncert-ingestion`).
SQL lives in `scripts/cleanup/ncert_dedupe_exact/`; nothing there has been run against Supabase. Every step was
tested end to end (backup → map → apply → unique index → rollback) on a throwaway local Postgres with a fixture
covering a practice-mark collision, a remap, a G4.5 pointer to a doomed row, fix-log rows and an all-hidden group.

## Dry run — live DB, 27 Nov 2026 (`scripts/audit/16d_dedupe_dryrun.sql`, read-only)

| Metric | Value |
|---|---|
| duplicate groups (exact, same class + subject + chapter) | 17,385 |
| rows that would be deleted | **53,378** (45,178 visible + 8,200 already hidden by G4.5) |
| rows kept | 38,528 |
| `ncert_practice_marks` rows / students | **0 / 0** |
| marks to remap / students affected / marks lost to collisions | **0 / 0 / 0** |
| groups whose canonical is chosen because of marks | 0 |
| groups where a hidden copy would be kept over a visible one | 0 |
| `ncert_question_fixes` rows on doomed rows (would cascade) | 2,206 of 2,775 |
| kept rows whose `duplicate_of` points at a doomed row | 0 |
| kept rows that stay hidden (G4.5 near-duplicates / all-hidden groups) | 936 / 425 groups |

**Student records today: none affected.** Verified it is not RLS: the session role (`supabase_read_only_user`)
bypasses RLS, the table has had 0 inserts since stats reset, and no frontend code on `main` calls
`record_practice_mark` yet (feature added in migrations on 2 Nov). The remap step stays in the plan because the
count will not be 0 once practice ships — **re-run `16d` the same day as step 3.**

Other dependents (checked): FKs only from `ncert_practice_marks` (CASCADE), `ncert_question_fixes` (CASCADE) and
`ncert_questions.duplicate_of` (SET NULL); one function, `record_practice_mark`; no views. Saved tests (`questions`)
copy the text and hold no id → unaffected. Papers saved in teachers' browsers hold ids → handled by redirects.

## Canonical row per group (priority order)
1. Most student practice marks: students' records stay where they are; nothing to remap in the common case.
2. Visible today (not G4.5 `duplicate`/`junk`): never resurrects a copy the quality pass hid.
3. Lowest `quality_rank` (G4.5 clean/fixed first).
4. Has `image_url`, then 5. has an answer, 6. has options, 7. has `question_table`.
8. Lowest id.

Ties in marks (two copies both marked) fall through to rules 2–8. Every student's marks are moved to the canonical,
and if a student marked several copies of one question, the **newest** mark wins (`record_practice_mark` upserts
with `updated_at`, so the newest is the student's latest answer).

## Steps — each needs GO (Tarun + Nitin)

**0. Preconditions.**
- Migration 41 applied.
- No extraction/ingest script running.
- Practice-marks feature status known: re-run `16d`.
- Before practice ships, add an index on `ncert_practice_marks(question_id)`: FK cascade lookups use it, and today
  only the `(user_id, question_id)` PK exists.

**1. Backup** (`01_backup.sql`).
- Creates schema `a4_backup_ncert_dedupe` with copies of `ncert_questions`, `ncert_practice_marks` and
  `ncert_question_fixes`, plus fingerprints.
- Fails if the schema already exists, and checks row counts.
- Also take an offline dump of the same three tables, e.g.
  `pg_dump "$DATABASE_URL" -t public.ncert_questions -t public.ncert_practice_marks -t public.ncert_question_fixes -Fc`,
  stored encrypted, outside the repo.

**2. Map** (`02_build_map.sql`).
- Writes `a4_backup_ncert_dedupe.dedupe_map (dup_id → canonical_id)`, only into the backup schema.
- Checks that no canonical row is itself a duplicate.
- Export to CSV and spot-check about 50 groups across subjects before GO.

**3. Apply** (`03_apply.sql`). One transaction; any failed check rolls everything back.
1. Stale-map guard: every mapped row must still exist, and the marks count must be unchanged since the backup.
2. **Marks:** collapse collisions (newest wins), then remap to the canonical.
3. **G4.5 pointers:** point any `duplicate_of` at the canonical.
4. **Fix log:** rows of doomed copies are deleted explicitly (copies kept in the backup), not silently by
   CASCADE.
5. **Redirects:** new `public.ncert_question_redirects(old_id → new_id)`, SELECT-only for `anon`/`authenticated`.
   Default privileges would otherwise grant them ALL.
6. **Delete** the 53,378 copies.
7. Asserts: exact row count; the delete cascaded into zero practice marks; no exact duplicate left.

Lock: `SHARE ROW EXCLUSIVE` on the three tables (blocks writes for the few seconds it runs, not reads).

Verify after commit:
- `/ncert-question-stats` totals per class+subject vs a snapshot taken before step 3. Expected equal, because the
  read-time dedupe already hides exact copies. It may be slightly lower where copies differed only in options.
- `/ncert-questions/by-ids` for a few deleted ids returns nothing until the backend follows redirects (see
  follow-ups).

**4. Unique constraint** (`04_unique_index.sql`, only after step 3 is verified).
`CREATE UNIQUE INDEX CONCURRENTLY ncert_questions_chapter_text_uniq ON ncert_questions (class_grade, subject,
chapter, md5(lower(btrim(regexp_replace(question_text, '\s+', ' ', 'g')))))`.
- All four functions are IMMUTABLE (checked live, `16f`).
- Run the file alone: it cannot run in a transaction.
- From then on a duplicate insert or text edit fails loudly (tested).

**Rollback** (tested; fingerprint identical to the backup afterwards):
- Before step 3 commits: `ROLLBACK`.
- After it: `99a_rollback_drop_unique_index.sql` (run alone, only if step 4 ran), then `99b_rollback_restore.sql`.
  This re-inserts deleted rows with their original ids and restores marks, fix log and `duplicate_of`, then drops
  the redirects.
- Marks recorded *after* step 1 must be merged by hand; the script says so.

## Not covered (next passes, separate GO)
- **Near-duplicates:** spacing/punctuation-only pairs can be auto-merged by the same mechanism. Digit/word
  differences need review.
- **Subject-variant copies:** 69 chapters, 3,053 rows under Physics-I/II, Chemistry-I/II and Political_Science.
  These are different subject strings, so they are outside these groups and the unique key; that needs the
  chapters table first.
- **Code follow-ups:**
  - `/ncert-questions/by-ids` follows `ncert_question_redirects`.
  - Any future ingest uses `INSERT … ON CONFLICT DO NOTHING` against the new index.
  - The G4.5 script must expect unique violations when it rewrites text.
