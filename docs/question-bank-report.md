# Question Bank → Paper Builder — report (14 Nov 2026)

Branch `feat/question-bank`, both repos, from `origin/main`. Committed per gate, **not pushed, not deployed, migration not applied**.
**Not run by me:** the real stack in a browser, and a live Gemini read of the second fixture (this session may not load the backend `.env`).

**Works (tests + real files inspected)**
- Contest from PDF tab: after review, "Save as Question Bank" and "Build paper from these questions"; "My question banks" (list, view / edit / leave out, delete, Build paper dialog → PDF or Word).
- Backend `/api/v1/question-banks`: create from reviewed questions, list, questions with filters, PATCH, DELETE, `build-paper`. Owner always from the token; bank tables go through the teacher's own JWT, so RLS decides.
- Papers render through the existing export (4 picker templates, all 14 pass), with real section headings, the answer key on its own page, and "— (no answer in the source PDF)" where the bank has none. Saved to Test History.
- Contest flow: parser files have zero diff; its full output on both fixtures is identical to the pre-change baseline (test added).

**Fixture numbers** (`scripts/qb_extract_check.py`, recorded parser replies)
| PDF | pages | questions | mcq / tf / short / long | needs_review | answers matched | duplicates |
|---|---|---|---|---|---|---|
| Economics (real, recorded live 24 Oct) | 5 | 50 | 50 / 0 / 0 / 0 | 0 | 50 | 0 |
| Business Studies (**synthetic; reply hand-written**) | 2 | 16 | 6 / 4 / 4 / 2 | 1 | 11 | 0 |

**Tests** Backend 437 passed, 1 failed, 1 skipped — the failure is `test_cors_attempt_header`, which fails the same way on untouched `main` in a full run and passes alone. 113 tests are new. Frontend: 275 passed (20 new), `vite build` OK, `tsc` 40 errors = the 40 already on `main`, none in new files. RLS: 43 checks pass on a local throw-away Postgres.

**To run:** `supabase/migrations/33_question_bank.sql` (frontend repo), then `notify pgrst, 'reload schema';`. Rollback file next to it.
**Env / config:** none required. Optional `QUESTION_BANK_MAX_BANKS` (default 50). Order: migration → backend → frontend.
**Demo steps:** `docs/question-bank-manual-test.md` (start with section 1, the live read).

**Decisions that differ from the brief**
- A written question with no model answer is *not* marked needs_review (otherwise no long-answer section could be built); MCQ / true-false without an answer is.
- An AI-guessed answer the teacher did not confirm is saved as *no answer*.
- `export_service.py` has two switches used only by bank papers; 160 existing export variants compared before / after: identical text.
- `main.py` exposes one response header (`X-Paper-Meta`); bank papers do not count against the plan's paper quota.

**Do not claim in front of the customer**
1. Hindi papers as PDF — the PDF fonts cannot print Hindi; the app refuses and asks for Word. Word output was checked as text, not opened in Word.
2. "₹" prints as "Rs." (existing exporter behaviour).
3. True/false detection and the year tag are rules, not AI: a true/false statement with no key answer is filed as short answer; a year is found only in a bracket like "[CBSE 2023]" or in the paper title.
4. Marks: the parser gives 1 when the PDF prints none, so "marks from the PDF" is not reliable. The paper uses the marks typed in the dialog.
5. Diagrams, tables as images, match-the-column, case studies with sub-parts: not supported. Assertion–Reason comes through as MCQ.
6. Limits: 15 MB, 30 pages (not 40), 10 imports a day, 300 questions per bank, 50 banks.
7. The existing reader can swap a word (seen: "for" ↔ "of" on 24 Oct) — the teacher's review is the safeguard, the bank stores what was reviewed.
8. Sharing a bank with the institute: the database allows read-only sharing, nothing in the app turns it on.
9. No chapter or difficulty filter, no "pattern learning". Random pick is plain random.
10. The Modern template prints an "a4ai" mark and, with no duration given, guesses a time from the marks.
