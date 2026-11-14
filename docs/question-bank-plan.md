# Question Bank → Paper Builder — plan (G0 recon)

14 Nov 2026 · branch `feat/question-bank` in both repos, cut from `origin/main`
(backend `7cba4eb`, frontend `2895673`). Nothing here touches `main`, the live database or a deployment.

## 1. What already exists: "Contest from PDF"

| | Where | What it does |
|---|---|---|
| (a) Frontend | `src/pages/TestGeneratorPage.tsx` tab `contest-pdf` → `src/components/contest/ImportContestFromPdf.tsx`, API client `src/lib/contestImportApi.ts` | Phases `pick → uploading → starting → processing → review → failed`. Review = editable `QuestionReviewCard` list + sticky "Create contest" bar → `CreateContestModal`. Draft is kept in `sessionStorage`. |
| (b) Backend | `app/routers/contest_import_router.py` (`POST /api/v1/contests/import-pdf {path}` → 202 `{job_id}`, `GET …/{job_id}`), `app/services/contest_import_service.py` (`check_pdf`, `import_paper`, `build_result`, `check_key`), `app/services/contest_import_gemini.py` (REST client) | The browser uploads the PDF to the private `answer-keys` bucket; the API only gets the path (Vercel caps request bodies at about 4.5 MB). The parse runs as a background task of the same invocation with a 270 s budget; the row in `contest_import_jobs` carries progress and the result. |
| (c) Output shape | `app/schemas/contest_import.py` `ImportResult` | `title, subject, class_grade, institute, board, duration_minutes, total_marks, answer_key_found, answer_key_count, questions[], warnings[]`. Question: `question_text, question_type (MCQ / Short / Long), options[{label, text}], correct_answer (bare label "B" for MCQ, text for written, or null), explanation, marks (int ≥ 1), difficulty, chapter, source_number, answer_source (key / ai / missing), usable`. Warning: `{question, code, message}`. |
| (d) Limits | `contest_import_service.py` | 15 MB, 30 pages, must be a real unprotected PDF. 10 imports per teacher per day (`contest_import_jobs`), 10 POSTs per minute. The PDF goes to Gemini as a document, so **scanned papers are read too** (no text layer needed). |
| (e) Answer key | `build_result` / `_key_entries` | Gemini transcribes the key rows; code maps them by question number. Table beats grid; disagreements are warnings (`KEY_GRID_MISMATCH`, `KEY_TEXT_MISMATCH`, `KEY_LETTER_NOT_IN_OPTIONS`, `KEY_COUNT_MISMATCH`). No key entry → `correct_answer = null`, `answer_source = "missing"`, `ANSWER_MISSING`. **If the PDF has no key at all, the model's own answer is used** (`answer_source = "ai"`, `AI_ANSWER` warning). An optional second pass solves keyed MCQs and only flags (`KEY_MISMATCH_SUSPECT`); it never changes the key. |

Model name: `CONTEST_IMPORT_MODELS` env (existing config). No model is hardcoded by this feature — it makes no Gemini call of its own.

## 2. Decisions

1. **The parser is not changed at all.** `qtype`, `exam_year`, `source_tag`, `needs_review` and `content_hash` are derived in the new bank layer from the parser's existing output. Zero diff in the contest code is the strongest way to keep that flow identical. Cost: true/false and the year tag come from rules, not from the model (section 5).
2. **`POST /question-banks` takes `{name, questions[]}` (reviewed JSON), not the PDF.** Smaller and safer than a second server-side parse: no new Gemini spend, no second long request, and what is saved is exactly what the teacher saw and corrected.
3. **Answers are never invented.** A bank stores an answer only when it came from the PDF's key (`answer_source = key`) or the teacher set / confirmed it in review (`teacher`). An unconfirmed AI guess (`ai`) is saved with `answer = null` and `needs_review = true`. For an MCQ the answer must resolve to one of its options, otherwise null.
4. **Bank tables are read and written with the caller's JWT** (`get_supabase_for_user`, the `require_caller` dependency the import router uses), so RLS is the enforcement, not only backend code. `owner_id` is always the token's user; the request models have no user-id field and reject unknown fields.
5. **Test history** is written like the AI generator does it: service-role insert into `tests` with `teacher_id` = token user, plus `questions` rows in the `/test-generator/save` shape.
6. **Answer key = its own page(s) at the end of the same file** (changed in G3 from "a second endpoint": one request, one download). `include_answers=True` in the existing exporter also highlights the correct option inside the question paper, which would turn the student paper into a teacher copy — so the paper is rendered without answers and the existing standalone answer-key export is appended (PDF pages merged; DOCX body appended after a page break). A question with no answer in the bank prints "— (no answer in the source PDF)".
7. **Two switches in `export_service.py`** (G3 / G5), both off for every other caller — 160 existing export variants (5 templates × sections × subject × answer modes × PDF/DOCX, plus answer keys) were dumped as text before and after: identical.
   - `section_meta` on the questions: the exporter's section titles and instructions are fixed CBSE text ("Section A — 1 mark each — MCQ / Assertion-Reason", "20 MCQs", "5 Sections", "Use of calculator is not permitted", "CBSE Pattern", and a hard-wired Accountancy layout). A bank paper brings its own headings and prints none of those claims.
   - `literal_text()`: bank text is a transcription, not generator output. Without it "$25 to $30" printed as "25 to 30", and `x_1`, `a^2`, `{…}`, `<b>` were treated as markup.
8. Migration number **33** (31 is the last on `main`; 32 is taken by an unmerged branch).

## 3. Files

Backend, new: `app/schemas/question_bank.py`, `app/services/question_bank_service.py`, `app/routers/question_bank_router.py`, `scripts/qb_extract_check.py`, `scripts/qb_contest_flow_snapshot.py`, `scripts/make_question_bank_fixture.py`, `tests/unit/test_question_bank_*.py`, `tests/integration/test_question_bank_rls.py`, `tests/fixtures/question_bank/`.
Backend, modified: `app/main.py` (router), `app/core/rate_limit.py` (limits), `app/services/export_service.py` (custom section headings only).
Backend, reused unchanged: the whole contest import, `app/core/auth.py`, `app/core/database.py`, `smart_assignment_router.require_caller / with_retry`, `test_generator.py` (`VALID_TEMPLATES`, the `tests` row shape), `generate_pdf / generate_docx / generate_answer_key_*`.
Frontend, new: `supabase/migrations/33_question_bank.sql` (+ rollback), `src/lib/questionBankApi.ts`, `src/components/contest/QuestionBankPanel.tsx`, `BuildPaperDialog.tsx`.
Frontend, modified: `ImportContestFromPdf.tsx` only (two extra buttons after review, the "My question banks" section).

## 4. Vercel limits

- `vercel.json` sets no `maxDuration`; the project default is 300 s (verified from the Vercel API on 13 Nov). Every new endpoint is a few database calls plus one ReportLab / python-docx render — seconds. No background work, no Gemini.
- Request and response bodies cap at about 4.5 MB. `POST /question-banks` is limited to 300 questions and bounded field lengths (well under 1 MB); a text-only paper PDF is tens of KB.
- Rate limits are in memory per instance (best effort), same as every other route.

## 5. Risks

- **Second fixture is synthetic and its recording is hand-written.** Only one real paper is in the repo (Economics, 50 MCQs, recorded live on 24 Oct). `mixed_business_studies.pdf` is made by `scripts/make_question_bank_fixture.py`; a live Gemini read of it could not be run in this session (the session may not load the backend `.env`). The live command is in the report.
- True/false is detected by rule (two options True/False, or a written question whose key answer is True/False, or a "True or False" stem). The parser itself has no such type.
- Year tag comes from text such as "[CBSE 2023]" in the question or the paper title. A paper that states the year only in a logo or footer gets no year.
- PDF fonts are Helvetica / Times: Hindi text and "₹" may not print in the PDF export (to be checked in G3; DOCX is not affected).
- Subject "Accountancy" routes to the exporter's own CBSE accountancy layout — to be checked in G3.
- `tests/unit/test_cors_attempt_header.py` fails in the full run on untouched `origin/main` (passes alone) — existing, order-dependent, not from this work. Baseline: 324 passed, 1 failed, 1 skipped.
- A bank saved from one browser tab is tied to the user, not to an institute: `institute_id` stays null in the MVP (the column and the read-only member policy exist, nothing sets it).

## 6. Regression guard for the contest flow

`tests/fixtures/question_bank/contest_flow_baseline/*.json` = the full `import_paper` result for both fixtures, written on untouched `origin/main` before any feature code. `tests/unit/test_question_bank_contest_regression.py` requires an exact match after every gate, and that the contest import files have no diff against `origin/main`.
