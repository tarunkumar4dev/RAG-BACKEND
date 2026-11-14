# Question Bank — manual test (about 15 minutes)

Branch `feat/question-bank` in both repos. Nothing is deployed and migration 33 is not applied yet.

## 0. Before you start

1. **Database** — Supabase SQL editor, run `supabase/migrations/33_question_bank.sql` (frontend repo), then
   `notify pgrst, 'reload schema';`. Run the three check queries at the bottom of that file.
   Undo: `33_question_bank_rollback.sql` (deletes all banks).
2. **Backend** — `cd test-generator-backend`, `uvicorn app.main:app --port 8000` on `feat/question-bank`.
3. **Frontend** — on `feat/question-bank`: `npx vite dev --port 8080 --strictPort` (8080 is the only port login works on).
4. Log in as a teacher and open `/dashboard/test-generator?tab=contest-pdf`.

Fixtures (backend repo, `test-generator-backend/tests/fixtures/`):
`Class12_Economics_ForeignExchangeRate_50MCQs_EducationBeast.pdf` and `question_bank/mixed_business_studies.pdf`.

## 1. First, the one thing I could not run: a live read of the second fixture

```
cd test-generator-backend
python scripts/qb_extract_check.py --live
```

Expected for `mixed_business_studies.pdf`: 16 questions, mcq 6 / true_false 4 / short 4 / long 2, needs_review 1,
answers matched 11. If true_false comes out lower, the live model read those statements differently from my
hand-written stand-in — tell me the numbers before the demo.

## 2. Economics PDF: upload → bank → paper

| Step | Expect |
|---|---|
| Upload the Economics PDF, wait 1–2 min | Review screen, 50 questions, answer key 50/50 |
| Look at the bottom bar | "Create contest · 50 questions" exactly as before, plus two new buttons |
| "Save as Question Bank" → Save | Toast "Saved … 50 questions"; button shows "✓ Saved as question bank" |
| Scroll to "My question banks" | One row: name, "50 questions", today's date |
| Open it → "Build paper" | Dialog; "In this bank: 50 MCQ · 0 True / False · 0 Short answer · 0 Long answer" |
| Sections: MCQ 20 × 1 and MCQ 5 × 2, answer key on, template Modern, PDF → Download | A PDF: Section A (20) and Section B (5), then an ANSWER KEY page. Check 3 answers against the original PDF's key. Check a question with "$" and "₹" (₹ prints as "Rs.") |
| Set MCQ to 60 | Red line "Section A needs 60 MCQ questions, bank has 50." and the button is disabled |
| Same paper as Word | A .docx with the paper, then the answer key after a page break |
| Test History page | The paper is listed (saved, 25 questions, 30 marks) |

Persistence check (SQL editor; replace the email):

```sql
select b.name, b.status, count(q.*) as questions, count(q.answer) as with_answer
from question_banks b join bank_questions q on q.bank_id = b.id
where b.owner_id = (select id from auth.users where email = 'YOUR_LOGIN_EMAIL')
group by 1, 2;
```

## 3. Business Studies PDF: mixed types, missing answers, Hindi

| Step | Expect |
|---|---|
| Upload it, review | 16 questions. Q6 "No answer"; Q13–Q16 "No answer" (written questions). "Create contest" is blocked by those, as before |
| "Build paper from these questions" → Save | The bank is saved and the Build paper dialog opens directly |
| Dialog | MCQ 5 (Q6 needs review), True / False 4, Short 4, Long 2; a year list (2023, 2022, 2021); an amber Hindi note |
| Default sections, Word → Download | Paper + answer key. Q14 is in Hindi. Short/long questions without a model answer say "— (no answer in the source PDF)" |
| Same, PDF | If the Hindi question is picked: clear message "…PDF export cannot print Hindi yet. Choose Word (DOCX) instead." |
| Year 2023, MCQ 1, True/False 1, Long 1 | Only the three "[CBSE 2023]" questions |
| In the bank: edit Q6, tap the correct option, Save | "Needs review" disappears; MCQ count in the dialog becomes 6 |
| Untick a question's checkbox | "Left out"; it is never used in a paper |
| Delete the bank | Row disappears. The papers stay in Test History |

## 4. Things that must still work

- Fix Q33/Q49-style problems in a paper and create a contest from the same review screen — unchanged flow.
- A second teacher account sees no banks of the first one.
- A scanned PDF or one over 30 pages / 15 MB: the existing error from the import, no crash.
- Logged out: the two new buttons open the login dialog; "My question banks" is not shown.

## 5. Automated checks (already green)

```
cd test-generator-backend && python -m pytest tests/unit -q          # backend
python scripts/qb_contest_flow_snapshot.py                           # contest flow identical on both fixtures
python scripts/qb_build_sample_papers.py out_dir                     # real PDFs / DOCX to look at, offline
cd <frontend> && npx vitest run && npx tsc -p tsconfig.app.json --noEmit && npx vite build
python supabase/tests/33_question_bank_test.py                       # RLS on a throw-away local Postgres
```
