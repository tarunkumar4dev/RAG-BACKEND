# Check Copies — eval cases

`scripts/eval_checker.py` grades every case with the real checker (same rubric + grading code as
production, no database) and compares it with the teacher's marks. Every checker change is judged by it.

## ⚠️ Privacy: REAL copies never go in this repo

This repository is **public**. Only **synthetic** copies (typed in a handwriting font, fictional names) live
in `scripts/eval_cases/checker/` and are committed. Their `expected.json` says `"synthetic": true`.

**Real student copies go in `scripts/eval_cases_private/`** — that folder is in `.gitignore`, so git never
picks it up. Before every commit, `git status` must not list a real copy. If one slips in, do not push;
remove it from the commit and tell the team.

## Add a real copy in 5 minutes

1. Make a folder (the folder name is the case name; no student name in it):

   ```
   scripts/eval_cases_private/class10_trigo_copy07/
   ```

2. Put in:
   - `question_paper.pdf` (or `.jpg` / `.png`)
   - `answer_key.pdf` — optional; without it the AI writes its own scheme (less reliable)
   - the copy: either `copy.pdf`, or the phone photos as `pages/1.jpg`, `pages/2.jpg`, … (page order = name order)

3. Write `expected.json` with the teacher's marks:

   ```json
   {
     "synthetic": false,
     "student": "copy07",
     "note": "anything a reader should know about this copy",
     "strictness": "medium",
     "total": 23.5,
     "groups": {
       "A": {"questions": ["1", "2", "3", "4", "5"], "marks": 4, "note": "MCQs; which one is wrong not recorded"}
     },
     "questions": {
       "6":  {"marks": 2},
       "7":  {"marks": 1.5, "note": "unit missing"},
       "8":  {"marks": 3, "expect_review": true, "note": "answer has no question number"},
       "9":  {"marks": 0, "expect": "crossed_out_zero", "note": "whole answer struck through"},
       "10": {"marks": 4, "expect": "spans_pages", "pages": [2, 3]}
     }
   }
   ```

   - `total` must equal the sum of all marks (the script warns otherwise).
   - Every question of the paper appears exactly once: either in `questions` or inside one group.
   - `expect_review: true` = the checker **should** send this answer to the teacher for review
     (missing / wrong question label, unreadable, unusual). Default `false`.
   - `expect`: `"crossed_out_zero"` = struck-out work must not earn marks (pass if AI ≤ teacher marks);
     `"spans_pages"` = the answer continues onto another page (pass if AI = teacher marks).

4. Run it:

   ```
   cd test-generator-backend
   python scripts/eval_checker.py --cases-dir scripts/eval_cases_private --cases class10_trigo_copy07
   ```

## How the numbers are counted

- A **group** (e.g. Section A MCQs) counts as **one unit** in per-question MAE, exact-match % and
  over/under-credit — use a group when the teacher's per-question marks are not known.
- **Total-marks MAE** = mean of |AI total − teacher total| over all graded copies.
- **Review recall** = flagged ÷ should-be-flagged. **Review precision** = right flags ÷ all flags.
  **New wrong flags** = flagged although the teacher's answer needs no review.
- **Over-credit** = AI gave more than the teacher (worse for students' trust in marks); **under-credit** = less.
- **₹** comes from the token counts and the price table at the top of `eval_checker.py`
  (output price includes thinking tokens). Prices marked UNVERIFIED are estimates.

## PASS / FAIL (printed at the end of every run)

1. every copy graded (no error, no timeout)
2. every copy's total within ±1 of the teacher
3. crossed-out work earns nothing (`crossed_out_zero`)
4. an answer split across pages is graded as one (`spans_pages`)
5. every copy < 90 s, and p95 < 90 s
6. every `expect_review: true` answer is flagged

## Useful flags

| flag | what it does |
|---|---|
| `--cases all` / `--cases a,b` | which cases |
| `--runs 3` | repeat; lists every question whose marks or flag changed between runs |
| `--rebuild-rubric` | build the rubric again (otherwise cached); with `--runs 3` also reports rubric stability |
| `--allow-fallback` | allow fallback models on 429/503 (default OFF: a copy graded by another model is "contaminated") |
| `--call-timeout 180` | seconds per Gemini call; a timeout fails that copy, the run continues |
| `--app-root <dir>` | grade with another version of `app/` (e.g. an older commit via `git archive`) |
| `--compare A.json B.json` | per-question diff of two saved runs from `eval_runs/` |

Run outputs (`eval_runs/`) are git-ignored.
