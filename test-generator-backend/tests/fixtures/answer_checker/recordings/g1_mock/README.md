# g1_mock — HAND-WRITTEN mock replies (not live Gemini recordings)

Used with `GEMINI_MOCK=1 GEMINI_MOCK_DIR=<this folder>` by the G1 tests and `scripts/g1_marked_copy_demo.py`.

- `RUBRIC.json` — the RUBRIC call's reply, written from `answer_key.pdf` (17 questions, Q15 OR-group, 40 marks).
- `GRADE.json` — the GRADE call's reply for `answer_sheet_messy.pdf` in the G1 (current) prompt format, incl. the
  `pages` report (student page numbers 1,2,4,3,5,6,7; image 2 = student page 4 needs 180°). Verdicts follow
  `expected_evaluation.json` (total 35). Positions (`page_index` = image sent, `y_start` 0–1000 on the upright
  page) were read off the fixture by hand.

They test the CODE around Gemini (scoring, page reorder/rotation, remapping, rendering), not the model.
Live recordings replace them in G3 (`scripts/eval_live.py`, only when the owner runs it).
