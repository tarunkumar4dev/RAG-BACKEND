# Real handwritten copies (G3 test set)

Copies from partner institutes, used to measure the checker against **teacher marks**. These are minors' answer
sheets: **they are never in git — not the originals, not the anonymised versions, not in any repo.** They live in
a local folder outside the repo (backed up to the owner's Google Drive), pointed to by `REAL_COPIES_DIR`.
This README and `scripts/anonymise_copies.py` are the only parts in the repo.

```
$REAL_COPIES_DIR/              e.g. C:\Users\login\a4ai_real_copies  (NOT inside a repo folder — the script refuses)
  raw/                         originals, exactly as received — never uploaded anywhere
    <any folder name>/         one folder per student copy
      pages/                   one PDF, or photos (jpg/png/webp) named in page order (01.jpg, 02.jpg …)
      rubric.json              marking scheme in the checker's rubric format   ─┐ one of
      test.json                {"test_id": "<uuid of the saved test in a4ai>"} ─┘ the two
      expected.json            teacher marks (format below)
      redact.json              optional: extra boxes to black out (format below)
    _mapping.json              copy_NN → original folder name (written by the script)
  anonymised/                  output of the script — what tests read
    copy_01/
      pages/p01.jpg …          normalised page images, name/roll box blacked out, no metadata
      rubric.json | test.json
      expected.json            student / teacher identifiers removed
      anonymise.json           what was blacked out (page + box), script version
```

## expected.json (teacher marks)

```json
{
  "max_marks": 40,
  "total": 35,
  "class": "10",
  "subject": "Science",
  "marked_by": "teacher",
  "questions": [
    {"q": "1", "max": 1, "marks": 1},
    {"q": "7", "max": 2, "marks": 1, "note": "(ii) blank"},
    {"q": "15", "max": 3, "marks": 3, "note": "both OR options attempted, first graded"}
  ]
}
```

- `q` uses the paper's numbering ("15", "15-OR" for the internal-choice alternative that the student answered).
- Marks are what the teacher actually gave (½ steps). Questions not attempted: `"marks": 0, "note": "not attempted"`.
- No names, roll numbers, school names or teacher names anywhere in this file — the script also strips the keys
  `name, student_name, roll, roll_no, school, institute, teacher, teacher_name, phone, email, parent_*`.

## What gets blacked out

1. Always: the top band of **page 1** (default 22 % of the page height — the name / class / roll / school header
   of a CBSE-style sheet). Change with `--header 0.30` if the header is taller.
2. `--every-page-header`: the same band on every page (sheets with "Name ___" printed on each page).
3. Anything listed in `redact.json` (normalised coordinates, 0–1, on the page image as uploaded — the script
   does not rotate pages; a page scanned upside down has its header at the bottom, so add a box for it):
   ```json
   [{"page": 1, "x0": 0.60, "y0": 0.25, "x1": 0.98, "y1": 0.32}]
   ```
Boxes are filled solid (not blurred: blur can sometimes be read back). Pages are re-encoded, so photo EXIF
(camera, GPS, time) is dropped.

## Workflow

1. PowerShell: `$env:REAL_COPIES_DIR = "C:\Users\login\a4ai_real_copies"` (any folder outside the repos).
2. Put each copy in `$REAL_COPIES_DIR\raw\<folder>\` as above.
3. `python scripts/anonymise_copies.py` (from `test-generator-backend/`).
4. **Look at every `anonymised\copy_NN\pages\p01.jpg`** — the header band is a default, not a detector. Add a
   `redact.json` and re-run with `--force` if anything identifying is still visible.
5. Tests that use real copies read `$REAL_COPIES_DIR/anonymised` and are **skipped** when the variable is not set.

Guards: `.gitignore` ignores everything in `tests/fixtures/real/` except this README; the script refuses a
`REAL_COPIES_DIR` inside the repo; `tests/unit/test_real_fixtures_guard.py` fails if anything but this README is
tracked here.
