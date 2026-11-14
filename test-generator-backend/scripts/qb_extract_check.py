"""
Question Bank check on the fixtures: existing contest parser → bank normalisation → numbers.

    python scripts/qb_extract_check.py            # offline: the recorded Gemini replies (GEMINI_MOCK), no network
    python scripts/qb_extract_check.py --live     # real Gemini calls through the existing parser (needs .env;
                                                  # about 2 calls per fixture)
    python scripts/qb_extract_check.py --live --pdf some.pdf

Prints per fixture: pages, questions found, per-type counts, needs_review, answers matched, duplicates removed.
"""

import argparse
import os
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--pdf", type=Path, default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.live:
        from dotenv import load_dotenv
        load_dotenv(BACKEND / ".env", override=False)
        os.environ["GEMINI_MOCK"] = "0"
    elif args.pdf:
        print("--pdf needs --live (there is no recording for an arbitrary PDF)")
        return 2

    import qb_contest_flow_snapshot as snap          # sets GEMINI_MOCK=1 unless --live set it to 0 above
    from app.services import contest_import_service as ci
    from app.services import question_bank_service as qb

    cases = {args.pdf.stem: args.pdf} if args.pdf else {name: pdf for name, (pdf, _) in snap.CASES.items()}
    for name, pdf in cases.items():
        pages = ci.check_pdf(str(pdf))
        start = time.monotonic()
        if args.live:
            result = ci.import_paper(str(pdf), pages, start + 270)
        else:
            result = snap.run_case(name)
        rows, s = qb.normalise_questions(qb.payload_from_import(result), result.get("title"))
        key = sum(1 for q in result["questions"] if q["answer_source"] == "key")
        print(f"\n{pdf.name}  ({'live' if args.live else 'recorded reply'}, {time.monotonic() - start:.1f}s)")
        print(f"  pages               {pages}")
        print(f"  questions found     {len(result['questions'])}  (saved to bank: {s['saved']})")
        print(f"  per type            " + ", ".join(f"{t} {n}" for t, n in s["per_type"].items()))
        print(f"  needs_review        {s['needs_review']}")
        print(f"  answers matched     {s['answers_matched']}  (parser matched {key} to the PDF's key; "
              f"AI guesses dropped {s['ai_answers_dropped']}, unresolved {s['unresolved_answers']})")
        print(f"  duplicates removed  {s['duplicates_removed']}")
        print(f"  with a year tag     {sum(1 for r in rows if r['exam_year'])}")
        print(f"  parser warnings     {len(result['warnings'])}")
        for w in qb.stats_warnings(s):
            print(f"  note: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
