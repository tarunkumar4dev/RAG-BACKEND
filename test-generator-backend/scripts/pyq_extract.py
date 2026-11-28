"""
Read CBSE board papers (PDF) into reviewable JSON — step 1 of putting PYQs into the Test Builder library.

    .venv/bin/python scripts/pyq_extract.py --subject Accountancy --class 12 --year 2026 \
        ~/Downloads/Accountancy/Accountancy/67_4_3_Accountancy_cl12.pdf [more PDFs…]

One JSON per paper in data/pyq/<subject>_<class>_<year>/<paper code>.json (67_4_3.json). Review / fix it by
hand, then load with scripts/pyq_load.py. An existing JSON is skipped unless --force, so re-running never
overwrites a file you corrected.

Each paper is read by Gemini (CONTEST_IMPORT_MODELS, key CHECKER_GEMINI_API_KEY else GEMINI_API_KEY) in four
calls by question range, so one reply never runs out of output tokens. The papers are bilingual: every
question comes back with its English and its Hindi version, both copied from the paper, never translated.
Internal-choice questions ("7 (a) OR 7 (b)") become two questions, "7(a)" and "7(b)"; Part B has two options
(Analysis of Financial Statements / Computerised Accounting), both are kept.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

API = "https://generativelanguage.googleapis.com/v1beta/models"
DATA_DIR = ROOT / "data" / "pyq"

# Chapter names exactly as stored in ncert_questions, so a PYQ lands in the builder's existing chapter list.
# Topic hints tell the model which syllabus unit goes where (NCERT keeps goodwill and change in profit-sharing
# ratio inside the admission chapter).
CHAPTERS = {
    ("accountancy", "12"): {
        "Accounting for Partnership Basic Concepts":
            "partnership deed, interest on capital/drawings/loan, salary/commission, P&L appropriation account, "
            "fixed/fluctuating capital, guarantee of profit, past adjustments",
        "Reconstitution of a Partnership Firm – Admission of a Partner":
            "goodwill valuation (average/super profit/capitalisation), change in profit-sharing ratio, sacrificing/"
            "gaining ratio on change, revaluation and reserves on change in ratio, admission, new ratio, premium "
            "for goodwill, capital adjustment on admission",
        "Reconstitution of a Partnership Firm – RetirementDeath of a Partner":
            "retirement or death of a partner, gaining ratio, executors, deceased partner's share of profit, "
            "retiring partner's loan account, workmen compensation on retirement",
        "Dissolution of a Partnership Firm": "dissolution, realisation account, settlement of accounts",
        "Accounting for Share Capital":
            "issue of shares, pro-rata allotment, calls in arrears/advance, forfeiture and reissue of shares, "
            "authorised/issued/subscribed capital, share capital in the balance sheet",
        "Issue and Redemption of Debentures":
            "issue of debentures (par/premium/discount), loss on issue, interest on debentures, debentures as "
            "purchase consideration, securities premium use, types of debentures",
        "Financial Statements of a Company": "Schedule III balance sheet heads and sub-heads",
        "Analysis of Financial Statements":
            "comparative and common-size statements, tools/users/objectives of analysis",
        "Accounting Ratios": "liquidity, solvency, activity/turnover and profitability ratios",
        "Cash Flow Statement": "cash flow statement, operating/investing/financing activities, cash equivalents",
        "Computerised Accounting":
            "Part B Option II: computerised accounting system, accounting software, Excel functions, charts, "
            "errors, codes, vouchers, data vault/security",
    },
}

RANGES = [(1, 16), (17, 22), (23, 26), (27, 34)]

TEXT_RULES = (
    "Copy the wording exactly as printed (do not paraphrase, do not translate). Write amounts with the rupee sign "
    "and Indian grouping as printed, e.g. ₹ 2,10,000. Fractions as 1/5, superscripts as plain text (1st April). "
    "Put every table (balance sheet, data table) inside the text as a markdown table at the place it appears: "
    "a header row, then |---| separator, then one row per printed line; keep sub-items (e.g. capitals of each "
    "partner, 'Less: Provision') as their own rows and put sub-amounts in the amount column of that row. "
    "List sub-parts (i), (ii)… on separate lines. Do not include the question number or the marks in the text."
)


def schema(chapters: list) -> dict:
    version = {
        "type": "OBJECT",
        "properties": {
            "text": {"type": "STRING"},
            "options": {"type": "ARRAY", "items": {"type": "STRING"}},
        },
        "required": ["text", "options"],
    }
    return {
        "type": "OBJECT",
        "properties": {
            "questions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "question_number": {"type": "STRING"},
                        "part": {"type": "STRING", "enum": ["A", "B-I", "B-II"]},
                        "chapter": {"type": "STRING", "enum": chapters},
                        "question_type": {"type": "STRING",
                                          "enum": ["mcq", "short_answer", "long_answer", "case_study"]},
                        "marks": {"type": "INTEGER"},
                        "difficulty": {"type": "STRING", "enum": ["easy", "medium", "hard"]},
                        "en": version,
                        "hi": version,
                        "answer": {"type": "STRING"},
                    },
                    "required": ["question_number", "part", "chapter", "question_type", "marks", "difficulty",
                                 "en", "hi", "answer"],
                },
            },
        },
        "required": ["questions"],
    }


def prompt(first: int, last: int, chapters: dict) -> str:
    chapter_lines = "\n".join(f"- {name}: {hint}" for name, hint in chapters.items())
    return f"""This PDF is a bilingual CBSE board question paper: each page is printed in Hindi and then in English.
Return every question numbered {first} to {last}, in paper order.

Rules:
- A question with an internal choice ("(a) … OR (b) …") is TWO questions: question_number "7(a)" and "7(b)".
  A question without a choice keeps its plain number, e.g. "13".
- Part B has two options (Option I and Option II) that reuse the numbers 27–34: return both, part "B-I" for
  Option I and "B-II" for Option II. Part A questions have part "A".
- A case-based question with sub-questions (i)…(vi) is ONE question of type case_study with all sub-questions and
  their options inside the text; its marks are the total of the sub-questions.
- question_type: mcq (one-mark choice questions, including assertion–reason and statement I/II questions),
  short_answer (3 marks), long_answer (4 or 6 marks), case_study.
- marks: as printed on the right margin.
- en: the English version; hi: the Hindi version (Devanagari, exactly as printed on the Hindi page).
  options: the four choices without the (A)/(B) labels, in order; [] when there are none (same for hi).
- {TEXT_RULES}
- chapter: the one chapter below that the question tests:
{chapter_lines}
- difficulty: easy (direct recall or one-step), medium (a few steps), hard (long multi-step working or a
  common trap), judged for a Class 12 student.
- answer: for mcq, the letter of the correct option, a colon, and one short line of working, e.g.
  "C: 3,00,000 × 8% = 24,000 …". For every other type, a short answer key with the key figures (final balances,
  ratios, amounts), not full journal entries. Work it out carefully; if the printed question is ambiguous say so.
"""


def call_gemini(pdf_b64: str, text: str, body_schema: dict, models: list, key: str) -> dict:
    body = {
        "contents": [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "application/pdf", "data": pdf_b64}}, {"text": text}]}],
        "generationConfig": {
            "responseMimeType": "application/json", "responseSchema": body_schema,
            "temperature": 0, "seed": 7, "maxOutputTokens": 60000,
            "thinkingConfig": {"thinkingBudget": 8192},
        },
    }
    last_error = None
    for attempt in range(8):
        for model in models:
            try:
                r = httpx.post(f"{API}/{model}:generateContent", json=body, timeout=600,
                               headers={"x-goog-api-key": key})
            except httpx.HTTPError as e:
                last_error = f"{model}: {type(e).__name__}"
                continue
            if r.status_code == 200:
                data = r.json()
                cand = (data.get("candidates") or [{}])[0]
                parts = (cand.get("content") or {}).get("parts") or []
                reply = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                if cand.get("finishReason") not in (None, "STOP"):
                    last_error = f"{model}: finishReason {cand.get('finishReason')}"
                    continue
                return json.loads(reply)
            last_error = f"{model}: HTTP {r.status_code} {r.text[:200]}"
            if r.status_code not in (404, 429, 500, 503, 504):
                raise RuntimeError(last_error)
        time.sleep(min(30 * (attempt + 1), 180))  # 503 "high demand" spells last minutes
    raise RuntimeError(f"Gemini failed: {last_error}")


def in_range(questions: list, first: int, last: int, seen: set) -> list:
    """The model sometimes adds questions outside the asked range (Part B again): keep each number+part once."""
    kept = []
    for q in questions:
        m = re.match(r"\s*(\d+)", str(q.get("question_number", "")))
        key = (str(q.get("question_number", "")).replace(" ", ""), q.get("part"))
        if m and first <= int(m.group(1)) <= last and key not in seen:
            seen.add(key)
            kept.append(q)
    return kept


def paper_code(pdf: Path) -> str:
    m = re.match(r"(\d+)_(\d+)_(\d+)", pdf.name)
    if not m:
        raise SystemExit(f"{pdf.name}: expected a name like 67_4_3_Accountancy_cl12.pdf")
    return "/".join(m.groups())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdfs", nargs="+", type=Path)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--class", dest="class_grade", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--exam", default="CBSE")
    ap.add_argument("--force", action="store_true", help="overwrite an existing JSON")
    args = ap.parse_args()

    chapters = CHAPTERS.get((args.subject.lower(), args.class_grade))
    if not chapters:
        raise SystemExit(f"No chapter list for {args.subject} class {args.class_grade}: add it to CHAPTERS")
    key = os.getenv("CHECKER_GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("Set CHECKER_GEMINI_API_KEY or GEMINI_API_KEY in .env")
    models = [m.strip() for m in os.getenv("CONTEST_IMPORT_MODELS", "gemini-3-flash-preview,gemini-3.8-flash")
              .split(",") if m.strip()]
    out_dir = DATA_DIR / f"{args.subject.lower()}_{args.class_grade}_{args.year}"
    out_dir.mkdir(parents=True, exist_ok=True)
    body_schema = schema(list(chapters))

    for pdf in args.pdfs:
        code = paper_code(pdf)
        out = out_dir / f"{code.replace('/', '_')}.json"
        if out.exists() and not args.force:
            print(f"{code}: {out.relative_to(ROOT)} exists, skipped (--force to redo)")
            continue
        pdf_b64 = base64.b64encode(pdf.read_bytes()).decode("ascii")
        questions, seen = [], set()
        for first, last in RANGES:
            t = time.time()
            reply = call_gemini(pdf_b64, prompt(first, last, chapters), body_schema, models, key)
            got = reply.get("questions") or []
            kept = in_range(got, first, last, seen)
            print(f"{code} Q{first}-{last}: {len(kept)} questions in {time.time() - t:.0f}s"
                  + (f" ({len(got) - len(kept)} outside the range dropped)" if len(got) > len(kept) else ""))
            questions.extend(kept)
        out.write_text(json.dumps({
            "exam": args.exam, "year": args.year, "subject": args.subject, "class_grade": args.class_grade,
            "paper": code, "source_pdf": pdf.name, "questions": questions,
        }, ensure_ascii=False, indent=1))
        print(f"{code}: {len(questions)} questions → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
