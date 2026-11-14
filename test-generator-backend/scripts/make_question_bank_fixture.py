"""
Builds tests/fixtures/question_bank/mixed_business_studies.pdf — the second Question Bank fixture.

A made-up 16-question paper (not a real institute's) with everything the Economics fixture does not have:
MCQ + True/False + short + long questions, marks per section, "[CBSE 2023]" year tags, one Hindi question, and an
answer key that is deliberately incomplete (Q6 has no key entry; Q13–Q16 have no model answer).

    python scripts/make_question_bank_fixture.py

Needs C:/Windows/Fonts/Nirmala.ttc for the Hindi line (skipped with a note when the font is missing).
"""

import sys
from pathlib import Path

import pymupdf

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "question_bank" / "mixed_business_studies.pdf"
FONT_DIR = Path("C:/Windows/Fonts")
HINDI_FONT = "Nirmala.ttc"

MCQS = [
    ("Which of the following is an economic activity? [CBSE 2023]",
     ["A mother cooking food for her family", "A teacher teaching in a school for a salary",
      "A person donating blood", "A student helping a friend with homework"]),
    ("The occupation in which people work for others and get remunerated in return is known as:",
     ["Business", "Profession", "Employment", "None of these"]),
    ("Which of the following is NOT a characteristic of business? [CBSE 2022]",
     ["Production or procurement of goods and services", "Sale or exchange of goods and services",
      "Wages or salaries", "Element of risk"]),
    ("Industries which provide support services to other industries are known as:",
     ["Primary industries", "Secondary industries", "Tertiary industries", "Commercial industries"]),
    ("Transfer of interest exists in the case of:",
     ["Profession", "Employment", "Business", "None of these"]),
    ("Which of the following cannot be classified as an auxiliary to trade?",
     ["Mining", "Insurance", "Warehousing", "Transport"]),
]
TRUE_FALSE = [
    "Business risk can be completely eliminated by taking insurance.",
    "Earning profit is the only objective of business. [CBSE 2023]",
    "Warehousing removes the hindrance of time.",
    "A lawyer practising in a court is engaged in a profession.",
]
SHORT = [
    "State any three characteristics of business activities.",
    "Distinguish between primary and secondary industries. [CBSE 2021]",
    "What is meant by business risk? Give two causes of business risk.",
    "व्यवसाय और पेशे में कोई दो अंतर लिखिए।",
]
LONG = [
    "Explain any five objectives of business, other than earning profit.",
    "Describe the various auxiliaries to trade and explain how each one removes a hindrance in the "
    "exchange of goods. [CBSE 2023]",
]
KEY_MCQ = ["b", "c", "c", "c", "c"]                      # Q6 deliberately missing
KEY_TF = ["False", "False", "True", "True"]
KEY_SHORT = {
    11: "An economic activity; production or procurement of goods and services; sale or exchange of goods "
        "and services (any three).",
    12: "Primary industries extract or produce natural resources (mining, farming); secondary industries "
        "process those materials into finished goods (manufacturing, construction).",
}


def build_html(hindi: bool) -> str:
    h = ['<h1>SUNRISE COMMERCE CLASSES</h1>',
         '<h2>Class XI — Business Studies</h2>',
         '<p class="c">Chapter Test: Nature and Purpose of Business · Previous Year Questions</p>',
         '<p class="c"><b>Time: 1 Hour 30 Minutes &nbsp;&nbsp;&nbsp; Maximum Marks: 32</b></p><hr/>']
    n = 0
    h.append('<h3>SECTION A — Multiple Choice Questions (Q.1–6 · 1 mark each)</h3>')
    for stem, opts in MCQS:
        n += 1
        h.append(f'<p class="q">{n}. {stem}</p>')
        h.append('<p class="o">' + "<br/>".join(f"({'abcd'[i]}) {o}" for i, o in enumerate(opts)) + "</p>")
    h.append('<h3>SECTION B — State whether the following statements are True or False (Q.7–10 · 1 mark each)</h3>')
    for stem in TRUE_FALSE:
        n += 1
        h.append(f'<p class="q">{n}. {stem}</p>')
    h.append('<h3>SECTION C — Short Answer Questions (Q.11–14 · 3 marks each)</h3>')
    for stem in SHORT:
        n += 1
        is_hindi = any("\u0900" <= ch <= "\u097f" for ch in stem)
        if is_hindi and not hindi:
            stem = "Write any two differences between business and profession."
        h.append(f'<p class="q{" hi" if is_hindi and hindi else ""}">{n}. {stem}</p>')
    h.append('<h3>SECTION D — Long Answer Questions (Q.15–16 · 5 marks each)</h3>')
    for stem in LONG:
        n += 1
        h.append(f'<p class="q">{n}. {stem}</p>')
    h.append('<h2 class="key">ANSWER KEY</h2>')
    h.append("<p>" + " &nbsp;&nbsp; ".join(f"{i + 1}. ({k})" for i, k in enumerate(KEY_MCQ)) + "</p>")
    h.append("<p>" + " &nbsp;&nbsp; ".join(f"{i + 7}. {k}" for i, k in enumerate(KEY_TF)) + "</p>")
    for num, text in KEY_SHORT.items():
        h.append(f"<p>{num}. {text}</p>")
    return "".join(h)


CSS = """
body { font-family: sans-serif; font-size: 10.5pt; }
h1 { text-align: center; font-size: 16pt; margin: 0; }
h2 { text-align: center; font-size: 13pt; margin: 2pt 0; }
h3 { font-size: 11pt; margin: 10pt 0 2pt 0; }
p.c { text-align: center; margin: 2pt 0; }
p.q { margin: 7pt 0 1pt 0; }
p.o { margin: 0 0 0 16pt; }
h2.key { margin-top: 18pt; }
"""
HINDI_CSS = "@font-face { font-family: hin; src: url(%s); } p.hi { font-family: hin; }" % HINDI_FONT


def write_recordings() -> None:
    """HAND-WRITTEN stand-ins for Gemini's replies (GEMINI_MOCK recordings), built from the same data as the PDF:
    what a correct transcription under contest_import_service.PARSE_SCHEMA looks like. They are NOT a live reading
    of this PDF — replace them with `python scripts/eval_contest_import.py --pdf <this pdf> --record <dir>`."""
    import json
    questions, n = [], 0

    def add(qtype, text, options, marks):
        nonlocal n
        n += 1
        questions.append({"number": n, "section_level": "Unknown", "type": qtype, "text": text,
                          "options": [{"label": "abcd"[i], "text": o} for i, o in enumerate(options)],
                          "marks": marks, "explanation": "", "ai_answer": ""})
    for stem, opts in MCQS:
        add("MCQ", stem, opts, 1)
    for stem in TRUE_FALSE:
        add("Short", stem, [], 1)
    for stem in SHORT:
        add("Short", stem, [], 3)
    for stem in LONG:
        add("Long", stem, [], 5)
    key = ([{"number": i + 1, "letter": f"({k})", "text": ""} for i, k in enumerate(KEY_MCQ)]
           + [{"number": i + 7, "letter": "", "text": k} for i, k in enumerate(KEY_TF)]
           + [{"number": num, "letter": "", "text": text} for num, text in KEY_SHORT.items()])
    parse = {"header": {"institute": "SUNRISE COMMERCE CLASSES", "board": "", "title": "Class XI — Business Studies",
                        "class_grade": "XI", "subject": "Business Studies",
                        "topic": "Nature and Purpose of Business", "duration_text": "1 Hour 30 Minutes",
                        "max_marks": 32},
             "questions": questions,
             "answer_key": {"found": True, "table": [], "list": key}}
    solve = {"answers": [{"number": i + 1, "label": k.upper()} for i, k in enumerate(KEY_MCQ)]}
    rec = OUT.parent / "recordings_business_studies"
    rec.mkdir(parents=True, exist_ok=True)
    (rec / "IMPORT-PARSE_1.json").write_text(json.dumps(parse, ensure_ascii=False, indent=1), encoding="utf-8")
    (rec / "IMPORT-SOLVE_1.json").write_text(json.dumps(solve, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"hand-written recordings → {rec}")


def main() -> int:
    write_recordings()
    hindi = (FONT_DIR / HINDI_FONT).exists()
    if not hindi:
        print(f"note: {HINDI_FONT} not found — Q14 is written in English instead of Hindi")
    story = pymupdf.Story(html=build_html(hindi), user_css=CSS + (HINDI_CSS if hindi else ""),
                          archive=pymupdf.Archive(str(FONT_DIR)) if hindi else None)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    writer = pymupdf.DocumentWriter(str(OUT))
    box = pymupdf.paper_rect("a4")
    more = True
    while more:
        dev = writer.begin_page(box)
        more, _ = story.place(box + (50, 50, -50, -50))
        story.draw(dev)
        writer.end_page()
    writer.close()
    doc = pymupdf.open(str(OUT))
    doc.subset_fonts()                       # the whole Nirmala collection is 1.6 MB otherwise
    data = doc.tobytes(garbage=4, deflate=True)
    doc.close()
    OUT.write_bytes(data)
    doc = pymupdf.open(str(OUT))
    print(f"{OUT.name}: {doc.page_count} pages, {OUT.stat().st_size} bytes")
    print(doc[0].get_text()[:200])
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
