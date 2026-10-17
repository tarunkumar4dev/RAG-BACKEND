"""
Checks for _parse_batch with the lean ("each field once") model output.

The prompt no longer asks the model to echo format / marks / chapter, to repeat case-study
sub-questions in the passage, or to write a long answer's marking twice. These checks make sure
the parser rebuilds what the views need, and still accepts the older, verbose output.

Run from test-generator-backend/:  python scripts/check_generation_parser.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

import app.services.test_generator_service as svc  # noqa: E402
from app.models.test_generator import ChapterSection, DifficultyLevel, QuestionFormat  # noqa: E402


class Req:
    subject = "Science"
    class_grade = "10"


def chapter(fmt, marks):
    return ChapterSection(chapter="Electricity", difficulty=DifficultyLevel("medium"), format=QuestionFormat(fmt),
                          marks_per_question=marks, quantity=1)


def parse(fmt, marks, q, section="E"):
    return svc._parse_batch(json.dumps({"questions": [q]}), chapter(fmt, marks), Req, section)


failures = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail and not cond else ""))
    if not cond:
        failures.append(name)


# 1. Case study, lean output: passage only in text, answers only in sub_parts, no format echo.
case = {
    "text": "Riya connected three identical bulbs in a circuit at home and noticed that one switch controlled all of them, "
            "while her brother's circuit let each bulb work on its own.",
    "explanation": "Series vs parallel behaviour.",
    "sub_parts": [
        {"label": "i", "text": "Which arrangement did Riya use?", "marks": 1, "answer": "Series arrangement."},
        {"label": "ii", "text": "Why do household circuits use parallel?", "marks": 1, "answer": "Each appliance gets full voltage."},
        {"label": "iii", "text": "Find the equivalent resistance of three 6 ohm resistors in parallel.", "marks": 2, "answer": "1/R = 3/6, so R = 2 ohm."},
    ],
    "bloom_level": "apply", "topic": "circuits",
}
qs = parse("case_based", 4, case)
check("case: kept", len(qs) == 1)
if qs:
    q = qs[0]
    check("case: format comes from the chapter, not 'mcq'", q.format.value == "case_based", q.format)
    check("case: key built from sub_parts", q.correct_answer.startswith("(i) Series arrangement.") and "(iii) 1/R" in q.correct_answer, q.correct_answer)
    check("case: marks from the section", q.marks == 4, q.marks)

# 2. Case study, old verbose output: sub-questions repeated in the passage are cut out of it.
verbose = dict(case, text=case["text"] + "\n(i) Which arrangement did Riya use?\n(ii) Why do household circuits use parallel?\n"
                                         "(iii) Find the equivalent resistance of three 6 ohm resistors in parallel.",
               correct_answer="(i) Series. (ii) Full voltage. (iii) 2 ohm.", format="case_based")
qs = parse("case_based", 4, verbose)
check("verbose case: kept", len(qs) == 1)
if qs:
    check("verbose case: sub-questions removed from passage", "(i)" not in qs[0].text and qs[0].text.endswith("on its own."), qs[0].text[-80:])
    check("verbose case: model's own key kept", qs[0].correct_answer.startswith("(i) Series. (ii)"), qs[0].correct_answer)

# 3. Long answer, lean output: no explanation field; explanation built from marking_scheme; half marks survive.
long_q = {
    "text": "(a) State Ohm's law and draw the V-I graph for a metallic conductor. (b) A 12 V battery drives 0.5 A through a wire. Find its resistance.",
    "correct_answer": "(a) At constant temperature the current through a conductor is proportional to the potential difference across it; "
                      "the V-I graph is a straight line through the origin. (b) R = V/I = 12/0.5 = 24 ohm.",
    "marking_scheme": [{"step": "Statement of Ohm's law", "marks": 1}, {"step": "Graph", "marks": 1.5},
                       {"step": "Formula", "marks": 1}, {"step": "Answer with unit", "marks": 1.5}],
    "bloom_level": "apply", "topic": "Ohm's law", "is_or": False,
}
qs = parse("long_answer", 5, long_q, section="D")
check("long: kept", len(qs) == 1)
if qs:
    q = qs[0]
    check("long: format from chapter", q.format.value == "long_answer", q.format)
    check("long: explanation built from marking_scheme", q.explanation.startswith("Marking: Statement of Ohm's law (1m); Graph (1.5m)"), q.explanation)
    check("long: half marks kept (not int()-ed to 0/1)", [s.marks for s in q.marking_scheme] == [1, 1.5, 1, 1.5], [s.marks for s in q.marking_scheme])

# 4. MCQ without a "format" field still parses as an MCQ.
mcq = {"text": "Which instrument is connected in series to measure current in a circuit?",
       "options": ["A) Voltmeter", "B) Ammeter", "C) Galvanometer in parallel", "D) Rheostat"],
       "correct_answer": "B) Ammeter", "explanation": "An ammeter has low resistance and is connected in series.",
       "bloom_level": "remember", "topic": "measurement"}
c = ChapterSection(chapter="Electricity", difficulty=DifficultyLevel("easy"), format=QuestionFormat("mcq"), marks_per_question=1, quantity=1)
qs = svc._parse_batch(json.dumps({"questions": [mcq]}), c, Req, "A")
check("mcq: kept without format echo", len(qs) == 1 and qs[0].format.value == "mcq")

# 5. A passage that only mentions "(i)" in passing is not cut.
check("stem: unrelated (i) left alone",
      svc._strip_subparts_from_stem("A wire (i) of copper is bent into a loop and current flows for a long time in the lab.",
                                    [{"text": "What is resistance?"}]).endswith("in the lab."))

print()
if failures:
    print(f"{len(failures)} check(s) failed")
    sys.exit(1)
print("All generation parser checks passed.")
