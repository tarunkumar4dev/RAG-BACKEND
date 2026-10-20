"""
Checks for app/services/ncert_cleanup.py — cases copied from live ncert_questions rows (18 Oct 2026)
and from AI-generated questions that lost their statement list.

Run from test-generator-backend/:  python scripts/check_ncert_cleanup.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.ncert_cleanup import clean_question, missing_items_reason, quality_issue  # noqa: E402

failures = []


def check(name, cond, detail=""):
    if not cond:
        failures.append(name)
        print(f"FAIL {name} {detail}")


# ── options repeated in the text + labels on the options ──
q = clean_question({
    "question_text": "In the modern periodic table, the period indicates the value of : (a) atomic number "
                     "(b) atomic mass (c) principal quantum number (d) azimuthal quantum number.",
    "options": ["(a) atomic number", "(b) atomic mass", "(c) principal quantum number", "(d) azimuthal quantum number."],
})
check("inline options cut", q["question_text"] == "In the modern periodic table, the period indicates the value of :", q["question_text"])
check("letter labels stripped", q["options"][0] == "atomic number", q["options"])

q = clean_question({"question_text": "Length PQ is : / (A)12 cm (B)13 cm (C)8.5 cm (D) 119 cm.",
                    "options": ["(A)12 cm", "(B)13 cm", "(C)8.5 cm", "(D) 119 cm."]})
check("capital labels", q["options"] == ["12 cm", "13 cm", "8.5 cm", "119 cm."], q["options"])

# statements the options refer to must stay in the text
stem = ("Consider the following statements about power sharing. / A. In Belgium, the Dutch-speaking majority. "
        "/ B. In Sri Lanka, the policies. / C. The Tamils. / D. The transformation. Which are correct?")
q = clean_question({"question_text": stem, "options": ["A, B, C and D", "A, B and D", "C and D", "B, C and D"]})
check("statement list kept", q["question_text"] == stem, q["question_text"])

# roman options become A–D when nothing refers to them; the answer follows
q = clean_question({"question_text": "What purpose does handpicking serve? (i) Filtration (ii) Sorting (iii) Evaporation (iv) Decantation",
                    "options": ["(i) Filtration", "(ii) Sorting", "(iii) Evaporation", "(iv) Decantation"],
                    "answer": "(ii) Sorting"})
check("roman labels stripped", q["options"] == ["Filtration", "Sorting", "Evaporation", "Decantation"], q["options"])
check("roman answer relabelled", q["answer"] == "(b) Sorting", q["answer"])
check("roman inline cut", q["question_text"] == "What purpose does handpicking serve?", q["question_text"])

opts = ["(i) Trial balance", "(ii) Profit and loss account", "(iii) Balance sheet", "(iv) (i) & (iii)", "(v) (ii) & (iii)"]
q = clean_question({"question_text": "The financial statements consist of:", "options": list(opts)})
check("roman labels kept when options refer to each other", q["options"] == opts, q["options"])

# glyph junk
q = clean_question({"question_text": "/square6Take about 1g solid NaCl. /square6Add some acid.", "options": []})
check("square bullets", q["question_text"] == "Take about 1g solid NaCl. Add some acid.", q["question_text"])
q = clean_question({"question_text": "nTake about 3 mL of solution. nMix the two solutions.", "options": []})
check("n bullets", q["question_text"] == "Take about 3 mL of solution. Mix the two solutions.", q["question_text"])
q = clean_question({"question_text": "Find the value of nCr when n = 5.", "options": []})
check("nCr untouched", q["question_text"] == "Find the value of nCr when n = 5.", q["question_text"])

# ── missing content ──
check("following reaction", quality_issue("Is there any change in the hybridisation of B and N atoms as a result of the following reaction?") == "incomplete")
check("empty labels", quality_issue("Identify the functional groups in the following compounds (a) (b) (c)") == "incomplete")
check("glyphs garbled", quality_issue("Describe the change. AlCl Cl AlCl34/unif02b/unif0ae/unif02d") == "garbled")
check("image fixes it", quality_issue("… the following reaction?", has_image=True) is None)
check("items present", quality_issue("Calculate the number of atoms in each of the following (i) 52 moles of Ar (ii) 52 u of He") is None)
check("following as verb", quality_issue("— tourist guides following people.") is None)
check("mcq options are the items", quality_issue("Which of the following is a plant hormone?", ["Insulin", "Thyroxin", "Oestrogen", "Cytokinin"]) is None)
check("table carries the items", quality_issue("Find the mean of the following data:", [], has_table=True) is None)
check("option text with 'and'", missing_items_reason("Identify the true statements.", [
    "The expression 4 m – 1 always gives odd numbers.", "All even numbers can be expressed as 6 j – 4.",
    "Both expressions 2 p + 1 and 2 q – 1 describe all odd numbers.", "None"]) is None)
check("both (b) and (c) refers to options", missing_items_reason("Favourable bank balance means:", [
    "Credit balance in the cash book", "Credit balance in passbook", "Debit balance in the cash book", "Both (b) and (c)"]) is None)

# AI output that dropped the statements
check("ai statements missing", missing_items_reason("Consider the following statements about covalent bonds. Which of them are correct?",
                                                     ["A) (i) and (ii) only", "B) (ii) and (iii) only", "C) (i) and (iii) only", "D) (i), (ii) and (iii)"]) is not None)
check("ai statements present", missing_items_reason("Consider the following statements:\n(i) Sigma bonds are stronger.\n(ii) Pi bonds form by sideways overlap.\n(iii) He2 exists.\nWhich are correct?",
                                                    ["A) (i) and (ii) only", "B) (ii) and (iii) only", "C) (i) and (iii) only", "D) (i), (ii) and (iii)"]) is None)
check("ai numbered statements missing", missing_items_reason("Which of the statements given are correct?",
                                                             ["A) 1 and 2 only", "B) 2 and 3 only", "C) 1 and 3 only", "D) 1, 2 and 3"]) is not None)
check("ai template copied", missing_items_reason("Consider the following statements: (I)... (II)... (III)... Which are correct?",
                                                 ["A) I only", "B) II only", "C) I and II", "D) All"]) is not None)
check("ai match without columns", missing_items_reason("Match the following and choose the correct option.",
                                                       ["A) A-2, B-1, C-4, D-3", "B) A-1, B-2, C-3, D-4", "C) A-3, B-4, C-1, D-2", "D) A-4, B-3, C-2, D-1"]) is not None)
check("ai match with columns", missing_items_reason("Match Column I with Column II:\nColumn I: (A) Na (B) Cl (C) Ar (D) Mg\nColumn II: (1) noble gas (2) alkali metal (3) halogen (4) alkaline earth",
                                                    ["A) A-2, B-3, C-1, D-4", "B) A-1, B-2, C-3, D-4", "C) A-3, B-4, C-1, D-2", "D) A-4, B-3, C-2, D-1"]) is None)

if failures:
    print(f"\n{len(failures)} check(s) failed")
    sys.exit(1)
print("All ncert_cleanup checks passed")
