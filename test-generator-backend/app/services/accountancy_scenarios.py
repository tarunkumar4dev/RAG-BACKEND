"""
SOLVE-FIRST numeric MCQ engine for CBSE Class 12 Accountancy — Fundamentals of
Partnership. Phase 1B.

Pipeline for a numeric MCQ, in order:
  1. pick scenario parameters deterministically (seeded RNG + difficulty)
  2. the deterministic solver (accountancy_solver) computes the EXACT answer
  3. three distractors are computed — each a NAMED student error (timing trap,
     opening-vs-closing, charge-vs-appropriation, rate trap, ...), also via the
     solver with a wrong parameter/path, never a random offset
  4. make_options() guarantees exactly one correct option and four distinct values
  5. the stem prose is generated from the parameters (template here; an optional
     LLM polish that MUST preserve every number is in polish_stem_with_llm)

The LLM never decides a number or which option is correct. If the LLM is used to
rewrite prose and drops/changes any given number, we keep the template stem.

This module is pure and deterministic given a seed. No I/O, no network.
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.services import accountancy_solver as S

_LETTERS = ["A", "B", "C", "D"]
_NAMES = [
    ("Arun", "Bela"), ("Chetan", "Disha"), ("Esha", "Farhan"),
    ("Gita", "Harish"), ("Ishan", "Juhi"), ("Karan", "Lata"),
    ("Mohan", "Nisha"), ("Priya", "Rakesh"), ("Sahil", "Tina"),
]
_TRIOS = [
    ("Arun", "Bela", "Charu"), ("Dev", "Esha", "Farhan"),
    ("Gaurav", "Hina", "Imran"), ("Jatin", "Kavya", "Lokesh"),
    ("Mira", "Nikhil", "Omkar"),
]


# ── money formatting (Indian grouping) ───────────────────────────────────

def inr(x: Fraction) -> str:
    """Fraction rupees -> '2,70,000' (Indian digit grouping). Rounds half-up."""
    n = int(S.rupees(x, 0))
    neg = n < 0
    n = abs(n)
    s = str(n)
    if len(s) <= 3:
        grouped = s
    else:
        last3, rest = s[-3:], s[:-3]
        parts: List[str] = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    return ("-" if neg else "") + grouped


def _rs(x: Fraction) -> str:
    return f"Rs. {inr(x)}"


# ── scenario draft ────────────────────────────────────────────────────────

@dataclass
class ScenarioDraft:
    family: str
    topic: str
    stem: str
    correct: Fraction
    distractors: List[Tuple[str, Fraction]]   # (error_label, value)
    working: List[str]


@dataclass
class NumericMCQ:
    text: str
    options: List[str]              # ["A) Rs. ...", ...]
    correct_answer: str             # full text of the correct option, e.g. "C) Rs. 2,70,000"
    correct_letter: str
    explanation: str
    topic: str
    family: str
    marks: int = 1
    difficulty: str = "medium"
    bloom_level: str = "apply"
    validation_status: str = "verified"   # solver-guaranteed; confirmed by the Phase 1C gate

    def as_question_dict(self) -> Dict:
        return {
            "text": self.text,
            "format": "mcq",
            "options": list(self.options),
            "correct_answer": self.correct_answer,
            "explanation": self.explanation,
            "marks": self.marks,
            "difficulty": self.difficulty,
            "bloom_level": self.bloom_level,
            "topic": self.topic,
            "family": self.family,
            "validation_status": self.validation_status,
        }


# ── family generators ─────────────────────────────────────────────────────
# Each returns a ScenarioDraft for a fresh set of params. build_numeric_mcq
# retries with an advanced RNG if the distractors collide with the answer.

def _pick(rng: random.Random, seq: Sequence):
    return seq[rng.randrange(len(seq))]


def _phrase(variant: int, templates: Sequence[str]) -> str:
    """Pick a stem phrasing by variant index. Multiple phrasings for the same
    cognitive task keep same-structure/different-number questions from reading
    near-identically (which the dedup gate would otherwise flag as duplicates)."""
    return templates[variant % len(templates)]


def _rate(rng: random.Random, difficulty: str) -> int:
    pool = {"easy": [6, 8, 10, 12], "medium": [6, 8, 9, 10, 12],
            "hard": [7, 9, 11, 12, 15]}.get(difficulty, [6, 8, 10, 12])
    return _pick(rng, pool)


def _amount(rng: random.Random, lo: int, hi: int, step: int) -> int:
    return rng.randrange(lo, hi + 1, step)


def gen_interest_on_capital(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    """Interest on capital, product method. Four structural sub-cases:
    0 full year (no movement) · 1 mid-year addition · 2 mid-year withdrawal ·
    3 addition and withdrawal."""
    a, b = _pick(rng, _NAMES)
    opening = _amount(rng, 200000, 900000, 50000)
    rate = _rate(rng, difficulty)
    v = variant % 4

    if v == 0:
        correct = S.interest_on_capital(opening, rate).value
        half_trap = Fraction(opening) * rate / 100 * Fraction(6, 12)       # charged for 6 months
        no_time_trap = Fraction(opening) * rate                            # forgot /100
        rate_plus = Fraction(opening) * (rate + 2) / 100                   # wrong rate
        stem = (f"{a}'s capital on 1 April is {_rs(Fraction(opening))} and stays unchanged "
                f"through the year. Interest on capital is allowed @ {rate}% p.a. "
                f"Interest on {a}'s capital for the year is:")
        working = [f"IOC = {opening} x {rate}% = {_rs(correct)}"]
        return ScenarioDraft("interest_on_capital", "Interest on Capital", stem, correct,
                             [("charged interest for only half the year", half_trap),
                              ("forgot to divide the rate by 100", no_time_trap),
                              (f"used {rate + 2}% instead of {rate}%", rate_plus)],
                             working)

    if v == 1:
        add_month = _pick(rng, [3, 4, 8, 9])
        add_amt = _amount(rng, 50000, 200000, 50000)
        months_left = 12 - add_month
        correct = S.interest_on_capital(opening, rate, [(add_month, add_amt)]).value
        ignored_add = S.interest_on_capital(opening, rate).value
        treated_full = S.interest_on_capital(opening + add_amt, rate).value
        wrong_period = (Fraction(opening) * rate / 100
                        + Fraction(add_amt) * rate * add_month / 1200)
        stem = (f"On 1 April {a}'s capital was {_rs(Fraction(opening))}. {a} introduced "
                f"further capital of {_rs(Fraction(add_amt))} on {_month_name(add_month)}. "
                f"Interest on capital is allowed @ {rate}% p.a. Interest on {a}'s capital "
                f"for the year is:")
        working = [f"{_rs(Fraction(opening))} for {add_month} mo + {_rs(Fraction(opening + add_amt))} for {months_left} mo",
                   f"IOC = {_rs(correct)}"]
        return ScenarioDraft("interest_on_capital", "Interest on Capital", stem, correct,
                             [("ignored the additional capital", ignored_add),
                              ("treated the addition as brought in for the whole year", treated_full),
                              ("counted the elapsed months, not the remaining months", wrong_period)],
                             working)

    if v == 2:
        wd_month = _pick(rng, [3, 4, 8, 9])
        wd_amt = _amount(rng, 50000, 150000, 50000)
        months_left = 12 - wd_month
        correct = S.interest_on_capital(opening, rate, [(wd_month, -wd_amt)]).value
        ignored_wd = S.interest_on_capital(opening, rate).value
        treated_full = S.interest_on_capital(opening - wd_amt, rate).value
        wrong_period = (Fraction(opening) * rate / 100
                        - Fraction(wd_amt) * rate * wd_month / 1200)
        stem = (f"{a}'s capital on 1 April was {_rs(Fraction(opening))}. {a} permanently "
                f"withdrew {_rs(Fraction(wd_amt))} of capital on {_month_name(wd_month)}. "
                f"Interest on capital is allowed @ {rate}% p.a. Interest on {a}'s capital "
                f"for the year is:")
        working = [f"{_rs(Fraction(opening))} for {wd_month} mo + {_rs(Fraction(opening - wd_amt))} for {months_left} mo",
                   f"IOC = {_rs(correct)}"]
        return ScenarioDraft("interest_on_capital", "Interest on Capital", stem, correct,
                             [("ignored the capital withdrawn", ignored_wd),
                              ("treated the reduced capital as the balance all year", treated_full),
                              ("counted the elapsed months, not the remaining months", wrong_period)],
                             working)

    # v == 3: addition and withdrawal
    m1 = _pick(rng, [3, 4])
    m2 = _pick(rng, [8, 9])
    add_amt = _amount(rng, 50000, 150000, 50000)
    wd_amt = _amount(rng, 50000, 100000, 50000)
    correct = S.interest_on_capital(opening, rate, [(m1, add_amt), (m2, -wd_amt)]).value
    ignored_both = S.interest_on_capital(opening, rate).value
    add_only = S.interest_on_capital(opening, rate, [(m1, add_amt)]).value
    wd_only = S.interest_on_capital(opening, rate, [(m2, -wd_amt)]).value
    stem = (f"{a} began the year with capital of {_rs(Fraction(opening))}, introduced "
            f"{_rs(Fraction(add_amt))} on {_month_name(m1)} and withdrew {_rs(Fraction(wd_amt))} "
            f"on {_month_name(m2)}. Interest on capital is allowed @ {rate}% p.a. Interest on "
            f"{a}'s capital for the year is:")
    working = [f"product method over three balances; IOC = {_rs(correct)}"]
    return ScenarioDraft("interest_on_capital", "Interest on Capital", stem, correct,
                         [("ignored both movements", ignored_both),
                          ("ignored the withdrawal", add_only),
                          ("ignored the addition", wd_only)],
                         working)


def gen_interest_on_drawings(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    """Equal periodic drawings; average-period method; timing trap is primary."""
    a, b = _pick(rng, _NAMES)
    # variant selects a distinct (frequency, timing) structure so recurrences differ.
    _COMBOS = [("monthly", "beginning"), ("monthly", "end"),
               ("quarterly", "beginning"), ("quarterly", "end"),
               ("half-yearly", "beginning"), ("half-yearly", "end")]
    freq, timing = _COMBOS[variant % len(_COMBOS)]
    rate = _rate(rng, difficulty)
    per = {"monthly": _amount(rng, 1000, 8000, 1000),
           "quarterly": _amount(rng, 6000, 20000, 2000),
           "half-yearly": _amount(rng, 12000, 40000, 4000)}[freq]

    correct = S.interest_on_drawings(per, rate, freq, timing).value
    other_timing = "end" if timing == "beginning" else "beginning"
    timing_trap = S.interest_on_drawings(per, rate, freq, other_timing).value
    # flat trap: forgot the time factor entirely (rate on total drawings)
    n = {"monthly": 12, "quarterly": 4, "half-yearly": 2}[freq]
    total = per * n
    flat_trap = S.interest_on_drawings_flat(total, rate).value
    # 6-month trap: used 6 months average regardless of timing
    six_trap = Fraction(total) * rate / 100 * Fraction(6, 12)

    when = {"beginning": "the beginning", "end": "the end"}[timing]
    fw = _freq_word(freq)
    stem = _phrase(variant, [
        (f"{a} withdrew {_rs(Fraction(per))} at {when} of every {fw} during the year. "
         f"Interest on drawings is charged @ {rate}% p.a. The interest on {a}'s drawings "
         f"for the year is:"),
        (f"Every {fw}, {a} drew {_rs(Fraction(per))} at {when} of the period. At {rate}% "
         f"p.a., the interest on {a}'s drawings for the year is:"),
        (f"{a} made regular drawings of {_rs(Fraction(per))} at {when} of each {fw}. "
         f"Charging interest @ {rate}% p.a., {a}'s interest on drawings comes to:"),
        (f"Drawings of {_rs(Fraction(per))} were made by {a} at {when} of every {fw}. "
         f"Interest on drawings @ {rate}% p.a. for the year is:"),
        (f"{a} took out {_rs(Fraction(per))} at {when} of each {fw} through the year. "
         f"The interest on drawings at {rate}% p.a. is:"),
        (f"At {when} of every {fw}, {a} withdrew {_rs(Fraction(per))}. Interest on drawings "
         f"is {rate}% p.a.; the interest on {a}'s drawings is:"),
    ])
    working = [
        f"total drawings = {_rs(Fraction(total))}; average-period method, drawn at {when}",
        f"IOD = {_rs(correct)}",
    ]
    return ScenarioDraft(
        "interest_on_drawings", "Interest on Drawings", stem, correct,
        [(f"used the {other_timing}-of-period average period (timing trap)", timing_trap),
         ("charged a flat rate with no time factor", flat_trap),
         ("used 6 months average regardless of timing", six_trap)],
        working,
    )


def gen_interest_on_drawings_flat(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    """Rate stated WITHOUT 'p.a.' -> flat; trap is applying a time factor."""
    a, b = _pick(rng, _NAMES)
    total = _amount(rng, 20000, 90000, 10000)
    rate = _pick(rng, [4, 5, 6, 8])
    correct = S.interest_on_drawings_flat(total, rate).value
    half_trap = Fraction(total) * rate / 100 * Fraction(6, 12)
    beg_trap = Fraction(total) * rate / 100 * Fraction(Fraction(13, 2), 12)
    end_trap = Fraction(total) * rate / 100 * Fraction(Fraction(11, 2), 12)
    stem = _phrase(variant, [
        (f"{a}'s total drawings during the year were {_rs(Fraction(total))}. Interest on "
         f"drawings is to be charged @ {rate}% (the deed does NOT say per annum). Interest "
         f"on {a}'s drawings is:"),
        (f"During the year {a} drew {_rs(Fraction(total))} in total. The deed charges "
         f"interest on drawings at a flat {rate}% (not per annum). The interest is:"),
        (f"{a} withdrew {_rs(Fraction(total))} in all. Interest on drawings is {rate}% "
         f"(the rate is not stated per annum). {a}'s interest on drawings is:"),
    ])
    working = [f"rate is not p.a. -> flat % on total drawings",
               f"IOD = {total} x {rate}/100 = {_rs(correct)}"]
    return ScenarioDraft(
        "interest_on_drawings_flat", "Interest on Drawings", stem, correct,
        [("applied a 6-month time factor though the rate is not p.a.", half_trap),
         ("treated it as drawn at the beginning of every month", beg_trap),
         ("treated it as drawn at the end of every month", end_trap)],
        working,
    )


def gen_interest_on_drawings_irregular(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    """Product method for unequal drawings on given dates."""
    a, b = _pick(rng, _NAMES)
    rate = _rate(rng, difficulty)
    # three irregular withdrawals (amount, months outstanding)
    items = [
        (_amount(rng, 8000, 20000, 2000), 10),
        (_amount(rng, 6000, 16000, 2000), 6),
        (_amount(rng, 4000, 12000, 2000), 2),
    ]
    correct = S.interest_on_drawings_irregular(items, rate).value
    total = sum(a_ for a_, _ in items)
    # simple-average trap: total x rate x 6/12
    avg_trap = Fraction(total) * rate / 100 * Fraction(6, 12)
    # flat trap
    flat_trap = Fraction(total) * rate / 100
    # dropped-last-item trap
    drop_trap = S.interest_on_drawings_irregular(items[:-1], rate).value
    rows = "; ".join(f"{_rs(Fraction(a_))} outstanding for {m} months" for a_, m in items)
    stem = _phrase(variant, [
        (f"{a} made the following drawings: {rows}. Interest on drawings is charged @ {rate}% "
         f"p.a. using the product method. Interest on {a}'s drawings is:"),
        (f"{a}'s drawings on different dates were: {rows}. Using the product method at "
         f"{rate}% p.a., the interest on drawings is:"),
        (f"During the year {a} withdrew {rows}. By the product method, interest on drawings "
         f"@ {rate}% p.a. works out to:"),
    ])
    working = [f"product = sum(amount x months) = {sum(a_*m for a_, m in items)}",
               f"IOD = product x {rate}/100/12 = {_rs(correct)}"]
    return ScenarioDraft(
        "interest_on_drawings_irregular", "Interest on Drawings", stem, correct,
        [("used a flat 6-month average on total drawings", avg_trap),
         ("charged a flat rate with no time factor", flat_trap),
         ("left out the last withdrawal", drop_trap)],
        working,
    )


def gen_pl_appropriation(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    """Divisible profit after appropriations."""
    np = _amount(rng, 120000, 400000, 20000)
    ioc = _amount(rng, 10000, 60000, 5000)
    salary = _amount(rng, 12000, 60000, 6000)
    iod = _amount(rng, 2000, 12000, 1000)
    correct = S.appropriation(np, interest_on_capital={"x": ioc},
                              salaries={"s": salary},
                              interest_on_drawings={"d": iod}).value
    # sign trap on IOD (subtracted)
    iod_sign = S.appropriation(np, interest_on_capital={"x": ioc},
                               salaries={"s": salary},
                               interest_on_drawings={"d": -iod}).value
    # ignored IOD
    no_iod = S.appropriation(np, interest_on_capital={"x": ioc},
                             salaries={"s": salary}).value
    # added IOC instead of subtracting
    ioc_sign = S.appropriation(np, interest_on_capital={"x": -ioc},
                               salaries={"s": salary},
                               interest_on_drawings={"d": iod}).value
    stem = _phrase(variant, [
        (f"A firm's net profit for the year is {_rs(Fraction(np))}. The deed provides "
         f"interest on capital {_rs(Fraction(ioc))}, partner's salary {_rs(Fraction(salary))}, "
         f"and interest on drawings recovered {_rs(Fraction(iod))}. The profit available for "
         f"distribution among the partners (divisible profit) is:"),
        (f"Before appropriations a firm earned {_rs(Fraction(np))}. After allowing interest "
         f"on capital of {_rs(Fraction(ioc))} and a salary of {_rs(Fraction(salary))}, and "
         f"charging {_rs(Fraction(iod))} as interest on drawings, the divisible profit is:"),
        (f"A partnership's net profit is {_rs(Fraction(np))}. Interest on capital "
         f"{_rs(Fraction(ioc))} and salary {_rs(Fraction(salary))} are to be allowed and "
         f"interest on drawings {_rs(Fraction(iod))} recovered. What profit is left to be "
         f"divided among the partners?"),
        (f"From a net profit of {_rs(Fraction(np))}, a firm allows interest on capital "
         f"{_rs(Fraction(ioc))} and partner's salary {_rs(Fraction(salary))}, and recovers "
         f"interest on drawings {_rs(Fraction(iod))}. The divisible profit works out to:"),
    ])
    working = [f"divisible = {np} + IOD {iod} - IOC {ioc} - salary {salary} = {_rs(correct)}"]
    return ScenarioDraft(
        "pl_appropriation", "P&L Appropriation Account", stem, correct,
        [("subtracted interest on drawings instead of adding it", iod_sign),
         ("ignored interest on drawings", no_iod),
         ("added interest on capital instead of subtracting it", ioc_sign)],
        working,
    )


def gen_profit_share(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    """Share of divisible profit in a stated ratio."""
    a, b, c = _pick(rng, _TRIOS)
    ratio = _pick(rng, [(3, 2, 1), (5, 3, 2), (4, 3, 2)])   # all three shares distinct
    unit = _pick(rng, [10000, 12000, 20000, 30000])
    divisible = unit * sum(ratio)
    shares = S.distribute(divisible, list(ratio))
    correct = shares[0]
    equal_trap = Fraction(divisible, 3)
    second_trap = shares[1]
    third_trap = shares[2]
    rr = f"{ratio[0]}:{ratio[1]}:{ratio[2]}"
    stem = _phrase(variant, [
        (f"{a}, {b} and {c} are partners sharing profits in the ratio {rr}. The divisible "
         f"profit for the year is {_rs(Fraction(divisible))}. {a}'s share of profit is:"),
        (f"{a}, {b} and {c} share profits {rr}. If the profit available for distribution is "
         f"{_rs(Fraction(divisible))}, how much does {a} receive?"),
        (f"The divisible profit of a firm is {_rs(Fraction(divisible))}. {a}, {b} and {c} "
         f"share profits in the ratio {rr}. {a}'s share comes to:"),
        (f"Partners {a}, {b} and {c} divide {_rs(Fraction(divisible))} of profit in the ratio "
         f"{rr}. The amount credited to {a} is:"),
    ])
    working = [f"{a}'s share = {divisible} x {ratio[0]}/{sum(ratio)} = {_rs(correct)}"]
    return ScenarioDraft(
        "profit_share", "Distribution of Profit", stem, correct,
        [("divided the profit equally instead of in the ratio", equal_trap),
         (f"gave {a} the share belonging to {b}", second_trap),
         (f"gave {a} the share belonging to {c}", third_trap)],
        working,
    )


def gen_commission_before(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    a, b = _pick(rng, _NAMES)
    profit = _amount(rng, 120000, 500000, 20000)
    rate = _pick(rng, [5, 10, 15, 20])
    correct = S.commission_before_charging(profit, rate).value
    after_trap = S.commission_after_charging(profit, rate).value
    # wrong-base trap: % of a "profit after some other charge" — approximate with half
    double_trap = Fraction(profit) * (2 * rate) / 100
    tenth_trap = Fraction(profit) * rate / 1000
    role = _phrase(variant, ["accountant", "works manager", "branch head"])
    stem = _phrase(variant, [
        (f"A firm allows its {role} {rate}% commission computed on net profit BEFORE the "
         f"commission is charged. Net profit is {_rs(Fraction(profit))}. The commission equals:"),
        (f"{rate}% of the pre-commission net profit of {_rs(Fraction(profit))} is payable to "
         f"the {role} as commission (charged before the commission). That amount is:"),
        (f"The {role}'s commission is {rate}% of profit, taken before charging the commission. "
         f"On a net profit of {_rs(Fraction(profit))}, the commission is:"),
    ])
    working = [f"commission = {profit} x {rate}% = {_rs(correct)}"]
    return ScenarioDraft(
        "commission_before", "Manager's Commission", stem, correct,
        [("used the after-charging formula", after_trap),
         ("doubled the rate", double_trap),
         ("misplaced the decimal (divided by 1000)", tenth_trap)],
        working,
    )


def gen_commission_after(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    a, b = _pick(rng, _NAMES)
    rate = _pick(rng, [10, 20, 25])
    # choose profit so the after-charging answer is clean (base = profit net of commission)
    base = _amount(rng, 10, 60, 5) * 1000
    profit = (base * (100 + rate)) // 100
    correct = S.commission_after_charging(profit, rate).value
    before_trap = S.commission_before_charging(profit, rate).value          # /100 instead of /(100+rate)
    wrong_denom = Fraction(profit) * rate / (100 + 2 * rate)                 # doubled rate in denominator
    minus_denom = Fraction(profit) * rate / (100 - rate)                     # subtracted instead of added
    role = _phrase(variant, ["manager", "works manager", "general manager", "sales manager"])
    stem = _phrase(variant, [
        (f"The {role} of a firm is entitled to a commission of {rate}% of the net profit "
         f"AFTER charging such commission. Net profit before commission is "
         f"{_rs(Fraction(profit))}. The commission is:"),
        (f"A firm's {role} earns {rate}% commission on profit remaining after the commission "
         f"itself. Profit before commission is {_rs(Fraction(profit))}. The commission is:"),
        (f"The {role} is paid {rate}% of the net profit after charging that commission. If "
         f"the profit before commission is {_rs(Fraction(profit))}, the commission works out to:"),
        (f"Commission to the {role} is {rate}% of profit after charging it. On a pre-commission "
         f"profit of {_rs(Fraction(profit))}, the commission is:"),
    ])
    working = [f"commission = {profit} x {rate}/(100+{rate}) = {_rs(correct)}"]
    return ScenarioDraft(
        "commission_after", "Manager's Commission", stem, correct,
        [("used the before-charging formula", before_trap),
         ("used a wrong denominator (100 + 2*rate)", wrong_denom),
         ("subtracted the rate in the denominator (100 - rate)", minus_denom)],
        working,
    )


def gen_guarantee_to_partner(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    a, b, c = _pick(rng, _TRIOS)
    ratio = _pick(rng, [(3, 2, 1), (5, 3, 2), (4, 3, 2)])   # bearers r0 != r1 (else equal-split trap == correct)
    unit = _pick(rng, [20000, 30000, 40000])
    divisible = unit * sum(ratio)
    shares = dict(zip([a, b, c], S.distribute(divisible, list(ratio))))
    guaranteed_share = shares[c]
    minimum = S.rupees(guaranteed_share) + _amount(rng, 10000, 40000, 5000)
    r = S.guarantee_to_partner(shares, c, minimum, {a: ratio[0], b: ratio[1]})
    correct = r.parts[a]                    # a bearer's final share
    # trap: deficiency split equally between bearers
    deficiency = Fraction(minimum) - guaranteed_share
    equal_bearer = shares[a] - deficiency / 2
    # trap: ignored the guarantee entirely
    no_guarantee = shares[a]
    # trap: gave the guaranteed partner's minimum to a
    guaranteed_min = Fraction(minimum)
    gr = f"{ratio[0]}:{ratio[1]}:{ratio[2]}"
    br = f"{ratio[0]}:{ratio[1]}"
    stem = _phrase(variant, [
        (f"{a}, {b} and {c} share profits {gr}. {c} is guaranteed a minimum profit of "
         f"{_rs(Fraction(minimum))}; any deficiency is borne by {a} and {b} in the ratio "
         f"{br}. Divisible profit is {_rs(Fraction(divisible))}. {a}'s final share is:"),
        (f"Profits of {_rs(Fraction(divisible))} are shared by {a}, {b} and {c} in {gr}, but "
         f"{c} must get at least {_rs(Fraction(minimum))}; {a} and {b} meet any shortfall in "
         f"{br}. {a}'s final share is:"),
        (f"{a}, {b} and {c} divide {_rs(Fraction(divisible))} in the ratio {gr}. {c} has a "
         f"guaranteed minimum of {_rs(Fraction(minimum))}, the deficiency being borne by {a} "
         f"and {b} in {br}. What is {a}'s final share?"),
    ])
    working = r.working + [f"{a}'s final share = {_rs(correct)}"]
    return ScenarioDraft(
        "guarantee_to_partner", "Guarantee of Profit", stem, correct,
        [("split the deficiency equally instead of in the given ratio", equal_bearer),
         ("ignored the guarantee", no_guarantee),
         ("reported the guaranteed minimum instead", guaranteed_min)],
        working,
    )


def gen_guarantee_to_firm(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    a, b, c = _pick(rng, _TRIOS)
    ratio = _pick(rng, [(2, 2, 1), (3, 2, 1), (1, 1, 1)])
    unit = _pick(rng, [20000, 30000, 40000])
    minimum = unit * sum(ratio)
    # keep the guarantor's net share positive: shortfall < c's base share (unit*ratio_c)
    max_short = unit * ratio[2] - 10000
    shortfall = _amount(rng, 10000, max(10000, max_short), 10000)
    actual = minimum - shortfall
    r = S.guarantee_to_firm(actual, minimum, {a: ratio[0], b: ratio[1], c: ratio[2]}, c)
    correct = r.parts[c]
    base_shares = dict(zip([a, b, c], S.distribute(minimum, list(ratio))))
    no_deduct = base_shares[c]                       # forgot to deduct shortfall
    deduct_from_actual = dict(zip([a, b, c], S.distribute(actual, list(ratio))))[c]
    full_shortfall_as_share = Fraction(shortfall)
    gr = f"{ratio[0]}:{ratio[1]}:{ratio[2]}"
    stem = _phrase(variant, [
        (f"{a}, {b} and {c} share profits {gr}. {c} has guaranteed the firm a minimum profit "
         f"of {_rs(Fraction(minimum))}. Actual profit is {_rs(Fraction(actual))}; {c} meets "
         f"the shortfall. {c}'s net share is:"),
        (f"{c} guarantees the firm a minimum profit of {_rs(Fraction(minimum))}, profits being "
         f"shared {gr} with {a} and {b}. The firm actually earns {_rs(Fraction(actual))}. After "
         f"making good the shortfall, {c}'s share is:"),
        (f"Partners {a}, {b} and {c} share {gr}. The firm's guaranteed minimum is "
         f"{_rs(Fraction(minimum))} (guaranteed by {c}) but it earned only {_rs(Fraction(actual))}. "
         f"{c}'s net share after bearing the shortfall is:"),
    ])
    working = r.working + [f"{c}'s net share = {_rs(correct)}"]
    return ScenarioDraft(
        "guarantee_to_firm", "Guarantee of Profit", stem, correct,
        [("did not deduct the shortfall from the guarantor's share", no_deduct),
         ("shared the actual profit, not the guaranteed minimum", deduct_from_actual),
         ("reported only the shortfall", full_shortfall_as_share)],
        working,
    )


def gen_opening_capital_plain(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    a, b = _pick(rng, _NAMES)
    closing = _amount(rng, 300000, 700000, 50000)
    drawings = _amount(rng, 30000, 80000, 10000)
    additional = _amount(rng, 50000, 150000, 50000)
    profit = _amount(rng, 40000, 120000, 20000)
    r = S.opening_capital_from_closing(closing, drawings=drawings,
                                       additional_capital=additional, share_of_profit=profit)
    correct = r.value
    sign_trap = S.opening_capital_from_closing(closing, drawings=-drawings,
                                               additional_capital=additional,
                                               share_of_profit=profit).value
    ignored_additional = S.opening_capital_from_closing(closing, drawings=drawings,
                                                        share_of_profit=profit).value
    ignored_profit = S.opening_capital_from_closing(closing, drawings=drawings,
                                                     additional_capital=additional).value
    stem = _phrase(variant, [
        (f"{a}'s capital account (fluctuating method) shows a closing balance of "
         f"{_rs(Fraction(closing))}. During the year {a} introduced additional capital "
         f"{_rs(Fraction(additional))}, withdrew {_rs(Fraction(drawings))}, and was credited "
         f"with a profit share of {_rs(Fraction(profit))}. {a}'s opening capital was:"),
        (f"At year end {a}'s capital stood at {_rs(Fraction(closing))}. In the year {a} added "
         f"{_rs(Fraction(additional))} of capital, drew {_rs(Fraction(drawings))}, and earned "
         f"a profit share of {_rs(Fraction(profit))}. Working back, {a}'s opening capital was:"),
        (f"{a} (fluctuating capital) closed the year with {_rs(Fraction(closing))} after "
         f"introducing {_rs(Fraction(additional))}, withdrawing {_rs(Fraction(drawings))}, and "
         f"being credited {_rs(Fraction(profit))} of profit. {a}'s opening capital was:"),
    ])
    working = r.working
    return ScenarioDraft(
        "opening_capital_plain", "Fundamentals of Partnership", stem, correct,
        [("added drawings instead of subtracting them", sign_trap),
         ("ignored the additional capital", ignored_additional),
         ("ignored the profit share", ignored_profit)],
        working,
    )


def gen_opening_capital_ioc_on_opening(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    a, b = _pick(rng, _NAMES)
    opening_true = _amount(rng, 200000, 500000, 50000)
    rate = _pick(rng, [8, 10, 12])
    drawings = _amount(rng, 20000, 60000, 10000)
    profit = _amount(rng, 30000, 90000, 10000)
    # construct a consistent closing from the true opening
    ioc = Fraction(opening_true) * rate / 100
    closing = Fraction(opening_true) + ioc + profit - drawings
    r = S.opening_capital_with_ioc_on_opening(closing, rate, drawings=drawings,
                                              share_of_profit=profit)
    correct = r.value
    no_div_trap = (Fraction(closing) - profit + drawings)                  # forgot to divide by (1+rate)
    over_div_trap = no_div_trap / (1 + Fraction(2 * rate, 100))            # divided by (1 + 2*rate)
    closing_as_opening = Fraction(closing)
    stem = _phrase(variant, [
        (f"{a} (fluctuating capital) has a closing capital of {_rs(closing)}. Interest on "
         f"capital @ {rate}% p.a. was allowed on the OPENING capital, profit credited "
         f"{_rs(Fraction(profit))}, drawings {_rs(Fraction(drawings))}. {a}'s opening capital was:"),
        (f"{a}'s closing capital is {_rs(closing)} after a profit of {_rs(Fraction(profit))}, "
         f"drawings of {_rs(Fraction(drawings))}, and interest on capital @ {rate}% p.a. "
         f"computed on the opening balance. {a}'s opening capital was:"),
        (f"With interest on capital @ {rate}% p.a. charged on the opening balance, {a} ends the "
         f"year with {_rs(closing)}, having drawn {_rs(Fraction(drawings))} and earned "
         f"{_rs(Fraction(profit))}. {a}'s opening capital was:"),
    ])
    working = r.working
    return ScenarioDraft(
        "opening_capital_ioc_on_opening", "Interest on Capital", stem, correct,
        [("forgot to divide by (1 + rate)", no_div_trap),
         ("divided by (1 + 2*rate) instead of (1 + rate)", over_div_trap),
         ("took the closing balance as the opening", closing_as_opening)],
        working,
    )


def gen_ioc_when_inadequate(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    a, b = _pick(rng, _NAMES)
    ioc_a = _amount(rng, 20000, 60000, 5000)
    ioc_b = _amount(rng, 15000, 50000, 5000)
    total = ioc_a + ioc_b
    profit = _amount(rng, max(10000, total // 4), total - 5000, 5000)
    r = S.ioc_when_inadequate(profit, {a: ioc_a, b: ioc_b})
    correct = r.parts[a]
    full_trap = Fraction(ioc_a)                           # gave full IOC despite shortage
    equal_trap = Fraction(profit, 2)                      # split profit equally
    b_share = r.parts[b]                                  # gave a what b gets
    stem = _phrase(variant, [
        (f"{a} and {b} are entitled to interest on capital of {_rs(Fraction(ioc_a))} and "
         f"{_rs(Fraction(ioc_b))} respectively. Interest on capital is an appropriation. The "
         f"firm's profit for the year is only {_rs(Fraction(profit))}. The interest on capital "
         f"actually allowed to {a} is:"),
        (f"Interest on capital due to {a} and {b} is {_rs(Fraction(ioc_a))} and "
         f"{_rs(Fraction(ioc_b))}, treated as an appropriation. As the year's profit is only "
         f"{_rs(Fraction(profit))}, the interest on capital allowed to {a} is:"),
        (f"The firm earned just {_rs(Fraction(profit))} this year. Interest on capital "
         f"(an appropriation) claimed by {a} and {b} is {_rs(Fraction(ioc_a))} and "
         f"{_rs(Fraction(ioc_b))}. How much interest on capital does {a} actually get?"),
    ])
    working = r.working + [f"{a} gets {profit} x {ioc_a}/{total} = {_rs(correct)}"]
    return ScenarioDraft(
        "ioc_when_inadequate", "Interest on Capital", stem, correct,
        [("allowed the full interest on capital despite the shortage", full_trap),
         ("split the available profit equally", equal_trap),
         (f"gave {a} the amount due to {b}", b_share)],
        working,
    )


def gen_partners_loan_interest(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    a, b = _pick(rng, _NAMES)
    loan = _amount(rng, 60000, 300000, 20000)
    months = _pick(rng, [6, 9])                       # never 12 (else full-year trap == correct)
    r = S.partners_loan_interest(loan, months)
    correct = r.value
    full_year = S.partners_loan_interest(loan, 12).value              # ignored the time factor
    deed_rate = Fraction(loan) * 10 / 100 * Fraction(months, 12)      # used 10% not 6%
    rate_twelve = Fraction(loan) * 12 / 100 * Fraction(months, 12)    # used 12% not 6%
    stem = _phrase(variant, [
        (f"{a} advanced a loan of {_rs(Fraction(loan))} to the firm {months} months before the "
         f"year end. The partnership deed is silent on loan interest. Interest on {a}'s loan "
         f"for the year is:"),
        (f"The firm took a loan of {_rs(Fraction(loan))} from {a} {months} months ago. With no "
         f"agreement on the rate, the interest on {a}'s loan for the year is:"),
        (f"{a} lent the firm {_rs(Fraction(loan))}, outstanding for {months} months. As the "
         f"deed says nothing about loan interest, {a}'s loan interest for the year is:"),
    ])
    working = [f"no deed -> 6% p.a. for {months} months", f"= {_rs(correct)}"]
    return ScenarioDraft(
        "partners_loan_interest", "Partner's Loan", stem, correct,
        [("charged a full year's interest", full_year),
         ("used 10% instead of the statutory 6%", deed_rate),
         ("used 12% instead of the statutory 6%", rate_twelve)],
        working,
    )


def gen_past_adjustment(rng: random.Random, difficulty: str, variant: int = 0) -> ScenarioDraft:
    a, b = _pick(rng, _NAMES)
    should_a = _amount(rng, 40000, 80000, 2000)
    should_b = _amount(rng, 30000, 70000, 2000)
    total = should_a + should_b
    got_each = Fraction(total, 2)
    r = S.past_adjustment({a: should_a, b: should_b}, {a: got_each, b: got_each})
    net_a = r.parts[a]
    correct = abs(net_a)
    wrong_total = abs(Fraction(should_a) - Fraction(total))       # subtracted correct share from whole profit
    half_trap = correct / 2                                       # halved the adjustment
    double_trap = correct * 2                                     # doubled the adjustment
    direction = "credited" if net_a > 0 else "debited"
    stem = _phrase(variant, [
        (f"On a past adjustment, {a} should have received {_rs(Fraction(should_a))} and {b} "
         f"{_rs(Fraction(should_b))}, but the profit of {_rs(Fraction(total))} was wrongly "
         f"shared equally. In the single adjusting entry, {a}'s capital account is {direction} by:"),
        (f"A profit of {_rs(Fraction(total))} was divided equally by mistake; it should have "
         f"gone {_rs(Fraction(should_a))} to {a} and {_rs(Fraction(should_b))} to {b}. In the "
         f"rectifying entry, {a}'s capital account is {direction} by:"),
        (f"{a} and {b} were each credited with half of {_rs(Fraction(total))}, though their "
         f"correct dues were {_rs(Fraction(should_a))} and {_rs(Fraction(should_b))}. The single "
         f"adjusting entry {direction}s {a}'s capital account by:"),
    ])
    working = r.working + [f"{a} net = {_rs(net_a)} ({direction})"]
    return ScenarioDraft(
        "past_adjustment", "Past Adjustments", stem, correct,
        [("subtracted the correct share from the whole profit", wrong_total),
         ("halved the adjustment", half_trap),
         ("doubled the adjustment", double_trap)],
        working,
    )


# ── month / word helpers ──────────────────────────────────────────────────

def _month_name(months_elapsed: int) -> str:
    # Apr-Mar year: months_elapsed=6 -> 1 October
    names = ["1 April", "1 May", "1 June", "1 July", "1 August", "1 September",
             "1 October", "1 November", "1 December", "1 January", "1 February", "1 March"]
    return names[months_elapsed % 12]


def _freq_word(freq: str) -> str:
    return {"monthly": "month", "quarterly": "quarter",
            "half-yearly": "half-year"}[freq]


# ── registry ────────────────────────────────────────────────────────────────

FAMILY_GENERATORS: Dict[str, Callable[[random.Random, str], ScenarioDraft]] = {
    "interest_on_capital": gen_interest_on_capital,
    "interest_on_drawings": gen_interest_on_drawings,
    "interest_on_drawings_flat": gen_interest_on_drawings_flat,
    "interest_on_drawings_irregular": gen_interest_on_drawings_irregular,
    "pl_appropriation": gen_pl_appropriation,
    "profit_share": gen_profit_share,
    "commission_before": gen_commission_before,
    "commission_after": gen_commission_after,
    "guarantee_to_partner": gen_guarantee_to_partner,
    "guarantee_to_firm": gen_guarantee_to_firm,
    "opening_capital_plain": gen_opening_capital_plain,
    "opening_capital_ioc_on_opening": gen_opening_capital_ioc_on_opening,
    "ioc_when_inadequate": gen_ioc_when_inadequate,
    "partners_loan_interest": gen_partners_loan_interest,
    "past_adjustment": gen_past_adjustment,
}

ALL_FAMILIES = list(FAMILY_GENERATORS.keys())

# How many structurally/phrasing distinct questions each family can produce before
# the dedup gate would start removing near-identical repeats. The blueprint planner
# never allocates a family more than this.
FAMILY_CAPACITY: Dict[str, int] = {
    "interest_on_capital": 4,            # 4 structural sub-cases
    "interest_on_drawings": 6,           # 3 frequencies x 2 timings
    "interest_on_drawings_flat": 3,
    "interest_on_drawings_irregular": 3,
    "pl_appropriation": 4,
    "profit_share": 4,
    "commission_before": 3,
    "commission_after": 4,
    "guarantee_to_partner": 3,
    "guarantee_to_firm": 3,
    "opening_capital_plain": 3,
    "opening_capital_ioc_on_opening": 3,
    "ioc_when_inadequate": 3,
    "partners_loan_interest": 3,
    "past_adjustment": 3,
}

# Rotation used when the teacher asks for N numeric MCQs on "Fundamentals of
# Partnership" without naming a sub-topic.
FUNDAMENTALS_ROTATION = [
    "interest_on_capital", "interest_on_drawings", "pl_appropriation",
    "profit_share", "commission_after", "interest_on_drawings_flat",
    "guarantee_to_partner", "opening_capital_plain", "commission_before",
    "interest_on_drawings_irregular", "ioc_when_inadequate",
    "guarantee_to_firm", "partners_loan_interest",
    "opening_capital_ioc_on_opening", "past_adjustment",
]

# Keyword -> family, for routing a teacher's sub-topic string.
_SUBTOPIC_KEYWORDS: List[Tuple[str, str]] = [
    ("interest on capital", "interest_on_capital"),
    ("interest on drawing", "interest_on_drawings"),
    ("appropriation", "pl_appropriation"),
    ("distribution of profit", "profit_share"),
    ("profit sharing", "profit_share"),
    ("commission", "commission_after"),
    ("guarantee", "guarantee_to_partner"),
    ("past adjustment", "past_adjustment"),
    ("loan", "partners_loan_interest"),
    ("opening capital", "opening_capital_plain"),
]

_FUNDAMENTALS_CHAPTER = re.compile(r"fundamental|partnership firm|partnership.*fundamental", re.I)


def detect_numeric_family(chapter: str, topic: Optional[str] = None) -> Optional[str]:
    """Return a family name if the chapter/topic is a numeric Fundamentals family,
    else None. A None here means 'not a solve-first numeric family' — the caller
    keeps the legacy path."""
    hay = f"{chapter or ''} {topic or ''}".lower()
    for kw, fam in _SUBTOPIC_KEYWORDS:
        if kw in hay:
            return fam
    return None


def is_fundamentals_chapter(chapter: str) -> bool:
    return bool(_FUNDAMENTALS_CHAPTER.search(chapter or "")) or \
        "fundamentals of partnership" in (chapter or "").lower()


# ── MCQ builder ───────────────────────────────────────────────────────────

def build_numeric_mcq(family: str, seed: int, difficulty: str = "medium",
                      correct_index: Optional[int] = None, variant: int = 0,
                      max_tries: int = 60) -> NumericMCQ:
    """Build one solver-verified numeric MCQ for `family`. Deterministic in `seed`.

    `correct_index` places the correct option at a specific letter (A=0..D=3) —
    the paper-level balancer (generate_numeric_mcqs, Phase 1E) uses this to keep
    the A/B/C/D distribution uniform. When None, a seed-derived index is used.

    Retries with an advanced RNG if the distractor set collides with the answer
    (make_options raises), so a given seed is stable but never ships a bad option set.
    """
    gen = FAMILY_GENERATORS.get(family)
    if gen is None:
        raise ValueError(f"unknown numeric family: {family}")

    last_err: Optional[Exception] = None
    for attempt in range(max_tries):
        rng = random.Random(f"{family}:{seed}:{variant}:{attempt}")
        draft = gen(rng, difficulty, variant)
        try:
            built = S.make_options(draft.correct, [v for _, v in draft.distractors])
        except ValueError as e:
            last_err = e
            continue

        values = built["options"]                         # index 0 == correct
        correct_idx = (correct_index if correct_index is not None
                       else random.Random(f"{family}:{seed}:pos").randrange(4))
        distractor_vals = list(values[1:])
        random.Random(f"{family}:{seed}:distr").shuffle(distractor_vals)
        shuffled: List = [None, None, None, None]
        shuffled[correct_idx] = values[0]
        di = 0
        for i in range(4):
            if i == correct_idx:
                continue
            shuffled[i] = distractor_vals[di]
            di += 1
        option_texts = [f"{_LETTERS[i]}) {_rs(shuffled[i])}" for i in range(4)]
        correct_letter = _LETTERS[correct_idx]

        # Build a readable distractor note mapping each wrong option to its error.
        label_by_value = {S.rupees(v): lbl for lbl, v in draft.distractors}
        wrong_notes = []
        for i in range(4):
            if i == correct_idx:
                continue
            lbl = label_by_value.get(S.rupees(shuffled[i]), "distractor")
            wrong_notes.append(f"{_LETTERS[i]} = {lbl}")
        explanation = (" ".join(draft.working)
                       + "  Correct option " + correct_letter + ". "
                       + "Distractors: " + "; ".join(wrong_notes) + ".")

        return NumericMCQ(
            text=draft.stem,
            options=option_texts,
            correct_answer=option_texts[correct_idx],
            correct_letter=correct_letter,
            explanation=explanation,
            topic=draft.topic,
            family=family,
            difficulty=difficulty,
        )

    raise RuntimeError(f"could not build a clean option set for {family} after "
                       f"{max_tries} tries: {last_err}")


def _balanced_letters(n: int, seed: int) -> List[int]:
    """A length-n list of correct-option indices (0..3) as uniform as possible
    (each letter within +/-1 of n/4), deterministically shuffled by seed."""
    counts = [n // 4] * 4
    for i in range(n % 4):
        counts[i] += 1
    seq: List[int] = []
    for letter, c in enumerate(counts):
        seq += [letter] * c
    random.Random(f"balance:{seed}:{n}").shuffle(seq)
    return seq


def generate_numeric_mcqs(count: int, *, seed: int = 0, difficulty: str = "medium",
                          families: Optional[Sequence[str]] = None,
                          family_sequence: Optional[Sequence[str]] = None) -> List[Dict]:
    """Build solver-verified numeric MCQs with the correct option position balanced
    across the paper (Phase 1E). The family of each question comes from, in order:
    `family_sequence` (an explicit blueprint plan, Phase 2B), else a flat rotation
    over `families`, else the Fundamentals rotation. Each recurrence of a family
    advances its variant so phrasing / sub-structure differs (no near-verbatim
    repeats). Returns at most `count` question dicts."""
    if family_sequence is not None:
        seq = list(family_sequence)[:count]
    else:
        rotation = list(families) if families else list(FUNDAMENTALS_ROTATION)
        seq = [rotation[i % len(rotation)] for i in range(count)]

    letters = _balanced_letters(len(seq), seed)
    out: List[Dict] = []
    fam_seen: Dict[str, int] = {}
    for i, fam in enumerate(seq):
        variant = fam_seen.get(fam, 0)
        fam_seen[fam] = variant + 1
        mcq = build_numeric_mcq(fam, seed=seed * 1000 + i, difficulty=difficulty,
                                correct_index=letters[i], variant=variant)
        out.append(mcq.as_question_dict())
    return out


# ── optional LLM prose polish (numbers are code-owned; LLM only rewrites text) ──

_NUM_RE = re.compile(r"\d[\d,]*")


def _numbers_in(text: str) -> set:
    return {m.group(0).replace(",", "") for m in _NUM_RE.finditer(text or "")}


def polish_stem_with_llm(stem: str, llm_call: Callable[[str], str]) -> str:
    """Ask an LLM to rewrite ONLY the prose of `stem`, keeping every number. If the
    rewrite drops or changes any number (or errors), return the original template
    stem unchanged. The numbers are owned by the solver, never by the model."""
    want = _numbers_in(stem)
    prompt = (
        "Rewrite the following CBSE Accountancy MCQ question stem to read naturally "
        "for a Class 12 student. Keep EVERY number, name, rate and date EXACTLY as "
        "given. Do not add options. Do not give the answer. Return only the rewritten "
        f"stem.\n\nSTEM:\n{stem}"
    )
    try:
        out = (llm_call(prompt) or "").strip()
    except Exception:
        return stem
    if not out:
        return stem
    if not want.issubset(_numbers_in(out)):
        return stem                      # numbers drifted -> keep the trusted template
    return out
