"""
Tests for accountancy_solver. EXPECTED VALUES ARE HAND-COMPUTED LITERALS, not
solver output — so the solver is not grading itself. Each literal is derived by the
standard CBSE method in the comment above it. Run: pytest test_accountancy_solver.py
"""
from fractions import Fraction
from app.services import accountancy_solver as s


def eq(result, expected_rupees):
    assert result.rupees() == Fraction(expected_rupees), (result.value, result.working)


# ---- interest on capital -------------------------------------------------- #
def test_ioc_with_additional_and_withdrawal():
    # 400000 full yr @10% =40000; +100000 on 1 Oct (6 mo left)=5000; -50000 on 1 Jan (3 mo left)=-1250
    r = s.interest_on_capital(400000, 10, [(6, 100000), (9, -50000)])
    eq(r, 40000 + 5000 - 1250)        # 43750

def test_ioc_full_year_simple():
    eq(s.interest_on_capital(300000, 10), 30000)

def test_opening_capital_ioc_on_opening():
    # closing 288000, drawings 40000, IOD 2000, profit 55000, IOC=10% of opening
    # 1.1*opening = 288000 - 55000 + 40000 + 2000 = 275000 -> opening 250000
    r = s.opening_capital_with_ioc_on_opening(288000, 10, drawings=40000,
                                              interest_on_drawings=2000, share_of_profit=55000)
    eq(r, 250000)

def test_opening_capital_plain():
    # closing 400000, drawings 50000, additional 100000, profit 80000 -> opening 270000
    eq(s.opening_capital_from_closing(400000, drawings=50000,
                                      additional_capital=100000, share_of_profit=80000), 270000)

def test_ioc_inadequate_profit():
    # profit 40000 < IOC(30000+20000=50000) -> allocate 40000 in 30:20 => A 24000, B 16000
    r = s.ioc_when_inadequate(40000, {"A": 30000, "B": 20000})
    assert s.rupees(r.parts["A"]) == Fraction(24000)
    assert s.rupees(r.parts["B"]) == Fraction(16000)


# ---- interest on drawings ------------------------------------------------- #
def test_iod_monthly_middle():
    # 10000/mo, middle, 12% -> sum months 72, product 720000, IOD 7200
    eq(s.interest_on_drawings(10000, 12, "monthly", "middle"), 7200)

def test_iod_monthly_end_full_year():
    # 5000/mo end, 12% -> sum 66, product 330000, IOD 3300
    eq(s.interest_on_drawings(5000, 12, "monthly", "end"), 3300)

def test_iod_quarterly_end():
    # 12000/qtr end, 10% -> months 9,6,3,0 sum18, product 216000, IOD 1800
    eq(s.interest_on_drawings(12000, 10, "quarterly", "end"), 1800)

def test_iod_monthly_begin():
    # 2000/mo begin, 9% -> sum 78, product 156000, IOD 1170
    eq(s.interest_on_drawings(2000, 9, "monthly", "beginning"), 1170)

def test_iod_half_yearly_begin():
    # 15000 half-yearly begin, 8% -> months 12,6 sum18, product 270000, IOD 1800
    eq(s.interest_on_drawings(15000, 8, "half-yearly", "beginning"), 1800)

def test_iod_partial_year_end():
    # 4000/mo end, Oct-Mar = 6 installments, months 5,4,3,2,1,0 sum15, 12% -> 600
    eq(s.interest_on_drawings(4000, 12, "monthly", "end", n=6, year_months=6), 600)

def test_iod_irregular_product_method():
    # 10000@11 + 6000@7 + 8000@4 + 4000@1 = 110000+42000+32000+4000=188000; 12% -> 1880
    items = [(10000, 11), (6000, 7), (8000, 4), (4000, 1)]
    eq(s.interest_on_drawings_irregular(items, 12), 1880)

def test_iod_flat_not_pa():
    eq(s.interest_on_drawings_flat(40000, 5), 2000)


# ---- commission ----------------------------------------------------------- #
def test_commission_after_charging():
    eq(s.commission_after_charging(220000, 10), 20000)

def test_commission_before_charging():
    eq(s.commission_before_charging(264000, 20), 52800)


# ---- appropriation -------------------------------------------------------- #
def test_appropriation_divisible_profit():
    # 180000 - IOC 20000 - salary 36000 + IOD 4000 = 128000
    r = s.appropriation(180000, interest_on_capital={"x": 20000},
                        salaries={"B": 36000}, interest_on_drawings={"all": 4000})
    eq(r, 128000)

def test_appropriation_shares():
    # divisible 100000 in 3:2 -> 60000 / 40000
    r = s.appropriation(100000, ratio={"A": 3, "B": 2})
    assert s.rupees(r.parts["A"]) == Fraction(60000)
    assert s.rupees(r.parts["B"]) == Fraction(40000)


# ---- guarantee ------------------------------------------------------------ #
def test_guarantee_to_partner():
    # A,B,C 3:2:1 of 240000 = 120000,80000,40000; C min 50000; deficiency 10000 from A,B 3:2
    shares = dict(zip(["A", "B", "C"], s.distribute(240000, [3, 2, 1])))
    r = s.guarantee_to_partner(shares, "C", 50000, {"A": 3, "B": 2})
    assert s.rupees(r.parts["A"]) == Fraction(114000)
    assert s.rupees(r.parts["B"]) == Fraction(76000)
    assert s.rupees(r.parts["C"]) == Fraction(50000)

def test_guarantee_to_firm():
    # 2:2:1, guarantee firm 200000, actual 170000, shortfall 30000 borne by C
    # base 200000 -> 80000,80000,40000; C = 40000 - 30000 = 10000
    r = s.guarantee_to_firm(170000, 200000, {"A": 2, "B": 2, "C": 1}, "C")
    assert s.rupees(r.parts["C"]) == Fraction(10000)
    assert s.rupees(r.parts["A"]) == Fraction(80000)


# ---- past adjustment ------------------------------------------------------ #
def test_past_adjustment_nets_to_zero():
    # should: A 20000+32000=52000-ish vs actually; use a clean balanced case
    # A should receive 52000, got 45000 -> +7000 ; B should 38000, got 45000 -> -7000
    r = s.past_adjustment({"A": 52000, "B": 38000}, {"A": 45000, "B": 45000})
    assert s.rupees(r.parts["A"]) == Fraction(7000)
    assert s.rupees(r.parts["B"]) == Fraction(-7000)


# ---- partner's loan ------------------------------------------------------- #
def test_partners_loan_interest_half_year():
    # 50000 @6% for 9 months = 2250
    eq(s.partners_loan_interest(50000, 9), 2250)


# ---- option builder integrity --------------------------------------------- #
def test_make_options_exactly_one_correct():
    o = s.make_options(2250, [3750, 3000, 1500])
    assert o["options"][o["correct_index"]] == Fraction(2250)
    assert len(set(o["options"])) == 4

def test_make_options_rejects_colliding_distractor():
    import pytest
    with pytest.raises(ValueError):
        s.make_options(2000, [2000, 1000, 500])   # trap equals answer -> must fail loud


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
    print(f"{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
