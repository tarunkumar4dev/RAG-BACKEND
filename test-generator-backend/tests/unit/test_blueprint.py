"""Phase 2B: weighted blueprint, capacity-aware planner, coverage gate."""
from collections import Counter

from app.services import blueprint as bp
from app.services import accountancy_scenarios as sc


FUND = bp.get_blueprint("Fundamentals of Partnership")


def test_blueprint_lookup_is_case_insensitive():
    assert bp.get_blueprint("FUNDAMENTALS OF PARTNERSHIP") is not None
    assert bp.get_blueprint("Dissolution of a Firm") is None


def test_plan_allocation_sums_to_n_within_capacity():
    for n in (10, 20, 34, 50):
        plan = bp.plan_allocation(FUND, n)
        total_cap = sum(sc.FAMILY_CAPACITY[k] for k, _ in FUND)
        assert sum(plan.values()) == min(n, total_cap)


def test_plan_allocation_never_exceeds_capacity():
    plan = bp.plan_allocation(FUND, 50)
    for fam, c in plan.items():
        assert c <= sc.FAMILY_CAPACITY[fam], (fam, c)


def test_plan_allocation_roughly_proportional_to_weight():
    plan = bp.plan_allocation(FUND, 50)
    # interest_on_capital (w3) should get more than commission_before (w1).
    assert plan["interest_on_capital"] > plan["commission_before"]


def test_plan_sequence_interleaves_and_matches_allocation():
    seq = bp.plan_sequence(FUND, 50)
    counts = Counter(seq)
    assert dict(counts) == {k: v for k, v in bp.plan_allocation(FUND, 50).items() if v}
    # No family should appear in a long consecutive run (interleaved).
    runs = max(len(list(g)) for _, g in __import__("itertools").groupby(seq))
    assert runs <= 2, f"a family clustered ({runs} in a row)"


def test_coverage_gate_passes_for_planned_paper():
    plan = bp.plan_allocation(FUND, 50)
    report = bp.coverage_report(dict(plan), plan)
    assert report.ok, report.summary()


def test_coverage_gate_flags_overcovered():
    plan = {"interest_on_capital": 6, "pl_appropriation": 6}
    actual = {"interest_on_capital": 20, "pl_appropriation": 6}   # 20 > 2*6
    report = bp.coverage_report(actual, plan)
    assert not report.ok
    assert report.violations[0]["kind"] == "over-covered"


def test_coverage_gate_flags_undercovered():
    plan = {"interest_on_capital": 6, "pl_appropriation": 6}
    actual = {"interest_on_capital": 2, "pl_appropriation": 6}    # 2 < 0.5*6
    report = bp.coverage_report(actual, plan)
    assert not report.ok
    assert report.violations[0]["kind"] == "under-covered"


def test_generated_paper_passes_its_own_coverage_gate():
    """A blueprint-planned 50-MCQ paper satisfies the coverage gate by construction."""
    seq = bp.plan_sequence(FUND, 50)
    qs = sc.generate_numeric_mcqs(50, seed=3, difficulty="hard", family_sequence=seq)
    actual = Counter(q["family"] for q in qs)
    plan = bp.plan_allocation(FUND, 50)
    assert bp.coverage_report(dict(actual), plan).ok
