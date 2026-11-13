"""
Blueprint + coverage gate (Phase 2B) for the simple (non-CBSE) path.

A blueprint is a weighted subtopic table per chapter. The planner allocates the
requested N questions across subtopics in proportion to the weights, capped at
each subtopic's distinct-question capacity (so it never asks a family for more
distinct questions than it can make). The coverage gate then checks an assembled
paper against the plan and flags any subtopic that is over- or under-weighted
(> 200% / < 50% of its planned share) — so a professional-looking paper with
lopsided coverage is blocked instead of shipped.

This replaces the free-pick-subtopics behaviour (test_generator_service
_build_chapter_prompt L1175 / "test a DIFFERENT sub-topic" L1206), where coverage
was left entirely to the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.services.accountancy_scenarios import FAMILY_CAPACITY, FUNDAMENTALS_ROTATION

# Chapter (normalised, lower-case) -> [(subtopic/family, weight)]. Weights are
# relative exam frequency; the planner normalises them. Keys are solve-first
# family names so the plan maps directly onto generated questions.
CHAPTER_BLUEPRINTS: Dict[str, List[Tuple[str, int]]] = {
    "fundamentals of partnership": [
        ("interest_on_capital", 3),
        ("interest_on_drawings", 3),
        ("interest_on_drawings_flat", 1),
        ("interest_on_drawings_irregular", 1),
        ("pl_appropriation", 3),
        ("profit_share", 2),
        ("commission_before", 1),
        ("commission_after", 2),
        ("guarantee_to_partner", 2),
        ("guarantee_to_firm", 1),
        ("opening_capital_plain", 1),
        ("opening_capital_ioc_on_opening", 1),
        ("ioc_when_inadequate", 1),
        ("partners_loan_interest", 1),
        ("past_adjustment", 2),
    ],
}

DEFAULT_LOW = 0.5     # < 50% of plan -> under-covered
DEFAULT_HIGH = 2.0    # > 200% of plan -> over-covered


def get_blueprint(chapter: str) -> Optional[List[Tuple[str, int]]]:
    return CHAPTER_BLUEPRINTS.get((chapter or "").strip().lower())


def _largest_remainder(ideals: Dict[str, float], n: int) -> Dict[str, int]:
    floors = {k: int(v) for k, v in ideals.items()}
    used = sum(floors.values())
    remainder = n - used
    # hand the remaining units to the largest fractional parts
    order = sorted(ideals, key=lambda k: ideals[k] - floors[k], reverse=True)
    for k in order[:max(0, remainder)]:
        floors[k] += 1
    return floors


def plan_allocation(blueprint: List[Tuple[str, int]], n: int,
                    capacity: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """Allocate `n` questions across the blueprint's subtopics by weight, capped
    at each subtopic's capacity. Overflow from capped subtopics is redistributed
    (by weight) to subtopics with spare capacity. If `n` exceeds total capacity,
    the plan sums to the total capacity (the paper cannot be filled further
    without repeating a structure)."""
    capacity = capacity or FAMILY_CAPACITY
    weights = {k: w for k, w in blueprint}
    total_w = sum(weights.values())
    if total_w == 0 or n <= 0:
        return {k: 0 for k in weights}

    ideals = {k: n * w / total_w for k, w in weights.items()}
    alloc = _largest_remainder(ideals, n)

    # Cap at capacity, collect overflow.
    overflow = 0
    for k in alloc:
        cap = capacity.get(k, alloc[k])
        if alloc[k] > cap:
            overflow += alloc[k] - cap
            alloc[k] = cap

    # Redistribute overflow to subtopics with spare capacity, by weight, round-robin.
    while overflow > 0:
        spare = [k for k in weights if alloc[k] < capacity.get(k, alloc[k])]
        if not spare:
            break
        # weight-ordered so heavier subtopics absorb more first
        spare.sort(key=lambda k: weights[k], reverse=True)
        progressed = False
        for k in spare:
            if overflow <= 0:
                break
            alloc[k] += 1
            overflow -= 1
            progressed = True
        if not progressed:
            break

    return alloc


@dataclass
class CoverageResult:
    ok: bool
    plan: Dict[str, int]
    actual: Dict[str, int]
    violations: List[dict] = field(default_factory=list)   # {subtopic, planned, actual, kind}

    def summary(self) -> str:
        if self.ok:
            return f"coverage within bounds across {len([v for v in self.plan.values() if v])} subtopics"
        return "; ".join(f"{v['subtopic']} {v['kind']} ({v['actual']} vs plan {v['planned']})"
                         for v in self.violations)


def coverage_report(actual: Dict[str, int], plan: Dict[str, int],
                    low: float = DEFAULT_LOW, high: float = DEFAULT_HIGH) -> CoverageResult:
    """Flag subtopics whose actual count is < low*plan or > high*plan. Subtopics
    planned at 0 are ignored (not part of the blueprint share); a subtopic planned
    > 0 but absent from the paper is an under-coverage violation."""
    violations = []
    for sub, planned in plan.items():
        if planned <= 0:
            continue
        got = actual.get(sub, 0)
        if got < low * planned:
            violations.append({"subtopic": sub, "planned": planned, "actual": got,
                               "kind": "under-covered"})
        elif got > high * planned:
            violations.append({"subtopic": sub, "planned": planned, "actual": got,
                               "kind": "over-covered"})
    return CoverageResult(ok=not violations, plan=plan, actual=actual, violations=violations)


def plan_sequence(blueprint: List[Tuple[str, int]], n: int,
                  capacity: Optional[Dict[str, int]] = None) -> List[str]:
    """A length-(<=n) list of family names following the plan, interleaved so the
    same family is spread across the paper rather than clustered. Each family
    appears exactly alloc[family] times."""
    alloc = plan_allocation(blueprint, n, capacity)
    # Interleave: repeatedly take one from each remaining family (heaviest first).
    remaining = dict(alloc)
    order = [k for k, _ in blueprint]
    seq: List[str] = []
    while any(remaining[k] > 0 for k in order):
        for k in order:
            if remaining[k] > 0:
                seq.append(k)
                remaining[k] -= 1
    return seq
