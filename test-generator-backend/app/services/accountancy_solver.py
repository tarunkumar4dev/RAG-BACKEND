"""
a4ai — Deterministic solver for CBSE Class 12 Accountancy:
"Accounting for Partnership Firms — Fundamentals".

Phase 1A reference prototype. PORT VERBATIM into
  app/services/accountancy_solver.py

Design rules:
  * Pure functions, no I/O, no LLM, no globals.
  * Exact arithmetic with fractions.Fraction end-to-end; round to rupee ONLY at
    the boundary via rupees(). This avoids the float drift that lets a wrong
    option look "close enough".
  * Every function returns a Result(value, working) so the stem/solution text can
    be generated from `working` instead of being re-reasoned by the LLM.
  * Families flagged  # SME-REVIEW  are multi-step; get a human SME sign-off on the
    benchmark fixtures for these before treating solver output as ground truth.

Covered families (audit Part C list):
  interest on capital (incl. opening-capital derivation, mid-year movements,
  inadequate-profit proportioning, charge vs appropriation);
  interest on drawings (monthly/quarterly/half-yearly x begin/mid/end,
  product method for irregular, flat "rate not p.a.");
  profit & loss appropriation; commission before/after charging;
  guarantee of profit + deficiency distribution (to a partner and to the firm);
  past adjustments (omitted IOC/salary/commission); partner's loan interest.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

Num = Fraction


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def F(x) -> Fraction:
    """Coerce int / float-free decimal string / Fraction to an exact Fraction."""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, str):
        return Fraction(x)
    # tolerate float input but go through its exact decimal string
    return Fraction(str(x))


def rupees(x: Fraction, places: int = 0) -> Fraction:
    """Round-half-up to `places` decimals, returned as Fraction for exact compare."""
    q = Fraction(10) ** places
    v = x * q
    # round half up (CBSE convention), handling negatives symmetrically
    floor = v.numerator // v.denominator
    frac = v - floor
    rounded = floor + (1 if frac >= Fraction(1, 2) else 0)
    return Fraction(rounded) / q


@dataclass
class Result:
    value: Fraction                      # the exact answer (rupees unless noted)
    working: List[str] = field(default_factory=list)
    parts: Dict[str, Fraction] = field(default_factory=dict)  # named sub-results

    def rupees(self, places: int = 0) -> Fraction:
        return rupees(self.value, places)


def distribute(amount, ratio: Sequence) -> List[Fraction]:
    """Split `amount` in `ratio`. Exact; no rounding."""
    amt = F(amount)
    r = [F(x) for x in ratio]
    total = sum(r)
    if total == 0:
        raise ValueError("ratio sums to zero")
    return [amt * x / total for x in r]


def interest(principal, rate_pa, months=12) -> Fraction:
    """Simple interest for `months` out of 12. Exact."""
    return F(principal) * F(rate_pa) / 100 * F(months) / 12


# --------------------------------------------------------------------------- #
# Interest on Capital
# --------------------------------------------------------------------------- #
def interest_on_capital(opening_capital, rate_pa, movements: Optional[Sequence[Tuple]] = None,
                        year_months: int = 12) -> Result:
    """
    IOC via the product (time) method.
      opening_capital : balance at the start of the year.
      movements       : list of (month_of_change, delta) where month_of_change is
                        months ELAPSED since year start (e.g. additional capital on
                        1 Oct of an Apr-Mar year -> 6), delta positive for capital
                        introduced, negative for capital withdrawn.
    Interest = sum over segments of (balance * segment_months) * rate/100/12.
    """
    moves = sorted([(F(m), F(d)) for m, d in (movements or [])], key=lambda t: t[0])
    bal = F(opening_capital)
    t = F(0)
    total = F(0)
    work = []
    for m, delta in moves:
        seg = m - t
        if seg < 0:
            raise ValueError("movement months must be within the year and ascending")
        total += bal * seg
        work.append(f"{bal} x {seg} mo")
        bal += delta
        t = m
    seg = F(year_months) - t
    total += bal * seg
    work.append(f"{bal} x {seg} mo")
    ioc = total * F(rate_pa) / 100 / 12
    return Result(ioc, [f"product = {total}", f"IOC = {total} x {rate_pa}/100/12 = {ioc}"] + work)


def opening_capital_from_closing(closing, *, drawings=0, interest_on_drawings=0,
                                 additional_capital=0, salary=0, interest_on_capital=0,
                                 share_of_profit=0) -> Result:
    """
    Fluctuating capital: reverse the year to recover opening capital.
      closing = opening + additional + salary + IOC + profit - drawings - IOD
    => opening = closing - additional - salary - IOC - profit + drawings + IOD
    Use this when IOC is a KNOWN number. If IOC itself depends on the unknown
    opening (IOC = rate% of opening), use opening_capital_with_ioc_on_opening().
    """
    opening = (F(closing) - F(additional_capital) - F(salary) - F(interest_on_capital)
               - F(share_of_profit) + F(drawings) + F(interest_on_drawings))
    return Result(opening, [
        f"opening = {closing} - {additional_capital} - {salary} - {interest_on_capital}"
        f" - {share_of_profit} + {drawings} + {interest_on_drawings} = {opening}"])


def opening_capital_with_ioc_on_opening(closing, rate_pa, *, drawings=0,
                                        interest_on_drawings=0, additional_capital=0,
                                        salary=0, share_of_profit=0) -> Result:
    """
    Fluctuating capital where IOC = rate% x opening (full year, no mid-year move).
      closing = opening + rate*opening + profit + salary + additional - drawings - IOD
    => opening = (closing - profit - salary - additional + drawings + IOD) / (1 + rate)
    """
    r = F(rate_pa) / 100
    num = (F(closing) - F(share_of_profit) - F(salary) - F(additional_capital)
           + F(drawings) + F(interest_on_drawings))
    opening = num / (1 + r)
    return Result(opening, [f"opening = {num} / (1 + {rate_pa}/100) = {opening}"],
                  parts={"ioc": opening * r})


def ioc_when_inadequate(net_profit, iocs: Dict[str, Fraction]) -> Result:  # SME-REVIEW
    """
    IOC treated as an APPROPRIATION and profit < total IOC claimed.
    Available profit is allocated in the ratio of the partners' IOC claims; no
    partner gets full IOC, and the firm shows no profit for distribution.
    """
    claims = {k: F(v) for k, v in iocs.items()}
    total = sum(claims.values())
    np = F(net_profit)
    if np >= total:
        return Result(np, [f"profit {np} >= total IOC {total}: full IOC allowed"], parts=claims)
    alloc = {k: np * v / total for k, v in claims.items()}
    return Result(np, [f"profit {np} < total IOC {total}: allocate in IOC ratio"], parts=alloc)


# --------------------------------------------------------------------------- #
# Interest on Drawings
# --------------------------------------------------------------------------- #
_INTERVAL = {"monthly": 1, "quarterly": 3, "half-yearly": 6, "halfyearly": 6}


def _months_list(n: int, interval: int, timing: str, year: int = 12) -> List[Fraction]:
    out = []
    for k in range(n):
        if timing == "beginning":
            m = year - k * interval
        elif timing == "end":
            m = year - (k + 1) * interval
        elif timing == "middle":
            m = F(year) - k * interval - Fraction(interval, 2)
        else:
            raise ValueError("timing must be beginning|middle|end")
        out.append(F(m))
    return out


def interest_on_drawings(amount, rate_pa, frequency: str, timing: str, *,
                         n: Optional[int] = None, year_months: int = 12) -> Result:
    """
    Equal `amount` withdrawn every period (monthly/quarterly/half-yearly) at the
    beginning/middle/end of each period, over `year_months`. Average-period method
    via explicit months-outstanding so partial years (e.g. 6 months) are exact.
    """
    interval = _INTERVAL[frequency.lower()]
    if n is None:
        if year_months % interval:
            raise ValueError("year not divisible by interval; pass n explicitly")
        n = year_months // interval
    months = _months_list(n, interval, timing, year_months)
    product = F(amount) * sum(months)          # total of (amount x months)
    iod = product * F(rate_pa) / 100 / 12
    return Result(iod, [
        f"installments={n}, months={[str(m) for m in months]}",
        f"sum(months)={sum(months)}, product={product}",
        f"IOD = {product} x {rate_pa}/100/12 = {iod}"])


def interest_on_drawings_irregular(items: Sequence[Tuple], rate_pa) -> Result:
    """
    Product method for unequal / irregular drawings.
      items : list of (amount, months_outstanding_to_year_end)
    """
    product = sum(F(a) * F(m) for a, m in items)
    iod = product * F(rate_pa) / 100 / 12
    return Result(iod, [f"sum(amount x months)={product}",
                        f"IOD = {product} x {rate_pa}/100/12 = {iod}"])


def interest_on_drawings_flat(total_drawings, rate) -> Result:
    """Rate given WITHOUT 'p.a.' -> flat % on total drawings, no time factor."""
    iod = F(total_drawings) * F(rate) / 100
    return Result(iod, [f"IOD = {total_drawings} x {rate}/100 = {iod} (flat, not p.a.)"])


# --------------------------------------------------------------------------- #
# Commission
# --------------------------------------------------------------------------- #
def commission_before_charging(profit, rate) -> Result:
    c = F(profit) * F(rate) / 100
    return Result(c, [f"commission = {profit} x {rate}% = {c}"])


def commission_after_charging(profit, rate) -> Result:
    c = F(profit) * F(rate) / (100 + F(rate))
    return Result(c, [f"commission = {profit} x {rate}/(100+{rate}) = {c}"])


# --------------------------------------------------------------------------- #
# Profit & Loss Appropriation
# --------------------------------------------------------------------------- #
def appropriation(net_profit, *, interest_on_capital: Optional[Dict] = None,
                  salaries: Optional[Dict] = None, commissions: Optional[Dict] = None,
                  interest_on_drawings: Optional[Dict] = None,
                  ratio: Optional[Dict[str, Fraction]] = None) -> Result:
    """
    Divisible profit = net profit + total IOD - (total IOC + salaries + commissions).
    If `ratio` (partner->share) given, also returns each partner's profit share.
    """
    ioc = {k: F(v) for k, v in (interest_on_capital or {}).items()}
    sal = {k: F(v) for k, v in (salaries or {}).items()}
    com = {k: F(v) for k, v in (commissions or {}).items()}
    iod = {k: F(v) for k, v in (interest_on_drawings or {}).items()}
    divisible = (F(net_profit) + sum(iod.values())
                 - sum(ioc.values()) - sum(sal.values()) - sum(com.values()))
    work = [f"divisible = {net_profit} + IOD {sum(iod.values())} "
            f"- IOC {sum(ioc.values())} - salary {sum(sal.values())} "
            f"- commission {sum(com.values())} = {divisible}"]
    parts = {}
    if ratio:
        keys = list(ratio.keys())
        shares = distribute(divisible, [ratio[k] for k in keys])
        parts = dict(zip(keys, shares))
    return Result(divisible, work, parts=parts)


# --------------------------------------------------------------------------- #
# Guarantee of profit
# --------------------------------------------------------------------------- #
def guarantee_to_partner(shares: Dict[str, Fraction], guaranteed: str, minimum,
                         bearers_ratio: Dict[str, Fraction]) -> Result:  # SME-REVIEW
    """
    `shares` = each partner's profit share BEFORE guarantee. If the guaranteed
    partner's share < minimum, the deficiency is taken from `bearers_ratio`
    partners in that ratio.
    """
    out = {k: F(v) for k, v in shares.items()}
    minimum = F(minimum)
    if out[guaranteed] >= minimum:
        return Result(out[guaranteed], [f"{guaranteed} share {out[guaranteed]} >= {minimum}; no deficiency"], parts=out)
    deficiency = minimum - out[guaranteed]
    out[guaranteed] = minimum
    keys = list(bearers_ratio.keys())
    cut = distribute(deficiency, [bearers_ratio[k] for k in keys])
    for k, c in zip(keys, cut):
        out[k] -= c
    return Result(out[guaranteed],
                  [f"deficiency {deficiency} borne by {keys} in {[str(bearers_ratio[k]) for k in keys]}"],
                  parts=out)


def guarantee_to_firm(actual_profit, minimum, ratio: Dict[str, Fraction],
                      guarantor: str) -> Result:  # SME-REVIEW
    """
    A partner guarantees the FIRM a minimum profit; any shortfall is contributed by
    the guarantor, then the (topped-up) profit is shared in `ratio`. The guarantor's
    net share = their ratio share - shortfall contributed.
    """
    actual = F(actual_profit)
    minimum = F(minimum)
    base = max(actual, minimum)
    keys = list(ratio.keys())
    shares = dict(zip(keys, distribute(base, [ratio[k] for k in keys])))
    shortfall = max(F(0), minimum - actual)
    shares[guarantor] -= shortfall
    return Result(shares[guarantor],
                  [f"base profit = max({actual},{minimum}) = {base}",
                   f"shortfall {shortfall} borne by {guarantor}"], parts=shares)


# --------------------------------------------------------------------------- #
# Past adjustments
# --------------------------------------------------------------------------- #
def past_adjustment(should_receive: Dict[str, Fraction],
                    actually_received: Dict[str, Fraction]) -> Result:  # SME-REVIEW
    """
    Single adjusting entry for omitted appropriations.
      net[p] = should_receive[p] - actually_received[p]
    net > 0  -> that partner's capital/current A/c is CREDITED (Dr others)
    net < 0  -> DEBITED.  The nets must sum to zero.
    """
    keys = set(should_receive) | set(actually_received)
    net = {k: F(should_receive.get(k, 0)) - F(actually_received.get(k, 0)) for k in keys}
    s = sum(net.values())
    if s != 0:
        raise ValueError(f"adjustment nets must sum to 0, got {s}")
    work = [f"{k}: {'Cr' if v > 0 else ('Dr' if v < 0 else '-')} {abs(v)}" for k, v in sorted(net.items())]
    return Result(F(0), work, parts=net)


# --------------------------------------------------------------------------- #
# Partner's loan
# --------------------------------------------------------------------------- #
def partners_loan_interest(amount, months=12, rate_pa=6) -> Result:
    """Absence of deed -> 6% p.a. as a CHARGE against profit (allowed even in loss)."""
    i = interest(amount, rate_pa, months)
    return Result(i, [f"loan interest = {amount} x {rate_pa}% x {months}/12 = {i}"])


# --------------------------------------------------------------------------- #
# Option builder (used by Phase 1B SOLVE-FIRST; kept here so traps stay next to math)
# --------------------------------------------------------------------------- #
def make_options(correct, distractors: Sequence, *, places: int = 0) -> Dict:
    """
    Build a 4-option MCQ payload from an exact correct answer + trap values.
    Guarantees: exactly one correct, all four distinct after rounding. Caller is
    responsible for shuffling/position-balancing across the paper (Phase 1E).
    Returns {"options": [Fraction x4], "correct_index": int} or raises if the traps
    collide with the answer (so a bad distractor set fails loudly, never silently).
    """
    c = rupees(F(correct), places)
    opts = [c]
    for d in distractors:
        r = rupees(F(d), places)
        if r == c:
            raise ValueError(f"distractor {d} equals correct answer after rounding")
        if r in opts:
            raise ValueError(f"duplicate distractor {d}")
        opts.append(r)
    if len(opts) != 4:
        raise ValueError("need exactly 3 distinct distractors")
    return {"options": opts, "correct_index": 0}  # index 0 == correct, pre-shuffle
