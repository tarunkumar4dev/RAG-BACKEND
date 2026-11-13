"""Five correctness metrics the harness scores a paper on (Phase 0).

Each metric is a pure function:  (paper, config, **kw) -> MetricResult.
No LLM calls. No network. Deterministic.

Metric status vocabulary:
  "pass"          every check passed
  "fail"          at least one check failed; see `failures`
  "needs_solver"  the metric needs the accountancy_solver (Phase 1A) which does not exist yet
  "skipped"       the metric does not apply to this paper (e.g. no MCQs)
  "error"         the metric could not run (missing dep, bad input)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

from .schema import BenchmarkConfig, Paper, PaperQuestion


# ── Result type ──────────────────────────────────────────────────────────

@dataclass
class MetricResult:
    metric: str
    status: str                       # see module docstring
    summary: str                      # one-line human description
    failures: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "skipped")


# ── Helpers ──────────────────────────────────────────────────────────────

_LETTER = re.compile(r"^\s*\(?([A-Da-d])\s*[\).:\]]")
_NUM = re.compile(r"-?\d+(?:,\d+)*(?:\.\d+)?")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _parse_amount(s: str) -> float | None:
    """'Rs. 2,70,000' / '₹2,70,000' / '12.5%' -> float. Returns None if no number."""
    if s is None:
        return None
    cleaned = str(s).replace("₹", "").replace("Rs.", "").replace("Rs", "").replace("INR", "")
    cleaned = cleaned.replace(",", "").replace("%", "").strip()
    m = _NUM.search(cleaned)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _mcq_only(paper: Paper) -> list[PaperQuestion]:
    return [q for q in paper.questions if q.format == "mcq"]


def _match_option(correct: str, options: list[str]) -> int | None:
    """Return the index of the option that matches `correct`, or None.

    Matches letter prefix (A)/B)/...), exact normalised text, or substring.
    """
    if not options or not correct:
        return None
    # Letter prefix
    m = _LETTER.match(correct)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(options):
            return idx
    nc = _norm(correct)
    for i, o in enumerate(options):
        if _norm(o) == nc:
            return i
    for i, o in enumerate(options):
        no = _norm(o)
        if nc and no and (nc in no or (len(nc) > 3 and no in nc)):
            return i
    return None


# ── 1. Exactly one correct option (lexical, no solver needed) ────────────

def exactly_one_correct_option(paper: Paper, config: BenchmarkConfig | None = None) -> MetricResult:
    mcqs = _mcq_only(paper)
    if not mcqs:
        return MetricResult("exactly_one_correct_option", "skipped", "no MCQs in paper")

    failures: list[dict[str, Any]] = []
    for q in mcqs:
        opts = q.options or []
        if len(opts) != 4:
            failures.append({"qid": q.qid, "reason": "option_count", "detail": f"{len(opts)} options"})
            continue
        if len({_norm(o) for o in opts}) != 4:
            failures.append({"qid": q.qid, "reason": "duplicate_options",
                             "detail": [o[:40] for o in opts]})
        matched = _match_option(q.correct_answer, opts)
        if matched is None:
            failures.append({"qid": q.qid, "reason": "no_match",
                             "detail": f"correct={q.correct_answer[:40]!r} options={[o[:30] for o in opts]}"})

    status = "pass" if not failures else "fail"
    return MetricResult(
        "exactly_one_correct_option", status,
        f"{len(mcqs) - len(failures)}/{len(mcqs)} MCQs have exactly one option matching the stated correct answer",
        failures=failures,
        stats={"mcq_count": len(mcqs)},
    )


# ── 2. Numeric correctness (needs solver — stubbed for Phase 0) ──────────

def numeric_correctness(paper: Paper, config: BenchmarkConfig) -> MetricResult:
    """Compares the value parsed from each MCQ's correct_answer against the
    hand-verified `expected_answer` in the fixture.

    Phase 0 scope: when the fixture ships `expected_numeric_items`, we check
    each one by `qid`. If no items are declared or the fixture's key_source is
    `pending_sme`, the metric returns `needs_solver` so CI knows the number is
    not yet enforceable for this config.
    """
    items = [i for i in config.expected_numeric_items if i.key_source != "pending_sme"]
    pending = [i for i in config.expected_numeric_items if i.key_source == "pending_sme"]

    if not items and pending:
        return MetricResult(
            "numeric_correctness", "needs_solver",
            f"{len(pending)} numeric items pending SME verification; solver (Phase 1A) not yet available",
            stats={"pending": len(pending)},
        )
    if not items:
        return MetricResult("numeric_correctness", "skipped",
                            "no numeric items declared in fixture")

    by_qid = {q.qid: q for q in paper.questions}
    failures: list[dict[str, Any]] = []
    for item in items:
        q = by_qid.get(item.qid)
        if q is None:
            failures.append({"qid": item.qid, "reason": "missing_in_paper"})
            continue
        got = _parse_amount(q.correct_answer)
        if got is None:
            failures.append({"qid": item.qid, "reason": "unparseable_answer",
                             "detail": q.correct_answer[:80]})
            continue
        if abs(got - item.expected_answer) > item.tolerance:
            failures.append({
                "qid": item.qid,
                "reason": "wrong_value",
                "expected": item.expected_answer,
                "got": got,
                "tolerance": item.tolerance,
                "description": item.description,
            })

    status = "pass" if not failures else "fail"
    return MetricResult(
        "numeric_correctness", status,
        f"{len(items) - len(failures)}/{len(items)} hand-verified numeric answers match",
        failures=failures,
        stats={"checked": len(items), "pending_sme": len(pending)},
    )


# ── 3. Semantic duplicates (lexical + numeric-pattern; Phase 2 adds embeddings) ──

def semantic_duplicate_count(paper: Paper, config: BenchmarkConfig | None = None,
                             max_per_structure: int | None = None) -> MetricResult:
    """Uses the production dedup gate (app.services.dedup_gate) so the benchmark
    scores exactly the policy that ships. A failure = the gate would remove a
    question as a hard duplicate (exact / lexical) or over the structure cap."""
    qs = paper.questions
    if len(qs) < 2:
        return MetricResult("semantic_duplicate_count", "skipped", "fewer than 2 questions")

    from app.services import dedup_gate as dg
    res = dg.dedup([{"text": q.text, "qid": q.qid} for q in qs],
                   max_per_structure=max_per_structure)
    failures = [{"kind": r["reason"], "index": r["index"], "dup_of": r["dup_of"],
                 "text": r["text"]} for r in res.removed]
    from collections import Counter
    reasons = dict(Counter(r["reason"] for r in res.removed))
    structures = dg.distinct_structures([{"text": q.text} for q in qs])
    status = "pass" if not failures else "fail"
    return MetricResult(
        "semantic_duplicate_count", status,
        f"{len(failures)} duplicate(s) would be removed {reasons}; "
        f"{structures} distinct structures across {len(qs)} questions",
        failures=failures,
        stats={"removed": len(failures), "distinct_structures": structures,
               "max_per_structure": max_per_structure},
    )


# ── 4. Count / marks reconciliation ──────────────────────────────────────

def count_marks_reconcile(paper: Paper, config: BenchmarkConfig) -> MetricResult:
    failures: list[dict[str, Any]] = []
    actual_count = len(paper.questions)
    actual_marks = sum(q.marks for q in paper.questions)

    if actual_count != config.expected_total_questions:
        failures.append({
            "kind": "question_count",
            "expected": config.expected_total_questions, "got": actual_count,
        })
    if actual_marks != config.expected_total_marks:
        failures.append({
            "kind": "total_marks",
            "expected": config.expected_total_marks, "got": actual_marks,
        })
    if paper.total_marks != actual_marks:
        failures.append({
            "kind": "declared_vs_summed_marks",
            "declared": paper.total_marks, "summed": actual_marks,
        })

    status = "pass" if not failures else "fail"
    return MetricResult(
        "count_marks_reconcile", status,
        f"{actual_count} questions totalling {actual_marks} marks "
        f"(expected {config.expected_total_questions} / {config.expected_total_marks})",
        failures=failures,
        stats={"questions": actual_count, "marks": actual_marks},
    )


# ── 5. PDF render integrity ──────────────────────────────────────────────

def pdf_render_integrity(paper: Paper, pdf_path: str) -> MetricResult:
    """Use PyMuPDF to extract text per page and verify:

      - every question stem appears somewhere in the PDF
      - for each MCQ, every option appears on the SAME page as its stem
      - total pages > 0

    PyMuPDF (fitz) is already a dependency (requirements.txt).
    """
    try:
        import fitz  # type: ignore
    except ImportError:
        return MetricResult("pdf_render_integrity", "error",
                            "PyMuPDF (fitz) not importable")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return MetricResult("pdf_render_integrity", "error", f"could not open PDF: {e}")

    if doc.page_count == 0:
        doc.close()
        return MetricResult("pdf_render_integrity", "fail", "PDF has 0 pages",
                            failures=[{"kind": "empty_pdf"}])

    page_texts = [doc[i].get_text("text") for i in range(doc.page_count)]
    doc.close()

    def _loose(s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    pages_loose = [_loose(t) for t in page_texts]
    all_text = " ".join(pages_loose)

    failures: list[dict[str, Any]] = []
    for q in paper.questions:
        stem_fragment = _loose(q.text)[:80]
        if not stem_fragment:
            continue
        stem_page = next((i for i, p in enumerate(pages_loose) if stem_fragment in p), None)
        if stem_page is None:
            failures.append({"qid": q.qid, "kind": "stem_missing",
                             "detail": stem_fragment[:60]})
            continue

        if q.format == "mcq" and q.options:
            for opt in q.options:
                of = _loose(opt)[:40]
                if not of:
                    continue
                if of in pages_loose[stem_page]:
                    continue
                if of in all_text:
                    failures.append({"qid": q.qid, "kind": "option_split_page",
                                     "detail": of[:40], "stem_page": stem_page})
                else:
                    failures.append({"qid": q.qid, "kind": "option_missing",
                                     "detail": of[:40]})

    status = "pass" if not failures else "fail"
    return MetricResult(
        "pdf_render_integrity", status,
        f"{doc.page_count} page(s); {len(paper.questions) - len({f['qid'] for f in failures})} questions fully present",
        failures=failures,
        stats={"pages": doc.page_count, "questions": len(paper.questions)},
    )


# ── 7. Blueprint coverage (2B) ────────────────────────────────────────────

def blueprint_coverage(paper: Paper, config: BenchmarkConfig) -> MetricResult:
    """For a full-chapter paper (no single sub-topic requested), check that the
    subtopic distribution matches the chapter blueprint within 50%-200%. Skipped
    when the fixture targets a single sub-topic or the chapter has no blueprint."""
    sd = (config.request.get("simpleData") or [{}])[0]
    if sd.get("subtopic"):
        return MetricResult("blueprint_coverage", "skipped", "single-subtopic fixture")

    try:
        from app.services import blueprint as bp
    except Exception as e:
        return MetricResult("blueprint_coverage", "error", f"blueprint import failed: {e}")

    chapter = sd.get("topic", "")
    bpt = bp.get_blueprint(chapter)
    if bpt is None:
        return MetricResult("blueprint_coverage", "skipped", f"no blueprint for '{chapter}'")

    from collections import Counter
    # Each question carries its family tag (solve-first). Fall back to skip if absent.
    fams = [getattr(q, "family", None) for q in paper.questions]
    # Paper is a normalised view; family is not a Paper field, so read from source.
    # (Golden papers written by this project include "family" on each question.)
    actual = Counter(f for f in fams if f)
    if not actual:
        return MetricResult("blueprint_coverage", "skipped",
                            "questions have no family tag")
    plan = bp.plan_allocation(bpt, len(paper.questions))
    report = bp.coverage_report(dict(actual), plan)
    status = "pass" if report.ok else "fail"
    return MetricResult("blueprint_coverage", status, report.summary(),
                        failures=report.violations,
                        stats={"plan": plan, "actual": dict(actual)})


# ── 6. Option-position balance (D4) ──────────────────────────────────────

def option_position_balance(paper: Paper, config: BenchmarkConfig | None = None,
                            tolerance: int = 1) -> MetricResult:
    """Each of A/B/C/D should be the correct answer within `tolerance` of a
    uniform share across the paper's MCQs (Phase 1E). With <4 MCQs, skipped."""
    mcqs = _mcq_only(paper)
    if len(mcqs) < 4:
        return MetricResult("option_position_balance", "skipped", "fewer than 4 MCQs")
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    for q in mcqs:
        m = _LETTER.match(q.correct_answer)
        if m:
            counts[m.group(1).upper()] += 1
    total = sum(counts.values())
    target = total / 4
    failures = []
    for letter, c in counts.items():
        if abs(c - target) > tolerance + 1e-9:
            failures.append({"letter": letter, "count": c, "target": round(target, 2)})
    status = "pass" if not failures else "fail"
    return MetricResult(
        "option_position_balance", status,
        f"A/B/C/D = {counts['A']}/{counts['B']}/{counts['C']}/{counts['D']} "
        f"(target ~{target:.1f}, tolerance +/-{tolerance})",
        failures=failures, stats={"counts": counts},
    )


ALL_METRICS = [
    "exactly_one_correct_option",
    "numeric_correctness",
    "option_position_balance",
    "semantic_duplicate_count",
    "count_marks_reconcile",
    "pdf_render_integrity",
]
