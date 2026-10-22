"""
What the job worker adds on top of grade_copy's result for an assignment-link copy (pure functions, no I/O):

  review()        per-question confidence threshold → check_sheets.needs_review_reasons (enum a4_review_reason)
                  and check_sheets.confidence (0..1)
  merge_chunks()  a long copy is graded in chunks of 10-12 pages; the chunk results are merged by question
"""

import os

from app.routers.copy_checker_router import _apply_overrides

CONFIDENCE_VALUE = {"low": 0.4, "medium": 0.75, "high": 0.95}
# A question whose confidence is below this needs review. A rubric question may set its own
# ("review_min_confidence": "high" for a 10-mark derivation, say); this is the default for the rest.
DEFAULT_MIN_CONFIDENCE = os.getenv("CHECKER_REVIEW_MIN_CONFIDENCE", "medium")
REASON_ORDER = ("empty_answer", "illegible", "low_confidence", "key_missing", "blank_pages", "bad_scan")

_NOT_FOUND = "Answer not found"
_ESTIMATED = "estimated"


def _threshold(q_no: str, rubric: dict) -> float:
    for q in (rubric or {}).get("questions") or []:
        if str(q.get("q_no")) == str(q_no) and q.get("review_min_confidence") in CONFIDENCE_VALUE:
            return CONFIDENCE_VALUE[q["review_min_confidence"]]
    return CONFIDENCE_VALUE.get(DEFAULT_MIN_CONFIDENCE, CONFIDENCE_VALUE["medium"])


def review(result: dict, rubric: dict) -> tuple:
    """(reasons, confidence, by_question). Only questions that count are looked at (the unused alternative of
    an OR pair being empty is fine)."""
    reasons, by_q, values = set(), {}, []
    for gq in result.get("graded_questions") or []:
        if gq.get("counted") is False:
            continue
        threshold = _threshold(gq.get("q_no"), rubric)
        mine = set()
        for leaf in gq.get("subpart_grades") or [gq]:
            conf = CONFIDENCE_VALUE.get(str(leaf.get("confidence") or "medium").lower(), CONFIDENCE_VALUE["medium"])
            values.append(conf)
            notes = " ".join(leaf.get("review_reasons") or [])
            if leaf.get("not_attempted") or _NOT_FOUND in notes:
                mine.add("empty_answer")
            if leaf.get("legibility_issue"):
                mine.add("illegible")
            if conf < threshold or "spans" in notes:
                mine.add("low_confidence")
            if _ESTIMATED in notes:
                mine.add("key_missing")
        if mine:
            by_q[str(gq.get("q_no"))] = sorted(mine, key=REASON_ORDER.index)
            reasons |= mine
    if any(f.get("code") == "BLANK_PAGE" for f in result.get("sheet_flags") or []):
        reasons.add("blank_pages")
    if str(result.get("readability_score") or "").lower() == "poor":
        reasons.add("bad_scan")
    confidence = round(sum(values) / len(values), 3) if values else None
    return sorted(reasons, key=REASON_ORDER.index), confidence, by_q


def _shift(item: dict, offset: int) -> None:
    if isinstance(item.get("page_index"), int):
        item["page_index"] += offset


def merge_chunks(chunks: list, rubric: dict) -> dict:
    """
    chunks — in page order: {"result": grade_copy result (page_index local to the chunk), "pages": n}
    Per question the chunk where it was answered wins. Answered in two chunks (an answer running over a chunk
    boundary, or written twice) → the higher marks count and the question is flagged for review.
    page_index is made global (offset by the pages of earlier chunks); OR groups and the total are decided again
    over the merged questions, with the same rule as a teacher override.
    """
    order, best, flags = [], {}, []
    offset = 0
    for ch in chunks:
        res = ch.get("result") or {}
        for gq in res.get("graded_questions") or []:
            gq = dict(gq)
            _shift(gq, offset)
            gq["subpart_grades"] = [dict(s) for s in gq.get("subpart_grades") or []]
            for s in gq["subpart_grades"]:
                _shift(s, offset)
            key = str(gq.get("q_no"))
            if key not in best:
                order.append(key)
                best[key] = gq
                continue
            cur = best[key]
            if gq.get("not_attempted"):
                continue
            if cur.get("not_attempted"):
                best[key] = gq
                continue
            keep = gq if float(gq.get("marks_awarded") or 0) > float(cur.get("marks_awarded") or 0) else cur
            keep = dict(keep)
            keep["needs_review"] = True
            keep["review_reasons"] = sorted(set(keep.get("review_reasons") or []) |
                                            {"Answer spans two parts of the copy — please check"})
            best[key] = keep
        for f in res.get("sheet_flags") or []:
            f = dict(f)
            if isinstance(f.get("page"), int):
                f["page"] += offset
            flags.append(f)
        offset += int(ch.get("pages") or 0)

    first = next((c.get("result") or {} for c in chunks if c.get("result")), {})
    merged = {"graded_questions": [best[k] for k in order], "max_marks": first.get("max_marks") or
              (rubric or {}).get("total_marks") or 0, "sheet_flags": flags,
              "overall_remarks": " ".join(r for r in ((c.get("result") or {}).get("overall_remarks") or ""
                                                       for c in chunks) if r)[:2000],
              "readability_score": _worst_readability(chunks)}
    total = _apply_overrides(merged, rubric)          # counted flags for OR groups + capped total
    for gq in merged["graded_questions"]:
        if gq.get("counted") is False and gq.get("not_attempted"):
            gq["needs_review"] = False
    max_marks = float(merged["max_marks"] or 0)
    merged["total_marks_awarded"] = total
    merged["percentage"] = round(total / max_marks * 100, 1) if max_marks else 0.0
    merged["needs_review_count"] = sum(1 for gq in merged["graded_questions"] if gq.get("needs_review") and gq.get("counted"))
    return merged


def _worst_readability(chunks: list) -> str:
    rank = {"poor": 0, "average": 1, "good": 2}
    seen = [str((c.get("result") or {}).get("readability_score") or "") for c in chunks]
    seen = [s for s in seen if s in rank]
    return min(seen, key=rank.get) if seen else ""
