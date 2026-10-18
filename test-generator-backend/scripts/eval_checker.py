"""
Check Copies eval — runs the real checker pipeline on test copies and compares it with the teacher's marks.

Run from test-generator-backend/:
    python scripts/eval_checker.py                              # all cases, 1 run
    python scripts/eval_checker.py --cases aman_trigo10 --runs 3   # stability on one copy
    python scripts/eval_checker.py --cases-dir scripts/eval_cases_private   # real copies (never committed)
    python scripts/eval_checker.py --app-root /tmp/app_7cf7265 --label 7cf7265   # grade with older code
    python scripts/eval_checker.py --rebuild-rubric --runs 3   # also measures the rubric's own stability
    python scripts/eval_checker.py --compare eval_runs/checker_A.json eval_runs/checker_B.json

Pipeline = production's Check Copies path without DB/Storage:
    build_rubric(question paper, answer key)  →  grade_sheet(page photos, rubric, strictness, require_identity=False)
The rubric is cached (key = rubric prompt + model + paper bytes + key bytes + strictness) so that a grading
change is measured on the same rubric. Fallback models are OFF by default (a 429/503 is retried on the primary
model or the copy fails) so a run never mixes models silently; --allow-fallback turns them back on and every
copy that used another model is marked "contaminated".

Case folder (see scripts/eval_cases/checker/README.md):
    question_paper.(pdf|png|jpg)  answer_key.(…) [optional]  pages/*.jpg|png  OR  copy.pdf  expected.json
"""

import argparse
import hashlib
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path

# ── Prices ──────────────────────────────────────────────────────────────────────────────────────────────
# USD per 1M tokens (input, output). Output is billed INCLUDING thinking tokens.
# Source: https://ai.google.dev/gemini-api/docs/pricing (Gemini Developer API, Standard tier), checked 17 Oct 2026.
#   gemini-3.8-flash: introductory rate until 31 Dec 2026 ($1.50 / $7.50 from 1 Jan 2027).
#   gemini-3-flash-preview: standard rate.
#   gemini-2.5-flash: NOT checked → UNVERIFIED (cost shown, marked).
PRICES = {
    "gemini-3.8-flash": (0.75, 3.75, "verified 2026-10-17, intro rate"),
    "gemini-3-flash-preview": (0.50, 3.00, "verified 2026-10-17"),
    "gemini-2.5-flash": (0.30, 2.50, "UNVERIFIED"),
}
PRICE_VALID_UNTIL = date(2026, 12, 31)
USD_INR_DEFAULT = 88.0  # UNVERIFIED exchange rate; override with --usd-inr

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
RUNS = HERE / "eval_runs"
DEFAULT_CASES = HERE / "eval_cases" / "checker"

LIMIT_TOTAL = 1.0   # pass: |AI total − teacher total| ≤ 1
LIMIT_TIME = 90.0   # pass: every copy and p95 < 90 s


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# Gemini call log: model, tokens and errors per call, attributed to the thread that made it
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

class CallLog(logging.Handler):
    USAGE = re.compile(r"\[checker\] (\S+) tokens: input=(\d+) output=(\d+) thinking=(\d+) model=(\S+) time=([\d.]+)s")
    BUSY = re.compile(r"\[(\w+)\] (\S+) (?:unavailable|not available) \((\d+)\)")

    def __init__(self):
        super().__init__()
        self.calls, self.errors = [], []
        self._lock = threading.Lock()

    def emit(self, record):
        msg = record.getMessage()
        m = self.USAGE.search(msg)
        if m:
            with self._lock:
                self.calls.append({"tag": m[1], "input": int(m[2]), "output": int(m[3]), "thinking": int(m[4]),
                                   "model": m[5], "seconds": float(m[6]), "thread": record.thread})
            return
        m = self.BUSY.search(msg)
        if m:
            with self._lock:
                self.errors.append({"tag": m[1], "model": m[2], "code": int(m[3]), "thread": record.thread})

    def take(self, thread_id):
        with self._lock:
            calls = [c for c in self.calls if c["thread"] == thread_id]
            errors = [e for e in self.errors if e["thread"] == thread_id]
            self.calls = [c for c in self.calls if c["thread"] != thread_id]
            self.errors = [e for e in self.errors if e["thread"] != thread_id]
        return calls, errors


def cost_inr(calls, usd_inr):
    usd, unverified = 0.0, set()
    for c in calls:
        price = PRICES.get(c["model"])
        if price is None:
            unverified.add(c["model"] + " (no price)")
            continue
        if price[2].startswith("UNVERIFIED"):
            unverified.add(c["model"])
        usd += (c["input"] * price[0] + (c["output"] + c["thinking"]) * price[1]) / 1e6
    return round(usd * usd_inr, 2), sorted(unverified)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# Cases
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

IMG = (".png", ".jpg", ".jpeg", ".webp")


def _one(case_dir: Path, stem: str, required=True):
    hits = sorted(p for p in case_dir.glob(stem + ".*") if p.suffix.lower() in (".pdf",) + IMG)
    if not hits and required:
        raise FileNotFoundError(f"{case_dir.name}: missing {stem}.pdf/.jpg/.png")
    return hits[0] if hits else None


def _natural(p: Path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]


def load_case(case_dir: Path) -> dict:
    exp = json.loads((case_dir / "expected.json").read_text(encoding="utf-8"))
    pages_dir = case_dir / "pages"
    if pages_dir.is_dir() and any(p.suffix.lower() in IMG for p in pages_dir.iterdir()):
        pages = sorted((p for p in pages_dir.iterdir() if p.suffix.lower() in IMG), key=_natural)
    elif (case_dir / "copy.pdf").exists():
        pages = render_pdf(case_dir / "copy.pdf", case_dir.name)
    else:
        raise FileNotFoundError(f"{case_dir.name}: needs pages/*.jpg|png or copy.pdf")

    grouped = {q for g in (exp.get("groups") or {}).values() for q in g["questions"]}
    units = []
    for name, g in (exp.get("groups") or {}).items():
        units.append({"unit": name, "questions": [str(q) for q in g["questions"]], "marks": float(g["marks"]),
                      "expect_review": bool(g.get("expect_review")), "expect": g.get("expect"), "note": g.get("note", "")})
    for q, spec in (exp.get("questions") or {}).items():
        if q in grouped:
            raise ValueError(f"{case_dir.name}: Q{q} is both in a group and in questions")
        units.append({"unit": str(q), "questions": [str(q)], "marks": float(spec["marks"]),
                      "expect_review": bool(spec.get("expect_review")), "expect": spec.get("expect"),
                      "pages": spec.get("pages"), "note": spec.get("note", "")})
    total = float(exp["total"])
    unit_sum = round(sum(u["marks"] for u in units), 2)
    warn = None if abs(unit_sum - total) < 0.01 else f"expected.json total {total:g} ≠ sum of marks {unit_sum:g}"
    return {"name": case_dir.name, "dir": case_dir, "expected": exp, "units": units, "total": total,
            "strictness": exp.get("strictness", "medium"), "pages": [str(p) for p in pages],
            "paper": _one(case_dir, "question_paper"), "key": _one(case_dir, "answer_key", required=False),
            "synthetic": bool(exp.get("synthetic")), "warning": warn}


def render_pdf(pdf: Path, case: str, dpi: int = 150) -> list:
    """Copies stored as one PDF are graded as page photos, like production (teachers upload photos)."""
    import pymupdf
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()[:10]
    out = RUNS / "pages_cache" / f"{case}_{digest}"
    out.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(str(pdf)) as doc:
        paths = []
        for i, page in enumerate(doc):
            p = out / f"p{i + 1}.png"
            if not p.exists():
                page.get_pixmap(dpi=dpi).save(str(p))
            paths.append(p)
    return paths


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# Rubric (cached)
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

def rubric_key(sas, case) -> str:
    h = hashlib.sha256()
    for part in (getattr(sas, "RUBRIC_PROMPT", ""), getattr(sas, "_MULTI_PAGE_NOTE", ""),
                 getattr(sas, "_NO_KEY_NOTE", ""), sas.CHECKER_MODEL, case["strictness"]):
        h.update(part.encode("utf-8") + b"\0")
    h.update(case["paper"].read_bytes() + b"\0")
    h.update(case["key"].read_bytes() if case["key"] else b"<no key>")
    return h.hexdigest()[:16]


def rubric_summary(rubric: dict) -> dict:
    per_q = {}
    for q in rubric.get("questions") or []:
        n = len(q.get("points") or []) + sum(len(s.get("points") or []) for s in q.get("subparts") or [])
        per_q[q["q_no"]] = {"marks": q["marks"], "points": n}
    return {"total_marks": rubric.get("total_marks"), "questions": len(per_q),
            "points": sum(v["points"] for v in per_q.values()), "per_question": per_q}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# Scoring one graded copy against expected.json
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

def _qkey(q_no) -> str:
    return re.sub(r"(?i)[\s-]*or$", "", str(q_no)).strip()


def score(case, result: dict) -> dict:
    ai, flags, seen = {}, {}, set()
    for gq in result.get("graded_questions") or []:
        if gq.get("counted") is False:
            continue
        k = _qkey(gq["q_no"])
        seen.add(k)
        m = gq["teacher_marks"] if gq.get("teacher_marks") is not None else gq["marks_awarded"]
        ai[k] = round(ai.get(k, 0) + float(m or 0), 2)
        if gq.get("needs_review"):
            flags.setdefault(k, []).extend(gq.get("review_reasons") or ["review"])
    rows = []
    for u in case["units"]:
        got = round(sum(ai.get(q, 0) for q in u["questions"]), 2)
        reasons = sorted({r for q in u["questions"] for r in flags.get(q, [])})
        missing = [q for q in u["questions"] if q not in seen]
        diff = round(got - u["marks"], 2)
        check = None
        if u.get("expect") == "crossed_out_zero":
            check = {"type": "crossed_out_zero", "pass": got <= u["marks"] + 1e-9}
        elif u.get("expect") == "spans_pages":
            check = {"type": "spans_pages", "pass": abs(diff) < 1e-9}
        rows.append({"unit": u["unit"], "teacher": u["marks"], "ai": got, "diff": diff,
                     "expect_review": u["expect_review"], "flagged": bool(reasons), "reasons": reasons,
                     "missing_in_result": missing, "check": check, "note": u["note"]})
    extra = sorted(seen - {q for u in case["units"] for q in u["questions"]})
    total_ai = round(sum(r["ai"] for r in rows), 2)
    return {"rows": rows, "total_ai": total_ai, "total_teacher": case["total"],
            "total_diff": round(total_ai - case["total"], 2), "questions_not_in_expected": extra}


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# Running
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

def run_with_timeout(fn, timeout_s):
    """Runs fn on a daemon thread; returns (result, error, thread_id). A hung call never blocks the eval."""
    box = {}

    def target():
        box["tid"] = threading.get_ident()
        try:
            box["result"] = fn()
        except Exception as e:  # report, never crash the whole eval
            box["error"] = f"{type(e).__name__}: {str(e)[:300]}"

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None, f"TIMEOUT after {timeout_s:.0f}s (copy abandoned)", box.get("tid", t.ident)
    return box.get("result"), box.get("error"), box.get("tid", t.ident)


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 1)


def summarize(copies, cases_by_name):
    ok = [c for c in copies if not c.get("error")]
    units = [r for c in ok for r in c["score"]["rows"]]
    tp = sum(1 for r in units if r["expect_review"] and r["flagged"])
    fn = sum(1 for r in units if r["expect_review"] and not r["flagged"])
    fp = sum(1 for r in units if not r["expect_review"] and r["flagged"])
    times = [c["wall_s"] for c in ok]
    return {
        "copies": len(copies), "failed": len(copies) - len(ok),
        "total_mae": round(statistics.mean(abs(c["score"]["total_diff"]) for c in ok), 3) if ok else None,
        "unit_mae": round(statistics.mean(abs(r["diff"]) for r in units), 3) if units else None,
        "unit_exact_pct": round(100 * sum(1 for r in units if r["diff"] == 0) / len(units), 1) if units else None,
        "over_credit": sum(1 for r in units if r["diff"] > 0),
        "under_credit": sum(1 for r in units if r["diff"] < 0),
        "review_recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "review_precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "false_flags": fp, "missed_flags": fn,
        "time_p50": pct(times, 50), "time_p95": pct(times, 95),
        "inr_per_copy": round(statistics.mean(c["inr"] for c in ok), 2) if ok else None,
        "contaminated": sum(1 for c in copies if c.get("fallback_models")),
        "http_429": sum(1 for c in copies for e in c.get("api_errors", []) if e["code"] == 429),
        "http_5xx": sum(1 for c in copies for e in c.get("api_errors", []) if e["code"] >= 500),
    }


def pass_fail(copies, summary):
    ok = [c for c in copies if not c.get("error")]
    out = []

    def item(name, passed, detail):
        out.append({"check": name, "pass": passed, "detail": detail})

    failed = [f"{c['case']}#{c['run']}: {c['error']}" for c in copies if c.get("error")]
    item("every copy graded (no error / timeout)", not failed, "; ".join(failed) or "all graded")
    bad = [f"{c['case']}#{c['run']} Δ{c['score']['total_diff']:+g}" for c in ok if abs(c["score"]["total_diff"]) > LIMIT_TOTAL]
    item(f"every copy total within ±{LIMIT_TOTAL:g} of teacher", not bad and bool(ok), ", ".join(bad) or "all within")
    for kind, label in (("crossed_out_zero", "crossed-out work gets no credit"),
                        ("spans_pages", "answer split across pages graded as one")):
        rows = [(c, r) for c in ok for r in c["score"]["rows"] if r["check"] and r["check"]["type"] == kind]
        if not rows:
            item(label, None, "n/a (no case declares it)")
            continue
        bad = [f"{c['case']} Q{r['unit']}: AI {r['ai']:g} vs {r['teacher']:g}" for c, r in rows if not r["check"]["pass"]]
        item(label, not bad, ", ".join(bad) or ", ".join(f"{c['case']} Q{r['unit']}" for c, r in rows))
    slow = [f"{c['case']}#{c['run']} {c['wall_s']}s" for c in ok if c["wall_s"] >= LIMIT_TIME]
    p95 = summary["time_p95"]
    item(f"every copy < {LIMIT_TIME:g} s and p95 < {LIMIT_TIME:g} s",
         not slow and p95 is not None and p95 < LIMIT_TIME, (", ".join(slow) or "all fast") + f"; p95 {p95}s")
    want = [(c, r) for c in ok for r in c["score"]["rows"] if r["expect_review"]]
    if want:
        missed = [f"{c['case']} Q{r['unit']}" for c, r in want if not r["flagged"]]
        item("expected review flags raised", not missed,
             ("MISSED: " + ", ".join(missed)) if missed else ", ".join(f"{c['case']} Q{r['unit']}" for c, r in want))
    else:
        item("expected review flags raised", None, "n/a")
    return out


def stability(copies):
    by_case = {}
    for c in copies:
        if c.get("error"):
            continue
        by_case.setdefault(c["case"], []).append(c)
    report = {}
    for case, runs in by_case.items():
        if len(runs) < 2:
            continue
        diffs = []
        for i, row in enumerate(runs[0]["score"]["rows"]):
            vals = [r["score"]["rows"][i]["ai"] for r in runs]
            flags = [r["score"]["rows"][i]["flagged"] for r in runs]
            if len(set(vals)) > 1 or len(set(flags)) > 1:
                diffs.append({"unit": row["unit"], "teacher": row["teacher"], "ai_per_run": vals, "flag_per_run": flags})
        report[case] = {"runs": len(runs), "totals": [r["score"]["total_ai"] for r in runs], "changed": diffs}
    return report


def rubric_stability(builds):
    """builds: {key: [summary per build]} → questions whose marks or point count changed between builds."""
    out = {}
    for key, sums in builds.items():
        if len(sums) < 2:
            continue
        changed = []
        qs = sorted({q for s in sums for q in s["per_question"]}, key=lambda x: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", x)])
        for q in qs:
            vals = [s["per_question"].get(q) for s in sums]
            if len({json.dumps(v, sort_keys=True) for v in vals}) > 1:
                changed.append({"q": q, "per_build": vals})
        out[key] = {"builds": len(sums), "total_marks": [s["total_marks"] for s in sums],
                    "points": [s["points"] for s in sums], "changed": changed}
    return out


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# Output
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

def md_copy(c) -> str:
    head = f"\n### {c['case']} — run {c['run']}"
    if c.get("error"):
        return head + f"\n\n**FAILED**: {c['error']}\n"
    s = c["score"]
    lines = [head, "",
             f"time {c['wall_s']} s · calls {len(c['calls'])} · tokens in/out/think "
             f"{c['tokens']['input']}/{c['tokens']['output']}/{c['tokens']['thinking']} · ₹{c['inr']}"
             + (f" · ⚠ fallback: {', '.join(c['fallback_models'])}" if c.get("fallback_models") else "")
             + (f" · ⚠ unverified price: {', '.join(c['unverified_price'])}" if c.get("unverified_price") else ""),
             "", "| Q | teacher | AI | Δ | review exp → got | check | note |", "|---|---:|---:|---:|---|---|---|"]
    for r in s["rows"]:
        d = "" if r["diff"] == 0 else f"{r['diff']:+g} {'over' if r['diff'] > 0 else 'under'}"
        rev = f"{'yes' if r['expect_review'] else 'no'} → {'yes' if r['flagged'] else 'no'}"
        if r["expect_review"] != r["flagged"]:
            rev += " ✗"
        chk = "" if not r["check"] else f"{r['check']['type']} {'✅' if r['check']['pass'] else '❌'}"
        note = "; ".join(x for x in [", ".join(r["reasons"])[:60], ("not in AI result: " + ",".join(r["missing_in_result"])) if r["missing_in_result"] else ""] if x)
        lines.append(f"| {r['unit']} | {r['teacher']:g} | {r['ai']:g} | {d} | {rev} | {chk} | {note} |")
    lines.append(f"| **total** | **{s['total_teacher']:g}** | **{s['total_ai']:g}** | **{s['total_diff']:+g}** | | | |")
    if s["questions_not_in_expected"]:
        lines.append(f"\n⚠ AI graded questions not in expected.json: {', '.join(s['questions_not_in_expected'])}")
    return "\n".join(lines) + "\n"


def md_summary(run) -> str:
    s = run["summary"]
    rows = [
        ("copies graded / failed", f"{s['copies'] - s['failed']} / {s['failed']}"),
        ("total-marks MAE", s["total_mae"]),
        ("per-question MAE (group = 1 unit)", s["unit_mae"]),
        ("per-question exact match", f"{s['unit_exact_pct']}%" if s["unit_exact_pct"] is not None else None),
        ("over-credit / under-credit (units)", f"{s['over_credit']} / {s['under_credit']}"),
        ("review recall (flagged when it should)", s["review_recall"]),
        ("review precision (flags that were right)", s["review_precision"]),
        ("new wrong flags / missed flags", f"{s['false_flags']} / {s['missed_flags']}"),
        ("time p50 / p95", f"{s['time_p50']} s / {s['time_p95']} s"),
        ("₹ per copy (grading)", s["inr_per_copy"]),
        ("₹ this run (incl. rubric builds)", run["cost_inr_total"]),
        ("contaminated copies (fallback model used)", s["contaminated"]),
        ("HTTP 429 / 5xx seen", f"{s['http_429']} / {s['http_5xx']}"),
    ]
    out = ["\n## Summary", "", "| metric | value |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows]
    out += ["", "## PASS / FAIL", ""]
    for p in run["pass_fail"]:
        mark = "➖ n/a" if p["pass"] is None else ("✅ PASS" if p["pass"] else "❌ FAIL")
        out.append(f"- {mark} — {p['check']}: {p['detail']}")
    out.append(f"- naye galat flags (flagged but teacher says fine): {s['false_flags']}")
    if run["stability"]:
        out += ["", "## Stability (same copy, repeated runs)", ""]
        for case, st in run["stability"].items():
            out.append(f"- {case}: totals {st['totals']}" + ("" if st["changed"] else " — no question changed"))
            for d in st["changed"]:
                out.append(f"  - Q{d['unit']} (teacher {d['teacher']:g}): AI {d['ai_per_run']} flags {d['flag_per_run']}")
    if run["rubric_stability"]:
        out += ["", "## Rubric stability (rebuilt each run)", ""]
        for key, st in run["rubric_stability"].items():
            out.append(f"- rubric {key}: total marks {st['total_marks']}, points {st['points']}"
                       + ("" if st["changed"] else " — identical"))
            for d in st["changed"]:
                out.append(f"  - Q{d['q']}: {d['per_build']}")
    for w in run["warnings"]:
        out.append(f"\n⚠ {w}")
    return "\n".join(out) + "\n"


# ════════════════════════════════════════════════════════════════════════════════════════════════════════
# Compare two saved runs
# ════════════════════════════════════════════════════════════════════════════════════════════════════════

def compare(path_a, path_b):
    a, b = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (path_a, path_b))
    la, lb = a.get("label") or Path(path_a).stem, b.get("label") or Path(path_b).stem
    out = [f"# Compare: {la}  →  {lb}", ""]
    first = lambda run, case: next((c for c in run["copies"] if c["case"] == case and not c.get("error")), None)  # noqa: E731
    for case in sorted({c["case"] for c in a["copies"]} & {c["case"] for c in b["copies"]}):
        ca, cb = first(a, case), first(b, case)
        if not ca or not cb:
            out.append(f"## {case}: missing / failed in one run\n")
            continue
        rows_b = {r["unit"]: r for r in cb["score"]["rows"]}
        lines = [f"## {case}", "", f"| Q | teacher | {la} | {lb} | change |", "|---|---:|---:|---:|---|"]
        for r in ca["score"]["rows"]:
            rb = rows_b.get(r["unit"])
            if not rb:
                continue
            ch = ""
            if rb["ai"] != r["ai"]:
                closer = abs(rb["diff"]) < abs(r["diff"])
                ch = f"{rb['ai'] - r['ai']:+g} ({'closer to teacher' if closer else 'further from teacher' if abs(rb['diff']) > abs(r['diff']) else 'same distance'})"
            if rb["flagged"] != r["flagged"]:
                ch += f" flag {'added' if rb['flagged'] else 'removed'}"
            lines.append(f"| {r['unit']} | {r['teacher']:g} | {r['ai']:g} | {rb['ai']:g} | {ch} |")
        lines.append(f"| **total** | {ca['score']['total_teacher']:g} | {ca['score']['total_ai']:g} | {cb['score']['total_ai']:g} | "
                     f"{cb['score']['total_ai'] - ca['score']['total_ai']:+g} |")
        out += lines + [""]
    keys = ["total_mae", "unit_mae", "unit_exact_pct", "over_credit", "under_credit", "review_recall",
            "review_precision", "false_flags", "time_p50", "time_p95", "inr_per_copy", "contaminated"]
    # Same footing on both sides: first run of every case only (a --runs 3 file would otherwise count 3×).
    common = {c["case"] for c in a["copies"]} & {c["case"] for c in b["copies"]}
    firsts = lambda run: [next(c for c in run["copies"] if c["case"] == k) for k in sorted(common)]  # noqa: E731
    sa, sb = summarize(firsts(a), None), summarize(firsts(b), None)
    out += ["## Summary (first run of each common case)", "", f"| metric | {la} | {lb} |", "|---|---|---|"]
    out += [f"| {k} | {sa.get(k)} | {sb.get(k)} |" for k in keys]
    pa = {p["check"]: p["pass"] for p in a["pass_fail"]}
    flips = [f"{k}: {pa.get(k)} → {p['pass']}" for p in b["pass_fail"] for k in [p["check"]] if pa.get(k) != p["pass"]]
    out += ["", "PASS/FAIL changes: " + ("; ".join(flips) if flips else "none")]
    return "\n".join(out)


# ════════════════════════════════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="all", help="all | name[,name]")
    ap.add_argument("--cases-dir", default=str(DEFAULT_CASES))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--app-root", default=str(BACKEND), help="folder that contains app/ (grade with other code)")
    ap.add_argument("--env-file", default=str(BACKEND / ".env"))
    ap.add_argument("--label", default=None)
    ap.add_argument("--rebuild-rubric", action="store_true", help="build the rubric fresh in every run")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--allow-fallback", action="store_true", help="let 429/503 fall back to other models (marks copies contaminated)")
    ap.add_argument("--call-timeout", type=float, default=180.0, help="seconds per Gemini call")
    ap.add_argument("--usd-inr", type=float, default=USD_INR_DEFAULT)
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.compare:
        print(compare(*args.compare))
        return 0

    warnings = []
    if date.today() > PRICE_VALID_UNTIL:
        warnings.append(f"price table purani ho sakti hai: gemini-3.8-flash intro rate ended {PRICE_VALID_UNTIL}")

    # ── environment, BEFORE the checker module is imported (it reads these at import time) ──
    from dotenv import load_dotenv
    if Path(args.env_file).exists():
        load_dotenv(args.env_file, override=False)
    else:
        warnings.append(f"env file not found: {args.env_file} (using the shell environment)")
    if not args.allow_fallback:
        os.environ["CHECKER_GEMINI_FALLBACK_MODELS"] = ""
    os.environ["CHECKER_GEMINI_DEADLINE_S"] = str(int(args.call_timeout))

    app_root = Path(args.app_root).resolve()
    sys.path.insert(0, str(app_root))
    log = CallLog()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    for h in root.handlers:
        h.setLevel(logging.WARNING)  # console: warnings only; CallLog still sees the INFO token lines
    root.addHandler(log)
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "google_genai", "urllib3", "app"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("app.services.smart_assignment_service").setLevel(logging.INFO)

    from app.services import smart_assignment_service as sas
    if not Path(sas.__file__).resolve().is_relative_to(app_root):
        print(f"ERROR: checker imported from {sas.__file__}, not from --app-root {app_root}")
        return 2
    from google import genai
    from google.genai import types
    key = sas.CHECKER_API_KEY or sas.settings.GEMINI_API_KEY
    sas._client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(args.call_timeout * 1000)))
    primary = sas.CHECKER_MODEL
    fallbacks = list(getattr(sas, "CHECKER_FALLBACK_MODELS", []))

    try:
        app_sha = subprocess.run(["git", "-C", str(app_root), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        app_sha = None
    label = args.label or app_sha or app_root.name

    cases_dir = Path(args.cases_dir).resolve()
    names = sorted(p.name for p in cases_dir.iterdir() if (p / "expected.json").exists()) if args.cases == "all" \
        else [n.strip() for n in args.cases.split(",") if n.strip()]
    cases = [load_case(cases_dir / n) for n in names]
    for c in cases:
        if c["warning"]:
            warnings.append(f"{c['name']}: {c['warning']}")
    print(f"eval '{label}': {len(cases)} case(s) × {args.runs} run(s) | model {primary} | fallback "
          f"{'ON ' + str(fallbacks) if args.allow_fallback else 'OFF'} | call timeout {args.call_timeout:g}s | app {app_root}")

    RUNS.mkdir(exist_ok=True)
    (RUNS / "rubrics").mkdir(exist_ok=True)
    rubric_calls, rubric_errors, rubric_info, rubric_builds = [], [], [], {}
    copies = []
    copy_timeout = args.call_timeout * 2 + 120  # 2 attempts + upload; each call is also capped by --call-timeout

    for run_no in range(1, args.runs + 1):
        rubrics = {}
        for case in cases:
            k = rubric_key(sas, case)
            if k in rubrics:
                case["_rubric"], case["_rubric_src"] = rubrics[k], "cache"
                continue
            path = RUNS / "rubrics" / f"{k}.json"
            if path.exists() and not args.rebuild_rubric:
                rubric, src = json.loads(path.read_text(encoding="utf-8"))["rubric"], "cache"
            else:
                def build():
                    return sas.build_rubric(str(case["paper"]), str(case["key"]) if case["key"] else None)
                rubric, err, tid = run_with_timeout(build, copy_timeout)
                calls, errs = log.take(tid)
                rubric_calls += calls
                rubric_errors += errs
                if err:
                    print(f"rubric for {case['name']} FAILED: {err}")
                    rubric = None
                else:
                    rubric = {kk: vv for kk, vv in rubric.items() if not kk.startswith("_")}
                    blob = json.dumps({"rubric": rubric, "built_at": datetime.now().isoformat(timespec="seconds"),
                                       "model": primary, "calls": calls}, indent=1, ensure_ascii=False)
                    # The first build becomes the cached rubric that later runs (and comparisons) share;
                    # --rebuild-rubric builds are kept beside it and never replace it.
                    if not path.exists():
                        path.write_text(blob, encoding="utf-8")
                    else:
                        (RUNS / "rubrics" / f"{k}_rebuild_{datetime.now():%Y%m%d-%H%M%S}_{run_no}.json").write_text(blob, encoding="utf-8")
                    rubric_builds.setdefault(k, []).append(rubric_summary(rubric))
                src = "built"
            rubrics[k] = rubric
            case["_rubric"], case["_rubric_src"] = rubric, src
            if rubric:
                rs = rubric_summary(rubric)
                rubric_info.append({"run": run_no, "case": case["name"], "key": k, "source": src,
                                    "total_marks": rs["total_marks"], "questions": rs["questions"], "points": rs["points"]})
                print(f"run {run_no}: rubric {k} ({src}) — {rs['questions']} questions, {rs['total_marks']} marks, {rs['points']} points")

        todo = [c for c in cases if c["_rubric"]]
        for c in cases:
            if not c["_rubric"]:
                copies.append({"case": c["name"], "run": run_no, "synthetic": c["synthetic"], "rubric": "failed",
                               "error": "rubric failed (see above)", "wall_s": None, "calls": [], "api_errors": [],
                               "models": [], "fallback_models": [], "tokens": {"input": 0, "output": 0, "thinking": 0},
                               "inr": 0.0, "unverified_price": []})
        for i in range(0, len(todo), max(1, args.parallel)):
            batch = todo[i:i + max(1, args.parallel)]
            threads = []
            for case in batch:
                def grade(case=case):
                    t0 = time.time()
                    res = sas.grade_sheet(case["pages"], case["_rubric"], case["strictness"], require_identity=False)
                    return res, round(time.time() - t0, 1)
                box = {}
                t = threading.Thread(target=lambda case=case, box=box, grade=grade: box.update(
                    zip(("value", "error", "tid"), run_with_timeout(grade, copy_timeout))), daemon=True)
                t.start()
                threads.append((case, t, box))
            for case, t, box in threads:
                t.join(copy_timeout + 5)
                value, err, tid = box.get("value"), box.get("error") or ("TIMEOUT" if not box else None), box.get("tid")
                calls, errs = log.take(tid) if tid else ([], [])
                inr, unverified = cost_inr(calls, args.usd_inr)
                models = sorted({c["model"] for c in calls})
                entry = {"case": case["name"], "run": run_no, "synthetic": case["synthetic"], "rubric": case["_rubric_src"],
                         "calls": calls, "api_errors": errs, "models": models,
                         "fallback_models": [m for m in models if m != primary],
                         "tokens": {k: sum(c[k] for c in calls) for k in ("input", "output", "thinking")},
                         "inr": inr, "unverified_price": unverified}
                if err or not value:
                    entry["error"] = err or "no result"
                    entry["wall_s"] = None
                else:
                    result, wall = value
                    entry["wall_s"] = wall
                    entry["score"] = score(case, result)
                    entry["result"] = result
                copies.append(entry)
                print(md_copy(entry))

    rubric_inr, rubric_unverified = cost_inr(rubric_calls, args.usd_inr)
    summary = summarize(copies, {c["name"]: c for c in cases})
    summary["http_429"] += sum(1 for e in rubric_errors if e["code"] == 429)
    summary["http_5xx"] += sum(1 for e in rubric_errors if e["code"] >= 500)
    run = {
        "label": label, "started": datetime.now().isoformat(timespec="seconds"), "app_root": str(app_root),
        "app_sha": app_sha, "model": primary, "fallback_allowed": args.allow_fallback, "fallback_models": fallbacks,
        "call_timeout_s": args.call_timeout, "cases_dir": str(cases_dir), "runs": args.runs,
        "rubrics": rubric_info, "rubric_calls": rubric_calls, "rubric_api_errors": rubric_errors,
        "cost_inr_total": round(sum(c["inr"] for c in copies) + rubric_inr, 2),
        "usd_inr": args.usd_inr, "prices": PRICES, "warnings": warnings,
        "summary": summary, "pass_fail": [], "stability": stability(copies),
        "rubric_stability": rubric_stability(rubric_builds), "copies": copies,
    }
    run["pass_fail"] = pass_fail(copies, summary)
    if rubric_unverified:
        run["warnings"].append("unverified price for rubric model(s): " + ", ".join(rubric_unverified))
    if any(c.get("unverified_price") for c in copies):
        run["warnings"].append("some copies used a model with an UNVERIFIED price — ₹ is approximate")
    print(md_summary(run))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label)
    path = RUNS / f"checker_{safe}_{stamp}.json"
    path.write_text(json.dumps(run, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"saved {path}")
    return 0 if all(p["pass"] is not False for p in run["pass_fail"]) else 1


if __name__ == "__main__":
    sys.exit(main())
