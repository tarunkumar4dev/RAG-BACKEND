"""
Copy-checker eval on the golden set (Downloads/golden_set.json + the copies it lists).

  python scripts/eval_checker.py --pipeline v3            # current single-call grader (baseline)
  python scripts/eval_checker.py --pipeline v4            # map → assemble → per-question grade
  python scripts/eval_checker.py --pipeline v4 --only Aman_Gupta_answer_sheet.pdf

For every copy: expected vs AI per question, total difference, how many questions are off (and by
more than 1 mark), whether the expected review flags were raised, wall time, tokens and ₹.
The rubric is built once from the paper + answer key and cached in scripts/eval_runs/.
Results go to scripts/eval_runs/checker_<pipeline>_<timestamp>.json.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv  # noqa: E402

# The worktree has no .env of its own; the main checkout's is the local one.
for env in (BACKEND / ".env", Path("C:/Users/login/Downloads/RAG-BACKEND/test-generator-backend/.env")):
    if env.exists():
        load_dotenv(env, override=False)
        break

GOLDEN = Path(os.getenv("GOLDEN_SET", "C:/Users/login/Downloads/golden_set.json"))
RUNS = HERE / "eval_runs"
# gemini-3.8-flash introductory list price (USD / 1M tokens) until 31 Dec 2026; output includes thinking.
PRICE = {"gemini-3.8-flash": (0.75, 3.75), "gemini-3-flash-preview": (0.50, 3.00)}
USD_INR = float(os.getenv("USD_INR", "88"))


class Usage(logging.Handler):
    """Collects the "[checker] <tag> tokens: input=… output=… thinking=… model=… time=…s" log lines."""

    PATTERN = re.compile(r"\[checker\] (\S+) tokens: input=(\d+) output=(\d+) thinking=(\d+) model=(\S+) time=([\d.]+)s")

    def __init__(self):
        super().__init__()
        self.calls = []

    def emit(self, record):
        m = self.PATTERN.search(record.getMessage())
        if m:
            self.calls.append({"tag": m[1], "input": int(m[2]), "output": int(m[3]), "thinking": int(m[4]),
                               "model": m[5], "seconds": float(m[6]), "thread": record.thread})


def cost_inr(calls) -> float:
    usd = 0.0
    for c in calls:
        pin, pout = PRICE.get(c["model"], PRICE["gemini-3.8-flash"])
        usd += (c["input"] * pin + (c["output"] + c["thinking"]) * pout) / 1e6
    return round(usd * USD_INR, 2)


def pdf_pages(pdf: Path, out_dir: Path, dpi: int = 150) -> list:
    """Copies are PDFs here; teachers upload phone photos, so grade page images like production."""
    import pymupdf
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with pymupdf.open(str(pdf)) as doc:
        for i, page in enumerate(doc):
            p = out_dir / f"{pdf.stem}_p{i + 1}.png"
            if not p.exists():
                page.get_pixmap(dpi=dpi).save(str(p))
            paths.append(str(p))
    return paths


def question_key(q_no: str) -> str:
    """Golden keys: 'A' = Section A (Q1–Q8) summed, else the question number."""
    n = re.sub(r"\D", "", str(q_no))
    return "A" if n.isdigit() and int(n) <= 8 else (n or str(q_no))


def score_copy(copy: dict, result: dict) -> dict:
    got: dict = {}
    flags: dict = {}
    for gq in result.get("graded_questions") or []:
        if gq.get("counted") is False:
            continue
        k = question_key(gq["q_no"])
        marks = gq["teacher_marks"] if gq.get("teacher_marks") is not None else gq["marks_awarded"]
        got[k] = round(got.get(k, 0) + marks, 2)
        if gq.get("needs_review"):
            flags[k] = gq.get("review_reasons") or ["review"]
    rows, off, off_big = [], 0, 0
    for k, exp in copy["expected"].items():
        ai = got.get(k, 0)
        diff = round(ai - exp, 2)
        off += diff != 0
        off_big += abs(diff) > 1
        rows.append({"q": k, "expected": exp, "ai": ai, "diff": diff, "flag": flags.get(k)})
    total = round(sum(r["ai"] for r in rows), 2)
    want_flags = [k for k, note in (copy.get("notes") or {}).items() if "flag" in note.lower()]
    return {
        "rows": rows, "total_ai": total, "total_expected": copy["total"], "total_diff": round(total - copy["total"], 2),
        "questions_off": off, "questions_off_by_more_than_1": off_big,
        "expected_flags": {k: bool(flags.get(k)) for k in want_flags},
        "unassigned": result.get("unassigned") or [],
    }


def get_rubric(sas, paper: Path, key: Path, rebuild: bool) -> dict:
    cache = RUNS / f"rubric_{paper.stem}.json"
    if cache.exists() and not rebuild:
        return json.loads(cache.read_text(encoding="utf-8"))
    rubric = sas.build_rubric(str(paper), str(key))
    rubric = {k: v for k, v in rubric.items() if not k.startswith("_")}
    cache.write_text(json.dumps(rubric, indent=1, ensure_ascii=False), encoding="utf-8")
    return rubric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", choices=["v3", "v4"], required=True)
    ap.add_argument("--only", help="run just this copy file")
    ap.add_argument("--rebuild-rubric", action="store_true")
    ap.add_argument("--parallel", type=int, default=3, help="copies checked at the same time")
    args = ap.parse_args()

    RUNS.mkdir(exist_ok=True)
    usage = Usage()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger().addHandler(usage)
    logging.getLogger().setLevel(logging.INFO)
    for noisy in ("httpx", "google_genai", "urllib3", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from app.services import smart_assignment_service as sas
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    base = GOLDEN.parent
    rubric = get_rubric(sas, base / golden["paper"], base / golden["answer_key"], args.rebuild_rubric)
    rubric_calls = list(usage.calls)
    usage.calls.clear()
    print(f"rubric: {rubric.get('total_questions')} questions, {rubric.get('total_marks')} marks "
          f"({'built' if rubric_calls else 'cached'})", flush=True)

    copies = [c for c in golden["copies"] if not args.only or c["file"] == args.only]

    def run(copy):
        import threading
        pages = pdf_pages(base / copy["file"], RUNS / "pages")
        start = time.time()
        try:
            if args.pipeline == "v3":
                result = sas.grade_sheet(pages, rubric, "medium", require_identity=False)
            else:
                from app.services import checker_v4
                result = checker_v4.check_copy(pages, rubric, "medium")
            error = None
        except Exception as e:  # keep going: one broken copy shouldn't hide the others
            result, error = {}, f"{type(e).__name__}: {e}"
        return copy, result, round(time.time() - start, 1), threading.get_ident(), error

    out = {"pipeline": args.pipeline, "model": sas.CHECKER_MODEL, "rubric_calls": rubric_calls, "copies": []}
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        for copy, result, wall, tid, error in pool.map(run, copies):
            # v3 makes its one call on the worker thread; v4 runs calls on its own pool and reports them
            calls = result.get("_calls") if args.pipeline == "v4" else [c for c in usage.calls if c["thread"] == tid]
            calls = calls or []
            entry = {"file": copy["file"], "wall_s": wall, "error": error, "calls": len(calls),
                     "tokens": {k: sum(c[k] for c in calls) for k in ("input", "output", "thinking")},
                     "inr": cost_inr(calls)}
            if result:
                entry.update(score_copy(copy, result))
                entry["result"] = result
            out["copies"].append(entry)
            print_copy(entry)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = RUNS / f"checker_{args.pipeline}_{stamp}.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    ok = [c for c in out["copies"] if not c.get("error")]
    print(f"\nSUMMARY {args.pipeline}: " + " | ".join(
        f"{c['file'].split('_')[0]} {c.get('total_ai')}/{c.get('total_expected')} (Δ{c.get('total_diff')}, "
        f"{c.get('questions_off')} off, {c.get('questions_off_by_more_than_1')} >1) {c['wall_s']}s ₹{c['inr']}"
        for c in ok))
    print(f"saved {path}")


def print_copy(e: dict):
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"\n=== {e['file']}  wall {e['wall_s']}s  calls {e['calls']}  tokens {e['tokens']}  ₹{e['inr']}")
    if e.get("error"):
        print("   ERROR", e["error"])
        return
    for r in e["rows"]:
        mark = "" if r["diff"] == 0 else ("  <-- " + ("+" if r["diff"] > 0 else "") + f"{r['diff']:g}")
        flag = f"  [review: {'; '.join(r['flag'])[:70]}]" if r["flag"] else ""
        print(f"   Q{r['q']:>3}  expected {r['expected']:>4g}  ai {r['ai']:>4g}{mark}{flag}")
    print(f"   TOTAL {e['total_ai']:g} / expected {e['total_expected']:g}  (Δ {e['total_diff']:+g}), "
          f"{e['questions_off']} questions off, {e['questions_off_by_more_than_1']} off by >1")
    if e["expected_flags"]:
        print(f"   expected review flags raised: {e['expected_flags']}")
    if e["unassigned"]:
        print(f"   unassigned answers: {len(e['unassigned'])}")


if __name__ == "__main__":
    main()
