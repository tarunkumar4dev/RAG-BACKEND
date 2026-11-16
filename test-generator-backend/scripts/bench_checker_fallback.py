"""Gate 2 #2 benchmark: grade the same copies with the checker's primary model and a candidate fallback, side by side.

    python scripts/bench_checker_fallback.py --sheets <sheets.json> --out <dir OUTSIDE the repo> \
        --models gemini-3.8-flash gemini-3.6-flash --budget-inr 95 --confirm-spend

<sheets.json>: rows of check_sheets joined with check_sessions (id, pages, source_bucket, ai_marks, max_marks,
model_version, strictness, rubric), exported read-only with `supabase db query --remote --read-only`.

What it does
- Downloads each copy's page files from Supabase storage (service client from .env; read-only) into <out>/pages.
- For every copy, grades it once per model with the production code path (answer_checker.copy_check.grade_copy),
  with fallbacks OFF so each column is that model alone.
- Writes <out>/benchmark.md (per question: max, marks per model, delta, feedback per model) and <out>/raw.json.

Spend control (paid Gemini calls)
- Refuses to run without --confirm-spend.
- Prices tokens at the list price (job_worker.price_*_per_m_inr: gemini-3.x-flash $0.75/$3.75 per M = Rs 63/315
  until 31 Dec 2026). Before every call it stops if spent + HEADROOM_INR would pass --budget-inr; HEADROOM_INR is
  above the most expensive October copy (Rs 5.65 measured).
- Copies are graded model-by-model per copy, so a budget stop leaves complete pairs only.

Privacy
- Copies are labelled "Copy 01..NN"; no sheet / student / submission ids, names or roll numbers are written.
  student_info is dropped from every result. Page images stay in <out>/pages: keep <out> outside the repo and delete
  it after review.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HEADROOM_INR = 7.0
ROOT = Path(__file__).resolve().parents[1]


def _setup_env():
    os.environ["LLM_PROVIDER"] = "gemini"
    os.environ["GEMINI_MOCK"] = "0"
    sys.path.insert(0, str(ROOT))


def _download(rows, out: Path) -> list[list[str]]:
    from app.core.database import get_supabase_admin
    sb = get_supabase_admin()
    copies = []
    for i, r in enumerate(rows, 1):
        d = out / "pages" / f"{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for j, p in enumerate(r["pages"], 1):
            ext = p.rsplit(".", 1)[-1].lower() if "." in p else "jpg"
            f = d / f"{j:02d}.{ext}"
            if not f.exists() or f.stat().st_size == 0:
                f.write_bytes(sb.storage.from_(r.get("source_bucket") or "submissions").download(p))
            paths.append(str(f))
        copies.append(paths)
    return copies


def _cost_inr(u: dict) -> float:
    from app.services import job_worker as jw
    out = (u.get("output_tokens") or 0) + (u.get("thinking_tokens") or 0)
    return ((u.get("input_tokens") or 0) * jw.price_in_per_m_inr() + out * jw.price_out_per_m_inr()) / 1e6


def _clean(result: dict) -> dict:
    result = dict(result or {})
    for k in ("student_info", "identity", "student_name", "roll_no"):
        result.pop(k, None)
    return result


def _grade(model: str, paths: list[str], rubric: dict, strictness: str) -> dict:
    from app.services import smart_assignment_service as svc
    from app.services.answer_checker.copy_check import grade_copy
    svc.CHECKER_MODEL = model
    svc.CHECKER_FALLBACK_MODELS = []          # each column = that model alone
    t0 = time.time()
    try:
        result, _ordered, usage = grade_copy(paths, rubric, strictness)
        err = None
    except Exception as e:  # noqa: BLE001
        result, usage, err = {}, {}, f"{type(e).__name__}: {str(e)[:160]}"
    usage = dict(usage or {})
    return {"model": model, "seconds": round(time.time() - t0, 1), "error": err, "usage": usage,
            "cost_inr": round(_cost_inr(usage), 4), "result": _clean(result)}


def _questions(run: dict) -> dict:
    out = {}
    for q in (run.get("result") or {}).get("graded_questions") or []:
        key = f'{q.get("q_no")}{("/" + str(q["sub_id"])) if q.get("sub_id") else ""}'
        out[key] = q
    return out


def _cell(text) -> str:
    return str(text if text is not None else "").replace("|", "/").replace("\n", " ").strip()


def _report(runs: list[dict], models: list[str], spent: float, budget: float, stopped: str | None) -> str:
    a, b = models
    L = [f"# Checker fallback benchmark: `{a}` vs `{b}`", "",
         f"Run {time.strftime('%Y-%m-%d %H:%M')} IST. {len(runs)} copies. Spend Rs {spent:.2f} of Rs {budget:.0f} "
         f"(list price, tokens measured from Gemini usage_metadata). Fallbacks OFF: each column is one model.",
         "Production path: answer_checker.copy_check.grade_copy, same rubric and strictness as October.",
         "Copies are anonymised (Copy 01..NN); no names, roll numbers or ids.", ""]
    if stopped:
        L += [f"**Stopped early:** {stopped}", ""]
    L += ["## Summary", "", f"| Copy | Pages | Oct prod marks | {a} | {b} | Δ total | {a} Rs | {b} Rs | {a} s | {b} s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    blank = {"error": "not run", "result": {}, "cost_inr": 0.0, "seconds": ""}
    for c in runs:
        ra, rb = c["runs"].get(a) or blank, c["runs"].get(b) or blank
        ta = (ra["result"] or {}).get("total_marks_awarded")
        tb = (rb["result"] or {}).get("total_marks_awarded")
        d = (tb - ta) if isinstance(ta, (int, float)) and isinstance(tb, (int, float)) else ""
        L.append(f"| {c['label']} | {c['pages']} | {c['prod_marks']}/{c['max_marks']} | "
                 f"{ta if ra['error'] is None else 'ERROR'} | {tb if rb['error'] is None else 'ERROR'} | {d} | "
                 f"{ra['cost_inr']:.2f} | {rb['cost_inr']:.2f} | {ra['seconds']} | {rb['seconds']} |")
    for c in runs:
        ra, rb = c["runs"].get(a) or blank, c["runs"].get(b) or blank
        L += ["", f"## {c['label']} ({c['pages']} pages, max {c['max_marks']})", ""]
        for r in (ra, rb):
            if r["error"]:
                L.append(f"- `{r['model']}` failed: {_cell(r['error'])}")
        qa, qb = _questions(ra), _questions(rb)
        L += [f"| Q | Max | {a} | {b} | Δ | {a} feedback | {b} feedback |", "|---|---|---|---|---|---|---|"]
        for key in sorted(set(qa) | set(qb), key=lambda k: (len(k.split('/')[0]), k)):
            x, y = qa.get(key, {}), qb.get(key, {})
            mx = x.get("max_marks", x.get("marks", y.get("max_marks", y.get("marks", ""))))
            ma, mb = x.get("marks_awarded"), y.get("marks_awarded")
            d = (mb - ma) if isinstance(ma, (int, float)) and isinstance(mb, (int, float)) else ""
            L.append(f"| {key} | {mx} | {'' if ma is None else ma} | {'' if mb is None else mb} | {d} | "
                     f"{_cell(x.get('feedback'))} | {_cell(y.get('feedback'))} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs=2, default=["gemini-3.8-flash", "gemini-3.6-flash"])
    ap.add_argument("--budget-inr", type=float, default=95.0)
    ap.add_argument("--confirm-spend", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="keep successful runs from <out>/raw.json, retry only the failed ones (spend carries over)")
    ap.add_argument("--only", nargs="*", default=None, help="copy numbers to (re)run, e.g. 1 4")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    if str(out).lower().startswith(str(ROOT.parent).lower()):
        sys.exit("--out must be outside the repo (page images are personal data)")
    if not args.confirm_spend:
        sys.exit("paid Gemini calls: pass --confirm-spend")
    _setup_env()
    rows = json.loads(Path(args.sheets).read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    copies = _download(rows, out)
    print(f"pages ready: {sum(len(c) for c in copies)} files for {len(copies)} copies", flush=True)

    prev = {}
    if args.resume and (out / "raw.json").exists():
        prev = {c["label"]: c for c in json.loads((out / "raw.json").read_text(encoding="utf-8"))}
    spent = sum(r["cost_inr"] for c in prev.values() for r in c["runs"].values())
    runs, stopped = [], None
    for i, (r, paths) in enumerate(zip(rows, copies), 1):
        label = f"Copy {i:02d}"
        c = prev.get(label) or {"label": label, "pages": len(paths), "prod_marks": r.get("ai_marks"),
                                "max_marks": r.get("max_marks"), "runs": {}}
        todo = [m for m in args.models if m not in c["runs"] or c["runs"][m]["error"]]
        if args.only is not None and str(i) not in args.only and f"{i:02d}" not in args.only:
            todo = []
        if todo and spent + len(todo) * HEADROOM_INR > args.budget_inr:
            stopped = f"budget: Rs {spent:.2f} spent, the next run could pass Rs {args.budget_inr:.0f}"
            runs.append(c)
            continue
        for m in todo:
            if spent + HEADROOM_INR > args.budget_inr:
                stopped = f"budget: Rs {spent:.2f} spent, the next run could pass Rs {args.budget_inr:.0f}"
                break
            run = _grade(m, paths, r["rubric"], r.get("strictness") or "medium")
            run["attempts"] = c["runs"].get(m, {}).get("attempts", 1 if m in c["runs"] else 0) + 1
            spent += run["cost_inr"]
            c["runs"][m] = run
            print(f"{c['label']} {m}: {'ERR ' + run['error'] if run['error'] else (run['result'] or {}).get('total_marks_awarded')} "
                  f"Rs {run['cost_inr']:.2f} {run['seconds']}s (total Rs {spent:.2f})", flush=True)
        runs.append(c)
        _save(out, runs, prev, args, spent, stopped)       # after every copy: a stop (cap, 09:00) loses nothing

    _save(out, runs, prev, args, spent, stopped)
    print(f"done: Rs {spent:.2f}; {out / 'benchmark.md'}", flush=True)


def _save(out: Path, runs: list, prev: dict, args, spent: float, stopped) -> None:
    done = {c["label"] for c in runs}
    merged = runs + [c for label, c in sorted(prev.items()) if label not in done]
    merged.sort(key=lambda c: c["label"])
    (out / "raw.json").write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "benchmark.md").write_text(_report(merged, args.models, spent, args.budget_inr, stopped),
                                      encoding="utf-8")


if __name__ == "__main__":
    main()
