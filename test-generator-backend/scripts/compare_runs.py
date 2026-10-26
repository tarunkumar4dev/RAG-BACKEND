"""
Run A vs run B, per copy and per question (two export_ai_marks.py CSVs). Stdlib only.

  python scripts/compare_runs.py --a prod.csv --b shadow.csv [--primary gemini-3.8-flash] [--out report.md]

Copies are matched by submission_id. Per copy: total A / B (counted questions), total diff, MAE over questions,
questions changed. Summary split by "original model" = A's model: primary vs fallback (the 24 Oct copies).
A question in only one run counts as 0 in the other and is listed.
"""

import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict

PRIMARY = os.getenv("CHECKER_GEMINI_MODEL", "gemini-3.8-flash")


def load(path: str) -> dict:
    """submission_id → {"roll", "name", "model", "q": {question: marks (counted only)}}"""
    out: dict = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = out.setdefault(r["submission_id"], {"roll": r["roll"], "name": r["name"], "model": r["model"], "q": {}})
            if r.get("counted", "true") != "false":
                c["q"][r["question"]] = float(r["marks"] or 0)
    return out


def compare(a: dict, b: dict, primary: str = PRIMARY) -> dict:
    copies, only_a, only_b = [], sorted(set(a) - set(b)), sorted(set(b) - set(a))
    for sid in sorted(set(a) & set(b), key=lambda s: (a[s]["roll"] or "", a[s]["name"] or "")):
        qa, qb = a[sid]["q"], b[sid]["q"]
        qs = sorted(set(qa) | set(qb), key=lambda q: (len(q), q))
        diffs = {q: round(qb.get(q, 0) - qa.get(q, 0), 2) for q in qs}
        ta, tb = round(sum(qa.values()), 2), round(sum(qb.values()), 2)
        copies.append({"submission_id": sid, "roll": a[sid]["roll"], "name": a[sid]["name"], "model_a": a[sid]["model"],
                       "group": "primary" if a[sid]["model"] == primary else "fallback",
                       "total_a": ta, "total_b": tb, "total_diff": round(tb - ta, 2),
                       "mae_q": round(statistics.mean(abs(d) for d in diffs.values()), 3) if diffs else 0.0,
                       "changed": sum(1 for d in diffs.values() if d), "questions": len(qs),
                       "diffs": {q: d for q, d in diffs.items() if d},
                       "missing": [q for q in qs if q not in qa or q not in qb]})
    groups = defaultdict(list)
    for c in copies:
        groups[c["group"]].append(c)
        groups["all"].append(c)
    summary = {}
    for g, cs in groups.items():
        summary[g] = {"copies": len(cs),
                      "mean_abs_total_diff": round(statistics.mean(abs(c["total_diff"]) for c in cs), 2),
                      "mean_total_diff": round(statistics.mean(c["total_diff"] for c in cs), 2),
                      "max_abs_total_diff": max(abs(c["total_diff"]) for c in cs),
                      "mae_per_question": round(statistics.mean(c["mae_q"] for c in cs), 3),
                      "questions_changed": sum(c["changed"] for c in cs),
                      "copies_changed": sum(1 for c in cs if c["changed"])}
    return {"copies": copies, "summary": summary, "only_a": only_a, "only_b": only_b}


def report(res: dict, label_a: str, label_b: str) -> str:
    lines = [f"# {label_b} vs {label_a}", "",
             "| group | copies | mean abs total diff | mean total diff (B−A) | max abs | MAE / question | questions changed | copies changed |",
             "|---|---|---|---|---|---|---|---|"]
    for g in ("all", "primary", "fallback"):
        s = res["summary"].get(g)
        if s:
            lines.append(f"| {g} | {s['copies']} | {s['mean_abs_total_diff']} | {s['mean_total_diff']:+} | "
                         f"{s['max_abs_total_diff']} | {s['mae_per_question']} | {s['questions_changed']} | {s['copies_changed']} |")
    lines += ["", "| roll | name | model A | total A | total B | diff | changed | per question (B−A) |", "|---|---|---|---|---|---|---|---|"]
    for c in sorted(res["copies"], key=lambda c: -abs(c["total_diff"])):
        per = ", ".join(f"Q{q} {d:+g}" for q, d in c["diffs"].items())
        lines.append(f"| {c['roll']} | {c['name']} | {c['model_a']} | {c['total_a']:g} | {c['total_b']:g} | "
                     f"{c['total_diff']:+g} | {c['changed']}/{c['questions']} | {per} |")
    if res["only_a"] or res["only_b"]:
        lines += ["", f"Only in A: {len(res['only_a'])} copies · only in B: {len(res['only_b'])} copies"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="baseline run CSV (e.g. prod)")
    ap.add_argument("--b", required=True, help="new run CSV (e.g. shadow)")
    ap.add_argument("--primary", default=PRIMARY)
    ap.add_argument("--out", help="write the markdown report here (holds names: outside the repo)")
    args = ap.parse_args()
    res = compare(load(args.a), load(args.b), args.primary)
    if not res["copies"]:
        sys.exit("no copies in both runs (match is by submission_id)")
    text = report(res, os.path.basename(args.a), os.path.basename(args.b))
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
