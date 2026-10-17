"""
Cost / speed / quality harness for the test generator.

Runs the real generation pipeline (same code path as /generate-frontend, no DB writes) on a
fixed set of CBSE papers, then the production answer-key checker, then an independent judge
model that re-solves every question. Writes a JSON + Markdown report so two runs (e.g. before
and after a prompt change) can be compared side by side.

Run from test-generator-backend/:
    python scripts/eval_generation.py --label baseline
    python scripts/eval_generation.py --label diet --papers Science10,Maths10 --no-judge
    python scripts/eval_generation.py --compare baseline,diet
"""
import argparse
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from app.core.config import settings  # noqa: E402
import app.services.test_generator_service as svc  # noqa: E402
from app.services.rag_service import retrieve_context  # noqa: E402
from app.api.v1.endpoints.test_generator import (  # noqa: E402
    FrontendGenerateRequest, _transform_frontend_to_backend, _transform_backend_to_frontend,
    _is_english_pseudo, _resolve_subject,
)

USD_TO_INR = 88.0
OUT_DIR = os.environ.get("EVAL_OUT_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_runs")
JUDGE_MODELS = os.environ.get("EVAL_JUDGE_MODELS", "gemini-3.6-flash,gemini-3.8-flash,gemini-3.7-flash").split(",")
JUDGE_BATCH = 14
JUDGE_SPACING_S = 13  # free tier allows ~5 requests/minute on these models

PAPERS = {
    "Science10": ("Science", "Class 10", ["Light Reflection and Refraction", "Electricity"]),
    "Maths10": ("Mathematics", "Class 10", ["Quadratic Equations", "Introduction to Trigonometry"]),
    "Physics12": ("Physics", "Class 12", ["Current Electricity", "Electrostatic Potential and Capacitance"]),
    "History10": ("History", "Class 10", ["The Rise of Nationalism in Europe", "Nationalism in India"]),
    "Accountancy12": ("Accountancy", "Class 12", ["Accounting for Partnership Basic Concepts", "Accounting for Share Capital"]),
}

# Content fields whose size we track (they drive output tokens).
FIELDS = ["text", "options", "correctAnswer", "explanation", "markingScheme", "subParts", "answerTable"]

JUDGE_PROMPT = """You are an expert CBSE {cls} {subject} examiner auditing an answer key that teachers will
use to grade students. Independently solve each question, then judge the key.

For each question decide:
- "correct": true only if the marked answer, any sub-part answers / answer table AND the working are all right
  (a wrong statement in the explanation of a right answer counts as incorrect).
- "sufficient": true if a teacher could award marks from this key alone (enough steps / points for the marks);
  false if it is a bare answer where working or points are expected.

Return JSON: {{"results":[{{"id":"...","correct":true|false,"sufficient":true|false,"problem":"one line, empty if fine"}}]}}

Questions:
{qs}"""


def _judge_view(q: dict) -> dict:
    v = {"id": q["id"], "format": q.get("format"), "marks": q.get("marks"), "question": q.get("text"),
         "options": q.get("options") or None, "marked_answer": q.get("correctAnswer"),
         "explanation": q.get("explanation") or ""}
    if q.get("subParts"):
        v["sub_parts"] = q["subParts"]
    if q.get("markingScheme"):
        v["marking_scheme"] = q["markingScheme"]
    t = q.get("answerTable")
    if t:
        v["answer_table"] = {"headers": t.get("headers"), "rows": (t.get("rows") or [])[:40], "total_row": t.get("total_row")}
    return v


class JudgeQuotaExhausted(Exception):
    pass


def _judge_call(prompt):
    """One judge request. Walks the model chain; a model that hits its quota is skipped for the rest of the run
    (free-tier projects get ~20 requests/day on the stronger Flash models)."""
    for model in list(JUDGE_MODELS):
        for attempt in range(3):
            r = httpx.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                           headers={"x-goog-api-key": settings.GEMINI_API_KEY},
                           json={"contents": [{"parts": [{"text": prompt}]}],
                                 "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"}},
                           timeout=180)
            if r.status_code == 200:
                d = r.json()
                text = "".join(p.get("text", "") for p in d["candidates"][0]["content"]["parts"])
                u = d.get("usageMetadata", {})
                return svc._extract_json(text), model, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)
            if r.status_code == 429 and "PerDay" in r.text:
                print(f"   judge: {model} daily quota used up", flush=True)
                JUDGE_MODELS.remove(model)
                break
            time.sleep(JUDGE_SPACING_S * (attempt + 1))  # per-minute quota or overload: wait and retry
        else:
            continue
    raise JudgeQuotaExhausted("no judge model has quota left")


def judge(questions, subject, cls):
    """Independent re-solve with stronger models, one request at a time. Returns (verdicts by id, usd, models used)."""
    short = {f"q{i + 1}": q["id"] for i, q in enumerate(questions)}  # short ids keep the judge prompt small
    views = [dict(_judge_view(q), id=k) for k, q in zip(short, questions)]
    out, usd, used = {}, 0.0, set()
    for i in range(0, len(views), JUDGE_BATCH):
        prompt = JUDGE_PROMPT.format(cls=cls, subject=subject, qs=json.dumps(views[i:i + JUDGE_BATCH], ensure_ascii=False))
        try:
            data, model, tin, tout = _judge_call(prompt)
        except JudgeQuotaExhausted:
            break
        except Exception as e:
            print(f"   judge batch failed: {str(e)[:150]}", flush=True)
            continue
        used.add(model)
        usd += (tin * 0.75 + tout * 3.75) / 1e6  # Flash-class list price
        res = data.get("results") if isinstance(data, dict) else data
        for r in res if isinstance(res, list) else []:
            if isinstance(r, dict) and str(r.get("id")) in short:
                out[short[str(r["id"])]] = r
        time.sleep(JUDGE_SPACING_S)
    return out, usd, sorted(used)


def judge_row(row):
    subject, cls, _ = PAPERS[row["paper"]]
    qs = row["final_questions"]
    verdicts, jusd, models = judge(qs, subject, cls)
    wrong = [(q["id"], q.get("format"), (verdicts.get(q["id"]) or {}).get("problem"))
             for q in qs if (verdicts.get(q["id"]) or {}).get("correct") is False]
    thin = [(q["id"], q.get("format"), (verdicts.get(q["id"]) or {}).get("problem"))
            for q in qs if (verdicts.get(q["id"]) or {}).get("sufficient") is False]
    judged = sum(1 for q in qs if q["id"] in verdicts)
    row["judge"] = {"wrong": len(wrong), "insufficient": len(thin), "judged": judged, "of": len(qs), "usd": round(jusd, 4),
                    "models": models, "wrong_items": wrong, "insufficient_items": thin}
    print(f"  judge {models}: {len(wrong)} wrong, {len(thin)} insufficient, {judged}/{len(qs)} judged", flush=True)


def rejudge(label):
    """Judge (or finish judging) a saved run without regenerating it."""
    path = os.path.join(OUT_DIR, f"{label}.json")
    rows = json.load(open(path, encoding="utf-8"))
    for row in rows:
        if row.get("failed") or (row.get("judge") and row["judge"]["judged"] >= row["judge"].get("of", 0) > 0):
            continue
        print(f"\n=== judge {row['paper']} [{label}] ===", flush=True)
        judge_row(row)
        json.dump(rows, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if not JUDGE_MODELS:
            print("judge quota exhausted for today; rerun --rejudge later", flush=True)
            break
    print("\n" + report([label]), flush=True)


def generate(key):
    subject, cls, chapters = PAPERS[key]
    req = FrontendGenerateRequest(examTitle=key, board="CBSE", classGrade=cls, subject=subject,
                                  simpleData=[{"topic": c, "quantity": 10, "marks": 1, "difficulty": "medium", "format": "mcq"} for c in chapters],
                                  cbsePattern=True)
    br = _transform_frontend_to_backend(req)
    real = [c.chapter for c in br.chapters if not _is_english_pseudo(br.subject, c.chapter)]
    topics = [c.topic for c in br.chapters if c.topic] or real
    chunks = retrieve_context(real, topics, br.subject, br.class_grade)
    stats = svc._PaperStats()
    token = svc._PAPER_STATS.set(stats)
    t0 = time.time()
    try:
        if _resolve_subject(subject).lower() in ("accountancy", "accounts", "accounting"):
            qs = svc.generate_cbse_accountancy_paper(br, chunks)
        else:
            qs = svc.generate_questions(br, chunks, cbse_pattern=True)
    finally:
        svc._PAPER_STATS.reset(token)
    secs = time.time() - t0
    resp = _transform_backend_to_frontend(qs, req)
    usd, _ = stats.cost_usd()
    calls = {m: {"calls": c[0], "in": c[1], "out": c[2], "cached": c[3]} for m, c in stats.calls.items()}
    return {"secs": round(secs, 1), "questions": [q.model_dump() for q in resp.questions], "main": resp.totalQuestions,
            "marks": resp.totalMarks, "usd": usd, "calls": calls, "generated": stats.generated, "drops": stats.dropped}


def check(questions, subject, cls):
    stats = svc._PaperStats()
    token = svc._PAPER_STATS.set(stats)
    t0 = time.time()
    try:
        res = svc.verify_answer_key(questions, subject, cls)
    finally:
        svc._PAPER_STATS.reset(token)
    usd, _ = stats.cost_usd()
    return {"secs": round(time.time() - t0, 1), "usd": usd, "fixed": res["fixed"], "flagged": res["flagged"],
            "unchecked": res["unchecked"], "results": res["results"]}


def apply_fixes(questions, results):
    by = {r["id"]: r for r in results}
    out = []
    for q in questions:
        r = by.get(q["id"])
        if r and r["verdict"] == "fix":
            q = dict(q, correctAnswer=r["correctAnswer"], explanation=r.get("explanation") or q.get("explanation"))
        out.append(q)
    return out


def run(label, papers, use_checker, use_judge):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{label}.json")
    # Re-running some papers under an existing label replaces just those rows.
    rows = [r for r in (json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []) if r.get("paper") not in papers]
    for key in papers:
        subject, cls, _ = PAPERS[key]
        print(f"\n=== {key} [{label}] ===", flush=True)
        try:
            g = generate(key)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            rows.append({"paper": key, "failed": str(e)[:300]})
            continue
        qs = g["questions"]
        tin = sum(c["in"] for c in g["calls"].values())
        tout = sum(c["out"] for c in g["calls"].values())
        tcached = sum(c["cached"] for c in g["calls"].values())
        chars = {f: sum(len(json.dumps(q.get(f), ensure_ascii=False)) for q in qs if q.get(f)) for f in FIELDS}
        row = {"paper": key, "label": label, "secs": g["secs"], "questions": len(qs), "main": g["main"], "marks": g["marks"],
               "generated": g["generated"], "drops": g["drops"], "in": tin, "out": tout, "cached": tcached,
               "gen_inr": round(g["usd"] * USD_TO_INR, 2), "chars": chars, "calls": g["calls"]}
        print(f"  gen: {g['secs']}s, {len(qs)} q ({g['main']} main), {g['marks']} marks, generated {g['generated']}, "
              f"drops {g['drops']}, in {tin} ({tcached} cached) / out {tout} -> Rs {row['gen_inr']}", flush=True)

        final_qs = qs
        if use_checker:
            c = check(qs, subject, cls)
            row["checker"] = {k: v for k, v in c.items() if k != "results"}
            row["checker"]["inr"] = round(c["usd"] * USD_TO_INR, 2)
            final_qs = apply_fixes(qs, c["results"])
            print(f"  checker: {c['secs']}s, fixed {c['fixed']}, flagged {c['flagged']}, unchecked {c['unchecked']}, "
                  f"Rs {row['checker']['inr']}", flush=True)

        row["final_questions"] = final_qs
        if use_judge:
            judge_row(row)
        rows.append(row)
        json.dump(rows, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n" + report([label]), flush=True)


def _load(label):
    return [r for r in json.load(open(os.path.join(OUT_DIR, f"{label}.json"), encoding="utf-8")) if not r.get("failed")]


def report(labels):
    lines = ["| Run | Paper | Time | Marks | Q | Gen/kept | Out tok | In tok (cached) | Gen Rs | Checker Rs | Wrong | Thin |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for label in labels:
        rows = _load(label)
        for r in rows:
            j = r.get("judge") or {}
            c = r.get("checker") or {}
            lines.append(f"| {label} | {r['paper']} | {r['secs']}s | {r['marks']} | {r['questions']} | {r['generated']}/{r['questions']} | "
                         f"{r['out']} | {r['in']} ({r['cached']}) | {r['gen_inr']} | {c.get('inr', '-')} | "
                         f"{j.get('wrong', '-')}/{j.get('judged', '-')} | {j.get('insufficient', '-')} |")
        n = len(rows) or 1
        judged = sum((r.get("judge") or {}).get("judged", 0) for r in rows) or 1
        wrong = sum((r.get("judge") or {}).get("wrong", 0) for r in rows)
        thin = sum((r.get("judge") or {}).get("insufficient", 0) for r in rows)
        lines.append(f"| **{label} avg** | | {sum(r['secs'] for r in rows) / n:.0f}s | | | | {sum(r['out'] for r in rows) / n:.0f} | "
                     f"{sum(r['in'] for r in rows) / n:.0f} | **{sum(r['gen_inr'] for r in rows) / n:.2f}** | "
                     f"{sum((r.get('checker') or {}).get('inr', 0) for r in rows) / n:.2f} | {100 * wrong / judged:.1f}% | {100 * thin / judged:.1f}% |")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--papers", default=",".join(PAPERS))
    ap.add_argument("--no-checker", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--compare", help="comma-separated labels to print side by side")
    ap.add_argument("--rejudge", help="judge a saved run (resumes where the daily quota stopped it)")
    a = ap.parse_args()
    if a.rejudge:
        rejudge(a.rejudge)
    elif a.compare:
        print(report(a.compare.split(",")))
    else:
        run(a.label, a.papers.split(","), not a.no_checker, not a.no_judge)
