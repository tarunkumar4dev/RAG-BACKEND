"""
Shadow re-grade of one assignment: the same copies, rubric and answer-key version as production, graded again on
the PRIMARY model only (no fallback, temperature 0, seed 7 — the production settings), stored in
check_shadow_results under --label (26_checker_p0.sql). Nothing a teacher or student sees changes: no check_sheets /
submissions / check_jobs write, no published result touched, no page images saved, no notification.

  python scripts/shadow_regrade.py --assignment-id <uuid> --label shadow-2026-10-27 --dry-run
  python scripts/shadow_regrade.py --assignment-id <uuid> --label shadow-2026-10-27

--dry-run prints copies, pages and the estimated ₹ (from the copies' own past usage in check_jobs.usage) and calls
nothing. A real run skips copies already stored under the label (safe to re-run after a crash).
Then: export_ai_marks.py --run-label <label>, and compare_runs.py against --run-label prod.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
import time

from checker_runs import connect, copies, dump, rows

from app.services import job_worker, smart_assignment_service as svc  # noqa: E402
from app.services.answer_checker import copy_check, portal_result, preprocess, upload_validation  # noqa: E402

PRIMARY = svc.CHECKER_MODEL


def inr(c: dict) -> float:
    return float((c.get("usage") or {}).get("est_cost_inr") or 0)


def estimate(cs: list) -> dict:
    """₹ for re-grading `cs` on the primary: the primary copies' own past cost; for copies first marked by a
    backup model, the primary's average ₹ per page × their pages (their own cost was another model's)."""
    prim = [c for c in cs if (c.get("usage") or {}).get("model") == PRIMARY and inr(c) > 0]
    pages = lambda c: max(1, len(c.get("files") or []))                       # noqa: E731
    per_page = (sum(inr(c) for c in prim) / sum(pages(c) for c in prim)) if prim else None
    per_copy = statistics.mean(inr(c) for c in prim) if prim else None
    est, how = 0.0, {"own_past_cost": 0, "per_page_of_primary": 0}
    for c in cs:
        if c in prim:
            est += inr(c)
            how["own_past_cost"] += 1
        elif per_page is not None:
            est += per_page * pages(c)
            how["per_page_of_primary"] += 1
    return {"estimate_inr": round(est, 2), "how": how, "primary_avg_per_copy": per_copy and round(per_copy, 2),
            "primary_avg_per_page": per_page and round(per_page, 2),
            "past_cost_all_models": round(sum(inr(c) for c in cs), 2)}


def grade_copy_like_the_worker(c: dict, asg: dict, rubric: dict, storage) -> dict:
    """The worker's path for one attempt (validate the real bytes → grade_copy, or chunks + merge above 12 pages),
    with nothing written back to Storage."""
    sheet, sub = c["sheet"], {"id": c["submission_id"], "student_id": c["student_id"], "submitter_key": c["submitter_key"]}
    folder = job_worker._owner_folder(asg, sub)
    allowed = set(sheet.get("pages") or [])
    blobs = {}
    for e in c.get("files") or []:
        path = str(e.get("storage_path") or "")
        if path in allowed and path.startswith(folder) and ".." not in path.split("/"):
            data = storage.download(job_worker.SUBMISSIONS_BUCKET, path)
            if data is not None:
                blobs[path] = data
    v = upload_validation.validate(c.get("files") or [], blobs, asg.get("allowed_mimes"), asg.get("max_pages"))
    if not v.ok:
        raise preprocess.BadFile(v.summary())
    temps = [job_worker._write_temp(f.data, "." + f.ext, "shadow_") for f in v.files]
    strictness = rubric.get("strictness") or "medium"
    try:
        with svc.call_settings(timeout_s=job_worker.GRADE_CALL_TIMEOUT_S):
            if v.total_pages <= job_worker.CHUNK_MAX_PAGES:
                result, _, usage = copy_check.grade_copy(temps, rubric, strictness)
                meta = {"pages": usage.get("pages"), "chunks": 1}
            else:
                norm = preprocess.normalize_sources(temps, max_pages=upload_validation.HARD_MAX_PAGES)
                n = math.ceil(len(norm) / job_worker.CHUNK_MAX_PAGES)
                bounds = [round(i * len(norm) / n) for i in range(n + 1)]
                parts, usage = [], {}
                for k in range(n):
                    chunk = norm[bounds[k]:bounds[k + 1]]
                    if all(p.blank for p in chunk):
                        parts.append({"result": {"graded_questions": [], "sheet_flags": []}, "pages": len(chunk)})
                        continue
                    ctemps = [job_worker._write_temp(p.jpeg, ".jpg", "shadow_c") for p in chunk]
                    try:
                        r, _, u = copy_check.grade_copy(ctemps, rubric, strictness)
                    finally:
                        job_worker._cleanup(ctemps)
                    parts.append({"result": r, "pages": len(chunk)})
                    usage = job_worker.add_usage(usage, u)
                result = portal_result.merge_chunks(parts, rubric)
                meta = {"pages": len(norm), "chunks": n}
    finally:
        job_worker._cleanup(temps)
    usage = job_worker.add_usage({}, usage)
    reasons, confidence, by_q = portal_result.review(result, rubric)
    keep = ("graded_questions", "total_marks_awarded", "max_marks", "percentage", "overall_remarks",
            "readability_score", "needs_review_count", "sheet_flags")
    ai = {k: result.get(k) for k in keep if k in result}
    ai["review"] = {"reasons": reasons, "by_question": by_q, "confidence": confidence}
    ai["meta"] = {**meta, "model_version": usage.get("model"), "models": usage.get("models"), "primary_model": PRIMARY,
                  "answer_key_version": c["answer_key_version"], "strictness": strictness}
    return {"ai_result": ai, "ai_marks": result.get("total_marks_awarded"), "max_marks": result.get("max_marks"),
            "reasons": reasons, "usage": usage, "model": usage.get("model")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assignment-id", required=True)
    ap.add_argument("--label", required=True, help="name of this run, e.g. shadow-2026-10-27 (not 'prod')")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N copies (by roll)")
    args = ap.parse_args()
    if args.label == "prod":
        sys.exit("'prod' is the production run's label; pick another")
    sys.stdout.reconfigure(encoding="utf-8")

    conn = connect()
    try:
        asg = (rows(conn, "select row_to_json(a) from public.assignments a where id = %s", (args.assignment_id,)) or [None])[0]
        session = (rows(conn, "select row_to_json(s) from public.check_sessions s where assignment_id = %s",
                        (args.assignment_id,)) or [None])[0]
        key = (rows(conn, "select row_to_json(k) from public.assignment_ai_keys k where assignment_id = %s",
                    (args.assignment_id,)) or [None])[0]
        if not asg or not session or not session.get("rubric"):
            sys.exit("assignment / its marking scheme not found")
        rubric = session["rubric"]
        cs = copies(conn, args.assignment_id)
        if args.limit:
            cs = cs[:args.limit]
        same_key = [c for c in cs if c["answer_key_version"] == rubric.get("key_version")]
        other_key = len(cs) - len(same_key)
        done = set()
        if not args.dry_run:
            done = set(rows(conn, "select submission_id::text from public.check_shadow_results where run_label = %s",
                            (args.label,)))
        todo = [c for c in same_key if str(c["submission_id"]) not in done]

        est = estimate(todo)
        print(f"assignment {args.assignment_id} · '{asg.get('title')}' · rubric key v{rubric.get('key_version')} "
              f"(current key v{(key or {}).get('version')}) · primary model {PRIMARY} · fallback "
              f"{'ON — refusing' if svc.CHECKER_ALLOW_FALLBACK else 'off'}")
        print(f"copies: {len(cs)} checked, {len(same_key)} with this rubric's key version"
              f"{f', {other_key} skipped (other key version)' if other_key else ''}"
              f"{f', {len(done)} already in run {args.label!r}' if done else ''}")
        print(f"to grade: {len(todo)} copies, {sum(len(c.get('files') or []) for c in todo)} uploaded pages/files")
        print(f"estimated cost: ₹{est['estimate_inr']} ({est['how']['own_past_cost']} from their own past primary cost, "
              f"{est['how']['per_page_of_primary']} at the primary's ₹{est['primary_avg_per_page']}/page) · "
              f"primary avg ₹{est['primary_avg_per_copy']}/copy · what these copies cost originally (all models): "
              f"₹{est['past_cost_all_models']}")
        if args.dry_run:
            print("dry run: nothing graded, nothing written")
            return
        if svc.CHECKER_ALLOW_FALLBACK:
            sys.exit("CHECKER_ALLOW_FALLBACK=true in this environment: a shadow run must be primary-only")

        from app.core.database import get_supabase_admin
        storage = job_worker.SupabaseStorage(get_supabase_admin)
        rubric_sha = hashlib.sha256(dump(rubric).encode("utf-8")).hexdigest()
        spent = 0.0
        for i, c in enumerate(todo, 1):
            t0 = time.time()
            try:
                with svc.usage_meter():
                    g = grade_copy_like_the_worker(c, asg, rubric, storage)
            except Exception as e:          # one copy failing must not stop the run
                print(f"  [{i}/{len(todo)}] roll {c['roll']}: FAILED {type(e).__name__}: {str(e)[:120]}")
                continue
            if g["model"] not in (PRIMARY, "mock"):
                print(f"  [{i}/{len(todo)}] roll {c['roll']}: answered by {g['model']}, not stored")
                continue
            raw = getattr(conn, "_conn", conn)
            cur = raw.cursor()
            cur.execute("""insert into public.check_shadow_results
                             (run_label, submission_id, sheet_id, assignment_id, answer_key_version, rubric_sha256, model,
                              ai_result, ai_marks, max_marks, needs_review_reasons, usage)
                           values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb)
                           on conflict (run_label, submission_id) do nothing""",
                        (args.label, c["submission_id"], c["sheet"]["id"], args.assignment_id, c["answer_key_version"],
                         rubric_sha, g["model"], json.dumps(g["ai_result"], ensure_ascii=False), g["ai_marks"],
                         g["max_marks"], list(g["reasons"]), json.dumps(g["usage"])))
            cur.close()
            spent += float(g["usage"].get("est_cost_inr") or 0)
            print(f"  [{i}/{len(todo)}] roll {c['roll']}: {g['ai_marks']:g}/{g['max_marks']:g} "
                  f"(prod {c['sheet'].get('ai_marks')}) {time.time() - t0:.0f}s ₹{g['usage'].get('est_cost_inr', 0):.2f}")
        print(f"done: est ₹{spent:.2f} spent · next: export_ai_marks.py --run-label {args.label}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
