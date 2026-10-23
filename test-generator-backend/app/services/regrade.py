"""
Re-grade after the teacher changed an assignment's answer key (spec 8A): the selected attempts get a new check
job pinned to the current key version. Their old marks go to the audit table (smart_assignment_checks,
result_json.type = 'regrade') first; a published attempt is unpublished (audited by unpublish_results) so the
student never sees marks from the old key next to the new check.
"""

from typing import Callable

# Attempts that already have a result (or gave up) can be re-graded. Queued / checking ones will use the new key
# anyway (the key trigger re-pins them); rejected ones have a file problem, not a key problem.
REGRADABLE = ("checked", "needs_review", "published", "failed")
UNPUBLISH_REASON = "Re-grade: answer key changed"


def regrade(store, assignment_id: str, submission_ids: list, teacher_id: str,
            unpublish: Callable[[list, str], dict]) -> dict:
    key = store.row("assignment_ai_keys", assignment_id, key="assignment_id")
    if not key:
        return {"state": "key_missing", "message": "Attach an answer key to this assignment first."}
    skipped, picked = [], []
    for sid in dict.fromkeys(str(s) for s in submission_ids):
        sub = store.row("submissions", sid)
        if not sub or str(sub["assignment_id"]) != str(assignment_id):
            skipped.append({"id": sid, "reason": "not_found"})
        elif sub.get("status") not in REGRADABLE:
            skipped.append({"id": sid, "reason": "in_progress" if sub.get("status") in
                            ("uploaded", "queued", "checking", "waiting_quota", "waiting_key") else "not_checked"})
        elif sub.get("answer_key_version") == key.get("version") and sub.get("status") != "failed":
            skipped.append({"id": sid, "reason": "key_unchanged"})
        else:
            picked.append((sub, store.row("check_sheets", sid, key="submission_id")))

    published = [str(s["id"]) for s, _ in picked if s.get("status") == "published"]
    if published:
        r = unpublish(published, UNPUBLISH_REASON) or {}
        done = {str(x) for x in r.get("unpublished") or []}
        for s, _ in list(picked):
            if str(s["id"]) in published and str(s["id"]) not in done:
                picked.remove((s, _))
                skipped.append({"id": str(s["id"]), "reason": "unpublish_failed"})

    queued = []
    for sub, sheet in picked:
        sheet = sheet or {}
        previous = sub.get("grade") if sub.get("grade") is not None else sheet.get("ai_marks")
        max_marks = float(sheet.get("max_marks") or 0)
        store.insert_audit({
            "teacher_id": teacher_id, "student_name": sheet.get("student_name"), "student_class": None,
            "student_roll": sheet.get("roll_no"), "subject": None, "class_grade": None, "qp_filename": None,
            "ss_filename": None, "is_valid": True, "total_obtained": float(previous or 0), "max_marks": max_marks,
            "percentage": round(float(previous or 0) / max_marks * 100, 1) if max_marks else 0,
            "needs_review_count": 0, "strictness": key.get("strictness") or "medium",
            "overall_remarks": f"Re-grade requested (answer key v{sub.get('answer_key_version')} → v{key.get('version')})",
            "result_json": {"type": "regrade", "by": teacher_id, "previous_status": sub.get("status"),
                            "previous_marks": previous, "previous_ai_marks": sheet.get("ai_marks"),
                            "previous_final_marks": sheet.get("final_marks"),
                            "previous_answer_key_version": sub.get("answer_key_version"),
                            "new_answer_key_version": key.get("version"),
                            "previous_model_version": sheet.get("model_version"),
                            "previous_per_question": sheet.get("per_question"),
                            "previous_review_reasons": sheet.get("needs_review_reasons")},
            "assignment_id": assignment_id, "submission_id": str(sub["id"]), "check_sheet_id": sheet.get("id"),
        })
        job_id = store.enqueue_submission(str(sub["id"]))
        queued.append({"id": str(sub["id"]), "job_id": job_id})
    return {"state": "ok", "queued": queued, "skipped": skipped, "answer_key_version": key.get("version")}
