"""
Checker job worker (checker-v4 G2) for copies students upload through an assignment link.
Contract: docs/upload-flow-api.md §4 · queue + triggers: 18_upload_flow.sql (applied on live).

One kick (POST /internal/jobs/run, from pg_net on insert or the pg_cron sweep every minute) = one run():
claim a job, do it, claim the next while more than MIN_LEFT_TO_CLAIM_S of the WORK_BUDGET_S are left (Vercel
Hobby: 300 s per invocation; nothing runs in the background after that).

Job kinds
  check (submission_id set)    one student's attempt. Limits → files → rubric → grade (v4 grade_copy) → save.
                               More than CHUNK_MAX_PAGES pages → the pages are split into chunk jobs and this job
                               waits (queued, its retry given back) until they are all done, then merges them.
  check (checkpoint.parent)    one chunk of a long copy (submission_id null, so the state triggers ignore it).
  retention                    hourly: unattached uploads → deleted through the Storage API.

Rules
  * The worker trusts only submission_id: the files are what submissions.files / check_sheets.pages name, read
    from the private bucket with the service key, and re-checked from their bytes (upload_validation).
  * Marks are stored once. A job whose checkpoint says graded (or a chunk with a stored result) is finished
    without calling Gemini again.
  * Errors go to a4_finish_check_job with an error code (retry with backoff / waiting_quota / rejected / failed)
    and a teacher-safe Hinglish message. Logs carry ids and counts, student names only masked.
"""

import logging
import math
import os
import tempfile
import time
from collections import Counter
from typing import Callable, Optional

from app.core.log_mask import PIIMaskFilter, mask_name
from app.routers.copy_checker_router import _compact
from app.routers.smart_assignment_router import DAILY_LIMIT, _storage_path, day_start_ist
from app.services import smart_assignment_service as svc
from app.services.answer_checker import copy_check, portal_result, preprocess, upload_validation

logger = logging.getLogger(__name__)
logger.addFilter(PIIMaskFilter())

WORK_BUDGET_S = float(os.getenv("CHECKER_WORK_BUDGET_S", "240"))
MIN_LEFT_TO_CLAIM_S = float(os.getenv("CHECKER_MIN_LEFT_TO_CLAIM_S", "120"))
MAX_RUNNING_JOBS = int(os.getenv("MAX_RUNNING_JOBS", "5"))
CHUNK_MAX_PAGES = 12                  # > 12 pages → chunks of 10-12 pages
PARK_RECHECK_S = 600                  # a waiting long copy looks at its chunks at least this often
RUBRIC_BUSY_RETRY_S = 20
# Daily Gemini spend cap for checking (estimate from token counts × price; the Google project cap stays the
# last line of defence). Over the cap the worker claims nothing until the next IST day.
DAILY_BUDGET_INR = float(os.getenv("CHECKER_DAILY_BUDGET_INR", "150"))
PRICE_IN_PER_M_INR = float(os.getenv("CHECKER_PRICE_IN_PER_M_INR", "26"))
PRICE_OUT_PER_M_INR = float(os.getenv("CHECKER_PRICE_OUT_PER_M_INR", "213"))

SUBMISSIONS_BUCKET = "submissions"
SHEETS_BUCKET = "answer-sheets"
_RETRY_CODES = {"GEMINI_429", "GEMINI_5XX", "TIMEOUT", "UNKNOWN"}
_MIME = {"pdf": "application/pdf", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp", "heic": "image/heic"}

MSG = {
    "QUOTA": "Teacher ki aaj ki AI checking limit ({limit} copies) poori ho gayi. Copy kal apne aap check hogi.",
    "KEY_MISSING": "Answer key abhi attach nahi hai. Key milte hi copy apne aap check hogi.",
    "KEY_UNREADABLE": "Question paper / answer key padhi nahi ja saki. Teacher answer key dobara attach karein; "
                      "copy uske baad apne aap check hogi.",
    "GEMINI_429": "AI abhi busy hai. Copy thodi der mein apne aap dobara check hogi.",
    "GEMINI_5XX": "AI service abhi down hai. Copy thodi der mein apne aap dobara check hogi.",
    "TIMEOUT": "Checking mein zyada time laga. Copy apne aap dobara check hogi.",
    "UNKNOWN": "Copy check karte waqt error aaya. Apne aap dobara try ho raha hai.",
    "FINAL": "{n} baar try karne ke baad bhi copy check nahi ho payi ({why}). Teacher 'Re-check' se dobara chala sakte hain.",
    "CALL_CAP": "Is copy pe AI calls ki limit poori ho gayi. Teacher ise khud check karein.",
    "SCHEMA": "Is copy ka data adhoora hai, check nahi ho sakti.",
}
_WHY = {"GEMINI_429": "AI busy", "GEMINI_5XX": "AI down", "TIMEOUT": "time limit", "UNKNOWN": "error"}


class JobError(Exception):
    """final=True: no retry left (a chunk of the copy already used its own retries)."""

    def __init__(self, code: str, message: str, final: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.final = final


# ── usage / cost ──────────────────────────────────────────────────────

_COUNTERS = ("gemini_calls", "input_tokens", "output_tokens", "thinking_tokens")


def est_cost_inr(u: dict) -> float:
    out = (u.get("output_tokens") or 0) + (u.get("thinking_tokens") or 0)    # thinking is billed as output
    return round(((u.get("input_tokens") or 0) * PRICE_IN_PER_M_INR + out * PRICE_OUT_PER_M_INR) / 1e6, 4)


def add_usage(total: dict, part: dict) -> dict:
    out = dict(total or {})
    for k in _COUNTERS:
        out[k] = int(out.get(k) or 0) + int((part or {}).get(k) or 0)
    if (part or {}).get("model"):
        out["model"] = part["model"]
    out["est_cost_inr"] = est_cost_inr(out)
    return out


# ── storage (Supabase Storage API with the service key) ───────────────

class SupabaseStorage:
    def __init__(self, client_factory: Callable):
        self._client = client_factory

    def _bucket(self, bucket: str):
        return self._client().storage.from_(bucket)

    def download(self, bucket: str, path: str) -> Optional[bytes]:
        """None when the object doesn't exist; other errors are raised (→ retry)."""
        from app.routers.smart_assignment_router import with_retry
        try:
            return with_retry(lambda: self._bucket(bucket).download(path))
        except Exception as e:
            text = str(e).lower()
            if "not found" in text or "404" in text or "not_found" in text:
                return None
            raise

    def upload(self, bucket: str, path: str, data: bytes, content_type: str) -> None:
        from app.routers.smart_assignment_router import with_retry
        with_retry(lambda: self._bucket(bucket).upload(path, data, {"content-type": content_type, "upsert": "true"}))

    def remove(self, bucket: str, paths: list) -> None:
        from app.routers.smart_assignment_router import with_retry
        with_retry(lambda: self._bucket(bucket).remove(paths))


# ── worker ────────────────────────────────────────────────────────────

class JobWorker:
    def __init__(self, store, storage, *, budget_s: float = WORK_BUDGET_S, daily_limit: int = DAILY_LIMIT,
                 daily_budget_inr: float = DAILY_BUDGET_INR, max_running: int = MAX_RUNNING_JOBS,
                 clock: Callable = time.monotonic):
        self.store = store
        self.storage = storage
        self.budget_s = budget_s
        self.daily_limit = daily_limit
        self.daily_budget_inr = daily_budget_inr
        self.max_running = max_running
        self.clock = clock
        self._run_usage: dict = {}

    # One invocation
    def run(self) -> dict:
        start = self.clock()
        stats = {"jobs": 0, "paused_budget": False}
        while self.budget_s - (self.clock() - start) > MIN_LEFT_TO_CLAIM_S:
            if self.over_budget():
                stats["paused_budget"] = True
                break
            job = self.store.claim(self.max_running)
            if not job:
                break
            stats["jobs"] += 1
            self.handle(job)
        return stats

    def over_budget(self) -> bool:
        spent = self.store.spend_since(day_start_ist())
        if spent >= self.daily_budget_inr:
            logger.error(f"[jobs] ALERT daily Gemini budget reached: est ₹{spent:.2f} of ₹{self.daily_budget_inr:.2f}. "
                         f"New checks are paused until IST midnight (raise CHECKER_DAILY_BUDGET_INR to resume now).")
            return True
        return False

    def handle(self, job: dict) -> None:
        self._run_usage = {}
        t0 = time.time()
        cp = job.get("checkpoint") or {}
        try:
            if job["kind"] == "retention":
                self._retention(job)
            elif job["kind"] == "check" and job.get("submission_id"):
                self._copy(job)
            elif job["kind"] == "check" and cp.get("parent"):
                self._chunk(job)
            else:
                raise JobError("SCHEMA", MSG["SCHEMA"])
        except Exception as e:
            self._fail(job, e)
        finally:
            logger.info(f"[jobs] {job['kind']} {str(job['id'])[:8]} run {job.get('retries')} "
                        f"in {time.time() - t0:.1f}s")

    # ── errors ────────────────────────────────────────────────────────
    def _classify(self, e: Exception) -> tuple:
        if isinstance(e, JobError):
            return e.code, e.message, None
        if isinstance(e, preprocess.BadFile):
            return "BAD_FILE", upload_validation.hinglish_bad_file(e), None
        if isinstance(e, svc.GeminiBusy):
            code = "GEMINI_429" if e.code == 429 else "GEMINI_5XX"
            return code, MSG[code], e.retry_after
        if isinstance(e, svc.CallCapExceeded):
            return "CALL_CAP", MSG["CALL_CAP"], None
        if isinstance(e, TimeoutError) or "timeout" in type(e).__name__.lower():
            return "TIMEOUT", MSG["TIMEOUT"], None
        if isinstance(e, svc.genai_errors.APIError) and (e.code or 0) >= 500:
            return "GEMINI_5XX", MSG["GEMINI_5XX"], None
        logger.warning(f"[jobs] job failed with {type(e).__name__}", exc_info=True)
        return "UNKNOWN", MSG["UNKNOWN"], None

    def _fail(self, job: dict, e: Exception) -> None:
        code, message, retry_after = self._classify(e)
        if getattr(e, "final", False):
            final = True
        else:
            final = code not in _RETRY_CODES or int(job.get("retries") or 0) >= int(job.get("max_retries") or 3)
            if code in _RETRY_CODES and final:
                message = MSG["FINAL"].format(n=job.get("retries"), why=_WHY[code])
        usage = add_usage(job.get("usage") or {}, self._run_usage) if self._run_usage else None
        try:
            row = self.store.finish(job["id"], False, code, message, usage)
        except Exception:
            logger.error(f"[jobs] could not finish job {str(job['id'])[:8]} ({code}); the sweep will retry it",
                         exc_info=True)
            return
        logger.info(f"[jobs] job {str(job['id'])[:8]} → {row.get('state')} ({code})")
        if row.get("state") == "queued" and retry_after:
            self.store.not_before(job["id"], retry_after)
        if row.get("state") == "failed" and job.get("sheet_id") and job.get("submission_id"):
            self.store.mark_sheet_failed(job["sheet_id"], message)
        if (job.get("checkpoint") or {}).get("parent") and row.get("state") == "failed":
            self._wake_parent(job["checkpoint"]["parent"])

    # ── one attempt ──────────────────────────────────────────────────
    def _copy(self, job: dict) -> None:
        cp = job.get("checkpoint") or {}
        if cp.get("graded"):           # marks already stored by an earlier run: never grade twice
            self.store.finish(job["id"], True, usage=job.get("usage"))
            return
        sheet = self.store.row("check_sheets", job["sheet_id"])
        sub = self.store.row("submissions", job["submission_id"])
        session = self.store.row("check_sessions", job["session_id"])
        if not sheet or not sub or not session or sheet.get("submission_id") != sub["id"]:
            raise JobError("SCHEMA", MSG["SCHEMA"])
        asg = self.store.row("assignments", sub["assignment_id"])
        if cp.get("stage") == "chunks":
            return self._collect(job, cp, sheet, sub, asg, session)

        used = self.store.checks_today(job["teacher_id"], day_start_ist(), job["id"])
        if used >= self.daily_limit:
            self.store.finish(job["id"], False, "QUOTA", MSG["QUOTA"].format(limit=self.daily_limit))
            logger.info(f"[jobs] teacher {str(job['teacher_id'])[:8]} at daily limit ({used}/{self.daily_limit})")
            return
        key = self.store.row("assignment_ai_keys", asg["id"], key="assignment_id")
        if not key:
            self.store.finish(job["id"], False, "KEY_MISSING", MSG["KEY_MISSING"])
            return
        self.store.mark_sheet_checking(sheet["id"])

        v = self._validate(sheet, sub, asg)
        if not v.ok:
            return self._reject(job, sheet, sub, [p.as_dict() for p in v.problems], v.summary())
        flags = self._duplicate_flags(v, sub, asg)

        rubric = self._rubric(job, session, asg, key)
        if rubric is None:
            return
        temps = self._temp_files(v.files)
        try:
            if v.total_pages > CHUNK_MAX_PAGES:
                return self._split(job, sub, key, v, temps, flags)
            t0 = time.time()
            result, ordered, gusage = copy_check.grade_copy(temps, rubric, key.get("strictness") or "medium")
            self._run_usage = add_usage(self._run_usage, gusage)
            result["norm_pages"] = self._save_pages(job, ordered, "p", 0)
            result["usage"] = gusage
            meta = {"pages": gusage.get("pages"), "grade_time_s": round(time.time() - t0, 2), "chunks": 1}
            self._save(job, sheet, sub, session, key, rubric, result, flags, meta)
        except preprocess.BadFile as e:
            msg = upload_validation.hinglish_bad_file(e)
            return self._reject(job, sheet, sub, [{"code": "bad_file", "page_order": 0, "message": msg}], msg)
        finally:
            _cleanup(temps)

    def _validate(self, sheet: dict, sub: dict, asg: dict) -> upload_validation.Validation:
        if sheet.get("source_bucket") != SUBMISSIONS_BUCKET:
            raise JobError("SCHEMA", MSG["SCHEMA"])
        entries = sub.get("files") or []
        folder = f"{asg['institute_id']}/{asg['id']}/{sub['student_id']}/"
        allowed = set(sheet.get("pages") or [])
        blobs = {}
        for e in entries:
            path = str(e.get("storage_path") or "")
            # only this student's own files that the attempt names; never a path from anywhere else
            if path in allowed and path.startswith(folder) and ".." not in path.split("/"):
                blobs[path] = self.storage.download(SUBMISSIONS_BUCKET, path)
        blobs = {k: v for k, v in blobs.items() if v is not None}
        v = upload_validation.validate(entries, blobs, asg.get("allowed_mimes"), asg.get("max_pages"))
        if v.ok:
            for f in v.files:
                if f.exif_removed:     # the original in Storage loses its GPS / camera data too
                    try:
                        self.storage.upload(SUBMISSIONS_BUCKET, f.storage_path, f.data, _MIME[f.kind])
                    except Exception as e:
                        logger.warning(f"[jobs] EXIF-free copy not saved for page {f.page_order}: {type(e).__name__}")
        return v

    def _duplicate_flags(self, v, sub: dict, asg: dict) -> list:
        """sha256 of the real bytes: the same file twice in this copy, or the same file as another student's
        upload. Teacher-only flags (never in what the student sees), no accusation."""
        flags = []
        seen = Counter(f.sha256 for f in v.files)
        for sha, n in seen.items():
            if n > 1:
                pages = ", ".join(str(f.page_order) for f in v.files if f.sha256 == sha)
                flags.append({"code": "DUPLICATE_PAGE", "page": None, "message": f"Pages {pages} are the same file."})
        try:
            others = set(self.store.other_students_with(asg["id"], sub["student_id"], list(seen)))
        except Exception:
            others = set()
        for f in v.files:
            if f.sha256 in others:
                flags.append({"code": "DUPLICATE_OTHER_STUDENT", "page": None,
                              "message": f"Page {f.page_order} is the same file another student uploaded."})
            if f.declared_sha256 and f.declared_sha256 != f.sha256:
                logger.warning(f"[jobs] page {f.page_order}: sha256 from the browser does not match the file")
        return flags

    def _reject(self, job: dict, sheet: dict, sub: dict, problems: list, summary: str) -> None:
        usage = add_usage(job.get("usage") or {}, self._run_usage) if self._run_usage else None
        self.store.finish(job["id"], False, "BAD_FILE", summary, usage)
        self.store.set_reject_reasons(sub["id"], problems)
        self.store.mark_sheet_failed(sheet["id"], summary)
        logger.info(f"[jobs] attempt {str(sub['id'])[:8]} rejected: {[p['code'] for p in problems]}")

    # ── rubric: built once per answer-key version, cached in check_sessions.rubric ──
    def _rubric(self, job: dict, session: dict, asg: dict, key: dict) -> Optional[dict]:
        r = session.get("rubric")
        if session.get("rubric_status") == "ready" and r and r.get("key_version") == key.get("version"):
            return r
        if not self.store.lock_rubric(session["id"]):
            self.store.requeue_unchanged(job["id"], RUBRIC_BUSY_RETRY_S)
            return None
        temps = []
        try:
            key_path = key["answer_key_path"]
            temps.append(self._download_temp("answer-keys", key_path))
            qp = []
            if asg.get("file_url"):
                qp_path = _storage_path(asg["file_url"], "assignments")
                qp = [self._download_temp("assignments", qp_path)]
                temps += qp
            with svc.usage_meter() as meter:
                # no question paper attached: the answer key usually prints the questions too
                rubric = svc.build_rubric(qp or [temps[0]], [temps[0]] if qp else None)
            self._run_usage = add_usage(self._run_usage, meter.as_dict())
        except svc.GeminiBusy:
            self.store.rubric_failed(session["id"], MSG["GEMINI_429"])
            raise
        except Exception as e:
            logger.warning(f"[jobs] rubric for session {str(session['id'])[:8]} failed: {type(e).__name__}")
            self.store.rubric_failed(session["id"], MSG["KEY_UNREADABLE"])
            raise JobError("KEY_MISSING", MSG["KEY_UNREADABLE"])
        finally:
            _cleanup(temps)
        rubric = {k: v for k, v in rubric.items() if not k.startswith("_")}
        rubric["key_version"] = key.get("version")
        rubric["strictness"] = key.get("strictness") or "medium"
        self.store.save_rubric(session["id"], rubric)
        return rubric

    def _download_temp(self, bucket: str, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        data = self.storage.download(bucket, path)
        if data is None:
            raise JobError("KEY_MISSING", MSG["KEY_UNREADABLE"])
        return _write_temp(data, ext, "rub_")

    # ── save ─────────────────────────────────────────────────────────
    def _save(self, job, sheet, sub, session, key, rubric, result, flags, meta) -> None:
        result["sheet_flags"] = (result.get("sheet_flags") or []) + flags
        reasons, confidence, by_q = portal_result.review(result, rubric)
        model = self._run_usage.get("model") or (job.get("usage") or {}).get("model") or svc.CHECKER_MODEL
        compact = _compact(result)
        compact["review"] = {"reasons": reasons, "by_question": by_q}
        compact["meta"] = {**meta, "answer_key_version": key.get("version"), "model_version": model,
                           "strictness": key.get("strictness") or "medium"}
        total = float(result.get("total_marks_awarded") or 0)
        max_marks = float(result.get("max_marks") or 0)
        fields = {"ai_result": compact, "ai_marks": total, "max_marks": max_marks, "needs_review_reasons": reasons,
                  "confidence": confidence, "model_version": model}
        audit = {
            "teacher_id": job["teacher_id"], "student_name": sheet.get("student_name"), "student_class":
            session.get("class_grade"), "student_roll": sheet.get("roll_no"),
            "subject": (session.get("subject") or session.get("title") or "")[:100] or None,
            "class_grade": session.get("class_grade"), "qp_filename": (session.get("title") or "")[:255],
            "ss_filename": f"{meta.get('pages')} page(s)", "is_valid": True, "total_obtained": total,
            "max_marks": max_marks, "percentage": round(total / max_marks * 100, 1) if max_marks else 0,
            "needs_review_count": int(result.get("needs_review_count") or 0),
            "strictness": key.get("strictness") or "medium", "overall_remarks": (result.get("overall_remarks") or "")[:2000],
            "result_json": compact, "assignment_id": sub["assignment_id"], "submission_id": sub["id"],
            "check_sheet_id": sheet["id"],
        }
        usage = add_usage(job.get("usage") or {}, self._run_usage)
        self.store.save_result(job["id"], sheet["id"], fields, audit, {**(job.get("checkpoint") or {}), "graded": True})
        self.store.finish(job["id"], True, usage=usage)
        logger.info(f"[jobs] copy {str(sheet['id'])[:8]} ({mask_name(sheet.get('student_name'))}) checked: "
                    f"{total:g}/{max_marks:g}, review={reasons or 'none'}, model={model}, "
                    f"est ₹{usage.get('est_cost_inr', 0):.2f}")

    def _save_pages(self, job: dict, ordered: list, prefix: str, offset: int) -> list:
        """Final (upright, ordered) pages → answer-sheets/<teacher>/<session>/<sheet>/norm/ — the red pen draws on
        exactly these images (same layout as the Check Copies router)."""
        folder = f"{job['teacher_id']}/{job['session_id']}/{job['sheet_id']}/norm"
        out = []
        for i, p in enumerate(ordered):
            path = f"{folder}/{prefix}{i}.jpg"
            self.storage.upload(SHEETS_BUCKET, path, p.jpeg, "image/jpeg")
            out.append({"index": offset + i, "source_index": offset + p.source_index, "student_page_no": p.student_page_no,
                        "rotation": p.rotation, "blank": p.blank, "path": path, "w": p.w, "h": p.h})
        return out

    # ── long copies: chunks ──────────────────────────────────────────
    def _split(self, job, sub, key, v, temps, flags) -> None:
        norm = preprocess.normalize_sources(temps, max_pages=upload_validation.HARD_MAX_PAGES)
        if all(p.blank for p in norm):
            raise preprocess.BadFile("Every page of this copy is blank.")
        n = math.ceil(len(norm) / CHUNK_MAX_PAGES)
        bounds = [round(i * len(norm) / n) for i in range(n + 1)]       # balanced: 20 → 10 + 10, 30 → 10 × 3
        folder = f"{job['teacher_id']}/{job['session_id']}/{job['sheet_id']}/src"
        paths = []
        for i, p in enumerate(norm):
            path = f"{folder}/p{i}.jpg"
            self.storage.upload(SHEETS_BUCKET, path, p.jpeg, "image/jpeg")
            paths.append(path)
        chunks = []
        for k in range(n):
            s, e = bounds[k], bounds[k + 1]
            cp = {"parent": str(job["id"]), "chunk": k, "of": n, "start": s, "pages": paths[s:e],
                  "key_version": key.get("version")}
            state = "queued"
            if all(p.blank for p in norm[s:e]):          # nothing to grade: done without a Gemini call
                cp["result"] = {"graded_questions": [], "sheet_flags": [
                    {"code": "BLANK_PAGE", "page": i, "message": f"Page {s + i + 1} is blank."} for i in range(e - s)]}
                cp["norm_pages"] = [{"index": s + i, "source_index": s + i, "student_page_no": None, "rotation": 0,
                                     "blank": True, "path": paths[s + i], "w": norm[s + i].w, "h": norm[s + i].h}
                                    for i in range(e - s)]
                state = "done"
            chunks.append({"state": state, "checkpoint": cp})
        self.store.add_chunks(job, chunks)
        self.store.requeue_unchanged(job["id"], PARK_RECHECK_S,
                                     {"stage": "chunks", "of": n, "pages": len(norm), "flags": flags},
                                     usage=add_usage(job.get("usage") or {}, self._run_usage))
        self.store.show_checking(sub["id"])
        logger.info(f"[jobs] copy {str(job['sheet_id'])[:8]}: {len(norm)} pages → {n} chunks")

    def _chunk(self, job: dict) -> None:
        cp = job["checkpoint"]
        if cp.get("result") is not None:         # stored by an earlier run
            self.store.finish(job["id"], True, usage=job.get("usage"))
            self._wake_parent(cp["parent"])
            return
        session = self.store.row("check_sessions", job["session_id"])
        rubric = (session or {}).get("rubric")
        if not rubric:
            raise JobError("KEY_MISSING", MSG["KEY_UNREADABLE"])
        temps = []
        try:
            for path in cp["pages"]:
                data = self.storage.download(SHEETS_BUCKET, path)
                if data is None:
                    raise JobError("UNKNOWN", MSG["UNKNOWN"])
                temps.append(_write_temp(data, ".jpg", "chunk_"))
            t0 = time.time()
            result, ordered, gusage = copy_check.grade_copy(temps, rubric, rubric.get("strictness") or "medium")
        finally:
            _cleanup(temps)
        self._run_usage = add_usage(self._run_usage, gusage)
        keep = ("graded_questions", "max_marks", "overall_remarks", "readability_score", "sheet_flags")
        cp = {**cp, "result": {k: result.get(k) for k in keep},
              "norm_pages": self._save_pages(job, ordered, f"c{cp['chunk']}_p", cp["start"]),
              "grade_time_s": round(time.time() - t0, 2)}
        self.store.set_checkpoint(job["id"], cp)
        self.store.finish(job["id"], True, usage=add_usage(job.get("usage") or {}, self._run_usage))
        self._wake_parent(cp["parent"])

    def _wake_parent(self, parent_id: str) -> None:
        chunks = self.store.chunks_of(parent_id)
        if chunks and all(c["state"] in ("done", "failed", "cancelled") for c in chunks):
            self.store.wake(parent_id)

    def _collect(self, job, cp, sheet, sub, asg, session) -> None:
        chunks = self.store.chunks_of(job["id"])
        failed = [c for c in chunks if c["state"] in ("failed", "cancelled")]
        if failed:          # the chunk already used its own retries: the copy fails with its reason
            c = failed[0]
            self.store.use_up_retries(job["id"])
            code = c.get("last_error_code") or "UNKNOWN"
            raise JobError(code, c.get("error") or MSG.get(code) or MSG["UNKNOWN"], final=True)
        if len(chunks) < int(cp.get("of") or 0) or any(c["state"] != "done" for c in chunks):
            self.store.requeue_unchanged(job["id"], PARK_RECHECK_S)
            self.store.show_checking(sub["id"])
            return
        key = self.store.row("assignment_ai_keys", asg["id"], key="assignment_id") or {}
        rubric = session.get("rubric") or {}
        parts = [{"result": c["checkpoint"]["result"], "pages": len(c["checkpoint"]["pages"])} for c in chunks]
        result = portal_result.merge_chunks(parts, rubric)
        result["norm_pages"] = [p for c in chunks for p in c["checkpoint"].get("norm_pages") or []]
        chunk_usage = {}
        for c in chunks:
            chunk_usage = add_usage(chunk_usage, c.get("usage") or {})
        result["usage"] = chunk_usage          # reporting only; each chunk job's own row holds its cost
        meta = {"pages": cp.get("pages"), "chunks": len(chunks),
                "grade_time_s": round(sum(float(c["checkpoint"].get("grade_time_s") or 0) for c in chunks), 2)}
        self._run_usage = add_usage(self._run_usage, {"model": chunk_usage.get("model")})
        self._save(job, sheet, sub, session, key, rubric, result, cp.get("flags") or [], meta)

    # ── retention ────────────────────────────────────────────────────
    def _retention(self, job: dict) -> None:
        paths = self.store.orphan_files(500)
        deleted = 0
        for i in range(0, len(paths), 100):
            batch = paths[i:i + 100]
            self.storage.remove(SUBMISSIONS_BUCKET, batch)
            deleted += len(batch)
        self.store.finish(job["id"], True, usage={"deleted": deleted})
        logger.info(f"[jobs] retention: {deleted} unattached upload(s) deleted")

    def _temp_files(self, files: list) -> list:
        return [_write_temp(f.data, "." + f.ext, f"p{f.page_order}_") for f in files]


def _write_temp(data: bytes, ext: str, prefix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix=prefix)
    tmp.write(data)
    tmp.close()
    return tmp.name


def _cleanup(paths: list) -> None:
    for p in paths or []:
        try:
            os.unlink(p)
        except OSError:
            pass


def default_worker() -> JobWorker:
    from app.core.database import get_supabase_admin
    from app.core.db_pool import get_db_connection
    from app.services.job_store import JobStore
    return JobWorker(JobStore(get_db_connection), SupabaseStorage(get_supabase_admin))
