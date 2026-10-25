# app/services/contest_service.py
# ──────────────────────────────────────────────────────────
# Business logic for Contest CRUD + Submission + Scoring
# Uses Supabase Python client (not asyncpg)
#
# SCORING (final):
#   Student sends bare labels ("B", "C") while correct_answer is stored as
#   "C) Silver" (label + text). The matcher parses BOTH sides into (label, text)
#   and matches on either — so it works regardless of which format each side uses.
#   It does NOT depend on the options array at all.
#
#   HARDENED (v2):
#   - re.DOTALL so multiline correct_answer text ("B) Because...\n...") still
#     parses into (label, text) instead of falling through to plain text.
#   - All whitespace (newlines, tabs, double spaces) collapsed to single
#     spaces before text comparison, on BOTH sides.
# ──────────────────────────────────────────────────────────

import re
import string
import random
import json
import logging
import time
import uuid

from typing import Optional, List
from datetime import datetime, timedelta, timezone

from app.core.database import get_supabase_admin

from app.schemas.contest import (
    CreateContestRequest,
    CreateContestResponse,
    ContestInfoResponse,
    ContestDataResponse,
    ContestQuestionOut,
    ContestQuestionWithAnswer,
    SubmitContestRequest,
    SubmitContestResponse,
    AttemptSummary,
    ContestLeaderboardResponse,
    StartAttemptRequest,
    ProgressRequest,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════

def _generate_short_code(length: int = 8) -> str:
    """Generate a random alphanumeric short code for contest URL"""
    chars = string.ascii_lowercase + string.digits
    return ''.join(random.choices(chars, k=length))


class ContestNotStarted(Exception):
    """now < contests.scheduled_at: /info and /start answer 403 CONTEST_NOT_STARTED with the start time."""

    def __init__(self, scheduled_at: datetime):
        super().__init__("This contest has not started yet")
        self.scheduled_at = scheduled_at


def _check_started(row: dict) -> None:
    """Raise ContestNotStarted while the contest's scheduled_at is in the future (NULL = open now)."""
    starts = _ts(row.get("scheduled_at"))      # aware UTC (a naive DB text is UTC: timestamptz, UTC session)
    if starts and _now() < starts:
        raise ContestNotStarted(starts)


# ── scoring helpers ────────────────────────────────────────

def _canon(s) -> str:
    """
    Canonical text for comparison:
    - lowercase
    - curly quotes/dashes -> straight
    - ALL punctuation stripped (trailing periods, commas, etc.)
    - whitespace (newlines/tabs/doubles) collapsed to single spaces
    """
    if s is None:
        return ""
    s = str(s)
    s = (s.replace("\u2019", "'").replace("\u2018", "'")
          .replace("\u201c", '"').replace("\u201d", '"')
          .replace("\u2013", "-").replace("\u2014", "-"))
    s = re.sub(r'[^a-z0-9\s]', ' ', s.lower())
    return " ".join(s.split())


def _norm(s) -> str:
    """Lowercase + collapse all whitespace to single spaces (no punctuation strip)."""
    if s is None:
        return ""
    return " ".join(str(s).split()).lower()


def _parse_choice(value):
    """
    Parse an answer value into (label, canonical_text). Both lowercased.
    Handles:
      "C"                    -> ("c", "")
      "C) Silver"            -> ("c", "silver")
      "C. Silver" / "(C) Silver" / "C - Silver"
      "B) multi\nline text"  -> ("b", "multi line text")   (DOTALL)
      "Silver."              -> (None, "silver")            (punctuation-insensitive)
    """
    if value is None:
        return (None, "")
    s = str(value).strip()
    if s == "":
        return (None, "")
    if len(s) == 1 and s.isalpha():
        return (s.lower(), "")
    m = re.match(r'^[\(\[]?\s*([A-Za-z])\s*[\)\].:\-]\s*(.+)$', s, re.DOTALL)
    if m:
        return (m.group(1).lower(), _canon(m.group(2)))
    return (None, _canon(s))


def _resolve_label(label, text, raw, options):
    """Resolve an answer to its option label using the question's options array."""
    if label:
        return label
    c = text or _canon(raw)
    if not c or not options:
        return None
    for o in options:
        if isinstance(o, dict):
            ol = str(o.get("label", "")).strip().lower() or None
            ot = _canon(o.get("text", ""))
        else:
            ol, ot = None, _canon(o)
        if c == ot:
            return ol
    return None


def _answers_match(selected, correct, options=None) -> bool:
    """
    True if the student's `selected` matches the stored `correct` answer.
    Match order:
      1. label vs label            ("B" vs "B) 60 C")
      2. canonical text vs text    ("60 C" vs "B) 60 C." — punctuation-safe)
      3. canonical exact fallback
      4. options-resolved labels   (full text on one side, bare label on other)
    """
    sl, st = _parse_choice(selected)
    cl, ct = _parse_choice(correct)
    if sl and cl and sl == cl:
        return True
    if st and ct and st == ct:
        return True
    if _canon(selected) and _canon(selected) == _canon(correct):
        return True
    if options:
        rl = _resolve_label(sl, st, selected, options)
        rcl = _resolve_label(cl, ct, correct, options)
        if rl and rcl and rl == rcl:
            return True
    return False


def _calculate_score(
    answers: List[dict],
    questions: List[dict],
) -> tuple:
    """
    Calculate score from student answers vs correct answers.
    total_marks counts only questions that have a correct_answer (auto-gradable).
    Returns (score, total_marks).
    """
    q_map = {}
    for q in questions:
        q_map[str(q["id"])] = {
            "correct": q.get("correct_answer"),
            "marks": q.get("marks", 1) or 1,
            "options": q.get("options"),
        }

    total = sum(info["marks"] for info in q_map.values() if _norm(info["correct"]))

    score = 0
    for ans in answers:
        qid = ans.get("questionId") or ans.get("question_id")
        selected = (
            ans.get("selected")
            or ans.get("selectedOption")
            or ans.get("textAnswer")
        )
        if not qid or selected is None or _norm(selected) == "":
            continue

        info = q_map.get(str(qid))
        if not info or not _norm(info["correct"]):
            continue

        if _answers_match(selected, info["correct"], info.get("options")):
            score += info["marks"]

    return score, total


# ═══════════════════════════════════════════════════════════
# CREATE CONTEST
# ═══════════════════════════════════════════════════════════

def create_contest(
    request: CreateContestRequest,
    teacher_id: Optional[str] = None,
) -> CreateContestResponse:
    """Create a new contest with questions and return share link"""

    db = get_supabase_admin()

    short_code = _generate_short_code()

    for _ in range(5):
        existing = db.table("contests").select("id").eq("short_code", short_code).execute()
        if not existing.data:
            break
        short_code = _generate_short_code()

    total_marks = sum(q.marks for q in request.questions)

    contest_data = {
        "teacher_id": teacher_id,
        "title": request.title,
        "subject": request.subject,
        "class_grade": request.class_grade,
        "board": request.board,
        "logo_base64": request.logo_base64,
        "short_code": short_code,
        "duration_minutes": request.duration_minutes,
        "answer_mode": request.answer_mode,
        "show_explanation": request.show_explanation,
        "enable_camera": request.enable_camera,
        "enable_tab_detection": request.enable_tab_detection,
        "allow_back_navigation": request.allow_back_navigation,
        "max_warnings": request.max_warnings,
        "max_attempts": request.max_attempts,
        "scheduled_at": request.scheduled_at.isoformat() if request.scheduled_at else None,
        "expires_at": request.expires_at.isoformat() if request.expires_at else None,
        "total_questions": len(request.questions),
        "total_marks": total_marks,
        "status": "active",
    }

    result = db.table("contests").insert(contest_data).execute()

    if not result.data:
        raise Exception("Failed to insert contest")

    contest_id = result.data[0]["id"]

    questions_to_insert = []
    for idx, q in enumerate(request.questions, start=1):
        questions_to_insert.append({
            "contest_id": contest_id,
            "question_number": idx,
            "question_text": q.question_text,
            "question_type": q.question_type,
            "options": q.options if q.options else None,
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
            "marks": q.marks,
            "difficulty": q.difficulty,
            "chapter": q.chapter,
        })

    if questions_to_insert:
        db.table("contest_questions").insert(questions_to_insert).execute()

    share_link = f"https://a4ai.in/contest/{short_code}"

    logger.info(f"Contest created: {contest_id} ({short_code}) with {len(request.questions)} questions")

    return CreateContestResponse(
        contest_id=contest_id,
        short_code=short_code,
        share_link=share_link,
        total_questions=len(request.questions),
        total_marks=total_marks,
    )


# ═══════════════════════════════════════════════════════════
# GET CONTEST INFO (public — for landing/preview page)
# ═══════════════════════════════════════════════════════════

def get_contest_info(short_code: str) -> Optional[ContestInfoResponse]:
    """Get public contest info by short code (student landing page)"""

    db = get_supabase_admin()

    result = db.table("contests").select(
        "id, title, subject, class_grade, board, logo_base64, "
        "duration_minutes, total_questions, total_marks, "
        "enable_camera, enable_tab_detection, allow_back_navigation, "
        "max_warnings, status, scheduled_at, expires_at"
    ).eq("short_code", short_code).execute()

    if not result.data:
        return None

    row = result.data[0]
    _check_started(row)

    if _expired(row):
        row["status"] = "ended"

    return ContestInfoResponse(
        contest_id=str(row["id"]),
        title=row["title"],
        subject=row["subject"],
        class_grade=row["class_grade"],
        board=row["board"],
        logo_base64=row.get("logo_base64"),
        duration_minutes=row["duration_minutes"],
        total_questions=row["total_questions"],
        total_marks=row["total_marks"],
        enable_camera=row["enable_camera"],
        enable_tab_detection=row["enable_tab_detection"],
        allow_back_navigation=row["allow_back_navigation"],
        max_warnings=row["max_warnings"],
        status=row["status"],
    )


# ═══════════════════════════════════════════════════════════
# ATTEMPT LIFECYCLE (25 Oct 2026: refresh-proof, server-enforced)
#
#   deadline   = started_at + duration_minutes, capped by contests.expires_at
#   grace      = GRACE_S after the deadline: submit is on time, progress is still saved
#   /start     resumes the caller's unsubmitted attempt (X-Attempt-Id, or the logged-in student's own);
#              409 ALREADY_SUBMITTED when they are done; 410 after expires_at
#   /progress  one RPC (contest_save_progress); the max_warnings-th warning finalizes from the saved answers
#   /submit    idempotent: a submitted attempt returns its stored result; late = after deadline + grace
#   stale      unsubmitted past deadline + grace → finalized from saved progress ('auto_timeout'), lazily when
#              /start, /submit, /progress or /leaderboard touches the contest (no cron)
#   retry      (26) beyond max_attempts, /start grants ONE more attempt (2 in all) when every earlier attempt of the
#              student ended auto_timeout and contests.allow_technical_retry; or once after an attempt the teacher
#              marked (allow-retry). The new row has retry_of; leaderboard shows the latest submitted per chain.
#   release    (26) answer_mode 'teacher_release': no score / answers (results_pending) until results_published_at
# Old clients (no X-Attempt-Id on /start, never call /progress) get exactly the old behaviour plus extra fields.
# ═══════════════════════════════════════════════════════════

GRACE_S = 120
SWEEP_EVERY_S = 30          # per contest per instance: the stale sweep is a query, not on every request
_last_sweep: dict = {}


class ContestExpired(Exception):
    """now > contests.expires_at: /start answers 410 like /info."""


class AlreadySubmitted(Exception):
    """The caller is done (max_attempts reached / this attempt submitted): 409 ALREADY_SUBMITTED."""

    def __init__(self, attempt_id: str, message: str):
        super().__init__(message)
        self.attempt_id = attempt_id


class AttemptForbidden(Exception):
    """A logged-in caller writing to another student's attempt."""


class NotTeacher(Exception):
    """Teacher-only contest route called by someone else (or on a contest created without a login)."""


class AttemptRunning(Exception):
    """allow-retry on an attempt that is still running."""


MAX_WITH_TECHNICAL_RETRY = 2


def _ts(v) -> Optional[datetime]:
    if not v:
        return None
    if isinstance(v, datetime):
        dt = v
    else:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _iso_z(v) -> Optional[str]:
    """Same text as the pydantic responses ("2026-10-25T11:00:00Z"), whatever format Postgres used."""
    dt = _ts(v)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if dt else None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _deadline(attempt: dict, contest: dict) -> datetime:
    deadline = _ts(attempt["started_at"]) + timedelta(minutes=int(contest["duration_minutes"]))
    expires = _ts(contest.get("expires_at"))
    return min(deadline, expires) if expires else deadline


def _expired(contest: dict) -> bool:
    expires = _ts(contest.get("expires_at"))
    return bool(expires and expires < _now())


def _load_questions(db, contest_id: str) -> list:
    rows = db.table("contest_questions").select("*").eq("contest_id", contest_id).order("question_number").execute()
    out = []
    for q in rows.data:
        opts = q.get("options")
        if isinstance(opts, str):
            opts = json.loads(opts)
        out.append({
            "id": str(q["id"]),
            "question_number": q["question_number"],
            "question_text": q["question_text"],
            "question_type": q["question_type"],
            "options": opts,
            "correct_answer": q.get("correct_answer"),
            "explanation": q.get("explanation"),
            "marks": q.get("marks", 1),
            "difficulty": q.get("difficulty"),
            "chapter": q.get("chapter"),
        })
    return out


def _answered(answers) -> int:
    return sum(1 for a in answers or [] if isinstance(a, dict)
               and (a.get("selected") or a.get("selectedOption") or a.get("textAnswer")))


def _results_pending(contest: dict) -> bool:
    return contest.get("answer_mode") == "teacher_release" and not contest.get("results_published_at")


def _result_response(attempt: dict, contest: dict, questions: list) -> SubmitContestResponse:
    """The submit response from what is STORED for a finished attempt (same body every time it is asked).
    teacher_release before results_published_at: no score, percentage or answers (results_pending=true)."""
    pending = _results_pending(contest)
    questions_with_answers = None
    if not pending and contest["answer_mode"] in ("after_test", "teacher_release"):
        questions_with_answers = [
            ContestQuestionWithAnswer(
                id=q["id"],
                question_number=q["question_number"],
                question_text=q["question_text"],
                question_type=q["question_type"],
                options=q["options"],
                correct_answer=q["correct_answer"],
                explanation=q["explanation"] if contest["show_explanation"] else None,
                marks=q["marks"],
                difficulty=q["difficulty"],
                chapter=q.get("chapter"),
            )
            for q in questions
        ]
    percentage = attempt.get("percentage")
    return SubmitContestResponse(
        attempt_id=str(attempt["id"]),
        status=attempt["status"],
        score=None if pending else attempt.get("score"),
        total_marks=attempt.get("total_marks") or 0,
        percentage=None if pending or percentage is None else float(percentage),
        answered_count=_answered(attempt.get("answers")),
        total_questions=len(questions),
        questions_with_answers=questions_with_answers,
        submitted_at=_ts(attempt.get("submitted_at")),
        submitted_via=attempt.get("submitted_via"),
        late=bool(attempt.get("late")),
        time_taken_seconds=attempt.get("time_taken_seconds"),
        retry_of=str(attempt["retry_of"]) if attempt.get("retry_of") else None,
        retry=bool(attempt.get("retry_of")),
        results_pending=pending,
    )


def _finalize(db, attempt: dict, contest: dict, questions: list, answers: list, *, via: str,
              warning_count: int, warning_log: list, client_time: Optional[int] = None,
              extra: Optional[dict] = None) -> SubmitContestResponse:
    """Grade `answers` and store the result ONCE: the UPDATE only matches an in_progress row, so a second submit,
    a parallel auto-finalize or a double click never grades twice — the loser returns what the winner stored."""
    now = _now()
    deadline = _deadline(attempt, contest)
    duration_s = int(contest["duration_minutes"]) * 60
    score, total_marks = _calculate_score(answers, questions)
    percentage = round((score / total_marks * 100), 2) if total_marks > 0 else 0.0
    status = "submitted"
    if via != "manual" or warning_count >= contest["max_warnings"]:
        status = "auto_submitted"
    taken = int((now - _ts(attempt["started_at"])).total_seconds())
    update_data = {
        "status": status,
        "submitted_at": now.isoformat(),
        "score": score,
        "total_marks": total_marks,
        "percentage": percentage,
        "time_taken_seconds": max(0, min(taken, duration_s)),     # server clock, never more than the duration
        "warning_count": warning_count,
        "warning_log": warning_log,
        "answers": answers,
        "submitted_via": via,
        "late": via == "manual" and now > deadline + timedelta(seconds=GRACE_S),
        "client_time_taken_seconds": client_time,
        **(extra or {}),
    }
    won = db.table("contest_attempts").update(update_data).eq("id", attempt["id"]) \
        .eq("status", "in_progress").execute()
    if not won.data:
        stored = db.table("contest_attempts").select("*").eq("id", attempt["id"]).execute().data[0]
        return _result_response(stored, contest, questions)
    logger.info(f"Contest {contest['id']} attempt {attempt['id']} {via}: score={score}/{total_marks} "
                f"({percentage}%) warnings={warning_count} late={update_data['late']}")
    return _result_response({**attempt, **update_data}, contest, questions)


def _saved(attempt: dict) -> tuple:
    """(answers, warning_count, warning_log) from an attempt's saved progress."""
    progress = attempt.get("progress") or {}
    return (progress.get("answers") or [], int(attempt.get("warning_count") or 0),
            attempt.get("warning_log") or [])


def _finalize_from_progress(db, attempt: dict, contest: dict, questions: list, via: str) -> SubmitContestResponse:
    answers, warnings, log = _saved(attempt)
    return _finalize(db, attempt, contest, questions, answers, via=via, warning_count=warnings, warning_log=log)


def _stale(attempt: dict, contest: dict) -> bool:
    return attempt.get("status") == "in_progress" and \
        _now() > _deadline(attempt, contest) + timedelta(seconds=GRACE_S)


def sweep_stale(db, contest: dict, force: bool = False) -> int:
    """Finalize this contest's unsubmitted attempts past deadline + grace ('auto_timeout'). Best effort, never
    raises; at most once per SWEEP_EVERY_S per contest per instance unless forced."""
    key = str(contest["id"])
    mono = time.monotonic()
    if not force and mono - _last_sweep.get(key, -SWEEP_EVERY_S) < SWEEP_EVERY_S:
        return 0
    _last_sweep[key] = mono
    try:
        q = db.table("contest_attempts").select("*").eq("contest_id", key).eq("status", "in_progress")
        if not _expired(contest):
            cutoff = _now() - timedelta(minutes=int(contest["duration_minutes"]), seconds=GRACE_S)
            q = q.lt("started_at", cutoff.isoformat())
        rows = [r for r in (q.execute().data or []) if _stale(r, contest)]
        if not rows:
            return 0
        questions = _load_questions(db, key)
        for row in rows:
            _finalize_from_progress(db, row, contest, questions, "auto_timeout")
        logger.info(f"Contest {key}: {len(rows)} stale attempt(s) finalized")
        return len(rows)
    except Exception as e:
        logger.warning(f"Contest {key}: stale sweep failed: {type(e).__name__}")
        return 0


def _progress_view(attempt: dict) -> dict:
    progress = attempt.get("progress") or {}
    return {"answers": progress.get("answers") or [], "current_index": progress.get("current_index") or 0,
            "warning_count": int(attempt.get("warning_count") or 0), "warning_log": attempt.get("warning_log") or []}


def _contest_data(contest: dict, attempt: dict, questions: list, resumed: bool) -> ContestDataResponse:
    return ContestDataResponse(
        contest_id=str(contest["id"]),
        attempt_id=str(attempt["id"]),
        title=contest["title"],
        subject=contest["subject"],
        class_grade=contest["class_grade"],
        duration_minutes=contest["duration_minutes"],
        enable_camera=contest["enable_camera"],
        enable_tab_detection=contest["enable_tab_detection"],
        allow_back_navigation=contest["allow_back_navigation"],
        max_warnings=contest["max_warnings"],
        answer_mode=contest["answer_mode"],
        show_explanation=contest["show_explanation"],
        questions=[
            ContestQuestionOut(
                id=q["id"],
                question_number=q["question_number"],
                question_text=q["question_text"],
                question_type=q["question_type"],
                options=q["options"],
                marks=q["marks"],
                difficulty=q["difficulty"],
                chapter=q.get("chapter"),
            )
            for q in questions
        ],
        started_at=_ts(attempt.get("started_at")),
        deadline_at=_deadline(attempt, contest) if attempt.get("started_at") else None,
        server_now=_now(),
        resumed=resumed,
        progress=_progress_view(attempt) if resumed else None,
        retry_of=str(attempt["retry_of"]) if attempt.get("retry_of") else None,
    )


def _is_uuid(v) -> bool:
    try:
        uuid.UUID(str(v))
        return True
    except (ValueError, TypeError):
        return False


def clean_name(v) -> Optional[str]:
    """A usable student name: trimmed, not empty, not the placeholder "Student"."""
    name = " ".join(str(v or "").split())[:120]
    return None if not name or name.lower() == "student" else name


def _clean_email(v) -> Optional[str]:
    email = str(v or "").strip()[:200]
    return email if "@" in email else None


def email_name(email) -> Optional[str]:
    """Local part of an email as a last-resort display name ("asha.k@x.in" → "asha.k")."""
    return clean_name(str(email).split("@")[0]) if _clean_email(email) else None


def _profile(db, student_id: str) -> dict:
    """{full_name, email} of a logged-in student: profiles, then the auth user's email. Best effort."""
    out: dict = {}
    try:
        rows = db.table("profiles").select("full_name, email").eq("id", student_id).execute().data or []
        out = dict(rows[0]) if rows else {}
    except Exception as e:
        logger.warning(f"profile lookup failed: {type(e).__name__}")
    if not _clean_email(out.get("email")):
        try:
            user = db.auth.admin.get_user_by_id(student_id).user
            out["email"] = getattr(user, "email", None)
        except Exception:
            pass
    return out


def resolve_student(db, request, student_id: Optional[str]) -> tuple:
    """(name, email) for an attempt: the request body first; else the logged-in student's profile full_name, then
    their email's local part. Never "Student"; None only when nothing is known (anonymous, no name typed)."""
    name, email = clean_name(getattr(request, "student_name", None)), _clean_email(getattr(request, "student_email", None))
    if student_id and (not name or not email):
        prof = _profile(db, student_id)
        email = email or _clean_email(prof.get("email"))
        name = name or clean_name(prof.get("full_name"))
    return name or email_name(email), email


def _latest_of_chain(db, attempt: dict) -> dict:
    """An anonymous browser may send an older attempt id: follow retries down to the newest attempt."""
    for _ in range(5):
        child = db.table("contest_attempts").select("*").eq("retry_of", attempt["id"]).execute().data
        if not child:
            break
        attempt = child[0]
    return attempt


def _chain(db, attempt: dict) -> list:
    """attempt + the attempts it retries (an anonymous student's history)."""
    out = [attempt]
    while out[-1].get("retry_of") and len(out) < 10:
        rows = db.table("contest_attempts").select("*").eq("id", out[-1]["retry_of"]).execute().data
        if not rows:
            break
        out.append(rows[0])
    return out


def _retry_grant(contest: dict, history: list) -> Optional[str]:
    """Attempt id a new attempt may retry, else None.
    Teacher override: an attempt the teacher allowed that has not been retried yet (no attempt limit).
    Technical: allow_technical_retry, fewer than MAX_WITH_TECHNICAL_RETRY attempts, and EVERY earlier attempt
    ended auto_timeout (never after manual / auto_warnings / an old backend's submit)."""
    if not history:
        return None
    used = {str(r["retry_of"]) for r in history if r.get("retry_of")}
    latest = max(history, key=lambda r: str(r.get("started_at") or ""))
    for r in sorted(history, key=lambda r: str(r.get("started_at") or ""), reverse=True):
        if r.get("retry_allowed_at") and str(r["id"]) not in used:
            return str(r["id"])
    if contest.get("allow_technical_retry", True) and len(history) < MAX_WITH_TECHNICAL_RETRY \
            and all(r.get("submitted_via") == "auto_timeout" for r in history):
        return str(latest["id"])
    return None


# ═══════════════════════════════════════════════════════════
# START ATTEMPT (student clicks "Start Test")
# ═══════════════════════════════════════════════════════════

def start_attempt(
    short_code: str,
    request: StartAttemptRequest,
    student_id: Optional[str] = None,
    resume_attempt_id: Optional[str] = None,
) -> Optional[ContestDataResponse]:
    """Start an attempt, or resume the caller's unsubmitted one (same attempt_id + saved progress).
    Caller = resume_attempt_id (X-Attempt-Id of an earlier /start, newer clients) or the logged-in student."""

    db = get_supabase_admin()

    contest_result = db.table("contests").select("*").eq(
        "short_code", short_code
    ).eq("status", "active").execute()

    if not contest_result.data:
        return None

    contest = contest_result.data[0]
    _check_started(contest)
    if _expired(contest):
        raise ContestExpired()
    sweep_stale(db, contest)

    mine = None
    if resume_attempt_id and _is_uuid(resume_attempt_id):
        rows = db.table("contest_attempts").select("*").eq("id", resume_attempt_id) \
            .eq("contest_id", contest["id"]).execute().data
        if rows and not (rows[0].get("student_id") and student_id and str(rows[0]["student_id"]) != str(student_id)):
            mine = _latest_of_chain(db, rows[0])
    if mine is None and student_id:
        rows = db.table("contest_attempts").select("*").eq("contest_id", contest["id"]) \
            .eq("student_id", student_id).eq("status", "in_progress").execute().data
        if rows:
            mine = max(rows, key=lambda r: str(r.get("started_at") or ""))

    questions = None
    if mine is not None and mine["status"] == "in_progress":
        questions = _load_questions(db, contest["id"])
        if not _stale(mine, contest):
            logger.info(f"Contest {contest['id']} attempt {mine['id']} resumed")
            if not clean_name(mine.get("student_name")):
                name, email = resolve_student(db, request, student_id)
                fill = {k: v for k, v in (("student_name", name), ("student_email", email))
                        if v and not mine.get(k)}
                if fill:
                    db.table("contest_attempts").update(fill).eq("id", mine["id"]).execute()
            return _contest_data(contest, mine, questions, resumed=True)
        _finalize_from_progress(db, mine, contest, questions, "auto_timeout")
        mine = {**mine, "status": "auto_submitted", "submitted_via": "auto_timeout"}

    max_attempts = contest["max_attempts"]
    retry_of = None
    history = None
    if student_id:
        history = db.table("contest_attempts").select("*").eq("contest_id", contest["id"]) \
            .eq("student_id", student_id).execute().data or []
    elif mine is not None and max_attempts <= 1:
        history = _chain(db, mine)          # anonymous: only this browser's own attempts can be counted
    if history is not None and len(history) >= max_attempts:
        for i, r in enumerate(history):     # an unsubmitted attempt past deadline + grace counts as auto_timeout
            if _stale(r, contest):
                questions = questions or _load_questions(db, contest["id"])
                _finalize_from_progress(db, r, contest, questions, "auto_timeout")
                history[i] = {**r, "status": "auto_submitted", "submitted_via": "auto_timeout"}
        retry_of = _retry_grant(contest, history)
        if retry_of is None:
            done = (mine or {}).get("id") or max(history, key=lambda r: str(r.get("started_at") or ""))["id"]
            raise AlreadySubmitted(str(done), f"Maximum attempts ({max_attempts}) reached")
        logger.info(f"Contest {contest['id']}: retry of attempt {retry_of} granted")

    previous = next((r for r in history or [] if str(r["id"]) == retry_of), {}) if retry_of else {}
    name, email = resolve_student(db, request, student_id)
    attempt_data = {
        "contest_id": contest["id"],
        "student_id": student_id,
        "student_name": name or clean_name(previous.get("student_name")),
        "student_email": email or previous.get("student_email"),
        "status": "in_progress",
    }
    if retry_of:
        attempt_data["retry_of"] = retry_of

    attempt_result = db.table("contest_attempts").insert(attempt_data).execute()

    if not attempt_result.data:
        raise Exception("Failed to create attempt")

    attempt = attempt_result.data[0]
    attempt.setdefault("started_at", _now().isoformat())

    if questions is None:
        questions = _load_questions(db, contest["id"])

    db.table("contests").update({
        "total_attempts": (contest.get("total_attempts") or 0) + 1
    }).eq("id", contest["id"]).execute()

    return _contest_data(contest, attempt, questions, resumed=False)


# ═══════════════════════════════════════════════════════════
# SUBMIT ATTEMPT
# ═══════════════════════════════════════════════════════════

def submit_attempt(
    contest_id: str,
    attempt_id: str,
    request: SubmitContestRequest,
) -> SubmitContestResponse:
    """Grade and store once; a repeated submit returns the stored result. Late (after deadline + grace) is still
    graded from the request and marked late. time_taken is measured by the server."""

    db = get_supabase_admin()
    if not _is_uuid(attempt_id) or not _is_uuid(contest_id):
        raise ValueError("Attempt not found")

    attempt_result = db.table("contest_attempts").select("*").eq(
        "id", attempt_id
    ).eq("contest_id", contest_id).execute()

    if not attempt_result.data:
        raise ValueError("Attempt not found")

    attempt = attempt_result.data[0]

    contest_result = db.table("contests").select("*").eq("id", contest_id).execute()
    if not contest_result.data:
        raise ValueError("Contest not found")

    contest = contest_result.data[0]
    questions = _load_questions(db, contest_id)

    if attempt["status"] != "in_progress":
        return _result_response(attempt, contest, questions)       # idempotent: same stored result

    stored_log = attempt.get("warning_log") or []
    names = {}                       # SubmitContestRequest always had student_name / email; they were dropped
    if not clean_name(attempt.get("student_name")) and clean_name(request.student_name):
        names["student_name"] = clean_name(request.student_name)
    if not attempt.get("student_email") and _clean_email(request.student_email):
        names["student_email"] = _clean_email(request.student_email)
    result = _finalize(
        db, attempt, contest, questions, request.answers, via="manual",
        warning_count=max(int(request.warning_count or 0), int(attempt.get("warning_count") or 0)),
        warning_log=request.warning_log if len(request.warning_log) >= len(stored_log) else stored_log,
        client_time=request.time_taken_seconds, extra=names,
    )
    sweep_stale(db, contest)
    return result


# ═══════════════════════════════════════════════════════════
# SAVE PROGRESS (every ~15 s per student)
# ═══════════════════════════════════════════════════════════

def save_progress(contest_id: str, attempt_id: str, request: ProgressRequest, user_id: Optional[str] = None) -> dict:
    """One RPC in the normal case. Raises ValueError (not found), AttemptForbidden, AlreadySubmitted.
    Returns {"saved", "finalized", ...}; finalized=True also carries the submit response fields."""
    if not _is_uuid(attempt_id) or not _is_uuid(contest_id):
        raise ValueError("Attempt not found")
    db = get_supabase_admin()
    r = db.rpc("contest_save_progress", {
        "p_contest_id": contest_id, "p_attempt_id": attempt_id, "p_user_id": user_id,
        "p_progress": {"answers": request.answers, "current_index": request.current_index},
        "p_warning_count": request.warning_count, "p_warning_log": request.warning_log,
    }).execute().data or {}
    reason = r.get("reason")
    if reason == "not_found":
        raise ValueError("Attempt not found")
    if reason == "forbidden":
        raise AttemptForbidden()
    if reason == "submitted":
        raise AlreadySubmitted(attempt_id, "Attempt already submitted")
    if r.get("ok") and not r.get("limit_reached"):
        return {"saved": True, "finalized": False, "warning_count": r.get("warning_count"),
                "max_warnings": r.get("max_warnings"), "deadline_at": _iso_z(r.get("deadline_at")),
                "server_now": _iso_z(r.get("server_now"))}

    # rare: the max_warnings-th warning, or a write after deadline + grace → finalize now
    attempt = db.table("contest_attempts").select("*").eq("id", attempt_id).execute().data[0]
    contest = db.table("contests").select("*").eq("id", contest_id).execute().data[0]
    questions = _load_questions(db, contest_id)
    via = "auto_warnings" if r.get("ok") else "auto_timeout"
    result = _finalize_from_progress(db, attempt, contest, questions, via)
    sweep_stale(db, contest)
    return {"saved": bool(r.get("ok")), "finalized": True, **result.model_dump(mode="json")}


# ═══════════════════════════════════════════════════════════
# GET LEADERBOARD (for teacher)
# ═══════════════════════════════════════════════════════════

def get_leaderboard(
    contest_id: str,
    teacher_id: Optional[str] = None,
) -> Optional[ContestLeaderboardResponse]:
    """Get all attempts for a contest (teacher view). Stale attempts are finalized first."""

    db = get_supabase_admin()
    if not _is_uuid(contest_id):
        return None

    query = db.table("contests").select("*").eq("id", contest_id)
    if teacher_id:
        query = query.eq("teacher_id", teacher_id)

    contest_result = query.execute()
    if not contest_result.data:
        return None

    contest = contest_result.data[0]
    sweep_stale(db, contest, force=True)
    # teacher_release before publishing: scores only for the contest's own teacher
    hide = _results_pending(contest) and not (teacher_id and str(contest.get("teacher_id")) == str(teacher_id))

    attempts_result = db.table("contest_attempts").select(
        "id, student_name, student_email, status, score, "
        "total_marks, percentage, time_taken_seconds, "
        "warning_count, submitted_at, submitted_via, late, started_at, retry_of"
    ).eq("contest_id", contest_id).order(
        "percentage", desc=True
    ).execute()

    attempt_list = [
        AttemptSummary(
            attempt_id=str(a["id"]),
            student_name=a.get("student_name"),
            student_email=a.get("student_email"),
            status=a["status"],
            score=None if hide else a.get("score"),
            total_marks=a.get("total_marks") or 0,
            percentage=None if hide or not a.get("percentage") else float(a["percentage"]),
            time_taken_seconds=a.get("time_taken_seconds"),
            warning_count=a.get("warning_count") or 0,
            submitted_at=a.get("submitted_at"),
            submitted_via=a.get("submitted_via"),
            late=bool(a.get("late")),
            retry=bool(a.get("retry_of")),
            retry_of=str(a["retry_of"]) if a.get("retry_of") else None,
        )
        for a in _latest_per_chain(attempts_result.data)
    ]

    return ContestLeaderboardResponse(
        contest_id=contest_id,
        title=contest["title"],
        total_attempts=len(attempt_list),
        attempts=attempt_list,
        results_pending=hide,
    )


def _latest_per_chain(rows: list) -> list:
    """One row per retry chain (attempts linked by retry_of): its latest SUBMITTED attempt, else its latest one.
    Attempts that are not retries of each other (max_attempts > 1) stay separate rows, as before. Keeps order."""
    by_id = {str(r["id"]): r for r in rows}

    def root(r):
        seen = set()
        while r.get("retry_of") and str(r["retry_of"]) in by_id and str(r["id"]) not in seen:
            seen.add(str(r["id"]))
            r = by_id[str(r["retry_of"])]
        return str(r["id"])

    groups: dict = {}
    for r in rows:
        groups.setdefault(root(r), []).append(r)
    keep = set()
    for members in groups.values():
        done = [m for m in members if m.get("status") != "in_progress"]
        pick = max(done or members, key=lambda m: str(m.get("started_at") or ""))
        keep.add(str(pick["id"]))
    return [r for r in rows if str(r["id"]) in keep]


# ═══════════════════════════════════════════════════════════
# TEACHER: attempts, allow-retry, publish results
# ═══════════════════════════════════════════════════════════

def _teacher_contest(db, contest_id: str, user_id: Optional[str]) -> Optional[dict]:
    if not _is_uuid(contest_id):
        return None
    rows = db.table("contests").select("*").eq("id", contest_id).execute().data
    if not rows:
        return None
    if not user_id or str(rows[0].get("teacher_id")) != str(user_id):
        raise NotTeacher()
    return rows[0]


def list_attempts(contest_id: str, user_id: Optional[str]) -> Optional[dict]:
    """Every attempt of the contest (teacher view; scores shown even before results are published)."""
    db = get_supabase_admin()
    contest = _teacher_contest(db, contest_id, user_id)
    if contest is None:
        return None
    sweep_stale(db, contest, force=True)
    rows = db.table("contest_attempts").select("*").eq("contest_id", contest_id).order("started_at").execute().data
    used = {str(r["retry_of"]) for r in rows if r.get("retry_of")}
    return {
        "contest_id": contest_id,
        "results_published_at": _iso_z(contest.get("results_published_at")),
        "attempts": [{
            "attempt_id": str(r["id"]),
            "student_name": r.get("student_name"),
            "student_email": r.get("student_email"),
            "status": r.get("status"),
            "submitted_via": r.get("submitted_via"),
            "late": bool(r.get("late")),
            "warning_count": r.get("warning_count") or 0,
            "score": r.get("score"),
            "total_marks": r.get("total_marks"),
            "percentage": float(r["percentage"]) if r.get("percentage") is not None else None,
            "time_taken_seconds": r.get("time_taken_seconds"),
            "started_at": _iso_z(r.get("started_at")),
            "submitted_at": _iso_z(r.get("submitted_at")),
            "retry_of": str(r["retry_of"]) if r.get("retry_of") else None,
            "retry_allowed_at": _iso_z(r.get("retry_allowed_at")),
            "retry_used": str(r["id"]) in used,
        } for r in rows],
    }


def allow_retry(contest_id: str, attempt_id: str, user_id: Optional[str]) -> Optional[dict]:
    """Teacher override: the student may start one more attempt after this one, whatever ended it.
    Idempotent (the first who/when is kept). None = contest or attempt not found."""
    db = get_supabase_admin()
    contest = _teacher_contest(db, contest_id, user_id)
    if contest is None or not _is_uuid(attempt_id):
        return None
    rows = db.table("contest_attempts").select("*").eq("id", attempt_id).eq("contest_id", contest_id).execute().data
    if not rows:
        return None
    attempt = rows[0]
    if attempt["status"] == "in_progress":
        if not _stale(attempt, contest):
            raise AttemptRunning()
        _finalize_from_progress(db, attempt, contest, _load_questions(db, contest_id), "auto_timeout")
    if not attempt.get("retry_allowed_at"):
        attempt["retry_allowed_at"], attempt["retry_allowed_by"] = _now().isoformat(), str(user_id)
        db.table("contest_attempts").update({"retry_allowed_at": attempt["retry_allowed_at"],
                                             "retry_allowed_by": attempt["retry_allowed_by"]}) \
            .eq("id", attempt_id).execute()
        logger.info(f"Contest {contest_id}: teacher allowed a retry after attempt {attempt_id}")
    used = bool(db.table("contest_attempts").select("id").eq("retry_of", attempt_id).execute().data)
    return {"attempt_id": attempt_id, "retry_allowed": True, "retry_allowed_at": _iso_z(attempt["retry_allowed_at"]),
            "retry_allowed_by": str(attempt["retry_allowed_by"]), "retry_used": used}


def publish_results(contest_id: str, user_id: Optional[str]) -> Optional[dict]:
    """Sets results_published_at (once; publishing again keeps the first time)."""
    db = get_supabase_admin()
    contest = _teacher_contest(db, contest_id, user_id)
    if contest is None:
        return None
    published = contest.get("results_published_at")
    if not published:
        published = _now().isoformat()
        db.table("contests").update({"results_published_at": published}).eq("id", contest_id).execute()
        logger.info(f"Contest {contest_id}: results published")
    return {"contest_id": contest_id, "answer_mode": contest.get("answer_mode"),
            "results_published_at": _iso_z(published)}


# ═══════════════════════════════════════════════════════════
# LIST CONTESTS (for teacher dashboard)
# ═══════════════════════════════════════════════════════════

def list_teacher_contests(teacher_id: str) -> List[dict]:
    """List all contests created by a teacher"""

    db = get_supabase_admin()

    result = db.table("contests").select(
        "id, title, subject, class_grade, short_code, "
        "total_questions, total_marks, total_attempts, "
        "duration_minutes, answer_mode, status, "
        "created_at, updated_at"
    ).eq("teacher_id", teacher_id).order("created_at", desc=True).execute()

    return [
        {
            "contest_id": str(r["id"]),
            "title": r["title"],
            "subject": r["subject"],
            "class_grade": r["class_grade"],
            "short_code": r["short_code"],
            "share_link": f"https://a4ai.in/contest/{r['short_code']}",
            "total_questions": r["total_questions"],
            "total_marks": r["total_marks"],
            "total_attempts": r["total_attempts"],
            "duration_minutes": r["duration_minutes"],
            "answer_mode": r["answer_mode"],
            "status": r["status"],
            "created_at": r["created_at"],
        }
        for r in result.data
    ]