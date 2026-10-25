"""Refresh-proof, server-enforced contest attempts (25 Oct 2026): resume on /start, PUT …/progress, warning limit,
idempotent / late submit, lazy stale finalization, 410 after expires_at — and old clients unchanged.

In-memory Supabase with a controllable clock. contest_save_progress is mirrored in Python here; the SQL itself is
tested in the product repo (supabase/tests/upload_flow/25_contest_hardening_test.sql)."""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.api.v1.endpoints import contest as endpoints
from app.services import contest_service as cs

T0 = datetime(2026, 10, 25, 10, 0, tzinfo=timezone.utc)
CID = "c0000000-0000-0000-0000-000000000001"
CODE = "eb2026"
S1 = "aaaaaaaa-0000-0000-0000-000000000001"
QIDS = ["q0000000-0000-0000-0000-00000000000" + str(i) for i in (1, 2, 3)]


class _Q:
    def __init__(self, db, table):
        self.db, self.table, self.op, self.payload, self.filters = db, table, "select", None, []
        self.count = None

    def select(self, *_, count=None):
        self.count = count
        return self

    def eq(self, k, v):
        self.filters.append(lambda r: str(r.get(k)) == str(v))
        return self

    def lt(self, k, v):
        self.filters.append(lambda r: cs._ts(r.get(k)) < cs._ts(v))
        return self

    def order(self, *_, **__):
        return self

    def insert(self, row):
        self.op, self.payload = "insert", row
        return self

    def update(self, fields):
        self.op, self.payload = "update", fields
        return self

    def execute(self):
        rows = self.db.tables[self.table]
        if self.op == "insert":
            row = {"id": str(uuid.uuid4()), "started_at": self.db.now.isoformat(), "submitted_at": None,
                   "warning_count": 0, "warning_log": [], "answers": [], "score": None, "total_marks": None,
                   "percentage": None, "time_taken_seconds": None, "progress": None, "submitted_via": None,
                   "late": False, **self.payload}
            rows.append(row)
            return SimpleNamespace(data=[dict(row)], count=None)
        hits = [r for r in rows if all(f(r) for f in self.filters)]
        if self.op == "update":
            for r in hits:
                r.update(self.payload)
            if self.table == "contest_attempts" and self.payload.get("score") is not None:
                self.db.gradings += len(hits)
        return SimpleNamespace(data=[dict(r) for r in hits], count=len(hits))


class FakeDB:
    def __init__(self):
        self.now = T0
        self.gradings = 0
        self.tables = {
            "contests": [{"id": CID, "short_code": CODE, "status": "active", "title": "Foreign Exchange Rate",
                          "subject": "Economics", "class_grade": "12", "board": "CBSE", "logo_base64": None,
                          "duration_minutes": 60, "total_questions": 3, "total_marks": 3, "enable_camera": True,
                          "enable_tab_detection": True, "allow_back_navigation": True, "max_warnings": 5,
                          "max_attempts": 1, "answer_mode": "after_test", "show_explanation": True,
                          "scheduled_at": None, "expires_at": None, "total_attempts": 0, "teacher_id": None}],
            "contest_questions": [
                {"id": qid, "contest_id": CID, "question_number": i + 1, "question_text": f"Q{i + 1}?",
                 "question_type": "MCQ", "options": [{"label": "A", "text": "a"}, {"label": "B", "text": "b"},
                                                     {"label": "C", "text": "c"}],
                 "correct_answer": "ABC"[i], "explanation": None, "marks": 1, "difficulty": "Easy", "chapter": None}
                for i, qid in enumerate(QIDS)],
            "contest_attempts": [],
        }
        self.rpc_calls = 0

    def table(self, name):
        return _Q(self, name)

    def rpc(self, name, p):
        """Python mirror of contest_save_progress (25_contest_hardening.sql)."""
        assert name == "contest_save_progress"
        self.rpc_calls += 1
        db = self

        def run():
            a = next((r for r in db.tables["contest_attempts"]
                      if r["id"] == p["p_attempt_id"] and r["contest_id"] == p["p_contest_id"]), None)
            if a is None:
                return {"ok": False, "reason": "not_found"}
            if a.get("student_id") and p["p_user_id"] and a["student_id"] != p["p_user_id"]:
                return {"ok": False, "reason": "forbidden"}
            c = db.tables["contests"][0]
            deadline = cs._deadline(a, c)
            if a["status"] != "in_progress":
                return {"ok": False, "reason": "submitted"}
            if db.now > deadline + timedelta(seconds=120):
                return {"ok": False, "reason": "late"}
            count = max(a.get("warning_count") or 0, p["p_warning_count"] or 0)
            log = p["p_warning_log"] if len(p["p_warning_log"]) >= len(a.get("warning_log") or []) \
                else a["warning_log"]
            a.update(progress=p["p_progress"], progress_updated_at=db.now.isoformat(), warning_count=count,
                     warning_log=log)
            return {"ok": True, "warning_count": count, "max_warnings": c["max_warnings"],
                    "limit_reached": count >= c["max_warnings"], "deadline_at": deadline.isoformat(),
                    "server_now": db.now.isoformat()}
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=run()))

    def attempt(self, attempt_id):
        return next(r for r in self.tables["contest_attempts"] if r["id"] == attempt_id)


@pytest.fixture
def env(monkeypatch):
    db = FakeDB()
    who = {"id": None}
    monkeypatch.setattr(cs, "get_supabase_admin", lambda: db)
    monkeypatch.setattr(cs, "_now", lambda: db.now)
    monkeypatch.setattr(cs, "_last_sweep", {})

    async def user(request):
        return who["id"]
    monkeypatch.setattr(endpoints, "get_optional_user_id", user)
    main._rate_store.clear()
    return SimpleNamespace(client=TestClient(main.app), db=db, who=who)


def start(env, attempt_id=None, **body):
    headers = {"X-Attempt-Id": attempt_id} if attempt_id else {}
    return env.client.post(f"/api/v1/contests/{CODE}/start", json=body, headers=headers)


def submit(env, attempt_id, answers, **extra):
    return env.client.post(f"/api/v1/contests/{CID}/submit", headers={"X-Attempt-Id": attempt_id},
                           json={"answers": answers, **extra})


def progress(env, attempt_id, answers=(), warning_count=0, warning_log=(), current_index=0):
    return env.client.put(f"/api/v1/contests/{CID}/attempts/{attempt_id}/progress",
                          headers={"X-Attempt-Id": attempt_id},
                          json={"answers": list(answers), "current_index": current_index,
                                "warning_count": warning_count, "warning_log": list(warning_log)})


def ans(*labels):
    return [{"questionId": q, "selected": l, "timeSpent": 5} for q, l in zip(QIDS, labels) if l]


def later(env, **kw):
    env.db.now = env.db.now + timedelta(**kw)


# ── resume ────────────────────────────────────────────────────────────

def test_refresh_resumes_same_attempt_with_progress_and_remaining_time(env):
    first = start(env, student_name="Asha").json()
    att = first["attempt_id"]
    assert first["resumed"] is False and first["progress"] is None
    assert first["deadline_at"] == "2026-10-25T11:00:00Z" and first["server_now"] == "2026-10-25T10:00:00Z"
    later(env, minutes=20)
    assert progress(env, att, ans("A", "C"), warning_count=2, warning_log=[{"type": "tab"}] * 2,
                    current_index=1).json()["saved"] is True
    later(env, minutes=5)
    again = start(env, att, student_name="Asha").json()                    # browser refresh
    assert again["attempt_id"] == att and again["resumed"] is True
    assert again["started_at"] == "2026-10-25T10:00:00Z" and again["deadline_at"] == "2026-10-25T11:00:00Z"
    remaining = (datetime.fromisoformat(again["deadline_at"].replace("Z", "+00:00"))
                 - datetime.fromisoformat(again["server_now"].replace("Z", "+00:00")))
    assert remaining == timedelta(minutes=35)
    assert again["progress"] == {"answers": ans("A", "C"), "current_index": 1, "warning_count": 2,
                                 "warning_log": [{"type": "tab"}] * 2}
    assert len(again["questions"]) == 3 and "correct_answer" not in again["questions"][0]
    assert len(env.db.tables["contest_attempts"]) == 1


def test_logged_in_student_resumes_without_header(env):
    env.who["id"] = S1
    att = start(env).json()["attempt_id"]
    later(env, minutes=3)
    again = start(env).json()
    assert again["attempt_id"] == att and again["resumed"] is True


def test_deadline_capped_by_expires_at(env):
    env.db.tables["contests"][0]["expires_at"] = (T0 + timedelta(minutes=30)).isoformat()
    assert start(env).json()["deadline_at"] == "2026-10-25T10:30:00Z"


# ── done / expired ────────────────────────────────────────────────────

def test_refresh_after_submit_is_409_already_submitted(env):
    att = start(env).json()["attempt_id"]
    submit(env, att, ans("A", "B", "C"))
    r = start(env, att)
    assert r.status_code == 409
    assert r.json() == {"code": "ALREADY_SUBMITTED", "attempt_id": att, "detail": "Maximum attempts (1) reached"}
    assert len(env.db.tables["contest_attempts"]) == 1


def test_logged_in_student_at_max_attempts_is_409(env):
    env.who["id"] = S1
    att = start(env).json()["attempt_id"]
    submit(env, att, ans("A"))
    r = start(env)
    assert r.status_code == 409 and r.json()["code"] == "ALREADY_SUBMITTED" and r.json()["attempt_id"] == att
    assert r.json()["detail"] == "Maximum attempts (1) reached"                        # old error text


def test_expired_contest_start_is_410(env):
    env.db.tables["contests"][0]["expires_at"] = (T0 - timedelta(minutes=1)).isoformat()
    r = start(env)
    assert r.status_code == 410 and r.json()["detail"] == "This contest has ended"
    assert env.db.tables["contest_attempts"] == []


# ── progress ──────────────────────────────────────────────────────────

def test_progress_cannot_lower_warning_count(env):
    att = start(env).json()["attempt_id"]
    first = progress(env, att, warning_count=3).json()
    assert first == {"saved": True, "finalized": False, "warning_count": 3, "max_warnings": 5,
                     "deadline_at": "2026-10-25T11:00:00Z", "server_now": "2026-10-25T10:00:00Z"}
    r = progress(env, att, warning_count=1).json()
    assert r["saved"] is True and r["warning_count"] == 3 and env.db.attempt(att)["warning_count"] == 3


def test_fifth_warning_finalizes_from_saved_answers(env):
    att = start(env).json()["attempt_id"]
    progress(env, att, ans("A", "B", "A"), warning_count=4)
    later(env, minutes=12)
    r = progress(env, att, ans("A", "B", "A"), warning_count=5)
    body = r.json()
    assert r.status_code == 200 and body["finalized"] is True and body["saved"] is True
    assert body["status"] == "auto_submitted" and body["submitted_via"] == "auto_warnings"
    assert body["score"] == 2 and body["total_marks"] == 3 and body["time_taken_seconds"] == 720
    assert body["questions_with_answers"][0]["correct_answer"] == "A"
    assert progress(env, att, warning_count=5).status_code == 409                     # no writes after it
    assert submit(env, att, ans("A", "B", "C")).json()["score"] == 2                  # stored result, not regraded


def test_progress_after_submit_is_409(env):
    att = start(env).json()["attempt_id"]
    submit(env, att, ans("A"))
    r = progress(env, att, ans("B"))
    assert r.status_code == 409 and r.json()["code"] == "ALREADY_SUBMITTED" and r.json()["attempt_id"] == att


def test_progress_needs_its_own_attempt_header(env):
    att = start(env).json()["attempt_id"]
    url = f"/api/v1/contests/{CID}/attempts/{att}/progress"
    assert env.client.put(url, json={}).status_code == 400
    assert env.client.put(url, json={}, headers={"X-Attempt-Id": str(uuid.uuid4())}).status_code == 403
    env.who["id"] = S1                        # someone else's logged-in session with the right attempt id
    env.db.attempt(att)["student_id"] = "bbbbbbbb-0000-0000-0000-000000000002"
    assert progress(env, att).status_code == 403


def test_progress_is_one_rpc_call(env):
    att = start(env).json()["attempt_id"]
    env.db.rpc_calls = 0
    progress(env, att, ans("A"))
    assert env.db.rpc_calls == 1


def test_progress_after_deadline_plus_grace_finalizes_from_saved(env):
    att = start(env).json()["attempt_id"]
    progress(env, att, ans("A", "B"))
    later(env, minutes=63)
    body = progress(env, att, ans("A", "B", "C")).json()
    assert body["saved"] is False and body["finalized"] is True
    assert body["submitted_via"] == "auto_timeout" and body["score"] == 2 and body["time_taken_seconds"] == 3600


# ── submit ────────────────────────────────────────────────────────────

def test_double_submit_returns_same_stored_result(env):
    att = start(env).json()["attempt_id"]
    later(env, minutes=10)
    first = submit(env, att, ans("A", "B", "C"), time_taken_seconds=5).json()
    later(env, minutes=1)
    second = submit(env, att, ans("C", "C", "A")).json()
    assert second == first and first["score"] == 3 and env.db.gradings == 1
    assert first["time_taken_seconds"] == 600 and env.db.attempt(att)["client_time_taken_seconds"] == 5


def test_submit_within_grace_is_on_time(env):
    att = start(env).json()["attempt_id"]
    later(env, minutes=61, seconds=30)
    body = submit(env, att, ans("A", "B")).json()
    assert body["late"] is False and body["score"] == 2 and body["time_taken_seconds"] == 3600


def test_submit_after_grace_is_graded_but_late(env):
    att = start(env).json()["attempt_id"]
    later(env, minutes=70)
    r = submit(env, att, ans("A", "B", "C"))
    body = r.json()
    assert r.status_code == 200 and body["late"] is True and body["score"] == 3
    assert body["time_taken_seconds"] == 3600 and body["submitted_via"] == "manual"


# ── stale attempts ────────────────────────────────────────────────────

def test_stale_attempt_finalized_lazily_by_another_start(env, monkeypatch):
    monkeypatch.setattr(cs, "SWEEP_EVERY_S", 0)       # the fake clock jumps 63 min; the throttle uses real time
    stale = start(env).json()["attempt_id"]
    progress(env, stale, ans("A", "B"), warning_count=1)
    later(env, minutes=63)
    other = start(env).json()                                       # another student opens the link
    assert other["attempt_id"] != stale
    row = env.db.attempt(stale)
    assert row["status"] == "auto_submitted" and row["submitted_via"] == "auto_timeout" and row["score"] == 2
    assert row["time_taken_seconds"] == 3600 and row["warning_count"] == 1


def test_stale_sweep_is_throttled_per_contest(env, monkeypatch):
    queries = []
    real = env.db.table

    def counting(name):
        q = real(name)
        if name == "contest_attempts":
            q.lt = lambda k, v, q=q, orig=q.lt: queries.append(v) or orig(k, v)
        return q
    monkeypatch.setattr(env.db, "table", counting)
    for _ in range(5):
        start(env)                                       # five students within the same 30 s
    assert len(queries) == 1


def test_leaderboard_finalizes_stale_and_shows_server_time(env):
    a1 = start(env).json()["attempt_id"]
    a2 = start(env).json()["attempt_id"]
    later(env, minutes=15)
    submit(env, a1, ans("A", "B", "C"), time_taken_seconds=1)
    later(env, minutes=50)
    board = env.client.get(f"/api/v1/contests/{CID}/leaderboard").json()
    by_id = {a["attempt_id"]: a for a in board["attempts"]}
    assert by_id[a1]["time_taken_seconds"] == 900 and by_id[a1]["submitted_via"] == "manual"
    assert by_id[a2]["status"] == "auto_submitted" and by_id[a2]["submitted_via"] == "auto_timeout"


def test_refresh_of_a_stale_attempt_finalizes_it_then_grants_the_technical_retry(env):
    att = start(env).json()["attempt_id"]
    later(env, minutes=63)
    r = start(env, att).json()
    assert r["attempt_id"] != att and r["retry_of"] == att and r["resumed"] is False
    assert env.db.attempt(att)["submitted_via"] == "auto_timeout"


# ── old clients ───────────────────────────────────────────────────────

def test_old_client_flow_unchanged(env):
    """No X-Attempt-Id on /start, never calls /progress: every /start is a new attempt (as before), submit grades
    the request, client warning_count still auto-submits, all old fields present."""
    a = start(env, student_name="Ravi").json()
    b = start(env, student_name="Ravi").json()
    assert a["attempt_id"] != b["attempt_id"] and len(env.db.tables["contest_attempts"]) == 2
    for key in ("contest_id", "attempt_id", "title", "subject", "class_grade", "duration_minutes", "enable_camera",
                "enable_tab_detection", "allow_back_navigation", "max_warnings", "answer_mode", "show_explanation",
                "questions"):
        assert key in a
    later(env, minutes=20)
    r = submit(env, a["attempt_id"], ans("A", "B", "B"), warning_count=5, warning_log=[{"reason": "tab"}],
               time_taken_seconds=1200)
    body = r.json()
    assert r.status_code == 200 and body["status"] == "auto_submitted" and body["score"] == 2
    assert body["answered_count"] == 3 and body["total_questions"] == 3 and body["percentage"] == 66.67
    assert body["questions_with_answers"][2]["correct_answer"] == "C"
    assert env.db.attempt(a["attempt_id"])["warning_log"] == [{"reason": "tab"}]
    assert submit(env, b["attempt_id"], [], time_taken_seconds=3).json()["status"] == "submitted"


def test_max_warnings_default_is_5():
    from app.schemas.contest import CreateContestRequest
    assert CreateContestRequest(subject="S", class_grade="12", questions=[]).max_warnings == 5
    assert CreateContestRequest(subject="S", class_grade="12", questions=[], max_warnings=3).max_warnings == 3
