"""26: technical-failure retry, teacher override, teacher-released results (same in-memory Supabase as
test_contest_attempts)."""

import pytest

from test_contest_attempts import CID, CODE, S1, ans, env, later, progress, start, submit  # noqa: F401  (env = fixture)

T1 = "tttttttt-0000-0000-0000-000000000001"
OTHER = "tttttttt-0000-0000-0000-000000000002"


def timed_out(env, attempt_id=None, **body):
    """Start an attempt and let it run past deadline + grace (unsubmitted)."""
    att = start(env, attempt_id, **body).json()["attempt_id"]
    later(env, minutes=63)
    return att


def as_user(env, uid):
    env.who["id"] = uid


# ── technical retry ───────────────────────────────────────────────────

def test_retry_allowed_after_auto_timeout(env):
    att = timed_out(env, student_name="Asha")
    r = start(env, att)                                         # the same browser comes back
    body = r.json()
    assert r.status_code == 200 and body["attempt_id"] != att and body["retry_of"] == att
    new = env.db.attempt(body["attempt_id"])
    assert new["retry_of"] == att and new["student_name"] == "Asha"
    assert env.db.attempt(att)["submitted_via"] == "auto_timeout"


def test_retry_resumes_like_any_attempt(env):
    att = timed_out(env)
    retry = start(env, att).json()["attempt_id"]
    again = start(env, att).json()                              # old id again → newest attempt of the chain
    assert again["attempt_id"] == retry and again["resumed"] is True and again["retry_of"] == att


@pytest.mark.parametrize("how", ["manual", "auto_warnings"])
def test_retry_refused_after_manual_or_auto_warnings(env, how):
    att = start(env).json()["attempt_id"]
    if how == "manual":
        submit(env, att, ans("A"))
    else:
        assert progress(env, att, ans("A"), warning_count=5).json()["submitted_via"] == "auto_warnings"
    later(env, minutes=70)
    r = start(env, att)
    assert r.status_code == 409 and r.json()["code"] == "ALREADY_SUBMITTED"


def test_logged_in_at_most_two_attempts(env):
    as_user(env, S1)
    first = timed_out(env)
    second = start(env).json()
    assert second["retry_of"] == first
    later(env, minutes=63)                                     # the retry times out too
    r = start(env)
    assert r.status_code == 409 and r.json()["attempt_id"] == second["attempt_id"]
    assert len(env.db.tables["contest_attempts"]) == 2


def test_contest_can_turn_technical_retry_off(env):
    env.db.tables["contests"][0]["allow_technical_retry"] = False
    att = timed_out(env)
    assert start(env, att).status_code == 409


def test_unknown_submitted_via_from_old_backend_is_not_retried(env):
    att = start(env).json()["attempt_id"]
    env.db.attempt(att).update(status="submitted", submitted_via=None)
    assert start(env, att).status_code == 409


# ── teacher override ──────────────────────────────────────────────────

def test_teacher_override_after_manual_submit(env):
    env.db.tables["contests"][0]["teacher_id"] = T1
    att = start(env).json()["attempt_id"]
    submit(env, att, ans("A"))
    as_user(env, T1)
    r = env.client.post(f"/api/v1/contests/{CID}/attempts/{att}/allow-retry")
    body = r.json()
    assert r.status_code == 200 and body["retry_allowed"] is True and body["retry_allowed_by"] == T1
    assert body["retry_allowed_at"] == "2026-10-25T10:00:00Z" and body["retry_used"] is False
    later(env, minutes=5)
    assert env.client.post(f"/api/v1/contests/{CID}/attempts/{att}/allow-retry").json()["retry_allowed_at"] \
        == "2026-10-25T10:00:00Z"                               # idempotent: first who/when kept
    as_user(env, None)
    retry = start(env, att).json()
    assert retry["retry_of"] == att
    submit(env, retry["attempt_id"], ans("A", "B"))
    assert start(env, retry["attempt_id"]).status_code == 409   # the override is used up
    as_user(env, T1)
    assert env.client.post(f"/api/v1/contests/{CID}/attempts/{att}/allow-retry").json()["retry_used"] is True


def test_allow_retry_only_by_the_contest_teacher(env):
    env.db.tables["contests"][0]["teacher_id"] = T1
    att = start(env).json()["attempt_id"]
    submit(env, att, ans("A"))
    url = f"/api/v1/contests/{CID}/attempts/{att}/allow-retry"
    assert env.client.post(url).status_code == 401
    as_user(env, OTHER)
    assert env.client.post(url).status_code == 403
    assert env.db.attempt(att).get("retry_allowed_at") is None
    env.db.tables["contests"][0]["teacher_id"] = None           # contest created without a login
    assert env.client.post(url).status_code == 403


def test_allow_retry_on_running_attempt_is_409(env):
    env.db.tables["contests"][0]["teacher_id"] = T1
    att = start(env).json()["attempt_id"]
    as_user(env, T1)
    r = env.client.post(f"/api/v1/contests/{CID}/attempts/{att}/allow-retry")
    assert r.status_code == 409 and r.json()["code"] == "ATTEMPT_IN_PROGRESS"


def test_teacher_attempts_list(env):
    env.db.tables["contests"][0]["teacher_id"] = T1
    a1 = start(env, student_name="Asha", student_email="asha@x.in").json()["attempt_id"]
    later(env, minutes=10)
    submit(env, a1, ans("A", "B", "C"))
    as_user(env, OTHER)
    assert env.client.get(f"/api/v1/contests/{CID}/attempts").status_code == 403
    as_user(env, T1)
    body = env.client.get(f"/api/v1/contests/{CID}/attempts").json()
    assert body["contest_id"] == CID and body["results_published_at"] is None
    assert body["attempts"] == [{
        "attempt_id": a1, "student_name": "Asha", "student_email": "asha@x.in", "status": "submitted",
        "submitted_via": "manual", "late": False, "warning_count": 0, "score": 3, "total_marks": 3,
        "percentage": 100.0, "time_taken_seconds": 600, "started_at": "2026-10-25T10:00:00Z",
        "submitted_at": "2026-10-25T10:10:00Z", "retry_of": None, "retry_allowed_at": None, "retry_used": False}]


# ── leaderboard: latest submitted attempt per chain ───────────────────

def test_leaderboard_shows_latest_submitted_attempt_of_a_retry_chain(env):
    other = start(env, student_name="Ravi").json()["attempt_id"]
    first = timed_out(env, student_name="Asha")                # Asha's phone died
    submit(env, other, ans("A"))
    retry = start(env, first).json()["attempt_id"]
    later(env, minutes=20)
    submit(env, retry, ans("A", "B", "C"))
    rows = env.client.get(f"/api/v1/contests/{CID}/leaderboard").json()["attempts"]
    by_id = {r["attempt_id"]: r for r in rows}
    assert set(by_id) == {other, retry}
    assert by_id[retry]["retry"] is True and by_id[retry]["retry_of"] == first and by_id[retry]["score"] == 3
    assert by_id[other]["retry"] is False


def test_leaderboard_keeps_the_submitted_attempt_while_the_retry_runs(env):
    first = timed_out(env)
    retry = start(env, first).json()["attempt_id"]
    rows = env.client.get(f"/api/v1/contests/{CID}/leaderboard").json()["attempts"]
    assert [r["attempt_id"] for r in rows] == [first]
    assert retry not in [r["attempt_id"] for r in rows]


# ── teacher_release ───────────────────────────────────────────────────

def _release_contest(env):
    c = env.db.tables["contests"][0]
    c.update(answer_mode="teacher_release", teacher_id=T1, results_published_at=None)
    return c


def test_teacher_release_withholds_score_and_answers_until_published(env):
    _release_contest(env)
    att = start(env).json()["attempt_id"]
    later(env, minutes=10)
    body = submit(env, att, ans("A", "B", "C")).json()
    assert body["results_pending"] is True and body["score"] is None and body["percentage"] is None
    assert body["questions_with_answers"] is None
    assert body["total_marks"] == 3 and body["answered_count"] == 3 and body["status"] == "submitted"
    assert env.db.attempt(att)["score"] == 3                    # graded and stored, only not shown

    board = env.client.get(f"/api/v1/contests/{CID}/leaderboard").json()
    assert board["results_pending"] is True and board["attempts"][0]["score"] is None
    as_user(env, T1)
    teacher_board = env.client.get(f"/api/v1/contests/{CID}/leaderboard").json()
    assert teacher_board["results_pending"] is False and teacher_board["attempts"][0]["score"] == 3

    as_user(env, OTHER)
    assert env.client.post(f"/api/v1/contests/{CID}/publish-results").status_code == 403
    as_user(env, T1)
    pub = env.client.post(f"/api/v1/contests/{CID}/publish-results").json()
    assert pub == {"contest_id": CID, "answer_mode": "teacher_release",
                   "results_published_at": "2026-10-25T10:10:00Z"}
    as_user(env, None)
    after = submit(env, att, []).json()                         # the same stored result, now released
    assert after["results_pending"] is False and after["score"] == 3
    assert after["questions_with_answers"][0]["correct_answer"] == "A"


def test_teacher_release_also_hides_auto_finalized_results(env):
    _release_contest(env)
    att = start(env).json()["attempt_id"]
    body = progress(env, att, ans("A"), warning_count=5).json()
    assert body["finalized"] is True and body["results_pending"] is True and body["score"] is None


def test_publish_requires_login(env):
    _release_contest(env)
    assert env.client.post(f"/api/v1/contests/{CID}/publish-results").status_code == 401


def test_existing_modes_unchanged(env):
    att = start(env).json()["attempt_id"]                      # after_test
    body = submit(env, att, ans("A")).json()
    assert body["results_pending"] is False and body["score"] == 1 and body["questions_with_answers"]


def test_create_contest_accepts_teacher_release():
    from app.schemas.contest import CreateContestRequest
    assert CreateContestRequest(subject="S", class_grade="12", questions=[],
                                answer_mode="teacher_release").answer_mode == "teacher_release"
