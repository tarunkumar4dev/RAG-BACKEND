"""GET /contests/{id}/leaderboard (P1 25 Oct): rank with ties, correct / wrong / left, is_me / my_rank, privacy for
students vs the contest's teacher, name fallback, teacher_release gating, the per-contest cache."""

from datetime import timedelta

from test_contest_attempts import CID, QIDS, S1, ans, env, later, start, submit  # noqa: F401  (env = fixture)
from app.services import contest_service as cs

T1 = "tttttttt-0000-0000-0000-000000000001"
URL = f"/api/v1/contests/{CID}/leaderboard"


def finished(env, name=None, labels=("A", "B", "C"), minutes=10, warnings=0, **body):
    """Start, wait `minutes`, submit `labels` with `warnings`: one finished attempt."""
    att = start(env, student_name=name, **body).json()["attempt_id"]
    later(env, minutes=minutes)
    submit(env, att, ans(*labels), warning_count=warnings)
    later(env, minutes=-minutes)                     # next attempt starts at the same clock
    return att


def board(env, attempt_id=None, user=None):
    env.who["id"] = user
    headers = {"X-Attempt-Id": attempt_id} if attempt_id else {}
    return env.client.get(URL, headers=headers).json()


def test_rank_order_and_ties(env):
    a = finished(env, "A-full-slow", minutes=30)                 # 3, 1800 s
    b = finished(env, "B-full-fast", minutes=10)                 # 3, 600 s
    c = finished(env, "C-full-fast-warned", minutes=10, warnings=2)
    d = finished(env, "D-tie-with-B", minutes=10)                # 3, 600 s, 0 warnings = B
    e = finished(env, "E-two-right", labels=("A", "B", "A"), minutes=5)
    rows = board(env)["attempts"]
    assert [(r["student_name"], r["rank"]) for r in rows] == [
        ("B-full-fast", 1), ("D-tie-with-B", 1), ("C-full-fast-warned", 3), ("A-full-slow", 4), ("E-two-right", 5)]
    assert {r["attempt_id"] for r in rows} == {a, b, c, d, e}


def test_correct_wrong_left(env):
    finished(env, "Asha", labels=("A", "C", None))               # Q1 right, Q2 wrong, Q3 left
    row = board(env)["attempts"][0]
    assert (row["correct"], row["wrong"], row["left"]) == (1, 1, 1) and row["score"] == 1


def test_student_view_is_private_and_marks_me(env):
    env.db.tables["contests"][0]["teacher_id"] = T1
    mine = finished(env, "Asha", student_email="asha@x.in", warnings=2)
    finished(env, "Ravi", student_email="ravi@x.in", labels=("A", "B", "A"))
    running = start(env, student_name="Meera").json()["attempt_id"]
    body = board(env, attempt_id=mine)
    assert body["total_participants"] == 2 and body["my_rank"] == 1 and body["results_pending"] is False
    assert [r["student_name"] for r in body["attempts"]] == ["Asha", "Ravi"]   # running attempt not listed
    for r in body["attempts"]:
        assert r["student_email"] is None and r["warning_count"] is None
        assert r["submitted_via"] is None and r["late"] is None
    assert [r["is_me"] for r in body["attempts"]] == [True, False]
    assert running not in [r["attempt_id"] for r in body["attempts"]]


def test_teacher_view_has_everything(env):
    env.db.tables["contests"][0]["teacher_id"] = T1
    finished(env, "Asha", student_email="asha@x.in", warnings=2)
    start(env, student_name="Meera")
    body = board(env, user=T1)
    asha, meera = body["attempts"]
    assert asha["student_email"] == "asha@x.in" and asha["warning_count"] == 2
    assert asha["submitted_via"] == "manual" and asha["late"] is False and asha["rank"] == 1
    assert meera["status"] == "in_progress" and meera["rank"] is None
    assert body["total_participants"] == 1 and body["my_rank"] is None


def test_logged_in_student_is_me_without_header(env):
    env.who["id"] = S1
    att = start(env, student_name="Asha").json()["attempt_id"]
    later(env, minutes=5)
    submit(env, att, ans("A"))
    body = board(env, user=S1)
    assert body["attempts"][0]["is_me"] is True and body["my_rank"] == 1


def test_is_me_follows_a_retry_chain(env):
    first = start(env).json()["attempt_id"]
    later(env, minutes=63)                                       # timed out → technical retry
    retry = start(env, first).json()["attempt_id"]
    later(env, minutes=5)
    submit(env, retry, ans("A", "B"))
    body = board(env, attempt_id=first)                          # the browser still has the old id
    assert body["attempts"][0]["attempt_id"] == retry and body["attempts"][0]["is_me"] is True
    assert body["attempts"][0]["retry"] is True and body["my_rank"] == 1


def test_name_fallback(env):
    finished(env, None, student_email="ravi.k@x.in")             # no name typed → email local part
    env.who["id"] = S1                                           # logged in, nothing typed, no email stored
    att = start(env).json()["attempt_id"]
    later(env, minutes=5)
    submit(env, att, ans("A"))
    env.db.attempt(att).update(student_name="Student", student_email=None)   # an old row
    env.db.tables["profiles"].append({"id": S1, "full_name": "Meera S", "email": "meera@x.in"})
    names = sorted(r["student_name"] for r in board(env)["attempts"])
    assert names == ["Meera S", "ravi.k"]
    finished(env, None)                                          # truly anonymous: nothing to show
    assert None in [r["student_name"] for r in board(env)["attempts"]]
    assert "Student" not in [r["student_name"] for r in board(env)["attempts"]]


def test_teacher_release_hides_rows_from_students_only(env):
    env.db.tables["contests"][0].update(answer_mode="teacher_release", teacher_id=T1)
    mine = finished(env, "Asha")
    student = board(env, attempt_id=mine)
    assert student["results_pending"] is True and student["attempts"] == [] and student["my_rank"] is None
    teacher = board(env, user=T1)
    assert teacher["results_pending"] is False and teacher["attempts"][0]["score"] == 3
    env.db.tables["contests"][0]["results_published_at"] = "2026-10-25T12:00:00+00:00"
    assert board(env, attempt_id=mine)["my_rank"] == 1


def test_logged_in_non_teacher_no_longer_gets_404(env):
    env.db.tables["contests"][0]["teacher_id"] = T1
    finished(env, "Asha")
    env.who["id"] = S1
    assert env.client.get(URL).status_code == 200                # was 404 (the contest was filtered by teacher_id)


def test_board_is_cached_and_dropped_on_finalize(env, monkeypatch):
    monkeypatch.setattr(cs, "BOARD_TTL_S", 60)
    reads = []
    real = env.db.table
    monkeypatch.setattr(env.db, "table", lambda name: (reads.append(name), real(name))[1])
    finished(env, "Asha")
    reads.clear()
    for _ in range(3):
        board(env)
    assert reads.count("contest_questions") == 1                 # one board build for three requests
    finished(env, "Ravi")                                        # a new result drops the cached board
    assert len(board(env)["attempts"]) == 2
