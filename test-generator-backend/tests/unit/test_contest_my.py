"""GET /contests/my (P1 25 Oct): every contest whose teacher_id is the caller, with schedule / release / attempt
counts; POST /contests reports whether a teacher got attached."""

from test_contest_attempts import CID, CODE, env, start, submit, ans  # noqa: F401  (env = fixture)

T1 = "399c4826-2c49-48b3-ab8b-eca0200e4acc"          # evyxsy6s's teacher on live
OTHER = "tttttttt-0000-0000-0000-000000000002"


def test_my_contests_lists_the_callers_contests_with_new_fields(env):
    c = env.db.tables["contests"][0]
    c.update(teacher_id=T1, created_at="2026-10-25T04:27:45.376594+00:00",
             updated_at="2026-10-25T05:29:00+00:00", scheduled_at="2026-10-25T06:06:00+00:00",
             expires_at=None, results_published_at=None)
    env.db.tables["contests"].append({**c, "id": "c0000000-0000-0000-0000-000000000009", "short_code": "other1",
                                      "teacher_id": OTHER})
    a = start(env).json()["attempt_id"]
    submit(env, a, ans("A"))
    start(env)
    env.who["id"] = T1
    body = env.client.get("/api/v1/contests/my").json()
    assert [x["short_code"] for x in body["contests"]] == [CODE]
    row = body["contests"][0]
    assert row["contest_id"] == CID and row["title"] and row["subject"] == "Economics" and row["class_grade"] == "12"
    assert row["created_at"] == "2026-10-25T04:27:45.376594+00:00" and row["answer_mode"] == "after_test"
    assert row["results_published_at"] is None and row["attempts_count"] == 2
    assert row["scheduled_at"] == "2026-10-25T06:06:00Z" and row["expires_at"] is None
    assert row["duration_minutes"] == 60 and row["share_link"] == f"https://a4ai.in/contest/{CODE}"


def test_my_contests_requires_login(env):
    assert env.client.get("/api/v1/contests/my").status_code == 401


def test_no_contests(env):
    env.who["id"] = OTHER
    assert env.client.get("/api/v1/contests/my").json() == {"contests": []}


def test_create_reports_whether_a_teacher_was_attached(env, monkeypatch):
    from app.services import contest_service as cs
    from app.schemas.contest import CreateContestResponse
    seen = {}

    def fake_create(request, teacher_id=None):
        seen["teacher_id"] = teacher_id
        return CreateContestResponse(contest_id="c", short_code="s", share_link="l", total_questions=1,
                                     total_marks=1, teacher_attached=bool(teacher_id))
    monkeypatch.setattr(cs, "create_contest", fake_create)
    body = {"subject": "S", "class_grade": "12", "questions": [{"question_text": "Q?"}]}
    assert env.client.post("/api/v1/contests", json=body,
                           headers={"Authorization": "Bearer expired.token.x"}).json()["teacher_attached"] is False
    env.who["id"] = T1
    assert env.client.post("/api/v1/contests", json=body).json()["teacher_attached"] is True
    assert seen["teacher_id"] == T1
