"""contests.scheduled_at: /info and /start answer 403 CONTEST_NOT_STARTED before it; NULL / past = unchanged;
expires_at keeps /info at 410."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.services import contest_service

CONTEST = {"id": "c1", "title": "T", "subject": "Economics", "class_grade": "12", "board": "CBSE",
           "logo_base64": None, "duration_minutes": 30, "total_questions": 1, "total_marks": 1,
           "enable_camera": False, "enable_tab_detection": False, "allow_back_navigation": True, "max_warnings": 3,
           "status": "active", "scheduled_at": None, "expires_at": None, "max_attempts": 1, "answer_mode": "after_test",
           "show_explanation": True, "total_attempts": 0}
QUESTION = {"id": "q1", "question_number": 1, "question_text": "Q?", "question_type": "MCQ",
            "options": [{"label": "A", "text": "x"}, {"label": "B", "text": "y"}], "marks": 1,
            "difficulty": "Easy", "chapter": None}


class _Q:
    def __init__(self, db, table):
        self.db, self.table, self.op = db, table, "select"

    def select(self, *a, **k):
        return self

    def eq(self, *a):
        return self

    def order(self, *a):
        return self

    def insert(self, row):
        self.op = "insert"
        self.db.inserts.append((self.table, row))
        return self

    def update(self, fields):
        self.op = "update"
        return self

    def execute(self):
        if self.op == "insert":
            return SimpleNamespace(data=[{"id": "a1"}], count=None)
        rows = {"contests": [self.db.contest], "contest_questions": [QUESTION], "contest_attempts": []}[self.table]
        return SimpleNamespace(data=rows, count=0)


@pytest.fixture
def env(monkeypatch):
    db = SimpleNamespace(contest=dict(CONTEST), inserts=[])
    db.table = lambda name: _Q(db, name)
    monkeypatch.setattr(contest_service, "get_supabase_admin", lambda: db)
    main._rate_store.clear()
    return SimpleNamespace(client=TestClient(main.app), db=db)


def _iso(delta):
    return (datetime.now(timezone.utc) + delta).isoformat()


def test_not_started_info_and_start_are_403(env):
    starts = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=2)
    env.db.contest["scheduled_at"] = starts.isoformat().replace("+00:00", "Z")
    for r in (env.client.get("/api/v1/contests/abc/info"), env.client.post("/api/v1/contests/abc/start", json={})):
        assert r.status_code == 403
        body = r.json()
        assert body["code"] == "CONTEST_NOT_STARTED"
        assert datetime.fromisoformat(body["scheduled_at"]) == starts
    assert env.db.inserts == []                        # no attempt row before the start time


@pytest.mark.parametrize("scheduled", [None, "past"])
def test_open_contest_unchanged(env, scheduled):
    env.db.contest["scheduled_at"] = _iso(-timedelta(minutes=1)) if scheduled else None
    assert env.client.get("/api/v1/contests/abc/info").status_code == 200
    r = env.client.post("/api/v1/contests/abc/start", json={"student_name": "S"})
    assert r.status_code == 200 and r.json()["attempt_id"] == "a1"


def test_expired_info_stays_410(env):
    env.db.contest["scheduled_at"] = _iso(-timedelta(days=2))
    env.db.contest["expires_at"] = _iso(-timedelta(days=1))
    assert env.client.get("/api/v1/contests/abc/info").status_code == 410


# ── P0 25 Oct: how scheduled_at is read, stored and compared ───────────

OPEN_UTC = datetime(2026, 10, 25, 6, 6, tzinfo=timezone.utc)          # = 11:36 IST (contest jahrdxes)


@pytest.mark.parametrize("sent", ["2026-10-25T06:06:00Z", "2026-10-25T11:36:00+05:30", "2026-10-25T11:36:00",
                                  "2026-10-25T11:36"])
def test_create_reads_z_offset_and_naive_ist_as_the_same_instant(sent):
    from app.schemas.contest import CreateContestRequest
    body = CreateContestRequest(subject="S", class_grade="12", questions=[], scheduled_at=sent, expires_at=sent)
    assert body.scheduled_at == OPEN_UTC and body.scheduled_at.utcoffset() == timedelta(0)
    assert body.expires_at == OPEN_UTC


def test_create_stores_utc_with_offset(monkeypatch):
    from app.schemas.contest import CreateContestRequest
    inserted = []

    class Q:
        def __init__(self, table):
            self.table = table

        def select(self, *a, **k):
            return self

        def eq(self, *a):
            return self

        def insert(self, row):
            inserted.append((self.table, row))
            return self

        def execute(self):
            return SimpleNamespace(data=[{"id": "c1"}] if inserted and inserted[-1][0] == "contests" else [])
    monkeypatch.setattr(contest_service, "get_supabase_admin", lambda: SimpleNamespace(table=Q))
    contest_service.create_contest(CreateContestRequest(
        subject="S", class_grade="12", scheduled_at="2026-10-25T11:36", questions=[{"question_text": "Q?"}]))
    row = next(r for t, r in inserted if t == "contests")
    assert row["scheduled_at"] == "2026-10-25T06:06:00+00:00"          # never a naive string for Postgres


@pytest.mark.parametrize("stored", ["2026-10-25T06:06:00+00:00", "2026-10-25T06:06:00Z", "2026-10-25 06:06:00+00",
                                    "2026-10-25T11:36:00+05:30", "2026-10-25T06:06:00"])
@pytest.mark.parametrize("now,status", [(OPEN_UTC - timedelta(seconds=1), 403), (OPEN_UTC, 200),
                                        (OPEN_UTC + timedelta(minutes=2), 200)])
def test_opens_exactly_at_scheduled_at(env, monkeypatch, stored, now, status):
    env.db.contest["scheduled_at"] = stored
    monkeypatch.setattr(contest_service, "_now", lambda: now)
    assert env.client.get("/api/v1/contests/abc/info").status_code == status
    r = env.client.post("/api/v1/contests/abc/start", json={})
    assert r.status_code == status
    if status == 403:
        assert r.json()["scheduled_at"] == "2026-10-25T06:06:00+00:00"
        assert r.json()["server_now"] == "2026-10-25T06:05:59+00:00"
