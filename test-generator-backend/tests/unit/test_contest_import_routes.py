"""POST/GET /api/v1/contests/import-pdf: path rule, owner-only reads, daily limit, job states, upload cleanup.
Supabase is an in-memory fake; the parser is stubbed (the parser has its own tests)."""

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.routers import contest_import_router as cir
from app.routers import smart_assignment_router as sar
from app.services import contest_import_service as importer
from app.services.smart_assignment_service import GeminiBusy

OWNER = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"
URL = "/api/v1/contests/import-pdf"
PDF = b"%PDF-1.4\n%fake\n"


def _path(uid=OWNER):
    return f"{uid}/contest-imports/{uuid.uuid4()}.pdf"


class _Query:
    def __init__(self, db, op=None, payload=None, count=None):
        self.db, self.op, self.payload, self.count, self.filters = db, op, payload, count, []

    def select(self, *_, count=None):
        self.op, self.count = "select", count
        return self

    def insert(self, row):
        self.op, self.payload = "insert", row
        return self

    def update(self, fields):
        self.op, self.payload = "update", fields
        return self

    def eq(self, k, v):
        self.filters.append(lambda r: str(r.get(k)) == str(v))
        return self

    def gte(self, k, v):
        self.filters.append(lambda r: datetime.fromisoformat(r[k]) >= datetime.fromisoformat(v))   # timestamps, as PG
        return self

    def in_(self, k, vs):
        self.filters.append(lambda r: r.get(k) in vs)
        return self

    def or_(self, expr):
        assert expr == "status.neq.failed,retryable.eq.false"
        self.filters.append(lambda r: r["status"] != "failed" or not r["retryable"])
        return self

    def limit(self, _):
        return self

    def execute(self):
        if self.op == "insert":
            if any(r["owner"] == self.payload["owner"] and r["path"] == self.payload["path"] for r in self.db.rows):
                raise RuntimeError("duplicate key value violates unique constraint (23505)")
            row = {"id": str(uuid.uuid4()), "status": "queued", "progress": 0, "error": None, "retryable": False,
                   "attempts": 1, "result": None, "created_at": datetime.now(timezone.utc).isoformat(),
                   "updated_at": datetime.now(timezone.utc).isoformat(), **self.payload}
            self.db.rows.append(row)
            return SimpleNamespace(data=[row], count=None)
        hits = [r for r in self.db.rows if all(f(r) for f in self.filters)]
        if self.op == "update":
            for r in hits:
                r.update(self.payload)
                self.db.updates.append(dict(self.payload))
        return SimpleNamespace(data=[dict(r) for r in hits], count=len(hits))


class FakeAdmin:
    def __init__(self):
        self.rows, self.updates, self.objects, self.removed = [], [], {}, []
        self.created = {}           # storage path → upload time
        self.storage = SimpleNamespace(from_=self._bucket)

    def table(self, name):
        assert name == "contest_import_jobs"
        return _Query(self)

    def _bucket(self, name):
        assert name == "answer-keys"

        def download(path):
            if path not in self.objects:
                raise RuntimeError("Object not found")
            return self.objects[path]

        def remove(paths):
            self.removed.extend(paths)
            for p in paths:
                self.objects.pop(p, None)

        def list_(folder, options=None):
            out = []
            for p in self.objects:
                head, _, name = p.rpartition("/")
                if head == folder:
                    out.append({"name": name, "id": name,
                                "created_at": self.created.get(p, datetime.now(timezone.utc)).isoformat()})
            return out
        return SimpleNamespace(download=download, remove=remove, list=list_)


@pytest.fixture
def env(monkeypatch):
    db = FakeAdmin()
    who = {"id": OWNER}
    main.app.dependency_overrides[sar.require_caller] = lambda: SimpleNamespace(
        user=SimpleNamespace(id=who["id"]), db=None, token="t")
    monkeypatch.setattr(cir, "get_supabase_admin", lambda: db)
    import app.routers.copy_checker_router as ccr
    monkeypatch.setattr(ccr, "get_supabase_admin", lambda: db)
    monkeypatch.setattr(importer, "check_pdf", lambda path: 5)
    main._rate_store.clear()
    yield SimpleNamespace(client=TestClient(main.app), db=db, who=who)
    main.app.dependency_overrides.clear()


RESULT = {"title": "T", "subject": "Economics", "class_grade": "12", "institute": None, "duration_minutes": 60,
          "total_marks": 1, "answer_key_found": True, "answer_key_count": 1,
          "questions": [{"question_text": "Q?", "question_type": "MCQ",
                         "options": [{"label": "A", "text": "x"}, {"label": "B", "text": "y"}],
                         "correct_answer": "B", "explanation": None, "marks": 1, "difficulty": "Easy",
                         "chapter": None, "source_number": 1, "answer_source": "key", "usable": True}],
          "warnings": []}


def _start(env, path=None):
    path = path or _path()
    env.db.objects[path] = PDF
    env.db.created[path] = datetime.now(timezone.utc)
    return path, env.client.post(URL, json={"path": path})


def _job(env, job_id):
    return env.client.get(f"{URL}/{job_id}").json()


def test_happy_path_202_then_done_and_upload_deleted(env, monkeypatch):
    seen = {}

    def fake_import(local, pages, deadline, progress):
        seen["bytes"] = open(local, "rb").read()
        progress(50)
        return RESULT
    monkeypatch.setattr(importer, "import_paper", fake_import)
    path, r = _start(env)
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    assert seen["bytes"] == PDF                      # TestClient runs the background task before returning
    body = env.client.get(f"{URL}/{job_id}").json()
    assert body["status"] == "done" and body["progress"] == 100 and body["error"] is None
    assert body["result"]["questions"][0]["correct_answer"] == "B"
    assert env.db.removed == [path]
    assert {"progress": 50} in env.db.updates


@pytest.mark.parametrize("bad", [
    f"{OTHER}/contest-imports/{uuid.uuid4()}.pdf",           # someone else's folder
    f"{OWNER}/other-folder/{uuid.uuid4()}.pdf",
    f"{OWNER}/contest-imports/{uuid.uuid4()}.png",
    f"{OWNER}/contest-imports/../{uuid.uuid4()}.pdf",
    "",
])
def test_rejects_paths_outside_own_import_folder(env, bad):
    r = env.client.post(URL, json={"path": bad})
    assert r.status_code == 400
    assert env.db.rows == []


def test_other_user_cannot_read_job(env, monkeypatch):
    monkeypatch.setattr(importer, "import_paper", lambda *a, **k: RESULT)
    _, r = _start(env)
    job_id = r.json()["job_id"]
    env.who["id"] = OTHER
    assert env.client.get(f"{URL}/{job_id}").status_code == 404
    assert env.client.get(f"{URL}/not-a-uuid").status_code == 404


def test_requires_login(monkeypatch):
    main.app.dependency_overrides.clear()
    main._rate_store.clear()
    r = TestClient(main.app).post(URL, json={"path": _path()})
    assert r.status_code == 401


def test_gemini_busy_is_retryable_failure(env, monkeypatch):
    def busy(*a, **k):
        raise GeminiBusy(429)
    monkeypatch.setattr(importer, "import_paper", busy)
    path, r = _start(env)
    body = env.client.get(f"{URL}/{r.json()['job_id']}").json()
    assert body == {"status": "failed", "progress": 10, "error": "AI is busy, please retry in a minute",
                    "retryable": True, "result": None}
    assert env.db.removed == [] and path in env.db.objects          # kept for the retry


def test_rejected_file_is_not_retryable(env, monkeypatch):
    def too_many(path):
        raise importer.ImportRejected("The PDF has 31 pages; the limit is 30.")
    monkeypatch.setattr(importer, "check_pdf", too_many)
    path, r = _start(env)
    body = env.client.get(f"{URL}/{r.json()['job_id']}").json()
    assert body["status"] == "failed" and body["retryable"] is False
    assert body["error"] == "The PDF has 31 pages; the limit is 30."
    assert env.db.removed == [path] and path not in env.db.objects  # no retry possible: deleted


def test_missing_upload(env):
    r = env.client.post(URL, json={"path": _path()})        # nothing uploaded at that path
    body = env.client.get(f"{URL}/{r.json()['job_id']}").json()
    assert body["status"] == "failed" and body["error"] == cir.MISSING_MSG


def test_daily_limit_ignores_retryable_failures(env, monkeypatch):
    monkeypatch.setattr(cir, "DAILY_LIMIT", 2)
    monkeypatch.setattr(importer, "import_paper", lambda *a, **k: (_ for _ in ()).throw(GeminiBusy(503)))
    for _ in range(3):                                       # busy failures don't count
        assert _start(env)[1].status_code == 202
    monkeypatch.setattr(importer, "import_paper", lambda *a, **k: RESULT)
    assert _start(env)[1].status_code == 202
    assert _start(env)[1].status_code == 202
    r = _start(env)[1]
    assert r.status_code == 429 and "2 papers a day" in r.json()["detail"]


def _running_row(env, minutes_ago):
    old = (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()
    env.db.rows.append({"id": str(uuid.uuid4()), "owner": OWNER, "path": _path(), "status": "processing",
                        "progress": 20, "error": None, "retryable": False, "result": None,
                        "created_at": old, "updated_at": old})
    return env.db.rows[-1]["id"]


def test_processing_over_6_minutes_reported_failed(env):
    job_id = _running_row(env, 6.1)
    body = env.client.get(f"{URL}/{job_id}").json()
    assert body["status"] == "failed" and body["retryable"] is True and body["error"] == cir.STALE_MSG
    assert env.db.rows[-1]["status"] == "failed"


def test_processing_under_6_minutes_still_running(env):
    job_id = _running_row(env, 5.5)
    assert env.client.get(f"{URL}/{job_id}").json()["status"] == "processing"


def test_same_path_returns_the_existing_job(env, monkeypatch):
    runs = []
    monkeypatch.setattr(importer, "import_paper", lambda *a, **k: runs.append(1) or RESULT)
    path, r1 = _start(env)
    r2 = env.client.post(URL, json={"path": path})
    assert r2.status_code == 202 and r2.json()["job_id"] == r1.json()["job_id"]
    assert len(env.db.rows) == 1 and runs == [1]


def test_parallel_post_race_returns_the_winner(env, monkeypatch):
    """Both POSTs saw no job; the second insert hits the unique (owner, path) index."""
    monkeypatch.setattr(importer, "import_paper", lambda *a, **k: RESULT)
    path, r1 = _start(env)
    real = cir._job_for_path
    calls = []

    def first_miss(uid, p):
        calls.append(p)
        return None if len(calls) == 1 else real(uid, p)
    monkeypatch.setattr(cir, "_job_for_path", first_miss)
    r2 = env.client.post(URL, json={"path": path})
    assert r2.status_code == 202 and r2.json()["job_id"] == r1.json()["job_id"] and len(env.db.rows) == 1


# ── retry on the same path (retryable failure keeps the file) ─────────

def test_retryable_failure_keeps_file_then_retry_on_same_path_succeeds(env, monkeypatch):
    outcomes = [GeminiBusy(503), RESULT]

    def flaky(*a, **k):
        o = outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o
    monkeypatch.setattr(importer, "import_paper", flaky)
    path, r1 = _start(env)
    job_id = r1.json()["job_id"]
    assert _job(env, job_id)["status"] == "failed" and path in env.db.objects
    r2 = env.client.post(URL, json={"path": path})                  # same path, file still there
    assert r2.status_code == 202 and r2.json()["job_id"] == job_id
    body = _job(env, job_id)
    assert body["status"] == "done" and body["error"] is None and body["retryable"] is False
    assert len(env.db.rows) == 1 and env.db.rows[0]["attempts"] == 2
    assert env.db.removed == [path]                                 # done → deleted


def test_retry_ending_in_non_retryable_failure_deletes_the_file(env, monkeypatch):
    monkeypatch.setattr(importer, "import_paper", lambda *a, **k: (_ for _ in ()).throw(GeminiBusy(429)))
    path, r1 = _start(env)
    assert path in env.db.objects

    def rejected(local):
        raise importer.ImportRejected("The PDF has no pages.")
    monkeypatch.setattr(importer, "check_pdf", rejected)
    env.client.post(URL, json={"path": path})
    body = _job(env, r1.json()["job_id"])
    assert body["status"] == "failed" and body["retryable"] is False
    assert path not in env.db.objects


@pytest.mark.parametrize("final", ["done", "rejected"])
def test_repost_after_done_or_non_retryable_failure_returns_same_job_without_rerun(env, monkeypatch, final):
    runs = []
    if final == "done":
        monkeypatch.setattr(importer, "import_paper", lambda *a, **k: runs.append(1) or RESULT)
    else:
        monkeypatch.setattr(importer, "check_pdf",
                            lambda p: runs.append(1) or (_ for _ in ()).throw(importer.ImportRejected("bad")))
    path, r1 = _start(env)
    before = _job(env, r1.json()["job_id"])
    r2 = env.client.post(URL, json={"path": path})
    assert r2.json()["job_id"] == r1.json()["job_id"] and runs == [1]
    assert _job(env, r1.json()["job_id"]) == before


def test_repost_while_running_returns_same_job(env, monkeypatch):
    monkeypatch.setattr(importer, "import_paper", lambda *a, **k: RESULT)
    path = _path()
    env.db.rows.append({"id": str(uuid.uuid4()), "owner": OWNER, "path": path, "status": "processing",
                        "progress": 20, "error": None, "retryable": False, "attempts": 1, "result": None,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "updated_at": datetime.now(timezone.utc).isoformat()})
    r = env.client.post(URL, json={"path": path})
    assert r.json()["job_id"] == env.db.rows[0]["id"] and env.db.rows[0]["status"] == "processing"


def test_stopped_job_can_be_retried_on_same_path(env, monkeypatch):
    monkeypatch.setattr(importer, "import_paper", lambda *a, **k: RESULT)
    job_id = _running_row(env, 7)
    path = env.db.rows[-1]["path"]
    env.db.objects[path] = PDF
    assert _job(env, job_id)["status"] == "failed"                 # GET: stopped → failed, retryable
    assert env.client.post(URL, json={"path": path}).json()["job_id"] == job_id
    assert _job(env, job_id)["status"] == "done"


def test_parallel_retries_start_one_run(env, monkeypatch):
    runs = []
    monkeypatch.setattr(importer, "import_paper", lambda *a, **k: runs.append(1) or (_ for _ in ()).throw(
        GeminiBusy(503)))
    path, r1 = _start(env)
    row = dict(env.db.rows[0])                                      # both POSTs read the failed row …
    monkeypatch.setattr(cir, "_job_for_path", lambda uid, p: dict(row))
    env.db.rows[0]["status"] = "queued"                             # … but the other one already reset it
    r2 = env.client.post(URL, json={"path": path})
    assert r2.json()["job_id"] == r1.json()["job_id"] and runs == [1]


# ── old uploads (best effort, own prefix only) ────────────────────────

def test_new_job_deletes_own_uploads_older_than_24h_only(env, monkeypatch):
    monkeypatch.setattr(importer, "import_paper", lambda *a, **k: (_ for _ in ()).throw(GeminiBusy(503)))
    day_ago = datetime.now(timezone.utc) - timedelta(hours=25)
    old_mine, new_mine = _path(), _path()
    old_other_user = _path(OTHER)
    old_other_prefix = f"{OWNER}/assignments/{uuid.uuid4()}.pdf"
    for p, t in ((old_mine, day_ago), (new_mine, datetime.now(timezone.utc)), (old_other_user, day_ago),
                 (old_other_prefix, day_ago)):
        env.db.objects[p] = PDF
        env.db.created[p] = t
    path, _ = _start(env)
    assert old_mine not in env.db.objects
    assert {new_mine, old_other_user, old_other_prefix, path} <= set(env.db.objects)


def test_cleanup_failure_never_fails_the_job(env, monkeypatch):
    monkeypatch.setattr(importer, "import_paper", lambda *a, **k: RESULT)
    real = env.db._bucket

    def broken(name):
        b = real(name)
        b.list = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("storage down"))
        return b
    env.db.storage = SimpleNamespace(from_=broken)
    _, r = _start(env)
    assert _job(env, r.json()["job_id"])["status"] == "done"
