"""
Job-worker tests against the REAL queue SQL: a throw-away Postgres 16 (pgserver, no Docker) built from the frontend
repo's migrations, exactly like supabase/tests/upload_flow/run_local.py there (never the live DB). Gemini is the
G1 mock recordings (tests/conftest.py), Storage is an in-memory fake.

Needs `pip install pgserver` and the frontend repo (A4AI_PRODUCT_DIR, default C:\\Users\\login\\OneDrive\\Desktop\\product);
skipped when either is missing.
"""

import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
import uuid

import pytest

pgserver = pytest.importorskip("pgserver")
import psycopg2  # noqa: E402

PRODUCT = pathlib.Path(os.getenv("A4AI_PRODUCT_DIR", r"C:\Users\login\OneDrive\Desktop\product"))
SUPA = PRODUCT / "supabase"
TESTS = SUPA / "tests" / "upload_flow"
MIG = SUPA / "migrations"
if not (MIG / "18_upload_flow.sql").exists():
    pytest.skip(f"frontend repo with 18_upload_flow.sql not found at {PRODUCT}", allow_module_level=True)

FILES = [TESTS / "00_supabase_stub.sql", TESTS / "01_baseline_from_audit.sql",
         *(MIG / f for f in ("01_access_helpers.sql", "02_rls_policies.sql", "05_student_portal.sql",
                             "05b_drop_anon_policies.sql", "10_smart_assignment_checks.sql", "11_assignment_ai_check.sql",
                             "12_copy_checker.sql", "17_check_copies_student_scope.sql", "18_upload_flow.sql"))]
if (MIG / "19_upload_link_portal.sql").exists():       # retry_checks (teacher Retry), applied on live 23 Oct
    FILES.append(MIG / "19_upload_link_portal.sql")
PSQL = pathlib.Path(pgserver.__file__).parent / "pginstall" / "bin" / "psql"

# fixtures of 18_upload_flow_test.sql (institute I1, teacher T1, batch B1, students S1/S2, assignments A1..A12)
INST = "00000000-0000-0000-0000-000000001001"
T1 = "00000000-0000-0000-0000-0000000000a2"
S1 = "00000000-0000-0000-0000-00000000c001"
S2 = "00000000-0000-0000-0000-00000000c002"
A1 = "00000000-0000-0000-0000-00000000d001"
A9 = "00000000-0000-0000-0000-00000000d009"     # auto_publish = true
KEY_PATH = "00000000-0000-0000-0000-0000000000a2/key.pdf"
FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "answer_checker"


def _psql(uri: str, sql_path: pathlib.Path) -> None:
    r = subprocess.run([str(PSQL), "-X", "-q", "-v", "ON_ERROR_STOP=1", "-d", uri, "-f", str(sql_path)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"{sql_path.name} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def _ensure_timezones() -> None:
    """pgserver's Windows build ships without share/postgresql/timezone, so 'Asia/Kolkata' (used by the QUOTA
    branch of a4_finish_check_job) is unknown. Postgres reads the same compiled TZif files as Python's tzdata."""
    tz_dir = pathlib.Path(pgserver.__file__).parent / "pginstall" / "share" / "postgresql" / "timezone"
    if (tz_dir / "Asia" / "Kolkata").exists():
        return
    tzdata = pytest.importorskip("tzdata")
    import shutil
    shutil.copytree(pathlib.Path(tzdata.__file__).parent / "zoneinfo", tz_dir, dirs_exist_ok=True)


@pytest.fixture(scope="session")
def pg_uri():
    _ensure_timezones()
    data = tempfile.mkdtemp(prefix="a4pg_worker_")
    srv = pgserver.get_server(data, cleanup_mode="stop")
    try:
        uri = srv.get_uri()
        for f in FILES:
            _psql(uri, f)
        text = (TESTS / "18_upload_flow_test.sql").read_text(encoding="utf-8")
        marker = text.index("-- verify_access_code + rate limit + lockout")
        seed = pathlib.Path(data) / "seed.sql"
        seed.write_text(text[:text.rindex("\n-- ═", 0, marker)], encoding="utf-8")
        _psql(uri, seed)
        yield uri
    finally:
        srv.cleanup()


class FakeStorage:
    def __init__(self):
        self.buckets: dict = {}
        self.removed: list = []
        self.uploads: list = []

    def put(self, bucket: str, path: str, data: bytes) -> None:
        self.buckets.setdefault(bucket, {})[path] = data

    def download(self, bucket: str, path: str):
        return self.buckets.get(bucket, {}).get(path)

    def upload(self, bucket: str, path: str, data: bytes, content_type: str) -> None:
        self.uploads.append((bucket, path))
        self.put(bucket, path, data)

    def remove(self, bucket: str, paths: list) -> None:
        self.removed += [(bucket, p) for p in paths]
        for p in paths:
            self.buckets.get(bucket, {}).pop(p, None)


class DB:
    def __init__(self, uri: str):
        self.uri = uri

    def connect(self):
        conn = psycopg2.connect(self.uri)
        conn.autocommit = True
        return conn

    def q(self, sql: str, args: tuple = ()):
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(sql, args)
            return cur.fetchall() if cur.description else None
        finally:
            conn.close()

    def one(self, sql: str, args: tuple = ()):
        rows = self.q(sql, args)
        return rows[0][0] if rows else None

    def job(self, job_id: str) -> dict:
        return self.one("select row_to_json(j) from check_jobs j where id = %s", (job_id,))

    def sub(self, sub_id: str) -> dict:
        return self.one("select row_to_json(s) from submissions s where id = %s", (sub_id,))

    def sheet(self, sub_id: str) -> dict:
        return self.one("select row_to_json(s) from check_sheets s where submission_id = %s", (sub_id,))

    def due_now(self, job_id: str) -> None:
        self.q("update check_jobs set next_run_at = now() where id = %s", (job_id,))


@pytest.fixture
def db(pg_uri):
    d = DB(pg_uri)
    d.q("""delete from check_jobs; delete from smart_assignment_checks; delete from check_sheets;
           delete from check_sessions; delete from submissions; delete from storage.objects;
           update assignment_ai_keys set answer_key_path = %s where assignment_id = any(%s::uuid[]);
           update assignments set max_pages = 30, auto_publish = (id = %s), max_mb = 10,
                  allowed_mimes = array['application/pdf','image/jpeg','image/png','image/webp']
            where id = any(%s::uuid[])""", (KEY_PATH, [A1, A9], A9, [A1, A9]))
    return d


@pytest.fixture
def storage():
    s = FakeStorage()
    s.put("answer-keys", KEY_PATH, (FIXTURES / "answer_key.pdf").read_bytes())
    return s


def make_attempt(db: DB, storage: FakeStorage, files: list, asg: str = A1, student: str = S1) -> tuple:
    """files: [(bytes, ext, mime)] → (submission_id, job_id). Storage rows + bytes like a real upload, the
    attempt row like create_submission writes it, then the real a4_enqueue_submission."""
    entries = []
    for i, (data, ext, mime) in enumerate(files, 1):
        path = f"{INST}/{asg}/{student}/{uuid.uuid4()}.{ext}"
        storage.put("submissions", path, data)
        db.q("insert into storage.objects (bucket_id, name, metadata) values ('submissions', %s, %s)",
             (path, json.dumps({"size": len(data), "mimetype": mime})))
        entries.append({"storage_path": path, "mime": mime, "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(), "page_order": i})
    n = db.one("select coalesce(max(attempt_no), 0) + 1 from submissions where assignment_id = %s and student_id = %s",
               (asg, student))
    sub_id = db.one("""insert into submissions (assignment_id, student_id, attempt_no, idempotency_key, status, files,
                                                submitted_at)
                       values (%s, %s, %s, %s, 'uploaded', %s, now()) returning id""",
                    (asg, student, n, uuid.uuid4().hex, json.dumps(entries)))
    job_id = db.one("select public.a4_enqueue_submission(%s)", (sub_id,))
    return str(sub_id), str(job_id)
