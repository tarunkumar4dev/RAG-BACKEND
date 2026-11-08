"""Real-data mode on a schema-faithful fixture (live-audit baseline + a4ai migrations, fake rows).
Expected numbers: tests/agent/real_fixture/EXPECTED.md (hand-computed)."""
import json
import os
import sys
import time
import uuid
from datetime import date
from pathlib import Path
from uuid import UUID

import jwt
import psycopg
import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent / "real_fixture"))
import build  # noqa: E402

from agent.llm import LLMResponse, ToolCall  # noqa: E402
from agent.tools.insights import (AttendanceStatsArgs, AttendanceTrendArgs, LowAttendanceArgs,  # noqa: E402
                                  MarksStatsArgs, ResolveEntityArgs)
from agent.tools.insights import TestsCreatedArgs as TCArgs  # noqa: E402

A = UUID("a0000000-0000-0000-0000-00000000000a")
B = UUID("b0000000-0000-0000-0000-00000000000b")
U = {k: f"10000000-0000-0000-0000-00000000000{v}" for k, v in
     dict(owner=1, admin=2, hod=3, proctor=4, sharma=5, verma=6, old=7, student=8, badmin=9, anil="a", multi="b").items()}
ECE2, ECE2A, CSE1, B_ECE2 = (UUID(f"ba000000-0000-0000-0000-00000000000{i}") for i in (1, 2, 3, 4))
S = {i: UUID(f"5a000000-0000-0000-0000-00000000000{i}") for i in range(1, 8)}
NOV = dict(date_from=date(2026, 11, 1), date_to=date(2026, 11, 5))
SUPA = "https://fixtureproj.supabase.co"
SECRET = "fixture-hs256-secret-not-a-real-one-0123456789"
PII = ["999000", "@example.test", "Mr Mehta", "Mrs Kumar", "Mr Rao", "Mr Gupta", "Mr Kumar", "@a.test", "JOINA1"]


@pytest.fixture(scope="module", autouse=True)
def real_env():
    owner, ro = build.build()
    keep = {k: os.environ.get(k) for k in ("AGENT_REAL_DB_MODE", "AGENT_REAL_DB_URL", "SUPABASE_URL",
                                           "SUPABASE_JWT_SECRET", "AGENT_DEV_MODE", "AGENT_DEV_AUTH")}
    os.environ.update(AGENT_REAL_DB_MODE="1", AGENT_REAL_DB_URL=ro, SUPABASE_URL=SUPA, SUPABASE_JWT_SECRET=SECRET)
    os.environ.pop("AGENT_DEV_MODE", None)
    os.environ.pop("AGENT_DEV_AUTH", None)
    yield {"owner": owner, "ro": ro}
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def ctx(who):
    from agent.real_identity import ctx_for_real_user
    return ctx_for_real_user(U[who])


def T():
    from agent.real import schema_adapter
    return schema_adapter


# ---------------- identity (DB rows only) ----------------
def test_identity_roles_from_db_rows():
    assert (ctx("owner").role, ctx("owner").institute_id) == ("admin", A)
    assert ctx("admin").role == "admin" and ctx("admin").teacher_id is None
    h = ctx("hod")
    assert h.role == "hod" and h.department_id == UUID("d0000000-0000-0000-0000-0000000000e1")
    assert ctx("proctor").role == "teacher" and ctx("sharma").role == "teacher"
    assert ctx("badmin").institute_id == B and ctx("admin").data_source == "live"
    for who in ("old", "student", "multi"):  # inactive member, student account, two institutes
        with pytest.raises(HTTPException) as e:
            ctx(who)
        assert e.value.status_code == 403


# ---------------- exact values (EXPECTED.md) ----------------
def test_r1_teacher_attendance_with_previous_period():
    r = T().attendance_stats(ctx("admin"), AttendanceStatsArgs(scope="teacher", entity_id=U["sharma"], **NOV))
    assert {k: r[k] for k in ("sessions", "marked", "present", "late", "absent", "leave", "pct", "prev_pct")} == \
        dict(sessions=3, marked=9, present=4, late=1, absent=3, leave=1, pct=55.6, prev_pct=100.0)
    assert r["entity"] == "Priya Sharma"


def test_r2_r3_section_and_student():
    r = T().attendance_stats(ctx("admin"), AttendanceStatsArgs(scope="section", entity_id=ECE2, **NOV))
    assert (r["pct"], r["marked"], r["entity"]) == (55.6, 9, "ECE-2")
    r = T().attendance_stats(ctx("admin"), AttendanceStatsArgs(scope="student", entity_id=S[1], **NOV))
    assert (r["pct"], r["marked"]) == (100.0, 4)


def test_r4_r5_low_attendance_by_scope():
    rows = lambda who: [(x["name"], x["section"], x["pct"]) for x in
                        T().low_attendance_list(ctx(who), LowAttendanceArgs(threshold=75, **NOV))["rows"]]
    assert rows("admin") == [("Ishita Rao", "ECE-2", 0.0), ("Rahul Kumar", "CSE-1", 0.0), ("Rahul Kumar", "ECE-2", 50.0)]
    assert rows("hod") == rows("proctor") == [("Ishita Rao", "ECE-2", 0.0), ("Rahul Kumar", "ECE-2", 50.0)]


def test_r6_r7_subject_teacher_scope():
    r = T().attendance_stats(ctx("sharma"), AttendanceStatsArgs(scope="student", entity_id=S[2], **NOV))
    assert (r["pct"], r["marked"]) == (66.7, 3)
    r = T().attendance_stats(ctx("verma"), AttendanceStatsArgs(scope="student", entity_id=S[1], **NOV))
    assert (r["pct"], r["marked"]) == (100.0, 1)
    assert T().attendance_stats(ctx("verma"), AttendanceStatsArgs(scope="student", entity_id=S[3], **NOV)) == \
        {"status": "not_found"}
    assert T().attendance_stats(ctx("sharma"), AttendanceStatsArgs(scope="teacher", entity_id=U["verma"], **NOV)) == \
        {"status": "forbidden"}
    assert T().attendance_stats(ctx("sharma"), AttendanceStatsArgs(scope="section", entity_id=CSE1, **NOV)) == \
        {"status": "not_found"}


def test_r8_trend():
    a = dict(scope="section", entity_id=ECE2, date_from=date(2026, 10, 26), date_to=date(2026, 11, 5))
    day = T().attendance_trend(ctx("admin"), AttendanceTrendArgs(granularity="day", **a))
    assert [(p["period"], p["pct"]) for p in day["points"]] == \
        [("2026-10-28", 100.0), ("2026-11-02", 66.7), ("2026-11-03", 33.3), ("2026-11-04", 66.7)]
    wk = T().attendance_trend(ctx("admin"), AttendanceTrendArgs(granularity="week", **a))
    assert [(p["period"], p["pct"], p["marked"]) for p in wk["points"]] == [("2026-10-26", 100.0, 3), ("2026-11-02", 55.6, 9)]


def test_r9_r10_r11_marks():
    exp_tests = [("Unit Test 1", "check_copies", "2026-10-30", 25.0, 62.5, 30.0, 20.0, 1),
                 ("Trig worksheet", "assignment", "2026-11-01", 15.0, 75.0, 18.0, 12.0, 1)]
    for who in ("admin", "sharma", "hod"):
        r = T().marks_stats(ctx(who), MarksStatsArgs(scope="section", entity_id=ECE2))
        assert [(t["test"], t["kind"], t["date"], t["avg"], t["avg_pct"], t["highest"], t["lowest"], t["absent"])
                for t in r["tests"]] == exp_tests, who
        assert r["overall_avg_pct"] == 68.8
    r = T().marks_stats(ctx("admin"), MarksStatsArgs(scope="student", entity_id=S[1]))
    assert [(t["test"], t["marks"], t["pct"]) for t in r["tests"]] == [("Unit Test 1", 30.0, 75.0), ("Trig worksheet", 18.0, 90.0)]
    assert r["overall_avg_pct"] == 82.5
    r = T().marks_stats(ctx("verma"), MarksStatsArgs(scope="section", entity_id=CSE1))
    assert [(t["avg"], t["avg_pct"], t["highest"], t["lowest"], t["absent"]) for t in r["tests"]] == [(6.5, 65.0, 9.0, 4.0, 0)]
    assert T().marks_stats(ctx("verma"), MarksStatsArgs(scope="section", entity_id=ECE2)) == {"status": "not_found"}


def test_r12_tests_created_ist_boundaries():
    r = T().tests_created_by_teacher(ctx("admin"), TCArgs(teacher_id=U["sharma"], **NOV))
    assert r["count"] == 4 and r["teacher"] == "Priya Sharma"
    assert {t["title"] for t in r["recent"]} == {"Trigonometry Unit Test", "Algebra quiz", "Late night paper", "Midnight paper"}
    inst = T().tests_created_by_teacher(ctx("admin"), TCArgs(**NOV))
    assert inst["count"] == 5 and inst["by_teacher"] == [{"teacher": "Priya Sharma", "count": 4}, {"teacher": "Rakesh Verma", "count": 1}]
    assert T().tests_created_by_teacher(ctx("sharma"), TCArgs(**NOV))["count"] == 4
    assert T().tests_created_by_teacher(ctx("sharma"), TCArgs(teacher_id=U["verma"], **NOV)) == {"status": "forbidden"}
    assert T().tests_created_by_teacher(ctx("admin"), TCArgs(teacher_id=U["anil"], **NOV)) == {"status": "not_found"}


def test_r13_r14_resolve_and_institute_isolation():
    res = lambda who, kind, name: T().resolve_entity(ctx(who), ResolveEntityArgs(kind=kind, name=name))
    r = res("admin", "student", "Rahul Kumar")
    assert r["status"] == "ambiguous" and [m["label"] for m in r["matches"]] == \
        ["Rahul Kumar (CSE-1, roll 12)", "Rahul Kumar (ECE-2, roll 03)"]
    assert [m["label"] for m in res("verma", "student", "Rahul Kumar")["matches"]] == \
        ["Rahul Kumar (CSE-1, roll 12)", "Rahul Kumar (ECE-2, roll 03)"]
    assert res("sharma", "student", "rahul kumar")["status"] == "ok"
    assert res("admin", "teacher", "Sharma ma'am")["matches"] == [{"id": U["sharma"], "label": "Priya Sharma"}]
    assert res("sharma", "teacher", "Verma sir")["status"] == "not_found"  # plain teacher: only self
    assert res("hod", "section", "cse 1")["status"] == "not_found"
    assert res("admin", "section", "ece 2")["matches"] == [{"id": str(ECE2), "label": "ECE-2"}]
    d = res("badmin", "section", "ECE-2")
    assert d["matches"] == [{"id": str(B_ECE2), "label": "ECE-2"}]
    assert res("badmin", "teacher", "Sharma")["matches"] == [{"id": U["anil"], "label": "Anil Sharma"}]
    assert T().attendance_stats(ctx("badmin"), AttendanceStatsArgs(scope="section", entity_id=ECE2, **NOV)) == {"status": "not_found"}
    assert T().attendance_stats(ctx("badmin"), AttendanceStatsArgs(scope="teacher", entity_id=U["sharma"], **NOV)) == {"status": "not_found"}


def test_no_pii_in_any_tool_output():
    outs = []
    for who in ("admin", "hod", "sharma", "verma"):
        c = ctx(who)
        outs += [T().resolve_entity(c, ResolveEntityArgs(kind="student", name="Rahul")),
                 T().low_attendance_list(c, LowAttendanceArgs(threshold=100, **NOV)),
                 T().marks_stats(c, MarksStatsArgs(scope="student", entity_id=S[1])),
                 T().attendance_stats(c, AttendanceStatsArgs(scope="student", entity_id=S[2], **NOV)),
                 T().tests_created_by_teacher(c, TCArgs(**NOV))]
    blob = json.dumps(outs, default=str)
    assert not [p for p in PII if p in blob]
    assert "Aarav Mehta" in blob  # names are needed to answer, contact data is not


# ---------------- read-only DB role ----------------
def test_startup_guard_accepts_agent_ro_and_refuses_writable_roles(real_env, monkeypatch):
    from agent.db_real import UnsafeRoleError, startup_check
    assert startup_check() == {"role": "agent_ro", "read_only": True}
    monkeypatch.setenv("AGENT_REAL_DB_URL", real_env["owner"])  # superuser
    with pytest.raises(UnsafeRoleError):
        startup_check()
    with psycopg.connect(real_env["owner"], autocommit=True) as c:
        c.execute("drop role if exists agent_rw_probe")
        c.execute("create role agent_rw_probe login")
        c.execute("grant usage on schema public to agent_rw_probe")
        c.execute("grant select on all tables in schema public to agent_rw_probe")
        c.execute("grant insert on public.tests to agent_rw_probe")
    monkeypatch.setenv("AGENT_REAL_DB_URL", build._with_db(real_env["owner"], build.FIXTURE_DB, user="agent_rw_probe")
                       .replace(build.FIXTURE_PASSWORD, "x"))
    with pytest.raises(UnsafeRoleError) as e:
        startup_check()
    assert "INSERT public.tests" in str(e.value) or "auth" in str(e.value)


def test_agent_ro_cannot_write_even_outside_read_only_tx(real_env):
    with psycopg.connect(real_env["ro"], autocommit=True) as c:
        c.execute("set default_transaction_read_only = off")
        for q in ("insert into public.tests (exam_title) values ('x')", "update public.students set name = 'x'",
                  "delete from public.attendance_records", "select email from auth.users", "select phone from public.students"):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                c.execute(q)


def test_verify_block_of_role_script_is_clean(real_env):
    with psycopg.connect(real_env["owner"]) as c:
        res = [c.execute(q).fetchall() for q in build.verify_queries()]
    assert res[:4] == [[], [], [], []] and res[4] == [(17,)]


def test_modes_are_mutually_exclusive():
    from agent.api import create_app
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        create_app({"AGENT_DEV_MODE": "1", "AGENT_REAL_DB_MODE": "1", "AGENT_REAL_DB_URL": "x", "SUPABASE_URL": SUPA})
    with pytest.raises(RuntimeError, match="AGENT_REAL_DB_URL"):
        create_app({"AGENT_REAL_DB_MODE": "1", "SUPABASE_URL": SUPA})
    with pytest.raises(RuntimeError, match="AGENT_DEV_AUTH"):
        create_app({"AGENT_REAL_DB_MODE": "1", "AGENT_REAL_DB_URL": "x", "SUPABASE_URL": SUPA, "AGENT_DEV_AUTH": "1"})


# ---------------- JWT ----------------
def tok(sub=U["admin"], secret=SECRET, iss=f"{SUPA}/auth/v1", exp_in=600, alg="HS256", key=None, **extra):
    now = int(time.time())
    claims = {"sub": sub, "aud": "authenticated", "role": "authenticated", "iss": iss, "iat": now, "exp": now + exp_in,
              "user_metadata": {"full_name": "Demo Admin"}, **extra}
    return jwt.encode(claims, key or secret, algorithm=alg)


def test_jwt_valid_expired_wrong_issuer_forged_and_alg_confusion():
    from agent.jwt_auth import AuthError, verify_supabase_jwt
    assert verify_supabase_jwt(tok())["sub"] == U["admin"]
    for bad, msg in [(tok(exp_in=-120), "expired"), (tok(iss="https://other.supabase.co/auth/v1"), "issuer"),
                     (tok(secret="forged-secret-forged-secret-forged-0000"), "invalid"),
                     (tok(role="anon"), "authenticated"), (tok(aud="anon"), "audience"), ("a.b", "malformed")]:
        with pytest.raises(AuthError, match=msg):
            verify_supabase_jwt(bad)
    none_tok = jwt.encode({"sub": U["admin"]}, None, algorithm="none")
    with pytest.raises(AuthError):
        verify_supabase_jwt(none_tok)
    with pytest.raises(AuthError, match="SUPABASE_JWT_SECRET"):  # HS256 with no secret configured
        verify_supabase_jwt(tok(), env={"SUPABASE_URL": SUPA})


def test_jwt_rs256_via_jwks():
    from cryptography.hazmat.primitives.asymmetric import rsa
    from agent.jwt_auth import AuthError, verify_supabase_jwt
    k = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    class FakeJWKS:
        def __init__(self, pub):
            self.pub = pub

        def get_signing_key_from_jwt(self, _t):
            return type("K", (), {"key": self.pub})()
    good = tok(alg="RS256", key=k)
    assert verify_supabase_jwt(good, jwks_client=FakeJWKS(k.public_key()))["sub"] == U["admin"]
    with pytest.raises(AuthError):
        verify_supabase_jwt(good, jwks_client=FakeJWKS(other.public_key()))


# ---------------- API end to end (real mode, scripted LLM) ----------------
class Scripted:
    model = "claude-sonnet-5-5"

    def __init__(self, responses):
        self.responses = list(responses)

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False):
        return self.responses.pop(0) if self.responses else LLMResponse(text="**Sharma ma'am** ki attendance thodi kam hai.", model=self.model)


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    import agent.api as api
    from agent import config
    monkeypatch.setattr(config, "TRACE_PATH", str(tmp_path / "trace.jsonl"))
    app = api.create_app(os.environ)
    return TestClient(app), api, tmp_path / "trace.jsonl"


def _events(resp):
    return [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]


def test_chat_live_end_to_end_no_writes_no_pii_in_trace(client, monkeypatch, real_env):
    c, api, trace = client
    script = Scripted([
        LLMResponse(tool_calls=[ToolCall("t1", "resolve_entity", {"kind": "teacher", "name": "Sharma ma'am"})], model="claude-sonnet-5-5"),
        LLMResponse(tool_calls=[ToolCall("t2", "attendance_stats", {"scope": "teacher", "entity_id": U["sharma"],
                                                                     "date_from": "2026-11-01", "date_to": "2026-11-05"})],
                    model="claude-sonnet-5-5"),
        LLMResponse(tool_calls=[ToolCall("t3", "save_draft", {"draft_id": str(uuid.uuid4())})], model="claude-sonnet-5-5"),
    ])
    monkeypatch.setattr(api, "get_provider", lambda: script)
    with psycopg.connect(real_env["owner"]) as o:
        before = o.execute("select count(*) from tests").fetchone()[0]
    h = {"Authorization": f"Bearer {tok()}"}
    r = c.post("/agent/chat", json={"message": "Sharma ma'am ki is mahine ki attendance kya hai?"}, headers=h)
    assert r.status_code == 200
    evs = _events(r)
    steps = [(e["key"], e["params"]["phase"]) for e in evs if e["type"] == "thinking_status"]
    assert ("tool.attendance_stats", "start") in steps and ("tool.attendance_stats", "done") in steps
    card = next(e for e in evs if e["type"] == "stat_card")
    assert (card["value"], card["period"], card["delta"]) == ("55.6", "1–5 Nov 2026", "-44.4 pts vs previous 5 days")
    text = next(e for e in evs if e["type"] == "text_delta")["text"]
    assert "**" not in text
    assert evs[-1]["type"] == "final" and [e["seq"] for e in evs] == list(range(len(evs)))
    sid = evs[-1]["session_id"]
    assert [s["session_id"] for s in c.get("/agent/sessions", headers=h).json()["sessions"]] == [sid]
    meta = c.get("/agent/meta", headers=h).json()
    assert meta == {"data_source": "live", "read_only": True, "role": "admin", "first_name": "Demo",
                    "institute_name": "Demo Institute of Technology"}
    assert c.get(f"/agent/sessions/{sid}", headers={"Authorization": f"Bearer {tok(sub=U['sharma'])}"}).status_code == 404
    assert c.post("/agent/approve", json={"session_id": sid, "approval_id": "x"}, headers=h).status_code == 403
    with psycopg.connect(real_env["owner"]) as o:
        assert o.execute("select count(*) from tests").fetchone()[0] == before
    body = trace.read_text()
    assert "Priya" not in body and "55.6" not in body and "Sharma" not in body
    assert '"tool":"attendance_stats"' in body


def test_real_mode_rejects_missing_or_dev_identity(client):
    c, _, _ = client
    assert c.get("/agent/meta").status_code == 401
    assert c.get("/agent/meta", headers={"X-Dev-Role": "admin"}).status_code == 401
    assert c.get("/agent/meta", headers={"Authorization": f"Bearer {tok(exp_in=-100)}"}).status_code == 401
    assert c.get("/agent/meta", headers={"Authorization": f"Bearer {tok(sub=U['student'])}"}).status_code == 403


def test_write_tools_disabled_in_live_mode():
    from agent.core import Agent
    from agent.tools.paper import AssignArgs, DraftArgs
    a = Agent(Scripted([]), ctx("sharma"), str(uuid.uuid4()))
    did = uuid.uuid4()
    for name, args in [("save_draft", DraftArgs(draft_id=did)), ("finalize_test", DraftArgs(draft_id=did)),
                       ("export_pdf", DraftArgs(draft_id=did)), ("assign_to_section", AssignArgs(draft_id=did, section_id=ECE2))]:
        r = a._paper_tool(name, args, lambda e: None)
        assert r["status"] == "read_only" and "read-only" in r["message"]


# ---- JWT auth on /agent/chat endpoint ----
def test_chat_with_valid_jwt(client, monkeypatch):
    c, api, _ = client
    script = Scripted([LLMResponse(text="Hello", model="claude-sonnet-5-5")])
    monkeypatch.setattr(api, "get_provider", lambda: script)
    resp = c.post("/agent/chat", json={"message": "Hi"}, headers={"Authorization": f"Bearer {tok(sub=U['admin'])}"})
    assert resp.status_code == 200
    events = _events(resp)
    assert any(e.get("role") == "assistant" for e in events)


def test_chat_rejects_expired_jwt(client):
    c, _, _ = client
    resp = c.post("/agent/chat", json={"message": "Hi"}, headers={"Authorization": f"Bearer {tok(exp_in=-100)}"})
    assert resp.status_code == 401
    assert "expired" in resp.text.lower()


def test_chat_rejects_missing_jwt(client):
    c, _, _ = client
    resp = c.post("/agent/chat", json={"message": "Hi"})
    assert resp.status_code == 401
    assert "authentication" in resp.text.lower() or "no authorization" in resp.text.lower()


def test_chat_rejects_malformed_jwt(client):
    c, _, _ = client
    resp = c.post("/agent/chat", json={"message": "Hi"}, headers={"Authorization": "Bearer not.a.token"})
    assert resp.status_code == 401


def test_chat_rejects_wrong_role_in_jwt(client):
    c, _, _ = client
    resp = c.post("/agent/chat", json={"message": "Hi"}, headers={"Authorization": f"Bearer {tok(role='anon')}"})
    assert resp.status_code == 401
    assert "authenticated" in resp.text.lower()
