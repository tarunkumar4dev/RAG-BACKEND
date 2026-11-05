"""Approval single-use + replay, marks-sum enforcement, verifier gate, PII-free trace, event schema, budgets,
mocked providers, API session isolation."""
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from _agent_ids import ADMIN, B10A, DECOY, DECOY_ADMIN, MAIN, SHARMA, VERMA
from agent import config
from agent.approval import ApprovalError, create_approval, execute_approval
from agent.core import Agent
from agent.db import connect
from agent.llm import AnthropicProvider, LLMResponse, ToolCall, _gemini_schema
from agent.schemas import AgentEventAdapter, SessionCtx
from agent.tools.paper import BlueprintArgs, blueprint_defaults
from agent.verify import can_finalize, verify_paper

ADMIN_CTX = SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN)
VERMA_CTX = SessionCtx(user_id=VERMA, role="teacher", institute_id=MAIN, teacher_id=VERMA)
SHARMA_CTX = SessionCtx(user_id=SHARMA, role="teacher", institute_id=MAIN, teacher_id=SHARMA)


class Scripted:
    """Fake LLMProvider: returns scripted responses in order."""
    model = "claude-sonnet-5-5"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False):
        self.calls.append(messages)
        return self.responses.pop(0) if self.responses else LLMResponse(text="ok", model=self.model)


def _session(conn, ctx):
    sid = str(uuid.uuid4())
    conn.execute("insert into agent_sessions(id,user_id,institute_id,role) values (%s,%s,%s,%s)",
                 (sid, str(ctx.user_id), str(ctx.institute_id), ctx.role))
    return sid


# ---------------- approval gate ----------------
def test_approval_single_use_and_replay():
    with connect(readonly=False) as conn:
        sid = _session(conn, SHARMA_CTX)
        a = create_approval(conn, SHARMA_CTX, sid, "finalize_test", {"draft_id": "d1"})
        calls = []

        def ex(action, args):
            calls.append(action)
            return {"status": "finalized"}
        r1 = execute_approval(conn, SHARMA_CTX, sid, a["approval_id"], ex)
        r2 = execute_approval(conn, SHARMA_CTX, sid, a["approval_id"], ex)
        assert r1 == {"replayed": False, "status": "finalized"}
        assert r2 == {"replayed": True, "status": "finalized"}
        assert calls == ["finalize_test"]  # executed exactly once


def test_approval_bound_to_session_user_args_and_ttl():
    with connect(readonly=False) as conn:
        sid = _session(conn, SHARMA_CTX)
        a = create_approval(conn, SHARMA_CTX, sid, "finalize_test", {"draft_id": "d2"})
        ex = lambda *_: {"status": "x"}
        with pytest.raises(ApprovalError) as e:  # other session
            execute_approval(conn, SHARMA_CTX, _session(conn, SHARMA_CTX), a["approval_id"], ex)
        assert e.value.code == "not_found"
        with pytest.raises(ApprovalError) as e:  # other user, same session id
            execute_approval(conn, VERMA_CTX, sid, a["approval_id"], ex)
        assert e.value.code == "not_found"
        conn.execute("update agent_approvals set args = '{\"draft_id\":\"evil\"}'::jsonb where id=%s", (a["approval_id"],))
        with pytest.raises(ApprovalError) as e:  # args tampered after issue
            execute_approval(conn, SHARMA_CTX, sid, a["approval_id"], ex)
        assert e.value.code == "forbidden"
        b = create_approval(conn, SHARMA_CTX, sid, "finalize_test", {"draft_id": "d3"})
        conn.execute("update agent_approvals set expires_at = now() - interval '1 second' where id=%s", (b["approval_id"],))
        with pytest.raises(ApprovalError) as e:
            execute_approval(conn, SHARMA_CTX, sid, b["approval_id"], ex)
        assert e.value.code == "expired"


def test_ttl_is_15_minutes():
    assert config.APPROVAL_TTL_S == 900


# ---------------- marks sum + verifier ----------------
def _q(i, qtype="mcq", marks=1, **kw):
    base = {"id": f"q{i}", "qtype": qtype, "marks": marks, "text": f"question {i}", "chapter": "Trig"}
    if qtype == "mcq":
        base.update(options=["$1$", "$0$", r"$\dfrac{1}{2}$", "$-1$"], answer="B")
    base.update(kw)
    return base


@pytest.mark.parametrize("total", [20, 25, 30, 40, 50, 80])
def test_blueprint_sums_exactly(total):
    bp = blueprint_defaults(None, BlueprintArgs(class_grade=10, subject="Mathematics", chapter="Trig", total_marks=total))
    assert sum(s["count"] * s["marks_each"] for s in bp["sections"]) == total and bp["sum_ok"]


def test_marks_sum_must_be_exact():
    paper = {"total_marks": 3, "questions": [_q(1, check_expr="(1-tan(45*deg)**2)/(1+tan(45*deg)**2)"), _q(2)]}
    s = verify_paper(paper, lambda qs: {"q1": "B", "q2": "B"})
    assert s["marks_sum"] == 2 and not s["marks_sum_ok"] and not can_finalize(s)


def test_minus_one_and_one_are_distinct_and_dfrac_parses():  # regression: trig demo q3 / q4
    s = verify_paper({"total_marks": 1, "questions": [_q(1, check_expr="0")]}, lambda qs: {"q1": "B"})
    assert s["per_question"]["q1"]["status"] == "verified"
    q4 = _q(1, options=[r"$\dfrac{1}{3}$", r"$\dfrac{2}{3}$", r"$\dfrac{1}{2}$", r"$\dfrac{3}{4}$"], answer="C", check_expr="1/2")
    assert verify_paper({"total_marks": 1, "questions": [q4]}, lambda qs: {"q1": "C"})["objective_verified"] == 1


def test_two_equal_options_fail():
    q = _q(1, options=["$1/2$", r"$\sin 30^\circ$", "$1$", "$0$"], answer="A", check_expr="sin(30*deg)")
    s = verify_paper({"total_marks": 1, "questions": [q]}, lambda qs: {"q1": "A"})
    assert s["per_question"]["q1"]["status"] == "failed"


def test_solver_disagreement_fails_and_solver_down_is_manual_review():
    p = {"total_marks": 1, "questions": [_q(1, check_expr="0")]}
    assert verify_paper(p, lambda qs: {"q1": "A"})["failed"] == ["q1"]

    def down(qs):
        raise TimeoutError()
    s = verify_paper({"total_marks": 1, "questions": [_q(1)]}, down)
    assert s["manual_review"] == ["q1"] and s["objective_verified"] == 0


def test_solver_never_sees_the_key():
    seen = []
    verify_paper({"total_marks": 1, "questions": [_q(1, solution="because", check_expr="0")]},
                 lambda qs: seen.extend(qs) or {"q1": "B"})
    assert seen and all("answer" not in q and "solution" not in q and "check_expr" not in q for q in seen)


def test_unparseable_numeric_goes_to_manual_review():
    q = {"id": "q1", "qtype": "numeric", "marks": 2, "text": "t", "answer": "approximately the height of a tower",
         "check_expr": "the tower"}
    s = verify_paper({"total_marks": 2, "questions": [q]}, lambda qs: {"q1": "40"})
    assert s["manual_review"] == ["q1"]


def test_multi_value_answers_compare_pairwise():  # regression: trig demo q13
    q = {"id": "q1", "qtype": "very_short", "marks": 2, "text": "find A and B", "answer": "A = 45°, B = 15°",
         "check_expr": "45"}
    assert verify_paper({"total_marks": 2, "questions": [q]}, lambda qs: {"q1": "A=45, B=15"})["objective_verified"] == 1
    assert verify_paper({"total_marks": 2, "questions": [dict(q)]}, lambda qs: {"q1": "A=60, B=30"})["failed"] == ["q1"]


def test_assertion_reason_requires_standard_options():
    q = {"id": "q1", "qtype": "assertion_reason", "marks": 1, "text": "Assertion (A): x. Reason (R): y.",
         "options": ["yes", "no", "maybe", "never"], "answer": "A"}
    assert verify_paper({"total_marks": 1, "questions": [q]}, lambda qs: {"q1": "A"})["failed"] == ["q1"]


def test_rubric_checks_and_duplicates():
    good = {"id": "q1", "qtype": "short_answer", "marks": 3, "text": "Prove identity", "chapter": "Trig",
            "marking_scheme": [{"step": "a", "marks": 1}, {"step": "b", "marks": 2}]}
    bad_marks = dict(good, id="q2", text="Other", marking_scheme=[{"step": "a", "marks": 1}])
    dup = dict(good, id="q3")
    off_syllabus = dict(good, id="q4", text="Photosynthesis", chapter="Life Processes")
    s = verify_paper({"total_marks": 12, "questions": [good, bad_marks, dup, off_syllabus]}, None, ["Trig"])
    assert s["rubric_checked"] == 1 and s["duplicates"] == ["q3"] and set(s["failed"]) == {"q2", "q3", "q4"}
    assert s["rubric_total"] == 4  # never a flat "x/x verified"


# ---------------- events, trace, budgets ----------------
def _run(agent, msg):
    return list(agent.run(msg))


def test_events_validate_seq_monotonic_and_trace_has_no_pii(tmp_path, monkeypatch):
    trace = tmp_path / "t.jsonl"
    monkeypatch.setattr(config, "TRACE_PATH", str(trace))
    prov = Scripted([
        LLMResponse(tool_calls=[ToolCall("t1", "low_attendance_list",
                                         {"threshold": 75, "date_from": "2026-09-07", "date_to": "2026-11-05",
                                          "section_id": str(B10A)})], tokens_in=10, tokens_out=5, model="claude-sonnet-5-5"),
        LLMResponse(tool_calls=[ToolCall("t2", "resolve_entity", {"kind": "student", "name": "Rahul Kumar"})],
                    model="claude-sonnet-5-5"),
    ])
    evs = _run(Agent(prov, ADMIN_CTX, str(uuid.uuid4())), "low attendance 10A")
    for e in evs:
        AgentEventAdapter.validate_python(e)
    assert [e["seq"] for e in evs] == list(range(len(evs)))
    assert len({e["request_id"] for e in evs}) == 1
    types = [e["type"] for e in evs]
    assert "table" in types and "clarify" in types and types[-1] == "final"
    table = next(e for e in evs if e["type"] == "table")
    assert table["source"] == "tool:low_attendance_list" and table["as_of"].endswith("+05:30")
    names = [r[0] for r in table["rows"]]
    text = trace.read_text()
    assert names and not any(n in text for n in names) and "Rahul" not in text
    for line in text.splitlines():
        rec = json.loads(line)
        assert set(rec) <= {"ts", "request_id", "kind", "tool", "arg_keys", "latency_ms", "ok", "status", "model",
                            "tokens_in", "tokens_out", "cost_usd"}


def test_clarify_never_auto_picks():
    prov = Scripted([LLMResponse(tool_calls=[ToolCall("t1", "resolve_entity", {"kind": "student", "name": "Rahul Kumar"})],
                                 model="claude-sonnet-5-5")])
    evs = _run(Agent(prov, ADMIN_CTX, str(uuid.uuid4())), "Rahul Kumar ke marks")
    clarify = [e for e in evs if e["type"] == "clarify"]
    assert len(clarify) == 1 and len(clarify[0]["chips"]) == 2
    assert len(prov.calls) == 1  # loop stopped: no further LLM step picked one


def test_invalid_event_rejected():
    with pytest.raises(ValidationError):
        AgentEventAdapter.validate_python({"type": "stat_card", "seq": 0, "request_id": "r", "title": "x", "value": "1"})
    with pytest.raises(ValidationError):
        AgentEventAdapter.validate_python({"type": "clarify", "seq": 0, "request_id": "r", "question": "q",
                                           "chips": [{"label": str(i), "value": str(i)} for i in range(6)]})


def test_budget_breach_emits_error_with_partial():
    loop = [LLMResponse(tool_calls=[ToolCall(f"t{i}", "resolve_entity", {"kind": "section", "name": "10A"})],
                        model="claude-sonnet-5-5") for i in range(20)]
    evs = _run(Agent(Scripted(loop), ADMIN_CTX, str(uuid.uuid4())), "loop")
    err = [e for e in evs if e["type"] == "error"]
    assert err and err[0]["code"] == "budget_exceeded" and "tools_used" in err[0]["partial"]
    assert evs[-1]["type"] == "final"


def test_ctx_cannot_be_overridden_by_model_args():
    # a model trying to pass institute_id/role is rejected by extra='forbid'
    prov = Scripted([LLMResponse(tool_calls=[ToolCall("t1", "attendance_stats", {
        "scope": "section", "entity_id": str(B10A), "date_from": "2026-10-01", "date_to": "2026-10-31",
        "institute_id": str(DECOY), "role": "admin"})], model="claude-sonnet-5-5")])
    a = Agent(prov, VERMA_CTX, str(uuid.uuid4()))
    evs = _run(a, "x")
    tool_msg = [m for m in a.history if m["role"] == "tool"][0]
    assert json.loads(tool_msg["results"][0]["content"])["status"] == "invalid_args"
    assert not any(e["type"] == "stat_card" for e in evs)


def test_finalize_blocked_until_verified():
    a = Agent(Scripted([]), SHARMA_CTX, str(uuid.uuid4()))
    a.drafts["d"] = {"blueprint": {"subject": "Mathematics", "class_grade": 10, "total_marks": 2.0, "chapters": ["T"],
                                   "duration_min": 90},
                     "paper": {"total_marks": 2, "questions": []},
                     "summary": {"marks_sum_ok": False, "failed": [], "duplicates": []}}
    from agent.tools.paper import DraftArgs
    did = uuid.uuid4()
    a.drafts[str(did)] = a.drafts.pop("d")
    res = a._paper_tool("finalize_test", DraftArgs(draft_id=did), lambda e: None)
    assert res["status"] == "blocked"


# ---------------- providers (mocked) ----------------
def test_anthropic_provider_with_mocked_client():
    captured = {}

    class Msgs:
        def create(self, **kw):
            captured.update(kw)
            return SimpleNamespace(usage=SimpleNamespace(input_tokens=7, output_tokens=3), content=[
                SimpleNamespace(type="text", text="Dekh raha hoon"),
                SimpleNamespace(type="tool_use", id="tu1", name="resolve_entity", input={"kind": "teacher", "name": "Sharma"})])
    p = AnthropicProvider(client=SimpleNamespace(messages=Msgs()), model="claude-sonnet-5-5")
    r = p.complete("sys", [{"role": "user", "text": "hi"},
                           {"role": "assistant", "text": "", "tool_calls": [{"id": "a", "name": "x", "args": {}}]},
                           {"role": "tool", "results": [{"id": "a", "name": "x", "content": "{}"}]}],
                   [{"name": "x", "description": "d", "parameters": {"type": "object"}}])
    assert captured["model"] == "claude-sonnet-5-5" and captured["tools"][0]["input_schema"] == {"type": "object"}
    assert captured["messages"][2]["content"][0]["type"] == "tool_result"
    assert r.tool_calls[0].name == "resolve_entity" and r.tokens_in == 7 and r.text == "Dekh raha hoon"


def test_default_model_and_key_env_name():
    assert config.AGENT_MODEL == os.getenv("AGENT_MODEL", "claude-sonnet-5-5")
    assert config.ANTHROPIC_KEY_ENV == os.getenv("AGENT_ANTHROPIC_KEY_ENV", "A4AI_ANTHROPIC_API_KEY")


def test_gemini_schema_strips_unsupported_keys():
    s = _gemini_schema({"type": "object", "additionalProperties": False, "title": "X", "properties": {
        "a": {"anyOf": [{"type": "string", "format": "uuid"}, {"type": "null"}], "default": None}}})
    assert "additionalProperties" not in s and s["properties"]["a"] == {"type": "string", "format": "uuid"}


# ---------------- API: session isolation ----------------
@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("AGENT_DEV_AUTH", "1")
    monkeypatch.delenv("VERCEL", raising=False)
    from fastapi.testclient import TestClient
    from agent.api import app
    return TestClient(app)


def test_get_session_owner_or_same_institute_admin(client):
    with connect(readonly=False) as conn:
        sid = _session(conn, SHARMA_CTX)
    h = lambda u: {"x-agent-dev-user": str(u)}
    assert client.get(f"/agent/sessions/{sid}", headers=h(SHARMA)).status_code == 200
    assert client.get(f"/agent/sessions/{sid}", headers=h(ADMIN)).status_code == 200
    assert client.get(f"/agent/sessions/{sid}", headers=h(VERMA)).status_code == 404
    assert client.get(f"/agent/sessions/{sid}", headers=h(DECOY_ADMIN)).status_code == 404
    assert client.get(f"/agent/sessions/not-a-uuid", headers=h(SHARMA)).status_code == 404
    assert client.get(f"/agent/sessions/{sid}").status_code == 401


def test_dev_auth_refused_on_vercel(client, monkeypatch):
    monkeypatch.setenv("VERCEL", "1")
    with connect(readonly=False) as conn:
        sid = _session(conn, SHARMA_CTX)
    assert client.get(f"/agent/sessions/{sid}", headers={"x-agent-dev-user": str(SHARMA)}).status_code == 401


def test_approve_endpoint_replay_never_writes_twice(client):
    with connect(readonly=False) as conn:
        sid = _session(conn, SHARMA_CTX)
        did = str(uuid.uuid4())
        conn.execute("insert into agent_drafts(id, session_id, institute_id, user_id, paper) values (%s,%s,%s,%s,'{}')",
                     (did, sid, str(MAIN), str(SHARMA)))
        a = create_approval(conn, SHARMA_CTX, sid, "finalize_test", {"draft_id": did})
    h = {"x-agent-dev-user": str(SHARMA)}
    r1 = client.post("/agent/approve", json={"session_id": sid, "approval_id": a["approval_id"]}, headers=h).json()
    r2 = client.post("/agent/approve", json={"session_id": sid, "approval_id": a["approval_id"]}, headers=h).json()
    assert r1["replayed"] is False and r2["replayed"] is True and r1["status"] == r2["status"] == "finalized"
    other = client.post("/agent/approve", json={"session_id": sid, "approval_id": a["approval_id"]},
                        headers={"x-agent-dev-user": str(VERMA)})
    assert other.status_code == 404


def test_frac_shorthand_parses():  # regression: trig demo run 3 q7 ('\\frac54')
    from agent.verify import sym_equal
    assert sym_equal(r"$\frac54$", "sin(60*deg)**2+cos(45*deg)**2") is True
