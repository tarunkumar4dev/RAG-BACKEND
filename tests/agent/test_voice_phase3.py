"""Voice phase 3 — 'open this feature': registry shape, role gating, typed params, the open_feature tool through the
agent loop (ui_action navigate), the frontend copy is byte-identical (when the sibling checkout is present), and
every registry path exists in the frontend's App.tsx (same condition)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from _agent_ids import ADMIN, MAIN, SHARMA
from agent import routes
from agent.core import Agent
from agent.llm import LLMResponse, ToolCall
from agent.schemas import AgentEventAdapter, SessionCtx
from agent.tools.registry import TOOLS, active_tools, tool_schemas

ADMIN_CTX = SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN)
HOD_CTX = SessionCtx(user_id=SHARMA, role="hod", institute_id=MAIN, teacher_id=SHARMA)
TEACHER_CTX = SessionCtx(user_id=SHARMA, role="teacher", institute_id=MAIN, teacher_id=SHARMA)
UUID = "c1000000-0000-0000-0000-000000000001"

FRONTEND_CANDIDATES = [os.getenv("AGENT_FRONTEND_DIR") or "", str(Path.home() / "OneDrive" / "Desktop" / "product"),
                       str(Path(__file__).resolve().parents[3] / "product"), str(Path(__file__).resolve().parents[3] / "a4ai")]


def frontend_dir():
    for d in FRONTEND_CANDIDATES:
        if d and (Path(d) / "src" / "App.tsx").exists():
            return Path(d)
    return None


class Scripted:
    model = "claude-sonnet-5-5"

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False):
        self.calls.append(messages)
        return self.responses.pop(0) if self.responses else LLMResponse(text="ok", model=self.model)


# ---- registry ------------------------------------------------------------------------------------------------------
def test_registry_loads_and_validates():
    reg = routes.load()
    assert reg["version"] == 1 and len(reg["routes"]) >= 20
    for k in routes.keys():
        assert re.match(r"^[a-z0-9_.]+$", k)
    bad = json.loads(json.dumps(reg))
    bad["routes"]["x"] = {**reg["routes"]["notes"], "path": "/notes?tab=1"}
    with pytest.raises(AssertionError):
        routes.validate(bad)
    bad = json.loads(json.dumps(reg))
    bad["routes"]["student.profile"]["params"] = {}
    with pytest.raises(AssertionError):  # a :param without a typed param
        routes.validate(bad)


def test_role_gating():
    assert routes.role_allows("admin", "institute.dashboard")
    assert not routes.role_allows("teacher", "institute.dashboard")
    assert not routes.role_allows("hod", "institute.dashboard")
    assert routes.role_allows("hod", "hod.dashboard") and routes.role_allows("admin", "hod.dashboard")
    assert not routes.role_allows("teacher", "hod.dashboard")
    assert routes.role_allows("teacher", "dashboard.attendance") and routes.role_allows("teacher", "test_generator.check")
    assert set(routes.allowed_keys("admin")) == set(routes.keys())
    assert routes.allowed_keys("student") == []


def test_typed_params_only():
    ok, bad = routes.check_params("test_generator.check", {"test_id": UUID, "evil": "x"})
    assert ok == {"test_id": UUID} and bad == []
    ok, bad = routes.check_params("test_generator.check", {"test_id": "../../etc"})
    assert ok == {} and bad == ["test_id invalid"]
    ok, bad = routes.check_params("student.profile", {})
    assert bad == ["student_id required"]
    ok, bad = routes.check_params("practice", {"class": "10", "subject": "Science"})
    assert ok == {"class": "10", "subject": "Science"}
    ok, bad = routes.check_params("practice", {"class": "13", "subject": "<script>"})
    assert sorted(bad) == ["class invalid", "subject invalid"]


def test_open_feature_tool_results():
    r = routes.open_feature(TEACHER_CTX, routes.OpenFeatureArgs(route_key="dashboard.attendance"))
    assert r["status"] == "ok" and r["deep_link"] is False and r["note"]
    r = routes.open_feature(TEACHER_CTX, routes.OpenFeatureArgs(route_key="institute.dashboard"))
    assert r["status"] == "forbidden"
    r = routes.open_feature(ADMIN_CTX, routes.OpenFeatureArgs(route_key="nope.page"))
    assert r["status"] == "not_found" and "settings" in r["known_keys"]
    r = routes.open_feature(ADMIN_CTX, routes.OpenFeatureArgs(route_key="student.profile", params={"student_id": "x"}))
    assert r["status"] == "invalid_args"
    with pytest.raises(Exception):
        routes.OpenFeatureArgs(route_key="settings", extra=1)  # extra="forbid"
    with pytest.raises(Exception):
        routes.OpenFeatureArgs(route_key="Settings; drop")


def test_tool_is_registered_in_both_modes_and_in_the_schema():
    assert TOOLS["open_feature"].kind == "read" and TOOLS["open_feature"].fn is routes.open_feature
    assert active_tools(True)["open_feature"].fn is routes.open_feature
    s = next(t for t in tool_schemas() if t["name"] == "open_feature")
    assert "attendance kholo" in s["description"] and "dashboard.attendance" in s["description"]
    assert set(s["parameters"]["properties"]) == {"route_key", "params"} and "SessionCtx" not in json.dumps(s)


def test_agent_emits_ui_action_navigate_and_never_for_forbidden(admin_ctx, sharma_ctx):
    prov = Scripted([LLMResponse(tool_calls=[ToolCall("t1", "open_feature", {"route_key": "test_generator.check", "params": {"test_id": UUID}})]),
                     LLMResponse(text="Checker khol diya.")])
    evs = list(Agent(prov, admin_ctx, "s1").run("checker kholo"))
    for e in evs:
        AgentEventAdapter.validate_python(e)
    ua = [e for e in evs if e["type"] == "ui_action"]
    assert ua == [{"type": "ui_action", "seq": ua[0]["seq"], "request_id": evs[0]["request_id"], "action": "navigate",
                   "route_key": "test_generator.check", "params": {"test_id": UUID}, "label": "Copy checker"}]
    assert evs[-1]["usage"]["tools_used"] == ["open_feature"]
    prov = Scripted([LLMResponse(tool_calls=[ToolCall("t1", "open_feature", {"route_key": "institute.dashboard"})]),
                     LLMResponse(text="Ye page aapke liye nahi hai.")])
    evs = list(Agent(prov, sharma_ctx, "s2").run("institute dashboard kholo"))
    assert not [e for e in evs if e["type"] == "ui_action"]
    tool_msg = next(m for m in prov.calls[1] if m["role"] == "tool")
    assert json.loads(tool_msg["results"][0]["content"])["status"] == "forbidden"


def test_voice_chat_navigates(monkeypatch, tmp_path):
    from test_voice_phase1 import H, events
    from agent.store import MemoryStore, set_store
    from agent.voice.providers import FakeSTT, FakeTTS, set_providers
    monkeypatch.setenv("AGENT_DEV_AUTH", "1")
    monkeypatch.setenv("VOICE_ENABLED", "1")
    monkeypatch.delenv("VOICE_INSTITUTE_ALLOWLIST", raising=False)
    from fastapi.testclient import TestClient
    import agent.api as api_mod
    from agent import config
    from agent.rate_limit import reset
    monkeypatch.setattr(config, "TRACE_PATH", str(tmp_path / "t.jsonl"))
    set_store(MemoryStore())
    reset()
    prov = Scripted([LLMResponse(tool_calls=[ToolCall("t1", "open_feature", {"route_key": "dashboard.attendance"})]),
                     LLMResponse(text="Attendance page khol di.\nSPEECH: Attendance page खोल दी।")])
    monkeypatch.setattr(api_mod, "get_provider", lambda: prov)
    set_providers([FakeSTT()], FakeTTS())
    c = TestClient(api_mod.create_app({}))
    try:
        r = c.post("/agent/voice/chat", headers=H(SHARMA), json={"message": "attendance kholo"})
        evs = events(r)
        types = [e["type"] for e in evs]
        assert "ui_action" in types and "speech" in types
        rec = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text(encoding="utf-8").splitlines() if '"kind":"voice"' in l]
        assert rec[-1]["outcome"] == "navigated"
    finally:
        set_providers(None, None)
        set_store(None)
        reset()


# ---- cross-repo checks (skipped when the frontend checkout is not next to this one) ------------------------------
def test_frontend_copy_is_byte_identical():
    fe = frontend_dir()
    if fe is None:
        pytest.skip("frontend checkout not found (AGENT_FRONTEND_DIR)")
    mine = routes.REGISTRY_PATH.read_bytes().replace(b"\r\n", b"\n")
    theirs = (fe / "src" / "lib" / "agent" / "routeRegistry.json").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(mine).hexdigest() == hashlib.sha256(theirs).hexdigest(), "run `npm run agent:routes` in the frontend"


def test_every_registry_path_exists_in_app_tsx():
    fe = frontend_dir()
    if fe is None:
        pytest.skip("frontend checkout not found (AGENT_FRONTEND_DIR)")
    app = (fe / "src" / "App.tsx").read_text(encoding="utf-8")
    paths = set(re.findall(r'path=\s*"([^"]+)"', app))
    redirects = set(re.findall(r'path=\s*"([^"]+)"\s*element=\{<Navigate', app))
    for k, r in routes.load()["routes"].items():
        assert r["path"] in paths, f"{k}: {r['path']} is not a frontend route"
        assert r["path"] not in redirects, f"{k}: {r['path']} is a redirect, point at its target"
