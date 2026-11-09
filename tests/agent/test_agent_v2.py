"""API v2 + agent store (docs/AGENT_API_V2.md): store CRUD / isolation / pagination / dedupe / retention, chat flows
(session event, regenerate, edit, cancel), suggestions ranking + latency + scope, feedback, privacy. Seed (dev) mode."""
import json
import logging
import threading
import time
import uuid
from datetime import date

import pytest

from agent.llm import LLMResponse, ToolCall
from agent.store import MemoryStore, Owner, SQLiteStore, auto_title, normalize_query, set_store
from _agent_ids import ADMIN, B10A, B10B, B9A, DECOY_ADMIN, MAIN, SHARMA, VERMA

A = Owner(str(SHARMA), str(MAIN))
B = Owner(str(VERMA), str(MAIN))
A_OTHER_INST = Owner(str(SHARMA), "22222222-2222-2222-2222-222222222222")  # same user id, other institute
SECRET = "ZQX-private-7731"  # seeded marker: must never reach logs or traces


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    clock = Clock()
    s = MemoryStore(clock=clock) if request.param == "memory" else SQLiteStore(tmp_path / "s.db", clock=clock)
    s.clock_obj = clock
    yield s
    s.close()


def sid():
    return str(uuid.uuid4())


# ---------------- store ----------------
def test_store_crud_and_isolation(store):
    s1 = sid()
    store.create_session(A, s1, "teacher", "First")
    assert store.add_message(A, s1, "user", "hi") == 1
    assert store.add_message(A, s1, "assistant", "hello", events=[{"type": "final"}], llm=[{"role": "assistant"}]) == 2
    for other in (B, A_OTHER_INST):  # another user, or the same user id in another institute
        assert store.get_session(other, s1) is None
        assert store.list_sessions(other) == ([], None)
        assert store.list_messages(other, s1) == ([], None) and store.all_messages(other, s1) == []
        assert store.get_message(other, s1, 1) is None
        assert store.patch_session(other, s1, title="hacked") is None
        assert store.add_message(other, s1, "user", "x") is None
        assert store.set_feedback(other, s1, 2, "down", None) is False
        assert store.truncate(other, s1, 1) == 0
        assert store.delete_session(other, s1) is False
    got = store.get_session(A, s1)
    assert (got["title"], got["message_count"], got["preview"], got["pinned"]) == ("First", 2, "hello", False)
    assert store.patch_session(A, s1, title="  Renamed   chat ", pinned=True)["title"] == "Renamed chat"
    store.set_auto_title(A, s1, "auto")  # never overwrites a user title
    assert store.get_session(A, s1)["title"] == "Renamed chat"
    msgs, _ = store.list_messages(A, s1)
    assert [(m["seq"], m["role"], m["text"]) for m in msgs] == [(1, "user", "hi"), (2, "assistant", "hello")]
    assert msgs[1]["events"] == [{"type": "final"}]
    assert store.set_feedback(A, s1, 1, "up", None) is False  # user message: no feedback
    assert store.set_feedback(A, s1, 2, "up", "good") and store.set_feedback(A, s1, 2, "down", None)
    assert store.get_feedback(A, s1) == [{"message_seq": 2, "rating": "down", "note": None}]
    assert store.truncate(A, s1, 2) == 1 and store.get_feedback(A, s1) == []
    assert store.delete_session(A, s1) is True and store.get_session(A, s1) is None
    assert store.all_messages(A, s1) == []


def test_store_session_pagination_search_and_pinned(store):
    ids = []
    for i in range(25):
        store.clock_obj.t += 1
        ids.append(sid())
        store.create_session(A, ids[-1], "teacher", f"chat {i}")
        store.add_message(A, ids[-1], "user", "ECE 1 attendance" if i % 5 == 0 else f"q {i}")
    store.create_session(B, sid(), "teacher", "chat of B")
    seen, cur, pages = [], None, 0
    while True:
        items, cur = store.list_sessions(A, cursor=cur, limit=10)
        seen += [x["id"] for x in items]
        pages += 1
        if cur is None:
            break
    assert pages == 3 and seen == list(reversed(ids))  # newest first, no duplicates, nothing skipped
    hits, _ = store.list_sessions(A, q="ece 1", limit=50)
    assert len(hits) == 5
    assert [x["title"] for x in store.list_sessions(A, q="CHAT 24")[0]] == ["chat 24"]
    store.patch_session(A, ids[3], pinned=True)
    assert [x["id"] for x in store.list_sessions(A, pinned=True)[0]] == [ids[3]]
    assert len(store.list_sessions(A, pinned=False, limit=50)[0]) == 24
    with pytest.raises(ValueError):
        store.list_sessions(A, cursor="not-a-cursor")


def test_store_message_pagination(store):
    s1 = sid()
    store.create_session(A, s1, "teacher", "t")
    for i in range(7):
        store.add_message(A, s1, "user" if i % 2 == 0 else "assistant", f"m{i}")
    a, cur = store.list_messages(A, s1, limit=3)
    b, cur2 = store.list_messages(A, s1, cursor=cur, limit=3)
    c, cur3 = store.list_messages(A, s1, cursor=cur2, limit=3)
    assert [m["seq"] for m in a + b + c] == [1, 2, 3, 4, 5, 6, 7] and cur3 is None


def test_store_query_dedupe_and_clear(store):
    for t in ("Meri  classes", "  meri classes ", "MERI CLASSES"):
        store.clock_obj.t += 10
        store.record_query(A, t)
    store.clock_obj.t += 10
    store.record_query(A, "ECE 1 ki attendance")
    store.record_query(B, "meri classes")
    rows = store.list_queries(A)
    assert [(r["text"], r["use_count"]) for r in rows] == [("ECE 1 ki attendance", 1), ("MERI CLASSES", 3)]
    assert [r["text"] for r in store.list_queries(A, q="ece")] == ["ECE 1 ki attendance"]
    assert store.delete_query(B, rows[0]["id"]) is False  # not B's
    assert store.delete_query(A, rows[0]["id"]) is True
    assert store.list_queries(A_OTHER_INST) == []
    assert store.clear_queries(A) == 1 and store.list_queries(A) == [] and len(store.list_queries(B)) == 1
    assert normalize_query("  A   b\tC ") == "a b c"


def test_store_retention_purge(store):
    old, new = sid(), sid()
    store.create_session(A, old, "teacher", "old")
    store.add_message(A, old, "assistant", "x")
    store.set_feedback(A, old, 1, "up", None)
    store.record_query(A, "old query")
    store.clock_obj.t += 91 * 86400
    store.create_session(A, new, "teacher", "new")
    store.record_query(A, "new query")
    assert store.purge() == {"sessions": 1, "queries": 1}
    assert store.get_session(A, old) is None and store.all_messages(A, old) == [] and store.get_feedback(A, old) == []
    assert store.get_session(A, new) is not None and [q["text"] for q in store.list_queries(A)] == ["new query"]


def test_sqlite_store_is_wal_and_persists(tmp_path):
    p = tmp_path / "agent_store.db"
    s = SQLiteStore(p)
    s.create_session(A, "11111111-1111-1111-1111-111111111111", "teacher", "persist")
    assert s._q("pragma journal_mode")[0][0] == "wal"
    s.close()
    s2 = SQLiteStore(p)
    assert s2.get_session(A, "11111111-1111-1111-1111-111111111111")["title"] == "persist"
    s2.close()


def test_auto_title_deterministic():
    assert auto_title("  Meri   classes  dikhao ") == "Meri classes dikhao"
    long = "Sharma ma'am ki is mahine ki attendance kya hai aur pichhle mahine se kitni kam hai?"
    t = auto_title(long)
    assert t == "Sharma ma'am ki is mahine ki attendance kya…" and len(t) <= 48 and t == auto_title(long)
    assert len(auto_title("x" * 100)) == 48 and auto_title("") == "New chat"


# ---------------- API (seed mode, dev header identity) ----------------
class Scripted:
    model = "claude-sonnet-5-5"

    def __init__(self, responses=(), default="Theek hai."):
        self.responses, self.calls, self.default = list(responses), [], default

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False):
        self.calls.append([dict(m) for m in messages])
        r = self.responses.pop(0) if self.responses else self.default
        return LLMResponse(text=r, model=self.model) if isinstance(r, str) else r


@pytest.fixture()
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_DEV_AUTH", "1")
    monkeypatch.delenv("VERCEL", raising=False)
    from fastapi.testclient import TestClient
    import agent.api as api_mod
    from agent import config
    from agent.suggest import ENTITIES
    monkeypatch.setattr(config, "TRACE_PATH", str(tmp_path / "trace.jsonl"))
    st = MemoryStore()
    set_store(st)
    ENTITIES.clear()
    prov = Scripted()
    monkeypatch.setattr(api_mod, "get_provider", lambda: prov)
    yield TestClient(api_mod.create_app({})), prov, st, tmp_path / "trace.jsonl"
    set_store(None)


def H(u):
    return {"x-agent-dev-user": str(u)}


def events(resp):
    return [json.loads(l[6:]) for l in resp.text.splitlines() if l.startswith("data: ")]


def chat(c, who, **body):
    r = c.post("/agent/chat", json=body, headers=H(who))
    return r, (events(r) if r.status_code == 200 else [])


def test_session_event_first_title_and_replay(api):
    c, prov, st, _ = api
    prov.responses = ["Aapki do classes hain."]
    r, evs = chat(c, SHARMA, message="Meri   classes kaun si hain?")
    assert r.status_code == 200 and r.headers["x-agent-request-id"] == evs[0]["request_id"]
    assert evs[0]["type"] == "session" and evs[0]["seq"] == 0 and evs[0]["title"] == "Meri classes kaun si hain?"
    assert evs[-1]["type"] == "final" and evs[-1]["cancelled"] is False and [e["seq"] for e in evs] == list(range(len(evs)))
    s = evs[0]["session_id"]
    lst = c.get("/agent/sessions", headers=H(SHARMA)).json()
    assert lst["items"][0]["id"] == s and lst["items"][0]["message_count"] == 2
    assert lst["items"][0]["preview"] == "Aapki do classes hain." and lst["next_cursor"] is None
    assert lst["sessions"][0]["session_id"] == s  # 1.1 alias
    m = c.get(f"/agent/sessions/{s}/messages", headers=H(SHARMA)).json()
    assert [(x["seq"], x["role"], x["text"]) for x in m["items"]] == \
        [(1, "user", "Meri   classes kaun si hain?"), (2, "assistant", "Aapki do classes hain.")]  # as typed
    assert m["items"][1]["events"] == evs  # exact replay
    assert c.get(f"/agent/sessions/{s}", headers=H(SHARMA)).json()["turns"] == [evs]  # 1.1 view
    # follow-up in the same session: history comes from the store
    prov.responses = ["Haan."]
    _, evs2 = chat(c, SHARMA, message="aur?", session_id=s)
    assert evs2[0] == {**evs2[0], "type": "session", "session_id": s, "title": "Meri classes kaun si hain?"}
    assert [x.get("text") for x in prov.calls[-1]] == ["Meri   classes kaun si hain?", "Aapki do classes hain.",
                                                       "aur?"]


def test_v2_endpoints_are_owner_only_404(api):
    c, prov, st, _ = api
    _, evs = chat(c, SHARMA, message="hello")
    s = evs[0]["session_id"]
    for who in (VERMA, ADMIN, DECOY_ADMIN):  # other teacher, same-institute admin, other institute
        assert c.get(f"/agent/sessions/{s}/messages", headers=H(who)).status_code == 404
        assert c.patch(f"/agent/sessions/{s}", json={"pinned": True}, headers=H(who)).status_code == 404
        assert c.delete(f"/agent/sessions/{s}", headers=H(who)).status_code == 404
        assert c.post("/agent/feedback", json={"session_id": s, "message_seq": 2, "rating": "up"},
                      headers=H(who)).status_code == 404
        assert c.post("/agent/chat", json={"session_id": s, "message": "x"}, headers=H(who)).status_code == 404
        assert c.post("/agent/chat", json={"session_id": s, "regenerate_from_seq": 2}, headers=H(who)).status_code == 404
        assert c.get("/agent/sessions", headers=H(who)).json()["items"] == []
        assert c.get("/agent/history/queries", headers=H(who)).json() == []
    qid = c.get("/agent/history/queries", headers=H(SHARMA)).json()[0]["id"]
    assert c.delete(f"/agent/history/queries/{qid}", headers=H(VERMA)).status_code == 404
    assert c.get("/agent/sessions/not-a-uuid/messages", headers=H(SHARMA)).status_code == 404
    assert c.get("/agent/sessions", headers=H(SHARMA), params={"cursor": "garbage"}).status_code == 422
    assert c.get("/agent/sessions").status_code == 401
    # owner can patch, then delete
    p = c.patch(f"/agent/sessions/{s}", json={"title": "Mine", "pinned": True}, headers=H(SHARMA)).json()
    assert (p["title"], p["pinned"]) == ("Mine", True)
    assert c.patch(f"/agent/sessions/{s}", json={}, headers=H(SHARMA)).status_code == 422
    assert c.delete(f"/agent/sessions/{s}", headers=H(SHARMA)).status_code == 204
    assert c.get(f"/agent/sessions/{s}/messages", headers=H(SHARMA)).status_code == 404


def test_regenerate_flow(api):
    c, prov, st, _ = api
    prov.responses = ["A1", "A2"]
    _, evs = chat(c, SHARMA, message="q1")
    s = evs[0]["session_id"]
    chat(c, SHARMA, message="q2", session_id=s)
    assert c.post("/agent/feedback", json={"session_id": s, "message_seq": 4, "rating": "down", "note": "galat"},
                  headers=H(SHARMA)).json() == {"ok": True}
    prov.responses = ["A2 better"]
    r, evs = chat(c, SHARMA, session_id=s, regenerate_from_seq=4)
    assert r.status_code == 200 and evs[0]["type"] == "session"
    assert [x.get("text") for x in prov.calls[-1]] == ["q1", "A1", "q2"]  # old answer not in the history
    items = c.get(f"/agent/sessions/{s}/messages", headers=H(SHARMA)).json()["items"]
    assert [(x["seq"], x["role"], x["text"]) for x in items] == \
        [(1, "user", "q1"), (2, "assistant", "A1"), (3, "user", "q2"), (4, "assistant", "A2 better")]
    assert st.get_feedback(Owner(str(SHARMA), str(MAIN)), s) == []  # feedback of the replaced answer is gone
    # regenerate by pointing at the user message works too; queries are not re-recorded
    prov.responses = ["A2 third"]
    chat(c, SHARMA, session_id=s, regenerate_from_seq=3)
    assert c.get(f"/agent/sessions/{s}/messages", headers=H(SHARMA)).json()["items"][-1]["text"] == "A2 third"
    assert {q["text"]: q["use_count"] for q in c.get("/agent/history/queries", headers=H(SHARMA)).json()} == \
        {"q1": 1, "q2": 1}
    assert c.post("/agent/chat", json={"session_id": s, "regenerate_from_seq": 99}, headers=H(SHARMA)).status_code == 404
    assert c.post("/agent/chat", json={"regenerate_from_seq": 2}, headers=H(SHARMA)).status_code == 422


def test_edit_flow_and_retitle(api):
    c, prov, st, _ = api
    prov.responses = ["A1", "A2"]
    _, evs = chat(c, SHARMA, message="ECE 1 attendance")
    s = evs[0]["session_id"]
    chat(c, SHARMA, message="aur marks?", session_id=s)
    prov.responses = ["A2 edited"]
    r, evs = chat(c, SHARMA, session_id=s, edit_of_seq=3, message="aur 10B ke marks?")
    assert r.status_code == 200
    assert [x.get("text") for x in prov.calls[-1]] == ["ECE 1 attendance", "A1", "aur 10B ke marks?"]
    items = c.get(f"/agent/sessions/{s}/messages", headers=H(SHARMA)).json()["items"]
    assert [(x["seq"], x["text"]) for x in items] == [(1, "ECE 1 attendance"), (2, "A1"), (3, "aur 10B ke marks?"),
                                                    (4, "A2 edited")]
    # editing the first message recomputes the automatic title ...
    prov.responses = ["B1"]
    _, evs = chat(c, SHARMA, session_id=s, edit_of_seq=1, message="10A attendance")
    assert evs[0]["title"] == "10A attendance" and len(c.get(f"/agent/sessions/{s}/messages", headers=H(SHARMA))
                                                       .json()["items"]) == 2
    # ... but never a title the user set
    c.patch(f"/agent/sessions/{s}", json={"title": "My title"}, headers=H(SHARMA))
    _, evs = chat(c, SHARMA, session_id=s, edit_of_seq=1, message="9A attendance")
    assert evs[0]["title"] == "My title"
    assert c.post("/agent/chat", json={"session_id": s, "edit_of_seq": 2, "message": "x"},
                  headers=H(SHARMA)).status_code == 422  # assistant message
    assert c.post("/agent/chat", json={"session_id": s, "edit_of_seq": 1}, headers=H(SHARMA)).status_code == 422
    assert c.post("/agent/chat", json={"session_id": s, "edit_of_seq": 1, "regenerate_from_seq": 2, "message": "x"},
                  headers=H(SHARMA)).status_code == 422
    texts = {q["text"] for q in c.get("/agent/history/queries", headers=H(SHARMA)).json()}
    assert {"10A attendance", "9A attendance", "aur 10B ke marks?"} <= texts


class Blocking:
    """Provider whose first call blocks until released (the run must still stop on cancel)."""
    model = "claude-sonnet-5-5"

    def __init__(self):
        self.started, self.release = threading.Event(), threading.Event()

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False):
        self.started.set()
        self.release.wait(10)
        return LLMResponse(text="late answer", model=self.model)


def test_cancel_stops_run_and_emits_final_cancelled(api, monkeypatch):
    c, _, st, _ = api
    import agent.api as api_mod
    from agent.runs import RUNS
    prov = Blocking()
    monkeypatch.setattr(api_mod, "get_provider", lambda: prov)
    out = {}
    t = threading.Thread(target=lambda: out.update(r=c.post("/agent/chat", json={"message": "slow one"},
                                                            headers=H(SHARMA))))
    t.start()
    try:
        assert prov.started.wait(5)
        rid = next(iter(RUNS._runs))
        assert c.post(f"/agent/chat/{rid}/cancel", headers=H(VERMA)).status_code == 404  # not yours
        assert c.post("/agent/chat/nope/cancel", headers=H(SHARMA)).status_code == 404
        t0 = time.time()
        r = c.post(f"/agent/chat/{rid}/cancel", headers=H(SHARMA))
        assert r.status_code == 202 and r.json() == {"request_id": rid, "cancelled": True}
        t.join(5)
        assert not t.is_alive() and time.time() - t0 < 2
    finally:
        prov.release.set()
    evs = events(out["r"])
    assert evs[0]["type"] == "session" and evs[-1]["type"] == "final" and evs[-1]["cancelled"] is True
    assert not [e for e in evs if e["type"] in ("error", "text_delta")]
    assert rid not in RUNS._runs and c.post(f"/agent/chat/{rid}/cancel", headers=H(SHARMA)).status_code == 404
    s = evs[0]["session_id"]
    items = c.get(f"/agent/sessions/{s}/messages", headers=H(SHARMA)).json()["items"]
    assert [x["role"] for x in items] == ["user", "assistant"] and items[1]["events"][-1]["cancelled"] is True
    llm = st.get_message(Owner(str(SHARMA), str(MAIN)), s, 2)["llm"]
    assert llm[-1]["role"] == "assistant" and not llm[-1]["tool_calls"]  # next turn gets a valid history


def test_cancel_during_tool_drops_dangling_tool_call(api, monkeypatch):
    from agent.core import Agent
    from agent.tools import registry
    from agent.schemas import SessionCtx
    started, release = threading.Event(), threading.Event()

    def slow(ctx, args):
        started.set()
        release.wait(10)
        return {"status": "ok"}
    tools = dict(registry.TOOLS)
    tools["my_scope"] = registry.ToolSpec("my_scope", "x", registry.insights.MyScopeArgs, "read", slow)
    ev = threading.Event()
    prov = Scripted([LLMResponse(tool_calls=[ToolCall("t1", "my_scope", {})], model="m")])
    ctx = SessionCtx(user_id=SHARMA, role="teacher", institute_id=MAIN, teacher_id=SHARMA)
    a = Agent(prov, ctx, sid(), tools=tools, cancel_event=ev, session_title="t")
    out = []
    th = threading.Thread(target=lambda: out.extend(a.run("meri classes")))
    th.start()
    try:
        assert started.wait(5)
        ev.set()
        th.join(3)
        assert not th.is_alive()
    finally:
        release.set()
    assert out[-1]["cancelled"] is True
    assert all(not m.get("tool_calls") for m in a.history if m["role"] == "assistant")


# ---------------- suggestions ----------------
def test_suggestion_ranking_prefix_then_substring_by_recency_and_use(api):
    from agent.suggest import EntityCache, suggest
    from agent.schemas import SessionCtx
    clock = Clock()
    st = MemoryStore(clock=clock)
    o = Owner(str(SHARMA), str(MAIN))
    for _ in range(3):
        st.record_query(o, "ECE 1 ki attendance")       # 3 uses, 20 days old -> 3 * 0.5^(20/14) = 1.11
    for _ in range(2):
        st.record_query(o, "ece 3 marks")               # 2 uses, 20 days old -> 0.74
    clock.t += 20 * 86400
    st.record_query(o, "ECE 2 marks")                   # 1 use, today -> 1.0
    st.record_query(o, "Meri ECE class")                # substring match only
    st.record_query(o, "Verma sir ki attendance")       # no match
    ctx = SessionCtx(user_id=SHARMA, role="teacher", institute_id=MAIN, teacher_id=SHARMA)
    ents = [{"kind": "batch", "id": "b1", "name": "ECE-1 Lab", "student_count": 22},
            {"kind": "teacher", "id": "t1", "name": "Ecevarma Rao"}]
    got = suggest(st, ctx, "ece", limit=8, loader=lambda c: ents, cache=EntityCache())
    assert [(g["type"], g["text"]) for g in got] == [
        ("recent", "ECE 1 ki attendance"), ("recent", "ECE 2 marks"), ("recent", "ece 3 marks"),
        ("recent", "Meri ECE class"),
        ("entity", "ECE-1 Lab ki is mahine ki attendance"), ("entity", "Ecevarma Rao ki is hafte ki attendance")]
    assert got[0]["hint"] == "Recent · 3×" and got[4]["hint"] == "Batch · 22 students" and got[5]["hint"] == "Teacher"
    # ece 3 marks: 2 uses, 20 days old -> 0.74 < ECE 2 marks (1.0); after one more use today it moves up
    st.record_query(o, "ece 3 marks")
    got = suggest(st, ctx, "ece", limit=3, loader=lambda c: ents, cache=EntityCache())
    assert [g["text"] for g in got] == ["ece 3 marks", "ECE 1 ki attendance", "ECE 2 marks"]
    # empty q: up to 4 recent then role templates; duplicates removed
    got = suggest(st, ctx, "", limit=8, loader=lambda c: ents, cache=EntityCache())
    assert [g["type"] for g in got] == ["recent"] * 4 + ["template"] * 4
    assert len({normalize_query(g["text"]) for g in got}) == len(got)


def test_suggestions_entities_only_from_scope_and_templates_by_role(api):
    c, _, _, _ = api
    s = lambda who, q: c.get("/agent/suggestions", params={"q": q}, headers=H(who)).json()
    ent = lambda who, q: [x["text"] for x in s(who, q) if x["type"] == "entity"]
    assert ent(SHARMA, "10") == ["10A ki is mahine ki attendance", "10B ki is mahine ki attendance"]
    assert ent(SHARMA, "9a") == []                       # 9A is Verma's batch, not Sharma's
    assert ent(ADMIN, "9a") == ["9A ki is mahine ki attendance"]
    assert ent(SHARMA, "verma") == [] and ent(ADMIN, "verma") == ["Rakesh Verma ki is hafte ki attendance"]
    assert ent(DECOY_ADMIN, "10a") == ["10A ki is mahine ki attendance"]  # its own 10A only
    hint = next(x["hint"] for x in s(ADMIN, "10a") if x["type"] == "entity")
    assert hint == "Batch · 20 students"
    assert "Anil" not in json.dumps(s(ADMIN, "sharma")) and "Anil Sharma" in json.dumps(s(DECOY_ADMIN, "sharma"))
    assert s(SHARMA, "")[0]["text"] == "Meri classes dikhao" and s(ADMIN, "")[0]["type"] == "template"
    assert len(c.get("/agent/suggestions", params={"limit": 3}, headers=H(ADMIN)).json()) == 3


def test_suggestions_p95_under_150ms(api):
    c, _, st, _ = api
    o = Owner(str(ADMIN), str(MAIN))
    for i in range(300):
        st.record_query(o, f"query number {i} about 10A attendance")
    c.get("/agent/suggestions", params={"q": "1"}, headers=H(ADMIN))  # warm the 5-minute entity cache
    lat = []
    for q in ["1", "10", "10a", "query", "sharma", "att", "x", "9", "query number 2", "s"] * 5:
        t0 = time.perf_counter()
        r = c.get("/agent/suggestions", params={"q": q}, headers=H(ADMIN))
        lat.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
    p95 = sorted(lat)[int(len(lat) * 0.95) - 1]
    assert p95 < 150, f"p95 {p95:.1f} ms"


def test_identity_cache_ttl_and_no_failure_caching(monkeypatch):
    from fastapi import HTTPException
    from agent.auth import cached_identity, clear_identity_cache
    from agent.schemas import SessionCtx
    clear_identity_cache()
    built = []
    ok = lambda: built.append(1) or SessionCtx(user_id=SHARMA, role="teacher", institute_id=MAIN)

    def bad():
        built.append(1)
        raise HTTPException(403, "staff only")
    cached_identity(("t", "a"), ok)
    cached_identity(("t", "a"), ok)
    assert len(built) == 1
    for _ in range(2):
        with pytest.raises(HTTPException):
            cached_identity(("t", "b"), bad)
    assert len(built) == 3  # a refusal is never cached
    monkeypatch.setenv("AGENT_IDENTITY_CACHE_S", "0")
    cached_identity(("t", "a"), ok)
    assert len(built) == 4
    clear_identity_cache()


def test_entity_cache_ttl():
    from agent.suggest import EntityCache
    from agent.schemas import SessionCtx
    t = [0.0]
    calls = []
    cache = EntityCache(ttl_s=300, clock=lambda: t[0])
    ctx = SessionCtx(user_id=SHARMA, role="teacher", institute_id=MAIN, teacher_id=SHARMA)
    loader = lambda c: calls.append(1) or [{"kind": "batch", "id": "x", "name": "10A"}]
    cache.get(ctx, loader); cache.get(ctx, loader)
    t[0] = 299
    cache.get(ctx, loader)
    t[0] = 301
    cache.get(ctx, loader)
    assert len(calls) == 2


def test_history_queries_endpoints(api):
    c, prov, _, _ = api
    for m in ("Meri classes", "meri   CLASSES", "10A attendance"):
        chat(c, SHARMA, message=m)
    q = c.get("/agent/history/queries", headers=H(SHARMA)).json()
    assert [(x["text"], x["use_count"]) for x in q] == [("10A attendance", 1), ("meri CLASSES", 2)]
    assert [x["text"] for x in c.get("/agent/history/queries", params={"q": "MERI"}, headers=H(SHARMA)).json()] == \
        ["meri CLASSES"]
    assert c.delete(f"/agent/history/queries/{q[0]['id']}", headers=H(SHARMA)).status_code == 204
    assert c.delete(f"/agent/history/queries/{q[0]['id']}", headers=H(SHARMA)).status_code == 404
    assert c.delete("/agent/history/queries", headers=H(SHARMA)).status_code == 204
    assert c.get("/agent/history/queries", headers=H(SHARMA)).json() == []
    assert len(c.get("/agent/sessions", headers=H(SHARMA)).json()["items"]) == 3  # sessions are separate


def test_feedback_endpoint(api):
    c, _, st, _ = api
    _, evs = chat(c, SHARMA, message="hi")
    s = evs[0]["session_id"]
    fb = lambda seq, rating="up", **kw: c.post("/agent/feedback", json={"session_id": s, "message_seq": seq,
                                                                        "rating": rating, **kw}, headers=H(SHARMA))
    assert fb(2).json() == {"ok": True} and fb(2, "down", note="numbers galat").status_code == 200
    assert st.get_feedback(Owner(str(SHARMA), str(MAIN)), s) == [{"message_seq": 2, "rating": "down",
                                                                   "note": "numbers galat"}]
    assert fb(1).status_code == 404 and fb(9).status_code == 404 and fb(2, "meh").status_code == 422
    assert fb(2, note="x" * 1001).status_code == 422


def test_startup_purges_old_rows(monkeypatch):
    import agent.api as api_mod
    clock = Clock()
    st = MemoryStore(clock=clock)
    st.create_session(A, sid(), "teacher", "ancient")
    st.record_query(A, "ancient query")
    clock.t += 91 * 86400
    set_store(st)
    try:
        api_mod.create_app({})
        assert st.list_sessions(A) == ([], None) and st.list_queries(A) == []
    finally:
        set_store(None)


# ---------------- privacy ----------------
def test_no_query_text_in_logs_or_traces(api, caplog):
    c, prov, st, trace = api
    caplog.set_level(logging.DEBUG)
    prov.responses = [LLMResponse(tool_calls=[ToolCall("t1", "resolve_entity", {"kind": "section", "name": "10A"})],
                                  model="claude-sonnet-5-5"), f"Answer {SECRET}"]
    _, evs = chat(c, SHARMA, message=f"10A {SECRET} ki attendance")
    s = evs[0]["session_id"]
    c.get("/agent/suggestions", params={"q": SECRET}, headers=H(SHARMA))
    c.get("/agent/sessions", params={"q": SECRET}, headers=H(SHARMA))
    c.get("/agent/history/queries", params={"q": SECRET}, headers=H(SHARMA))
    c.patch(f"/agent/sessions/{s}", json={"title": f"t {SECRET}"}, headers=H(SHARMA))
    c.post("/agent/feedback", json={"session_id": s, "message_seq": 2, "rating": "up", "note": SECRET}, headers=H(SHARMA))
    chat(c, SHARMA, session_id=s, edit_of_seq=1, message=f"edit {SECRET}")
    server_logs = [r for r in caplog.records if not r.name.startswith("httpx")]  # httpx = the test client itself
    assert server_logs is not None and not [r for r in server_logs if SECRET in r.getMessage()]
    body = trace.read_text(encoding="utf-8")
    assert '"tool":"resolve_entity"' in body and SECRET not in body and "10A" not in body
    assert SECRET in json.dumps(st.list_queries(Owner(str(SHARMA), str(MAIN))))  # stored for its owner only


def test_access_log_redacts_agent_query_strings():
    from agent.api import RedactAgentQueryStrings
    rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                            ("1.2.3.4:5", "GET", f"/agent/suggestions?q={SECRET}", "1.1", 200), None)
    assert RedactAgentQueryStrings().filter(rec) and SECRET not in rec.getMessage()
    assert "/agent/suggestions" in rec.getMessage()
    other = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                              ("1.2.3.4:5", "GET", "/health?x=1", "1.1", 200), None)
    RedactAgentQueryStrings().filter(other)
    assert "/health?x=1" in other.getMessage()


# ---------------- new tools on the seed (events + consistency) ----------------
def test_seed_my_scope_and_comparison(admin_ctx, sharma_ctx, verma_ctx, decoy_ctx):
    from agent.tools.insights import (AttendanceStatsArgs, MyScopeArgs, SectionComparisonArgs, attendance_stats,
                                      my_scope, section_comparison)
    sc = my_scope(sharma_ctx, MyScopeArgs())
    assert [(b["name"], b["student_count"], b["role"], b["subjects"]) for b in sc["batches"]] == \
        [("10A", 20, "teacher", ["Mathematics"]), ("10B", 20, "teacher", ["Mathematics"])]
    assert [(b["name"], b["subjects"]) for b in my_scope(verma_ctx, MyScopeArgs())["batches"]] == \
        [("10B", ["Science"]), ("9A", ["Science"])]
    adm = my_scope(admin_ctx, MyScopeArgs())
    assert [(b["name"], b["subjects"]) for b in adm["batches"]] == \
        [("10A", ["Mathematics"]), ("10B", ["Mathematics", "Science"]), ("9A", ["Science"])]
    assert adm["students_in_scope"] == 60 and [b["name"] for b in my_scope(decoy_ctx, MyScopeArgs())["batches"]] == ["10A"]
    oct_ = dict(date_from=date(2026, 10, 1), date_to=date(2026, 10, 31))
    r = section_comparison(admin_ctx, SectionComparisonArgs(metric="attendance", **oct_))
    row = {x["name"]: x for x in r["rows"]}
    assert (row["10A"]["pct"], row["10A"]["marked"]) == (80.7, 540)  # G2
    for x in r["rows"]:
        s = attendance_stats(admin_ctx, AttendanceStatsArgs(scope="section", entity_id=x["id"], **oct_))
        assert (s["pct"], s["marked"], s["sessions"]) == (x["pct"], x["marked"], x["sessions"])
    assert r["lowest"] == [r["rows"][-1]["name"]] and r["rows"][0]["rank"] == 1
    assert {x["name"] for x in section_comparison(sharma_ctx, SectionComparisonArgs(**oct_))["rows"]} == {"10A", "10B"}
    m = section_comparison(admin_ctx, SectionComparisonArgs(metric="marks", date_from=date(2026, 6, 1),
                                                            date_to=date(2026, 11, 5)))
    assert all(x["pct"] is None or 0 <= x["pct"] <= 100 for x in m["rows"]) and m["highest"]


def test_seed_ambiguous_student_chips_have_hints(admin_ctx):
    from agent.core import Agent
    a = Agent(Scripted([LLMResponse(tool_calls=[ToolCall("t1", "resolve_entity",
                                                         {"kind": "student", "name": "Rahul Kumar"})], model="m")]),
              admin_ctx, sid())
    evs = list(a.run("Rahul Kumar ki attendance"))
    chips = next(e for e in evs if e["type"] == "clarify")["chips"]
    assert sorted((c["label"], c["hint"]) for c in chips) == [("Rahul Kumar (10A)", "10A · roll 01"),
                                                             ("Rahul Kumar (10B)", "10B · roll 01")]
    assert all(c["value"] == c["label"] for c in chips)  # labels differ -> no id needed


def test_new_tools_emit_table_and_bar_chart(admin_ctx):
    from agent.core import Agent
    prov = Scripted([LLMResponse(tool_calls=[ToolCall("t1", "section_comparison", {
        "metric": "attendance", "date_from": "2026-10-01", "date_to": "2026-10-31"})], model="m"),
        LLMResponse(tool_calls=[ToolCall("t2", "my_scope", {})], model="m"), "Sabse kam 9A ki hai."])
    evs = list(Agent(prov, admin_ctx, sid()).run("Sabse kam attendance kis class ki hai?"))
    tables = [e for e in evs if e["type"] == "table"]
    chart = next(e for e in evs if e["type"] == "chart")
    assert tables[0]["source"] == "tool:section_comparison" and tables[0]["columns"][:3] == ["#", "Class", "%"]
    assert chart["kind"] == "bar" and [p[0] for p in chart["series"][0]["points"]] == [r[1] for r in tables[0]["rows"]]
    assert tables[1]["source"] == "tool:my_scope" and [r[0] for r in tables[1]["rows"]] == ["10A", "10B", "9A"]


def test_empty_scope_has_no_table_and_honest_message():
    from agent.core import Agent
    from agent.schemas import SessionCtx
    from agent.tools.insights import MyScopeArgs, my_scope
    nobody = SessionCtx(user_id=uuid.UUID("aaaaaaaa-0000-0000-0000-0000000000ff"), role="teacher", institute_id=MAIN,
                        teacher_id=uuid.UUID("aaaaaaaa-0000-0000-0000-0000000000ff"))
    r = my_scope(nobody, MyScopeArgs())
    assert r["empty"] and r["batches"] == [] and "admin" in r["message"]
    evs = list(Agent(Scripted([LLMResponse(tool_calls=[ToolCall("t", "my_scope", {})], model="m")]), nobody, sid())
               .run("meri classes"))
    assert not [e for e in evs if e["type"] == "table"]


def test_system_prompt_routing_rules(admin_ctx):
    from agent.core import system_prompt
    p = system_prompt(admin_ctx)
    for s in ("my_scope", "section_comparison(metric='attendance')", "NEVER use low_attendance_list", "(id <uuid>)",
              "admin or HOD can assign"):
        assert s in p
