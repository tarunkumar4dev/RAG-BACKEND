"""Prompt caching (phase 4): the request prefix (tools -> system) is byte-identical across users, turns and
steps of one turn, so Anthropic's prefix cache can hit; breakpoints sit where the docs say. No API call: this
checks the exact Messages API kwargs (AnthropicProvider.request). Live hit rates: scripts/cache_probe.py."""
from __future__ import annotations

import json
import uuid

from _agent_ids import ADMIN, DECOY, DECOY_ADMIN, MAIN, SHARMA
from agent import config
from agent.core import active_tools, system_prompt
from agent.llm import AnthropicProvider
from agent.schemas import SessionCtx
from agent.tools.registry import tool_schemas

A = SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN)
B = SessionCtx(user_id=DECOY_ADMIN, role="admin", institute_id=DECOY)
T = SessionCtx(user_id=SHARMA, role="teacher", institute_id=MAIN, teacher_id=SHARMA)
MIN_CACHEABLE_TOKENS = 512          # Claude Sonnet 5.5 / Haiku 5.5 minimum (platform.claude.com, prompt caching)


def _req(ctx: SessionCtx, msgs: list[dict], voice: bool = False, model: str = "claude-haiku-5-5") -> dict:
    prov = AnthropicProvider(client=object(), model=model, cache=True)
    return prov.request(system_prompt(ctx, voice), msgs, tool_schemas(active_tools(ctx.data_source == "live")), 2048)


def _prefix(kw: dict) -> str:
    return json.dumps({"tools": kw.get("tools"), "system": kw["system"]}, sort_keys=True)


def test_prefix_identical_across_users_turns_and_repeats():
    """Three requests from two admins (different institutes): same bytes before the messages."""
    kws = [_req(A, [{"role": "user", "text": "attendance"}]), _req(B, [{"role": "user", "text": "marks 10B"}]),
           _req(A, [{"role": "user", "text": "attendance"}, {"role": "assistant", "text": "x", "tool_calls": []},
                    {"role": "user", "text": "aur?"}])]
    assert len({_prefix(k) for k in kws}) == 1
    # a teacher's prefix differs only by role (expected: one cache entry per role)
    assert _prefix(_req(T, [{"role": "user", "text": "hi"}])) != _prefix(kws[0])


def test_breakpoints_where_the_docs_say():
    kw = _req(A, [{"role": "user", "text": "attendance"}])
    assert kw["tools"][-1]["cache_control"] == {"type": "ephemeral"} and all("cache_control" not in t for t in kw["tools"][:-1])
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert kw["messages"][-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_prefix_is_long_enough_to_cache():
    kw = _req(A, [{"role": "user", "text": "hi"}])
    approx_tokens = len(_prefix(kw)) // 4
    assert approx_tokens >= MIN_CACHEABLE_TOKENS, approx_tokens


def test_nothing_volatile_in_the_prefix():
    """No uuid / timestamp / session id in tools or system (today's DATE is allowed: it changes once a day)."""
    kw = _req(A, [{"role": "user", "text": "hi"}])
    p = _prefix(kw)
    assert str(ADMIN) not in p and str(MAIN) not in p and str(uuid.uuid4())[:8] not in p
    import re
    assert not re.search(r"\d{2}:\d{2}:\d{2}", p)            # no clock time
    assert config.AGENT_MODEL and kw["model"] == "claude-haiku-5-5"


def test_steps_of_one_turn_grow_the_messages_only():
    """Tool loop: step 2 = step 1 + assistant tool call + tool result; the prefix and earlier messages are unchanged
    (the breakpoint moves to the newest block, the old prefix stays cached)."""
    m1 = [{"role": "user", "text": "attendance 10B"}]
    k1 = _req(A, m1)
    m2 = m1 + [{"role": "assistant", "text": "", "tool_calls": [{"id": "t1", "name": "attendance_stats", "args": {}}]},
               {"role": "tool", "results": [{"id": "t1", "name": "attendance_stats", "content": "{}"}]}]
    k2 = _req(A, m2)
    assert _prefix(k1) == _prefix(k2)
    strip = lambda m: {**m, "content": [{k: v for k, v in b.items() if k != "cache_control"} for b in (m["content"] if isinstance(m["content"], list) else [{"type": "text", "text": m["content"]}])]}
    assert [strip(m) for m in k2["messages"][:1]] == [strip(m) for m in k1["messages"][:1]]


def test_voice_turns_share_one_prefix_too():
    assert _prefix(_req(A, [{"role": "user", "text": "a"}], voice=True)) == _prefix(_req(B, [{"role": "user", "text": "b"}], voice=True))
