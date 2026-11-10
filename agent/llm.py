"""LLMProvider interface: AnthropicProvider (default) + GeminiProvider (fallback).

Neutral message format:
  {"role":"user","text":str}
  {"role":"assistant","text":str,"tool_calls":[{"id","name","args"}]}
  {"role":"tool","results":[{"id","name","content":str}]}

v3:
- Prompt caching (Anthropic): cache_control breakpoints on the last tool definition, the system prompt and the last
  message (3 of the 4 allowed). Render order is tools -> system -> messages, so the stable prefix (tool schemas +
  system prompt, which only changes per role / per day) is reused across turns and users of the same role, and the
  growing conversation is reused between the steps of one turn. AGENT_PROMPT_CACHE=0 turns it off. Gemini 2.5 caches
  implicitly; its cached token count is reported the same way.
- complete(..., cancel=Event): streams the response and closes the HTTP stream as soon as `cancel` is set, so the
  provider stops generating (raises LLMAborted with an estimate of the tokens produced so far).
- The SDK's own retries are off (max_retries=0): agent/core.py retries 429/5xx/overloaded itself (jittered backoff,
  cancel-aware), so a turn never waits on two retry policies stacked.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from . import config


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0      # uncached input tokens
    tokens_out: int = 0
    model: str = ""
    cache_read: int = 0     # input tokens served from the prompt cache
    cache_write: int = 0    # input tokens written to the prompt cache (Anthropic)


class LLMAborted(Exception):
    """The stream was closed because the run was cancelled."""

    def __init__(self, model: str, tokens_out_est: int, max_tokens: int):
        super().__init__("aborted")
        self.model, self.tokens_out_est, self.max_tokens = model, tokens_out_est, max_tokens


class LLMProvider(Protocol):
    model: str

    def complete(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 2048,
                 json_mode: bool = False) -> LLMResponse: ...


def prompt_cache_enabled() -> bool:
    return os.getenv("AGENT_PROMPT_CACHE", "1") != "0"


_CC = {"type": "ephemeral"}


class AnthropicProvider:
    supports_cancel = True

    def __init__(self, client: Any = None, model: Optional[str] = None, cache: Optional[bool] = None):
        self.model = model or config.AGENT_MODEL
        self.cache = prompt_cache_enabled() if cache is None else cache
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ[config.ANTHROPIC_KEY_ENV], max_retries=0,
                                         timeout=anthropic.Timeout(90.0, connect=5.0))
        self.client = client

    @staticmethod
    def _messages(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                content: list[dict] = []
                if m.get("text"):
                    content.append({"type": "text", "text": m["text"]})
                for tc in m.get("tool_calls", []):
                    content.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["args"]})
                out.append({"role": "assistant", "content": content or [{"type": "text", "text": "."}]})
            elif m["role"] == "tool":
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"]} for r in m["results"]]})
        return out

    def request(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int) -> dict:
        """The exact Messages API kwargs (no client call): tests check cache breakpoints and prefix stability."""
        msgs = self._messages(messages)
        kwargs: dict = dict(model=self.model, max_tokens=max_tokens, messages=msgs)
        tdefs = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools]
        if self.cache:
            kwargs["system"] = [{"type": "text", "text": system, "cache_control": _CC}]
            if tdefs:
                tdefs[-1] = {**tdefs[-1], "cache_control": _CC}
            if msgs:
                last = msgs[-1]
                content = last["content"]
                blocks = [{"type": "text", "text": content}] if isinstance(content, str) else [dict(b) for b in content]
                blocks[-1]["cache_control"] = _CC
                msgs[-1] = {**last, "content": blocks}
        else:
            kwargs["system"] = system
        if tdefs:
            kwargs["tools"] = tdefs
        return kwargs

    def _parse(self, r) -> LLMResponse:
        u = r.usage
        res = LLMResponse(model=self.model, tokens_in=getattr(u, "input_tokens", 0) or 0,
                          tokens_out=getattr(u, "output_tokens", 0) or 0,
                          cache_read=getattr(u, "cache_read_input_tokens", 0) or 0,
                          cache_write=getattr(u, "cache_creation_input_tokens", 0) or 0)
        for b in r.content:
            if b.type == "text":
                res.text += b.text
            elif b.type == "tool_use":
                res.tool_calls.append(ToolCall(b.id, b.name, dict(b.input)))
        return res

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False,
                 cancel: Optional[threading.Event] = None) -> LLMResponse:
        kwargs = self.request(system, messages, tools, max_tokens)
        if cancel is None:
            return self._parse(self.client.messages.create(**kwargs))
        chars = 0
        with self.client.messages.stream(**kwargs) as stream:  # leaving the block closes the HTTP stream
            for event in stream:
                if cancel.is_set():
                    raise LLMAborted(self.model, chars // 4, max_tokens)
                t = getattr(event, "type", "")
                if t == "text":
                    chars += len(getattr(event, "text", "") or "")
                elif t == "input_json":
                    chars += len(getattr(event, "partial_json", "") or "")
            msg = stream.get_final_message()
        return self._parse(msg)


def _gemini_schema(s: Any) -> Any:
    """Gemini function declarations accept an OpenAPI subset: drop unsupported keys."""
    if isinstance(s, dict):
        drop = {"additionalProperties", "title", "$defs", "default", "examples", "exclusiveMinimum", "exclusiveMaximum"}
        out = {k: _gemini_schema(v) for k, v in s.items() if k not in drop}
        if "anyOf" in out:  # Optional[X] -> X
            non_null = [x for x in out["anyOf"] if x.get("type") != "null"]
            base = out.pop("anyOf")
            out.update(non_null[0] if non_null else base[0])
        return out
    if isinstance(s, list):
        return [_gemini_schema(x) for x in s]
    return s


class GeminiProvider:
    supports_cancel = True

    def __init__(self, client: Any = None, model: Optional[str] = None):
        self.model = model or config.GEMINI_MODEL
        if client is None:
            from google import genai
            client = genai.Client(api_key=os.environ[config.GEMINI_KEY_ENV])
        self.client = client

    def _args(self, system, messages, tools, max_tokens, json_mode):
        from google.genai import types
        contents = []
        for m in messages:
            if m["role"] == "user":
                contents.append(types.Content(role="user", parts=[types.Part(text=m["text"])]))
            elif m["role"] == "assistant":
                parts = [types.Part(text=m["text"])] if m.get("text") else []
                parts += [types.Part(function_call=types.FunctionCall(name=tc["name"], args=tc["args"]))
                          for tc in m.get("tool_calls", [])]
                contents.append(types.Content(role="model", parts=parts or [types.Part(text=".")]))
            elif m["role"] == "tool":
                contents.append(types.Content(role="user", parts=[
                    types.Part(function_response=types.FunctionResponse(name=r["name"], response={"result": r["content"]}))
                    for r in m["results"]]))
        cfg: dict = dict(system_instruction=system, max_output_tokens=max_tokens, temperature=0)
        if tools:
            cfg["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters=_gemini_schema(t["parameters"])) for t in tools])]
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        return dict(model=self.model, contents=contents, config=types.GenerateContentConfig(**cfg))

    def _absorb(self, res: LLMResponse, r) -> None:
        um = getattr(r, "usage_metadata", None)
        if um and (um.prompt_token_count or um.candidates_token_count):
            cached = getattr(um, "cached_content_token_count", 0) or 0
            res.cache_read = cached
            res.tokens_in = max((um.prompt_token_count or 0) - cached, 0)
            res.tokens_out = um.candidates_token_count or 0
        for c in (r.candidates or [])[:1]:
            for p in (c.content.parts if c.content and c.content.parts else []):
                if p.function_call:
                    res.tool_calls.append(ToolCall("g_" + uuid.uuid4().hex[:8], p.function_call.name,
                                                   dict(p.function_call.args or {})))
                elif p.text:
                    res.text += p.text

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False,
                 cancel: Optional[threading.Event] = None) -> LLMResponse:
        kw = self._args(system, messages, tools, max_tokens, json_mode)
        res = LLMResponse(model=self.model)
        if cancel is None:
            self._absorb(res, self.client.models.generate_content(**kw))
            return res
        stream = self.client.models.generate_content_stream(**kw)
        try:
            for chunk in stream:
                if cancel.is_set():
                    raise LLMAborted(self.model, len(res.text) // 4, max_tokens)
                self._absorb(res, chunk)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
        return res


def get_provider(name: Optional[str] = None, model: Optional[str] = None) -> LLMProvider:
    name = name or config.PROVIDER
    return GeminiProvider(model=model) if name == "gemini" else AnthropicProvider(model=model)


def fast_provider_for(main: LLMProvider) -> Optional[LLMProvider]:
    """The routing partner of `main` (same client, AGENT_FAST_MODEL / AGENT_GEMINI_FAST_MODEL), or None when it would
    be the same model or `main` is not a known provider (tests' fakes): then every turn uses `main`."""
    if isinstance(main, AnthropicProvider) and config.AGENT_FAST_MODEL != main.model:
        return AnthropicProvider(client=main.client, model=config.AGENT_FAST_MODEL, cache=main.cache)
    if isinstance(main, GeminiProvider) and config.GEMINI_FAST_MODEL != main.model:
        return GeminiProvider(client=main.client, model=config.GEMINI_FAST_MODEL)
    return None


def complete_json(provider: LLMProvider, system: str, prompt: str, max_tokens: int = 8000) -> Any:
    """One-shot JSON completion (used by generator + independent solver)."""
    r = provider.complete(system, [{"role": "user", "text": prompt}], [], max_tokens=max_tokens, json_mode=True)
    t = r.text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    s, e = min([i for i in (t.find("{"), t.find("[")) if i >= 0] or [0]), max(t.rfind("}"), t.rfind("]"))
    return json.loads(t[s:e + 1]), r
