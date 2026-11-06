"""LLMProvider interface: AnthropicProvider (default) + GeminiProvider (fallback).

Neutral message format:
  {"role":"user","text":str}
  {"role":"assistant","text":str,"tool_calls":[{"id","name","args"}]}
  {"role":"tool","results":[{"id","name","content":str}]}
"""
from __future__ import annotations

import json
import os
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
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


class LLMProvider(Protocol):
    model: str

    def complete(self, system: str, messages: list[dict], tools: list[dict], max_tokens: int = 2048,
                 json_mode: bool = False) -> LLMResponse: ...


class AnthropicProvider:
    def __init__(self, client: Any = None, model: Optional[str] = None):
        self.model = model or config.AGENT_MODEL
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ[config.ANTHROPIC_KEY_ENV])
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

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False) -> LLMResponse:
        kwargs: dict = dict(model=self.model, max_tokens=max_tokens, system=system, messages=self._messages(messages))
        if tools:
            kwargs["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                               for t in tools]
        r = self.client.messages.create(**kwargs)
        res = LLMResponse(model=self.model, tokens_in=getattr(r.usage, "input_tokens", 0),
                          tokens_out=getattr(r.usage, "output_tokens", 0))
        for b in r.content:
            if b.type == "text":
                res.text += b.text
            elif b.type == "tool_use":
                res.tool_calls.append(ToolCall(b.id, b.name, dict(b.input)))
        return res


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
    def __init__(self, client: Any = None, model: Optional[str] = None):
        self.model = model or config.GEMINI_MODEL
        if client is None:
            from google import genai
            client = genai.Client(api_key=os.environ[config.GEMINI_KEY_ENV])
        self.client = client

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False) -> LLMResponse:
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
        r = self.client.models.generate_content(model=self.model, contents=contents,
                                                config=types.GenerateContentConfig(**cfg))
        res = LLMResponse(model=self.model)
        um = getattr(r, "usage_metadata", None)
        if um:
            res.tokens_in, res.tokens_out = um.prompt_token_count or 0, um.candidates_token_count or 0
        for c in (r.candidates or [])[:1]:
            for p in (c.content.parts if c.content and c.content.parts else []):
                if p.function_call:
                    res.tool_calls.append(ToolCall("g_" + uuid.uuid4().hex[:8], p.function_call.name,
                                                   dict(p.function_call.args or {})))
                elif p.text:
                    res.text += p.text
        return res


def get_provider(name: Optional[str] = None) -> LLMProvider:
    name = name or config.PROVIDER
    return GeminiProvider() if name == "gemini" else AnthropicProvider()


def complete_json(provider: LLMProvider, system: str, prompt: str, max_tokens: int = 8000) -> Any:
    """One-shot JSON completion (used by generator + independent solver)."""
    r = provider.complete(system, [{"role": "user", "text": prompt}], [], max_tokens=max_tokens, json_mode=True)
    t = r.text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1].rsplit("```", 1)[0]
    s, e = min([i for i in (t.find("{"), t.find("[")) if i >= 0] or [0]), max(t.rfind("}"), t.rfind("]"))
    return json.loads(t[s:e + 1]), r
