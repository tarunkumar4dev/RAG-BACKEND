"""Typed route registry (agent/routes/registry.json) + the `open_feature` tool.

The registry is shared with the frontend (byte-identical copy, `npm run agent:routes`). The tool accepts only keys
the caller's role may open and only params that match the registry's types; it never touches the database. The
agent loop turns an ok result into a `ui_action navigate` event; the frontend executes only keys in ITS copy.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..schemas import SessionCtx

REGISTRY_PATH = Path(__file__).resolve().parent / "registry.json"
_KEY_RE = re.compile(r"^[a-z0-9_.]+$")
_ROLES = ("admin", "hod", "proctor", "teacher")
# backend roles (SessionCtx.role) -> frontend roles they cover. A backend "teacher" may be a proctor (the backend has
# no proctor role), so proctor pages are allowed for teachers: the page's own RoleRoute still decides.
_ROLE_COVERS = {"admin": {"admin"}, "hod": {"hod"}, "teacher": {"teacher", "proctor"}}


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        reg = json.load(f)
    validate(reg)
    return reg


def validate(reg: dict[str, Any]) -> None:
    """Shape rules both copies must satisfy (the frontend test checks the same things in TypeScript)."""
    assert isinstance(reg.get("routes"), dict) and reg["routes"], "routes missing"
    types = reg.get("param_types") or {}
    for t, pat in types.items():
        re.compile(pat)
    for key, r in reg["routes"].items():
        assert _KEY_RE.match(key) and len(key) <= 64, key
        assert r["path"].startswith("/") and "?" not in r["path"], key
        assert r["roles"] and set(r["roles"]) <= set(_ROLES), key
        assert isinstance(r["query"], dict) and all(isinstance(v, str) for v in r["query"].values()), key
        assert isinstance(r["deep_link"], bool) and r["label"].get("hi") and r["label"].get("en"), key
        assert r["aliases"] and all(isinstance(a, str) for a in r["aliases"]), key
        path_params = set(re.findall(r":(\w+)", r["path"]))
        seen_path = set()
        for pname, p in r["params"].items():
            assert re.match(r"^[a-z_]+$", pname) and p["type"] in types and p["in"] in ("query", "path"), (key, pname)
            assert re.match(r"^[A-Za-z_]+$", p["name"]) and isinstance(p["required"], bool), (key, pname)
            if p["in"] == "path":
                assert p["name"] in path_params and p["required"], (key, pname)
                seen_path.add(p["name"])
            else:
                assert p["name"] not in r["query"], (key, pname)
        assert seen_path == path_params, (key, "every :param needs a typed param")


def keys() -> list[str]:
    return list(load()["routes"].keys())


def allowed_keys(role: str) -> list[str]:
    cover = _ROLE_COVERS.get(role, set())
    return [k for k, r in load()["routes"].items() if cover & set(r["roles"])]


def role_allows(role: str, key: str) -> bool:
    return key in allowed_keys(role)


def check_params(key: str, params: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """-> (accepted params, problems). Unknown params are dropped (never passed through); bad values are problems."""
    reg = load()
    r = reg["routes"][key]
    types = reg["param_types"]
    ok: dict[str, str] = {}
    bad: list[str] = []
    for pname, p in r["params"].items():
        v = params.get(pname)
        if v is None or v == "":
            if p["required"]:
                bad.append(f"{pname} required")
            continue
        v = str(v).strip()
        if len(v) > 80 or not re.match(types[p["type"]], v):
            bad.append(f"{pname} invalid")
            continue
        ok[pname] = v
    return ok, bad


def _describe() -> str:
    reg = load()
    lines = []
    for k, r in reg["routes"].items():
        lines.append(f"{k} ({r['label']['en']}; say: {', '.join(r['aliases'][:4])})")
    return "; ".join(lines)


class OpenFeatureArgs(BaseModel):
    """route_key must be one of the registry keys; params only as typed in the registry (ids from earlier tool
    results, never guessed)."""
    model_config = ConfigDict(extra="forbid")
    route_key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_.]+$")
    params: dict[str, str] = Field(default_factory=dict)


def open_feature(ctx: SessionCtx, args: OpenFeatureArgs) -> dict[str, Any]:
    reg = load()
    r = reg["routes"].get(args.route_key)
    if r is None:
        return {"status": "not_found", "known_keys": keys()}
    if not role_allows(ctx.role, args.route_key):
        return {"status": "forbidden", "message": "Ye page aapke role ke liye nahi hai."}
    params, bad = check_params(args.route_key, args.params)
    if bad:
        return {"status": "invalid_args", "errors": bad}
    return {"status": "ok", "route_key": args.route_key, "params": params, "label": r["label"],
            "deep_link": r["deep_link"],
            "note": None if r["deep_link"] else "page opens; the tab/item must be picked on screen"}


TOOL_DESCRIPTION = (
    "Open an a4ai page/feature on the user's screen ('attendance kholo', 'open the test generator', 'checker "
    "dikhao', 'settings', 'Rahul ka profile kholo'). Only opens pages — never saves, assigns or approves. Pick the "
    "route_key from this list: " + _describe() + ". For student.profile first resolve_entity(kind=student) and pass "
    "its id as params.student_id. After the tool returns ok, say in one short line which page opened."
)
