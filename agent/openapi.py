"""The agent's OpenAPI document, built from the app's own routes + every SSE event type.

Served at /openapi.json and committed as docs/agent_openapi_v2.json (python scripts/export_openapi.py). The frontend
generates its TypeScript types from that file; tests/agent/test_contract.py fails when the committed copy drifts.

Additions over FastAPI's default document:
- components.schemas.AgentEvent: oneOf every event model with a `type` discriminator (the SSE frame payload);
- POST /agent/chat 200 is `text/event-stream` whose frames are AgentEvent;
- MyScopeTable / SectionComparisonTable / SectionComparisonChart: typed views of tool-specific table/chart events
  (refinements for clients, not extra event types).
"""
from __future__ import annotations

from fastapi.openapi.utils import get_openapi
from pydantic import TypeAdapter

from .schemas import CONTRACT_VERSION, TOOL_EVENT_VIEWS, AgentEvent

REF = "#/components/schemas/{model}"
DESCRIPTION = ("a4ai Agent API (contract " + CONTRACT_VERSION + "). Human-readable spec: docs/AGENT_API_V2.md. "
               "Auth: Authorization: Bearer <Supabase access token>. Foreign ids return 404.")


def _merge(comps: dict, defs: dict) -> None:
    """Add event models FastAPI did not emit. Existing components are kept: FastAPI's rendering of the same model
    differs only cosmetically (drops `default: null`, writes minimum 0.0), and its copy is what responses reference."""
    for name, schema in defs.items():
        if name in comps and comps[name].get("required") != schema.get("required"):
            raise RuntimeError(f"OpenAPI component {name} differs between FastAPI and the event union")
        comps.setdefault(name, schema)


def build(app) -> dict:
    spec = get_openapi(title="a4ai Agent", version=CONTRACT_VERSION, description=DESCRIPTION, routes=app.routes)
    comps = spec.setdefault("components", {}).setdefault("schemas", {})
    ev = TypeAdapter(AgentEvent).json_schema(ref_template=REF, mode="serialization")
    _merge(comps, ev.pop("$defs", {}))
    ev["description"] = "One SSE frame of POST /agent/chat (`data: <json>`). Discriminated by `type`."
    comps["AgentEvent"] = ev
    for model in TOOL_EVENT_VIEWS:
        s = model.model_json_schema(ref_template=REF, mode="serialization")
        _merge(comps, s.pop("$defs", {}))
        comps[model.__name__] = s
    chat = spec["paths"]["/agent/chat"]["post"]["responses"]["200"]
    chat["content"] = {"text/event-stream": {"schema": {"$ref": REF.format(model="AgentEvent")}}}
    spec["components"]["schemas"] = dict(sorted(comps.items()))
    return spec


def install(app) -> None:
    """Serve the same document at /openapi.json."""
    app.openapi = lambda: app.openapi_schema or setattr(app, "openapi_schema", build(app)) or app.openapi_schema
