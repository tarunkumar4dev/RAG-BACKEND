"""Contract as source of truth: docs/agent_openapi_v2.json must equal what the app generates (the frontend builds its
TypeScript types from that file). Fix a failure with: python scripts/export_openapi.py"""
import json
import typing
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
OPENAPI = ROOT / "docs" / "agent_openapi_v2.json"


def _generated() -> str:
    import importlib.util
    spec = importlib.util.spec_from_file_location("export_openapi", ROOT / "scripts" / "export_openapi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate()


def test_committed_openapi_matches_generated():
    assert OPENAPI.read_text(encoding="utf-8") == _generated(), \
        "docs/agent_openapi_v2.json is stale: run python scripts/export_openapi.py"


def test_every_sse_event_is_a_component_in_the_discriminated_union():
    from agent.schemas import AgentEvent, CONTRACT_VERSION
    d = json.loads(OPENAPI.read_text(encoding="utf-8"))
    comps = d["components"]["schemas"]
    union = typing.get_args(typing.get_args(AgentEvent)[0])
    mapping = comps["AgentEvent"]["discriminator"]["mapping"]
    assert comps["AgentEvent"]["discriminator"]["propertyName"] == "type"
    for model in union:
        t = model.model_fields["type"].default
        assert model.__name__ in comps, model.__name__
        assert mapping[t] == f"#/components/schemas/{model.__name__}"
    assert {"session", "file", "final", "error", "table", "chart", "pending_approval"} <= set(mapping)
    assert d["info"]["version"] == CONTRACT_VERSION
    # SSE frames typed as AgentEvent
    sse = d["paths"]["/agent/chat"]["post"]["responses"]["200"]["content"]["text/event-stream"]["schema"]
    assert sse == {"$ref": "#/components/schemas/AgentEvent"}
    assert "RateLimitResponse" in comps and "429" in d["paths"]["/agent/chat"]["post"]["responses"]


def test_contract_covers_tool_views_meta_scope_feedback_and_files():
    d = json.loads(OPENAPI.read_text(encoding="utf-8"))
    comps = d["components"]["schemas"]
    for name in ("MyScopeTable", "SectionComparisonTable", "SectionComparisonChart", "FileReady", "SessionStarted"):
        assert name in comps, name
    assert comps["MyScopeTable"]["properties"]["source"]["const"] == "tool:my_scope"
    meta = d["paths"]["/agent/meta"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    assert meta == {"$ref": "#/components/schemas/MetaResponse"}
    assert "scope" in comps["MetaResponse"]["properties"] and "level" in comps["MetaScope"]["properties"]
    fb = d["paths"]["/agent/feedback"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert fb == {"$ref": "#/components/schemas/FeedbackRequest"}
    assert set(comps["FileReady"]["required"]) >= {"file_id", "name", "rows", "expires_at", "download_path"}
    assert "retryable" in comps["Error"]["properties"]
    assert {"status", "result_ref"} <= set(comps["PendingApproval"]["properties"])
    assert "/agent/files/{file_id}" in d["paths"] and "/healthz" in d["paths"] and "/readyz" in d["paths"]


def test_app_serves_the_same_document():
    from agent.api import create_app
    from agent.openapi import build
    app = create_app(env={}, check_real_db=False)
    served = TestClient(app).get("/openapi.json").json()
    assert served == json.loads(json.dumps(build(app)))
