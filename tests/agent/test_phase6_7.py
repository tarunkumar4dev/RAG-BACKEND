"""Phase 6+7 tests: deploy artifacts exist, download endpoint, full integration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


# ── Phase 6: deploy artifacts ────────────────────────────────────────────────

class TestDeployArtifacts:
    def test_dockerfile_exists(self):
        assert (ROOT / "agent" / "Dockerfile").is_file()

    def test_dockerfile_non_root(self):
        text = (ROOT / "agent" / "Dockerfile").read_text()
        assert "USER agent" in text
        assert "groupadd" in text

    def test_dockerfile_healthcheck(self):
        text = (ROOT / "agent" / "Dockerfile").read_text()
        assert "HEALTHCHECK" in text
        assert "/healthz" in text

    def test_gunicorn_conf_exists(self):
        assert (ROOT / "agent" / "gunicorn.conf.py").is_file()

    def test_gunicorn_conf_uvicorn_worker(self):
        text = (ROOT / "agent" / "gunicorn.conf.py").read_text()
        assert "UvicornWorker" in text
        assert "preload_app = True" in text

    def test_dockerignore_exists(self):
        assert (ROOT / "agent" / ".dockerignore").is_file()

    def test_readme_deploy_exists(self):
        assert (ROOT / "docs" / "README_DEPLOY.md").is_file()

    def test_readme_deploy_covers_env_vars(self):
        text = (ROOT / "docs" / "README_DEPLOY.md").read_text()
        for var in ["AGENT_DB_URL", "AGENT_CORS_ORIGINS", "AGENT_STORE", "WEB_CONCURRENCY"]:
            assert var in text, f"{var} not documented in README_DEPLOY.md"


# ── Phase 7: download endpoint via TestClient ────────────────────────────────

class TestDownloadEndpoint:
    def test_download_unknown_file(self, monkeypatch):
        monkeypatch.setenv("AGENT_DEV_MODE", "1")
        from starlette.testclient import TestClient
        from agent.api import create_app
        app = create_app(check_real_db=False)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/agent/files/nonexistent",
                          headers={"Authorization": "Bearer dev_admin", "X-Dev-Role": "admin"})
        assert resp.status_code == 404

    def test_download_requires_auth(self, monkeypatch):
        monkeypatch.setenv("AGENT_DEV_MODE", "1")
        from starlette.testclient import TestClient
        from agent.api import create_app
        app = create_app(check_real_db=False)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/agent/files/abc123")
        assert resp.status_code in (401, 403, 422)


# ── Phase 7: openapi includes new endpoints ──────────────────────────────────

class TestOpenAPICompleteness:
    def test_files_endpoint_in_openapi(self):
        spec = json.loads((ROOT / "docs" / "agent_openapi_v2.json").read_text())
        paths = spec.get("paths", {})
        file_paths = [p for p in paths if "/files/" in p]
        assert file_paths, "GET /agent/files/{id} not in OpenAPI spec"

    def test_metrics_endpoint_in_openapi(self):
        spec = json.loads((ROOT / "docs" / "agent_openapi_v2.json").read_text())
        paths = spec.get("paths", {})
        assert any("/metrics" in p for p in paths), "/agent/metrics not in OpenAPI spec"

    def test_file_event_in_schema(self):
        spec = json.loads((ROOT / "docs" / "agent_openapi_v2.json").read_text())
        schemas = spec.get("components", {}).get("schemas", {})
        assert "FileReady" in schemas, "FileReady event not in OpenAPI components"


# ── Phase 7: all tools have test coverage ─────────────────────────────────────

class TestToolCoverage:
    def test_every_tool_has_valid_args_model(self):
        from agent.tools.registry import TOOLS
        for name, spec in TOOLS.items():
            assert spec.args is not None, f"{name} has no args model"
            schema = spec.args.model_json_schema()
            assert "properties" in schema or schema.get("type") == "object", f"{name} args schema invalid"

    def test_every_insight_tool_has_fn(self):
        from agent.tools.registry import TOOLS, INSIGHT_TOOLS
        for name in INSIGHT_TOOLS:
            assert name in TOOLS, f"{name} listed in INSIGHT_TOOLS but not in TOOLS"
            assert TOOLS[name].fn is not None, f"{name} has no fn"
