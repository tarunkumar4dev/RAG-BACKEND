"""CORS on the standalone app + the dev-mode production guard + X-Dev-Role."""
import pytest
from fastapi.testclient import TestClient

from agent.api import create_app
from agent.auth import check_dev_mode_safe

LOCAL = "postgresql://postgres@/a4agent?host=/var/tmp&port=54329"


def _preflight(client, origin):
    return client.options("/agent/chat", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                                  "Access-Control-Request-Headers": "authorization,content-type"})


def test_cors_allowed_origin_passes_other_origin_does_not():
    client = TestClient(create_app({"AGENT_CORS_ORIGINS": "http://localhost:5173"}))
    ok = _preflight(client, "http://localhost:5173")
    assert ok.status_code == 200 and ok.headers["access-control-allow-origin"] == "http://localhost:5173"
    bad = _preflight(client, "https://evil.example")
    assert bad.status_code == 400 and "access-control-allow-origin" not in bad.headers
    with pytest.raises(RuntimeError):
        create_app({"AGENT_CORS_ORIGINS": "*"})


@pytest.mark.parametrize("var,url", [
    ("AGENT_DATABASE_URL", "postgresql://postgres:x@db.abcd.supabase.co:5432/postgres"),
    ("DATABASE_URL", "postgresql://u:p@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"),
    ("SUPABASE_URL", "https://abcd.supabase.co"),
    ("AGENT_DATABASE_URL", "postgresql://postgres@/a4agent?host=db.example.com"),
])
def test_dev_mode_refuses_to_start_with_production_db(var, url):
    env = {"AGENT_DEV_MODE": "1", "AGENT_DATABASE_URL": LOCAL, var: url}
    with pytest.raises(RuntimeError) as e:
        create_app(env)
    assert var in str(e.value) and url not in str(e.value)  # names only, never the URL
    check_dev_mode_safe({**env, "AGENT_DEV_MODE": "0"})  # dev mode off: no refusal
    check_dev_mode_safe({"AGENT_DEV_MODE": "1", "AGENT_DATABASE_URL": LOCAL,
                         "AGENT_DATABASE_RO_URL": "postgresql://agent_ro@localhost:54329/a4agent"})


def test_dev_role_header_only_in_dev_mode(monkeypatch):
    for k in ("SUPABASE_URL", "DATABASE_URL", "VERCEL"):
        monkeypatch.delenv(k, raising=False)
    client = TestClient(create_app({}))
    monkeypatch.setenv("AGENT_DEV_MODE", "1")
    r = client.get("/agent/sessions/00000000-0000-0000-0000-000000000000", headers={"X-Dev-Role": "admin"})
    assert r.status_code == 404  # authenticated as the seeded admin; session simply doesn't exist
    assert client.get("/agent/sessions/x", headers={"X-Dev-Role": "root"}).status_code == 400
    monkeypatch.setenv("AGENT_DEV_MODE", "0")
    assert client.get("/agent/sessions/x", headers={"X-Dev-Role": "admin"}).status_code == 401
