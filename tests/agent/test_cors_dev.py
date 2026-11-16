"""Dev mode accepts the frontend from any local / LAN origin (Vite listens on every interface); real mode keeps the
exact AGENT_CORS_ORIGINS list. A refused preflight is logged with its Origin, so "Connecting…" has a cause."""
from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

import agent.api as api_mod

PRE = {"Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "authorization,x-dev-role"}


def _app(env: dict):
    return TestClient(api_mod.create_app(env, check_real_db=False))


@pytest.mark.parametrize("origin", ["http://localhost:8080", "http://127.0.0.1:8080", "http://[::1]:5173",
                                    "http://192.168.1.7:8080", "http://10.0.0.3:3000", "http://localhost:4173"])
def test_dev_mode_allows_local_and_lan_origins(origin):
    c = _app({"AGENT_DEV_MODE": "1", "AGENT_STORE": "memory"})
    r = c.options("/agent/meta", headers={"Origin": origin, **PRE})
    assert r.status_code == 200, (origin, r.text)
    assert r.headers.get("access-control-allow-origin") == origin


def test_dev_mode_still_refuses_the_public_internet():
    c = _app({"AGENT_DEV_MODE": "1", "AGENT_STORE": "memory"})
    assert c.options("/agent/meta", headers={"Origin": "https://evil.example", **PRE}).status_code == 400


def test_real_mode_keeps_the_exact_list(caplog):
    env = {"AGENT_STORE": "memory", "AGENT_CORS_ORIGINS": "https://a4ai.in"}
    c = _app(env)
    real = {"Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "authorization"}  # no dev header
    assert c.options("/agent/meta", headers={"Origin": "https://a4ai.in", **real}).status_code == 200
    with caplog.at_level(logging.WARNING):
        r = c.options("/agent/meta", headers={"Origin": "http://127.0.0.1:8080", **real})
    assert r.status_code == 400
    assert any("127.0.0.1:8080" in rec.getMessage() and "CORS" in rec.getMessage() for rec in caplog.records)
