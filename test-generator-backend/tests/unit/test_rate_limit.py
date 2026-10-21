"""Rate-limit keys: route template (not raw path), user id for authenticated requests, IP otherwise."""

import base64
import json

import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute

from app.core import rate_limit as rl


def _token(payload: dict) -> str:
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")  # noqa: E731
    return f"Bearer {enc({'alg': 'HS256'})}.{enc(payload)}.sig"


@pytest.fixture
def app():
    a = FastAPI()

    @a.post("/api/v1/copy-checker/sheets/{sheet_id}/check")
    def check(sheet_id: str):
        return {}

    @a.get("/api/v1/test-generator/chapters")
    def chapters():
        return {}
    return a


def test_ids_share_one_bucket(app):
    t1 = rl.route_template(app, "POST", "/api/v1/copy-checker/sheets/aaa/check")
    t2 = rl.route_template(app, "POST", "/api/v1/copy-checker/sheets/bbb/check")
    assert t1 == t2 == "/api/v1/copy-checker/sheets/{sheet_id}/check"
    assert rl.route_template(app, "GET", "/nope/123") == rl.UNMATCHED
    assert rl.route_template(app, "GET", "/api/v1/copy-checker/sheets/x/check") == t1   # other method → same template


def test_user_key_vs_ip_key():
    tpl = "/api/v1/copy-checker/sheets/{sheet_id}/check"
    a = rl.bucket_key("POST", tpl, _token({"sub": "user-a"}), "1.2.3.4")
    b = rl.bucket_key("POST", tpl, _token({"sub": "user-b"}), "1.2.3.4")
    assert a != b                              # same school IP, different teachers → separate buckets
    assert a.startswith("u:user-a|")
    assert rl.bucket_key("POST", tpl, None, "1.2.3.4").startswith("ip:1.2.3.4|")
    for bad in ("Bearer nonsense", "Basic abc", _token({"no_sub": 1}), _token({"sub": "x" * 100}), "Bearer a.%%%.c"):
        assert rl.bucket_key("POST", tpl, bad, "9.9.9.9").startswith("ip:9.9.9.9|")


def test_exempt_and_limits():
    assert rl.is_exempt("/internal/jobs/kick")
    assert rl.is_exempt("/api/v1/whatsapp/webhook")
    assert rl.is_exempt("/api/v1/payment/webhook")
    assert rl.is_exempt("/health")
    assert not rl.is_exempt("/api/v1/copy-checker/sheets/x/check")
    assert rl.limit_for("POST", "/api/v1/copy-checker/sheets/{sheet_id}/check", 300) == 60
    assert rl.limit_for("GET", "/something/else", 300) == 300


def test_middleware_limits_per_user_across_ids(monkeypatch):
    """Real middleware: unmatched paths (no route runs, no network) with a limit of 2 per minute."""
    from fastapi.testclient import TestClient
    from app import main

    monkeypatch.setitem(rl.ROUTE_LIMITS, f"GET {rl.UNMATCHED}", 2)
    main._rate_store.clear()
    client = TestClient(main.app)
    a, b = {"Authorization": _token({"sub": "teacher-a"})}, {"Authorization": _token({"sub": "teacher-b"})}
    assert client.get("/nope/1", headers=a).status_code == 404
    assert client.get("/nope/2", headers=a).status_code == 404
    assert client.get("/nope/3", headers=a).status_code == 429          # ids don't create new buckets
    assert client.get("/nope/4", headers=b).status_code == 404          # same IP, other teacher
    main._rate_store.clear()


def test_every_configured_limit_is_a_real_route():
    """A typo in ROUTE_LIMITS would silently fall back to the default limit."""
    from app.main import app as real_app
    real = {f"{m} {r.path}" for r in real_app.router.routes if isinstance(r, APIRoute) for m in r.methods}
    future = {"GET /r/{token}"}                # G6 share page
    missing = set(rl.ROUTE_LIMITS) - real - future
    assert not missing, missing
