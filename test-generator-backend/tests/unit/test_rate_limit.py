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
    assert rl.limit_for("POST", "/api/v1/contests/{contest_id}/submit", 300, "att:x") == 30
    assert rl.limit_for("POST", "/api/v1/contests/{contest_id}/submit", 300, "ip:1.2.3.4") == 1200
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


def test_attempt_identity_and_ip_fallback():
    tpl = "/api/v1/contests/{contest_id}/submit"
    assert rl.checks("POST", tpl, None, "1.1.1.1", "att-1", 60) == [
        ("att:att-1|POST " + tpl, 30), ("attip:1.1.1.1|POST " + tpl, rl.ATTEMPT_IP_CEILING)]
    assert rl.checks("POST", tpl, None, "1.1.1.1", None, 60) == [("ip:1.1.1.1|POST " + tpl, 1200)]
    # a logged-in student is counted as the user, not the attempt
    assert rl.checks("POST", tpl, _token({"sub": "stu"}), "1.1.1.1", "att-1", 60)[0][0].startswith("u:stu|")
    assert rl.attempt_id("bad id with spaces") is None


def _client_with_limit(monkeypatch, limit: int, ceiling: int = None):
    from fastapi.testclient import TestClient
    from app import main
    monkeypatch.setitem(rl.ROUTE_LIMITS, f"POST {rl.UNMATCHED}", limit)
    if ceiling is not None:
        monkeypatch.setattr(rl, "ATTEMPT_IP_CEILING", ceiling)
    main._rate_store.clear()
    return TestClient(main.app), main


def test_50_students_one_school_ip_no_429(monkeypatch):
    """40+ students behind one WiFi IP, no login: each counted by its attempt id. Unmatched path = no route
    runs (no network); the per-identity limit is set to 5 so 5 requests each is exactly at the edge."""
    client, main = _client_with_limit(monkeypatch, limit=5)
    codes = [client.post(f"/nope/submit/{s}", headers={"X-Attempt-Id": f"attempt-{s}"}).status_code
             for s in range(50) for _ in range(5)]
    assert codes.count(429) == 0 and len(codes) == 250
    # the same 250 requests without an attempt id would be one IP bucket → 429 after the 5th
    main._rate_store.clear()
    codes = [client.post("/nope/submit/x").status_code for _ in range(10)]
    assert codes.count(429) == 5
    # one student over their own limit is still stopped
    main._rate_store.clear()
    codes = [client.post("/nope/submit/x", headers={"X-Attempt-Id": "attempt-0"}).status_code for _ in range(6)]
    assert codes[-1] == 429
    main._rate_store.clear()


def test_made_up_attempt_ids_hit_the_ip_ceiling(monkeypatch):
    client, main = _client_with_limit(monkeypatch, limit=5, ceiling=20)
    codes = [client.post("/nope/x", headers={"X-Attempt-Id": f"fake-{i}"}).status_code for i in range(25)]
    assert codes.count(429) == 5
    main._rate_store.clear()


def test_every_configured_limit_is_a_real_route():
    """A typo in ROUTE_LIMITS would silently fall back to the default limit."""
    from app.main import app as real_app
    real = {f"{m} {r.path}" for r in real_app.router.routes if isinstance(r, APIRoute) for m in r.methods}
    future = {"GET /r/{token}"}                # G6 share page
    missing = (set(rl.ROUTE_LIMITS) | set(rl.IP_FALLBACK_LIMITS)) - real - future
    assert not missing, missing


def test_cors_allows_the_attempt_header():
    """Contest submit sends X-Attempt-Id from the browser; without it in allow_headers the preflight is a 400."""
    from fastapi.middleware.cors import CORSMiddleware
    from app import main
    cors = next(m for m in main.app.user_middleware if m.cls is CORSMiddleware)
    assert "X-Attempt-Id" in cors.kwargs["allow_headers"]
