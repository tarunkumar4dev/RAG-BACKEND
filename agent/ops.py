"""Operational endpoints: GET /healthz (process up), GET /readyz (store + data DB reachable), GET /agent/metrics
(Prometheus text, no PII; AGENT_METRICS_TOKEN set -> Bearer token required)."""
from __future__ import annotations

import hmac
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from .api_models import HealthResponse, ReadyResponse

router = APIRouter(tags=["ops"])


@router.get("/healthz", response_model=HealthResponse)
def healthz():
    return {"status": "ok"}


def _check_store() -> bool:
    try:
        from .store import get_store
        return get_store().ping()
    except Exception:  # noqa: BLE001
        return False


def _check_db() -> bool:
    from .mode import real_mode
    try:
        if real_mode():
            from .db_real import connect_real
            with connect_real() as c:
                c.execute("select 1").fetchone()
        else:
            from .db import connect
            with connect(readonly=True) as c:
                c.execute("select 1").fetchone()
        return True
    except Exception:  # noqa: BLE001
        return False


@router.get("/readyz", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}})
def readyz():
    checks = {"store": _check_store(), "db": _check_db()}
    ok = all(checks.values())
    return JSONResponse({"status": "ready" if ok else "not_ready", "checks": checks}, status_code=200 if ok else 503)


@router.get("/agent/metrics", response_class=PlainTextResponse,
            responses={200: {"content": {"text/plain": {"schema": {"type": "string"}}},
                             "description": "Prometheus text exposition (version 0.0.4)"}})
def metrics(request: Request):
    token = os.getenv("AGENT_METRICS_TOKEN")
    if token:
        got = request.headers.get("authorization", "")
        if not hmac.compare_digest(got.encode(), f"Bearer {token}".encode()):
            raise HTTPException(401, "metrics token required")
    from .metrics import METRICS
    return PlainTextResponse(METRICS.render(), media_type="text/plain; version=0.0.4")
