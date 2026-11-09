"""Write docs/agent_openapi_v2.json from the routers (no DB, no env needed): python -m agent.openapi_dump"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI


def build() -> dict:
    from . import api, api_v2
    a = FastAPI(title="a4ai Agent", version="1.2")
    a.include_router(api_v2.router)  # v2 owns GET /agent/sessions
    for r in api.router.routes:
        if not (getattr(r, "path", "") == "/agent/sessions" and "GET" in getattr(r, "methods", set())):
            a.router.routes.append(r)
    return a.openapi()


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "docs" / "agent_openapi_v2.json"
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
