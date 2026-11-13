"""Cross-platform dev entry point (used by scripts/run_agent_dev.ps1):
    python -m agent.dev_server [--port 8000] [--keep-data]
Embedded Postgres (pgserver) unless AGENT_DATABASE_URL points at another LOCAL server, fake seed,
AGENT_DEV_MODE=1, CORS for Vite (5173 + 8080), uvicorn. Refuses to run next to a production DB URL.
Keys are read by the app from A4AI_ANTHROPIC_API_KEY / GEMINI_API_KEY; only their presence is reported.
"""
from __future__ import annotations

import argparse
import os
import sys

DEFAULT_CORS = "http://localhost:5173,http://localhost:8080"


def prepare_env(env=os.environ) -> None:
    """Set dev defaults, then run the production guard BEFORE anything starts or connects."""
    from .auth import check_dev_mode_safe
    env["AGENT_DEV_MODE"] = "1"
    env.setdefault("AGENT_DB", "embedded")
    env.setdefault("AGENT_CORS_ORIGINS", DEFAULT_CORS)
    check_dev_mode_safe(env)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.getenv("AGENT_PORT", "8000")))
    ap.add_argument("--keep-data", action="store_true", help="do not re-seed the fake data")
    a = ap.parse_args(argv)
    prepare_env()
    if os.environ["AGENT_DB"] == "embedded":
        from .embedded_pg import ensure_env
        ensure_env()
        print("DB: embedded Postgres (pgserver) at", os.getenv("AGENT_PGDATA") or "agent/.devdata/pg")
    from .auth import check_dev_mode_safe
    check_dev_mode_safe()  # again, with the final DB URLs
    if not a.keep_data:
        from .db import reset_and_seed
        reset_and_seed()
        print("Seeded fake data (schema agent_demo)")
    for k in ("A4AI_ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        print(f"{k}: {'set' if os.getenv(k) else 'NOT SET'}")
    print(f"a4ai Agent dev server on http://localhost:{a.port}  CORS: {os.environ['AGENT_CORS_ORIGINS']}  "
          "auth: X-Dev-Role admin|teacher", flush=True)
    import uvicorn
    uvicorn.run("agent.api:app", host="127.0.0.1", port=a.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
