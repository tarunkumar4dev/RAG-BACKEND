"""Shared local environment for scripts/bench_agent.py and scripts/loadtest_agent.py.

Starts the embedded Postgres (pgserver, local only), seeds the demo schema and builds the schema-faithful real-data
fixture (tests/agent/real_fixture, fake rows), then switches the process to real read-only mode against it. Never
connects anywhere else. Optional: load GEMINI_API_KEY from the repo .env for --provider gemini (value never printed).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "agent" / "real_fixture"))

SUPA = "https://fixtureproj.supabase.co"
SECRET = "fixture-hs256-secret-not-a-real-one-0123456789"  # fixture only, same as tests/agent/test_real_mode.py
U = {k: f"10000000-0000-0000-0000-00000000000{v}" for k, v in
     dict(owner=1, admin=2, hod=3, proctor=4, sharma=5, verma=6).items()}
ECE2, ECE2A, CSE1 = (f"ba000000-0000-0000-0000-00000000000{i}" for i in (1, 2, 3))


def load_dotenv_key(name: str = "GEMINI_API_KEY") -> bool:
    if os.environ.get(name):
        return True
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith(name + "="):
                os.environ[name] = line.split("=", 1)[1].strip().strip('"').strip("'")
                return True
    return False


def setup(data_dir: str | None = None) -> dict:
    os.environ.setdefault("AGENT_STORE", "memory")
    from agent.embedded_pg import ensure_env
    ensure_env(data_dir or (ROOT / "agent" / ".devdata" / "pg_bench"))
    from agent.db import reset_and_seed
    reset_and_seed()
    import build  # tests/agent/real_fixture/build.py
    owner, ro = build.build()
    os.environ.update(AGENT_REAL_DB_MODE="1", AGENT_REAL_DB_URL=ro, SUPABASE_URL=SUPA, SUPABASE_JWT_SECRET=SECRET)
    os.environ.pop("AGENT_DEV_MODE", None)
    os.environ.pop("AGENT_DEV_AUTH", None)
    return {"owner": owner, "ro": ro}


def token(sub: str, exp_in: int = 3600) -> str:
    import time

    import jwt
    now = int(time.time())
    return jwt.encode({"sub": sub, "aud": "authenticated", "role": "authenticated", "iss": f"{SUPA}/auth/v1",
                       "iat": now, "exp": now + exp_in, "user_metadata": {"full_name": "Bench User"}},
                      SECRET, algorithm="HS256")
