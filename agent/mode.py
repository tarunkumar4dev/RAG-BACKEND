"""Agent data mode. Seed/dev (AGENT_DEV_MODE=1, fake local DB) and real (AGENT_REAL_DB_MODE=1, Supabase via a
read-only role) are mutually exclusive: the app refuses to start if both are set."""
from __future__ import annotations

import os


def dev_mode(env=os.environ) -> bool:
    return env.get("AGENT_DEV_MODE") == "1"


def real_mode(env=os.environ) -> bool:
    return env.get("AGENT_REAL_DB_MODE") == "1"


def check_modes(env=os.environ) -> None:
    if dev_mode(env) and real_mode(env):
        raise RuntimeError("AGENT_DEV_MODE=1 and AGENT_REAL_DB_MODE=1 are mutually exclusive")
    if real_mode(env):
        missing = [k for k in ("AGENT_REAL_DB_URL", "SUPABASE_URL") if not env.get(k)]
        if missing:
            raise RuntimeError(f"AGENT_REAL_DB_MODE=1 needs {', '.join(missing)}")
        if env.get("AGENT_DEV_AUTH") == "1":
            raise RuntimeError("AGENT_DEV_AUTH=1 (header identity) is not allowed with AGENT_REAL_DB_MODE=1")


READ_ONLY_MESSAGE = ("Abhi read-only mode hai: draft save, finalize aur section assign band hain. "
                     "Paper ka preview dekh sakte hain, kuch bhi save nahi hoga.")
