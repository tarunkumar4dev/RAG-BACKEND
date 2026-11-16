"""Prompt-cache probe (phase 4): three identical agent requests to Anthropic, then the usage fields.

    set A4AI_ANTHROPIC_API_KEY=...   (never committed)
    python scripts/cache_probe.py [--model claude-haiku-5-5]

Paid: ~3 x 4k input tokens (Haiku: < $0.01 in total). Prints cache_creation_input_tokens / cache_read_input_tokens
per request. Expected: request 1 writes the prefix (cache_write > 0), requests 2 and 3 read it (cache_read > 0) and
the trace file gets the same numbers. Refuses to run without the key.
"""
from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    from agent import config
    if not os.getenv(config.ANTHROPIC_KEY_ENV):
        print(f"refusing: {config.ANTHROPIC_KEY_ENV} is not set (paid call)")
        return 2
    from _agent_ids import ADMIN, MAIN  # noqa: E402  tests/agent ids (demo data)
    from agent.core import Agent
    from agent.llm import AnthropicProvider
    from agent.schemas import SessionCtx
    ctx = SessionCtx(user_id=ADMIN, role="admin", institute_id=MAIN)
    prov = AnthropicProvider(model=args.model)
    for i in range(1, 4):
        evs = list(Agent(prov, ctx, str(uuid.uuid4())).run("meri classes batao"))
        fin = evs[-1]
        u = fin.get("usage", {})
        print(f"request {i}: model={prov.model} tokens_in={u.get('tokens_in')} cache_read={u.get('cache_read_tokens')} "
              f"cost=${u.get('cost_usd')} latency={u.get('latency_s')}s route={u.get('route')}")
    print(f"trace: {config.TRACE_PATH} (kind=llm rows have cache_read / cache_write)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests" / "agent"))
    raise SystemExit(main())
