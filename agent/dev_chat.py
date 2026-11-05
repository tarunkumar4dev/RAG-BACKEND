"""Tiny dev client: POST /agent/chat with X-Dev-Role and print the SSE events (no shell quoting issues).
    python -m agent.dev_chat [--role admin|teacher] [--url http://localhost:8000] "message"
"""
from __future__ import annotations

import argparse
import sys

import httpx


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("message")
    ap.add_argument("--role", default="admin", choices=["admin", "teacher"])
    ap.add_argument("--url", default="http://localhost:8000")
    a = ap.parse_args(argv)
    with httpx.stream("POST", a.url.rstrip("/") + "/agent/chat", json={"message": a.message},
                      headers={"X-Dev-Role": a.role, "Accept": "text/event-stream"}, timeout=180) as r:
        if r.status_code != 200:
            print(f"HTTP {r.status_code}: {r.read().decode(errors='replace')[:300]}")
            return 1
        for line in r.iter_lines():
            if line:
                print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
