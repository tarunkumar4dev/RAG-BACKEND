"""Write docs/agent_openapi_v2.json from the app (no DB, no secrets, no network).

    python scripts/export_openapi.py          # write
    python scripts/export_openapi.py --check  # exit 1 if the committed file is stale (CI)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "agent_openapi_v2.json"


def generate() -> str:
    os.environ.setdefault("AGENT_STORE", "memory")  # importing the app must not touch agent/.devdata
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from agent.api import create_app
    from agent.openapi import build
    return json.dumps(build(create_app(env={}, check_real_db=False)), indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    text = generate()
    if "--check" in argv:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != text:
            print(f"{OUT.relative_to(ROOT)} is stale: run python scripts/export_openapi.py")
            return 1
        print("openapi up to date")
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
