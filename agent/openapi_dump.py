"""Deprecated alias: python -m agent.openapi_dump == python scripts/export_openapi.py"""
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "export_openapi.py"), run_name="__main__")


if __name__ == "__main__":
    main()
