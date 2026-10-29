#!/usr/bin/env python3
"""
Checks app/services/ncert_text_hash.py against the shared fixtures (no pytest needed).

    python scripts/check_ncert_text_hash.py

scripts/fixtures/ncert_normalize_fixtures.json is the contract: the frontend implementation
must pass the same 10 strings (expected normalized text + SHA-256).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.services.ncert_text_hash import content_hash, normalize  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "ncert_normalize_fixtures.json")


def main() -> int:
    with open(FIXTURES, encoding="utf-8") as f:
        cases = json.load(f)
    failures = 0
    for i, case in enumerate(cases, 1):
        got = normalize(case["input"])
        got_hash = content_hash(case["input"])
        ok = got == case["normalized"] and got_hash == case["sha256"]
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {i:2d}  {case['note']}")
        if not ok:
            print(f"      expected {case['normalized']!r}\n      got      {got!r}")
    print(f"\n{len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
