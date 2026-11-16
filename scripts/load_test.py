"""Load test: 50 concurrent simulated users hitting the agent chat endpoint.

Usage: AGENT_DB=embedded python scripts/load_test.py [--users 50] [--turns 3] [--base-url http://localhost:8000]

Each user sends `turns` messages sequentially (each waits for the SSE stream to finish).
All users run in parallel. The test requires the dev server (or embedded mode) running.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTIONS = [
    "Sharma ji ki attendance kya hai?",
    "ECE-2 ki attendance trend dikhao",
    "75% se kam attendance wale students?",
    "Kitne tests bane last month?",
    "Meri classes dikhao",
    "Which class has lowest attendance?",
    "Class 10 Math ke marks kya hain?",
    "Section comparison by attendance",
    "Verma sir ki attendance stats",
    "Low attendance list for October",
]


async def run_user(client, base_url: str, user_idx: int, turns: int, results: list):
    """Simulate one user's chat session."""
    import httpx

    uid = str(uuid.uuid4())
    session_id = None

    for turn in range(turns):
        question = QUESTIONS[(user_idx * turns + turn) % len(QUESTIONS)]
        body = {"message": question}
        if session_id:
            body["session_id"] = session_id

        t0 = time.monotonic()
        events = []
        error = None

        try:
            async with client.stream(
                "POST", f"{base_url}/agent/chat", json=body,
                headers={"Authorization": f"Bearer dev_{uid}", "X-Dev-Role": "admin"},
                timeout=120,
            ) as resp:
                if resp.status_code == 429:
                    results.append({"user": user_idx, "turn": turn, "status": 429, "latency_ms": 0, "events": 0})
                    continue
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            ev = json.loads(line[6:])
                            events.append(ev)
                            if ev.get("type") == "session_start":
                                session_id = ev.get("session_id", session_id)
                        except json.JSONDecodeError:
                            pass

            latency = (time.monotonic() - t0) * 1000
            results.append({
                "user": user_idx, "turn": turn, "status": resp.status_code,
                "latency_ms": round(latency, 1), "events": len(events), "error": error,
            })
        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            results.append({
                "user": user_idx, "turn": turn, "status": 0,
                "latency_ms": round(latency, 1), "events": 0, "error": str(e),
            })


async def main():
    import httpx

    parser = argparse.ArgumentParser(description="Agent load test")
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    print(f"Load test: {args.users} users × {args.turns} turns against {args.base_url}")

    results: list[dict] = []
    t0 = time.monotonic()

    async with httpx.AsyncClient() as client:
        tasks = [run_user(client, args.base_url, i, args.turns, results) for i in range(args.users)]
        await asyncio.gather(*tasks)

    elapsed = time.monotonic() - t0

    # Summarize
    ok = [r for r in results if r["status"] == 200]
    rate_limited = [r for r in results if r["status"] == 429]
    errors = [r for r in results if r["status"] not in (200, 429)]
    latencies = sorted(r["latency_ms"] for r in ok) if ok else [0]

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Total requests: {len(results)}")
    print(f"  OK (200):         {len(ok)}")
    print(f"  Rate limited (429): {len(rate_limited)}")
    print(f"  Errors:           {len(errors)}")
    if ok:
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        print(f"Latency (OK): p50={p50:.0f}ms  p95={p95:.0f}ms  p99={p99:.0f}ms")
    if errors:
        print("Errors:")
        for e in errors[:5]:
            print(f"  user={e['user']} turn={e['turn']} status={e['status']} error={e.get('error')}")

    # Write results JSON
    out_path = ROOT / "docs" / "audit" / "load_test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"config": vars(args), "elapsed_s": round(elapsed, 1),
                                     "ok": len(ok), "rate_limited": len(rate_limited), "errors": len(errors),
                                     "latency_p50": latencies[len(latencies) // 2] if ok else None,
                                     "latency_p95": latencies[int(len(latencies) * 0.95)] if ok else None,
                                     "results": results}, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
