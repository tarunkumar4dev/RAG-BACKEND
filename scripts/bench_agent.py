"""Latency bench: replay 10 typical trial questions against the real-data fixture (embedded Postgres, fake rows).

    python scripts/bench_agent.py                     # mocked LLM (0 ms): agent + tools + DB overhead only
    python scripts/bench_agent.py --llm-ms 400        # mocked LLM that sleeps 400 ms per call
    python scripts/bench_agent.py --provider gemini   # real model (Gemini; GEMINI_API_KEY from .env); 5 RPM free tier
    python scripts/bench_agent.py --json out.json     # also write raw numbers

Reports p50 / p95 per tool (from the trace), per turn (end to end, first event to `final`), and GET /agent/suggestions
over HTTP (JWT verify + identity + store + entity cache). Nothing here talks to production.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent_fixture_env as fx  # noqa: E402

NOV = {"date_from": "2026-11-01", "date_to": "2026-11-05"}
OCT_NOV = {"date_from": "2026-10-01", "date_to": "2026-11-05"}
# (caller, question, LLM plan: list of steps, each a list of (tool, args) called in ONE model response)
QUESTIONS = [
    ("sharma", "Meri classes dikhao", [[("my_scope", {})]]),
    ("admin", "Sharma ji ki is hafte ki attendance",
     [[("resolve_entity", {"kind": "teacher", "name": "Sharma"})],
      [("attendance_stats", {"scope": "teacher", "entity_id": fx.U["sharma"], **NOV})]]),
    ("admin", "Is mahine sabse kam attendance kis class ki hai?",
     [[("section_comparison", {"metric": "attendance", **NOV})]]),
    ("admin", "75% se kam attendance wale students", [[("low_attendance_list", {"threshold": 75, **NOV})]]),
    ("hod", "ECE-2 ka attendance trend", [[("resolve_entity", {"kind": "section", "name": "ECE-2"})],
                                           [("attendance_trend", {"scope": "section", "entity_id": fx.ECE2,
                                                                  "granularity": "day", **NOV})]]),
    ("admin", "ECE-2 ke marks", [[("marks_stats", {"scope": "section", "entity_id": fx.ECE2})]]),
    ("admin", "Is mahine kitne tests bane?", [[("tests_created_by_teacher", OCT_NOV)]]),
    ("hod", "Classes ke marks compare karo", [[("section_comparison", {"metric": "marks", **OCT_NOV})]]),
    ("admin", "Rahul Kumar ki attendance", [[("resolve_entity", {"kind": "student", "name": "Rahul Kumar"})]]),
    ("admin", "ECE-2 ki attendance aur marks dono",  # two independent reads in one model response
     [[("attendance_stats", {"scope": "section", "entity_id": fx.ECE2, **NOV}),
       ("marks_stats", {"scope": "section", "entity_id": fx.ECE2})]]),
]


class ScriptedLLM:
    """Mock LLM: follows the plan, then answers with one sentence. Optional fixed latency per call."""
    supports_cancel = False

    def __init__(self, plan, llm_ms: float, model: str = "mock-llm"):
        self.plan, self.llm_ms, self.model, self.i = plan, llm_ms, model, 0

    def complete(self, system, messages, tools, max_tokens=2048, json_mode=False, **kw):
        from agent.llm import LLMResponse, ToolCall
        if self.llm_ms:
            time.sleep(self.llm_ms / 1000)
        step = self.plan[self.i] if self.i < len(self.plan) else None
        self.i += 1
        if step is None:
            return LLMResponse(text="Attendance theek hai.", tokens_in=1200, tokens_out=20, model=self.model)
        return LLMResponse(tool_calls=[ToolCall(f"t{self.i}_{k}", n, a) for k, (n, a) in enumerate(step)],
                           tokens_in=1200, tokens_out=40, model=self.model)


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = max(0, min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1)))))
    return round(xs[k], 1)


def run(args) -> dict:
    from agent import config
    from agent.core import Agent
    from agent.real_identity import ctx_for_real_user
    trace = Path(tempfile.mkdtemp()) / "trace.jsonl"
    config.TRACE_PATH = str(trace)
    ctxs = {k: ctx_for_real_user(v) for k, v in fx.U.items() if k in ("admin", "hod", "sharma")}
    turns: dict[str, list[float]] = {}
    real = None
    if args.provider == "gemini":
        from agent.llm import get_provider
        real = get_provider("gemini")
    for _ in range(args.reps):
        for who, q, plan in QUESTIONS:
            prov = real or ScriptedLLM(plan, args.llm_ms)
            agent = Agent(prov, ctxs[who], str(uuid.uuid4()), request_id=uuid.uuid4().hex[:16])
            t0 = time.perf_counter()
            evs = list(agent.run(q))
            turns.setdefault(q, []).append((time.perf_counter() - t0) * 1000)
            assert evs[-1]["type"] == "final", evs[-1]
            errs = [e for e in evs if e["type"] == "error"]
            if errs:
                print(f"  ! {q}: {errs[0]['code']} {errs[0]['message'][:60]}")
            if real is not None:
                time.sleep(args.min_interval)
    tools: dict[str, list[float]] = {}
    llm: list[float] = []
    cache = {"in": 0, "cache_read": 0}
    for line in trace.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("kind") == "tool":
            tools.setdefault(r["tool"], []).append(r["latency_ms"])
        elif r.get("kind") == "llm":
            llm.append(r["latency_ms"])
            cache["in"] += r.get("tokens_in", 0) + r.get("cache_read", 0) + r.get("cache_write", 0)
            cache["cache_read"] += r.get("cache_read", 0)
    all_turns = [x for v in turns.values() for x in v]
    return {"tools": {k: {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95)} for k, v in sorted(tools.items())},
            "turns": {k: {"n": len(v), "p50": pct(v, 50), "p95": pct(v, 95)} for k, v in turns.items()},
            "turn_all": {"n": len(all_turns), "p50": pct(all_turns, 50), "p95": pct(all_turns, 95)},
            "llm_calls": {"n": len(llm), "p50": pct(llm, 50), "p95": pct(llm, 95)},
            "prompt_cache": {**cache, "hit_rate": round(cache["cache_read"] / cache["in"], 3) if cache["in"] else None}}


def bench_suggestions(n: int) -> dict:
    from fastapi.testclient import TestClient

    from agent import api
    app = api.create_app(os.environ, check_real_db=False)
    c = TestClient(app)
    h = {"Authorization": f"Bearer {fx.token(fx.U['admin'])}"}
    assert c.get("/agent/suggestions?q=ec", headers=h).status_code == 200  # warm (entity cache, identity cache)
    xs = []
    for i in range(n):
        t0 = time.perf_counter()
        r = c.get(f"/agent/suggestions?q={'ece'[: 1 + i % 3]}", headers=h)
        xs.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
    return {"n": n, "p50": pct(xs, 50), "p95": pct(xs, 95)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--llm-ms", type=float, default=0)
    ap.add_argument("--provider", choices=["mock", "gemini"], default="mock")
    ap.add_argument("--min-interval", type=float, default=13.0, help="seconds between real-model turns (5 RPM)")
    ap.add_argument("--suggest-n", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args()
    if a.provider == "gemini":
        os.environ["AGENT_PROVIDER"] = "gemini"
        if not fx.load_dotenv_key("GEMINI_API_KEY"):
            sys.exit("GEMINI_API_KEY not found")
        a.reps = min(a.reps, 1)
    fx.setup()
    t0 = time.perf_counter()
    out = run(a)
    out["suggestions_http"] = bench_suggestions(a.suggest_n)
    out["env"] = {"provider": a.provider, "llm_ms": a.llm_ms, "reps": a.reps,
                  "db_pool": os.getenv("AGENT_DB_POOL", "1"), "tool_concurrency": os.getenv("AGENT_TOOL_CONCURRENCY"),
                  "wall_s": round(time.perf_counter() - t0, 1)}
    print(f"\nprovider={a.provider} llm_ms={a.llm_ms} reps={a.reps}")
    print(f"{'tool':28} {'n':>4} {'p50 ms':>8} {'p95 ms':>8}")
    for k, v in out["tools"].items():
        print(f"{k:28} {v['n']:>4} {v['p50']:>8} {v['p95']:>8}")
    print(f"\n{'turn (end to end)':52} {'n':>4} {'p50 ms':>8} {'p95 ms':>8}")
    for k, v in out["turns"].items():
        print(f"{k[:52]:52} {v['n']:>4} {v['p50']:>8} {v['p95']:>8}")
    t = out["turn_all"]
    print(f"{'ALL TURNS':52} {t['n']:>4} {t['p50']:>8} {t['p95']:>8}")
    print(f"LLM calls: {out['llm_calls']}  prompt cache: {out['prompt_cache']}")
    s = out["suggestions_http"]
    print(f"\nGET /agent/suggestions over HTTP: n={s['n']} p50={s['p50']} ms p95={s['p95']} ms")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
