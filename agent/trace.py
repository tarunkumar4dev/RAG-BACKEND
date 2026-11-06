"""JSONL trace: tool name, arg KEYS (never values), latency, tokens, cost. No student names / marks."""
from __future__ import annotations

import json
import time
from typing import Optional

from . import config


class Tracer:
    def __init__(self, path: Optional[str] = None, request_id: str = ""):
        self.path = path or config.TRACE_PATH
        self.request_id = request_id

    def _write(self, rec: dict) -> None:
        rec = {"ts": round(time.time(), 3), "request_id": self.request_id, **rec}
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except OSError:  # tracing must never break a chat turn (bad path, locked file on Windows)
            pass

    def tool(self, name: str, args: dict, latency_ms: float, ok: bool, status: str = "") -> None:
        self._write({"kind": "tool", "tool": name, "arg_keys": sorted(args.keys()), "latency_ms": round(latency_ms, 1),
                     "ok": ok, "status": status[:32]})

    def llm(self, model: str, tokens_in: int, tokens_out: int, latency_ms: float) -> float:
        pin, pout = config.PRICES.get(model, (0.0, 0.0))
        cost = (tokens_in * pin + tokens_out * pout) / 1e6
        self._write({"kind": "llm", "model": model, "tokens_in": tokens_in, "tokens_out": tokens_out,
                     "latency_ms": round(latency_ms, 1), "cost_usd": round(cost, 6)})
        return cost
