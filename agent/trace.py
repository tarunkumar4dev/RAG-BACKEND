"""JSONL trace: tool name, arg KEYS (never values), latency, tokens, cost, route. No student names / marks / text.

Every record also feeds the in-process metrics (agent/metrics.py), so /agent/metrics and the trace never disagree.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Optional

from . import config


_WRITE_LOCK = threading.Lock()  # parallel tools trace from several threads: one line per write, never interleaved


def _family(model: str) -> str:
    return "gemini" if model.startswith("gemini") else "anthropic"


def llm_cost(model: str, tokens_in: int, tokens_out: int, cache_read: int = 0, cache_write: int = 0) -> float:
    pin, pout = config.PRICES.get(model, (0.0, 0.0))
    read_x = config.CACHE_READ_X.get(_family(model), 0.1)
    return (tokens_in * pin + cache_read * pin * read_x + cache_write * pin * config.CACHE_WRITE_X
            + tokens_out * pout) / 1e6


class Tracer:
    def __init__(self, path: Optional[str] = None, request_id: str = ""):
        self.path = path or config.TRACE_PATH
        self.request_id = request_id

    def _write(self, rec: dict) -> None:
        rec = {"ts": round(time.time(), 3), "request_id": self.request_id, **rec}
        try:
            with _WRITE_LOCK:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except OSError:  # tracing must never break a chat turn (bad path, locked file on Windows)
            pass

    def tool(self, name: str, args: dict, latency_ms: float, ok: bool, status: str = "") -> None:
        self._write({"kind": "tool", "tool": name, "arg_keys": sorted(args.keys()), "latency_ms": round(latency_ms, 1),
                     "ok": ok, "status": status[:32]})
        from .metrics import METRICS
        METRICS.tool(name, latency_ms / 1000, ok, status)

    def llm(self, model: str, tokens_in: int, tokens_out: int, latency_ms: float, cache_read: int = 0,
            cache_write: int = 0) -> float:
        cost = llm_cost(model, tokens_in, tokens_out, cache_read, cache_write)
        self._write({"kind": "llm", "model": model, "tokens_in": tokens_in, "tokens_out": tokens_out,
                     "cache_read": cache_read, "cache_write": cache_write,
                     "latency_ms": round(latency_ms, 1), "cost_usd": round(cost, 6)})
        from .metrics import METRICS
        METRICS.llm(model, latency_ms / 1000, tokens_in, tokens_out, cache_read, cache_write, cost)
        return cost

    def route(self, route: str, reason: str, model: str) -> None:
        self._write({"kind": "route", "route": route, "reason": reason, "model": model})
        from .metrics import METRICS
        METRICS.route(route, reason)

    def retry(self, model: str, attempt: int, error: str, delay_s: float) -> None:
        self._write({"kind": "llm_retry", "model": model, "attempt": attempt, "error": error[:40],
                     "delay_s": round(delay_s, 2)})
        from .metrics import METRICS
        METRICS.inc("agent_llm_retries_total", {"model": model})

    def llm_abort(self, model: str, tokens_out_est: int, max_tokens: int) -> None:
        """Cancel closed the provider stream. tokens_out_est = output streamed before the close (chars/4);
        tokens_saved_max = the rest of the max_tokens budget that can no longer be generated (an upper bound)."""
        saved = max(max_tokens - tokens_out_est, 0)
        self._write({"kind": "llm_abort", "model": model, "tokens_out_est": tokens_out_est,
                     "tokens_saved_max": saved})
        from .metrics import METRICS
        METRICS.inc("agent_llm_aborts_total", {"model": model})

    def voice(self, endpoint: str, outcome: str, stages_ms: dict, provider: str = "", model: str = "",
              audio_bytes: int = 0, duration_ms: int = 0, chars: int = 0, cost_inr: float = 0.0, **extra) -> None:
        """Voice layer record: endpoint (transcribe / speak / chat), outcome (ok / empty / clarified / rejected /
        failed / cached / answered / refused / navigated), per-stage ms, audio size + duration, provider, estimated
        INR cost. `extra` holds ids, flags, counts and reasons only — never transcript or speech text."""
        rec = {"kind": "voice", "endpoint": endpoint, "outcome": outcome, "provider": provider, "model": model,
               "audio_bytes": audio_bytes, "duration_ms": duration_ms, "chars": chars, "cost_inr": round(cost_inr, 5),
               "stages_ms": {k: round(v, 1) for k, v in stages_ms.items()}}
        for k, v in extra.items():
            if isinstance(v, (bool, int, float)) or v is None:
                rec[k] = v
            elif isinstance(v, str):
                rec[k] = v[:32]
            elif isinstance(v, (list, tuple)):
                rec[k] = [str(x)[:32] for x in v][:10]
        self._write(rec)
        from .metrics import METRICS
        METRICS.voice(endpoint, outcome, provider, stages_ms, cost_inr)
