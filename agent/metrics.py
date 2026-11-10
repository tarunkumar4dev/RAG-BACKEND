"""In-process Prometheus metrics (text exposition 0.0.4), served by GET /agent/metrics.

No PII by construction: label values come from fixed vocabularies only — route templates (never raw paths), tool
names from the registry (anything else becomes "unknown"), configured model ids, error codes, route reasons. No user,
institute, session or request ids, no text.

Per process: with several gunicorn workers each worker reports its own numbers (scrape each worker, or run one
worker per container; see agent/README_DEPLOY.md).
"""
from __future__ import annotations

import threading
from typing import Optional

BUCKETS = {
    "agent_tool_seconds": (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
    "agent_llm_seconds": (0.25, 0.5, 1, 2, 3, 5, 8, 13, 20, 40, 90),
    "agent_chat_turn_seconds": (0.25, 0.5, 1, 1.5, 2, 2.5, 3, 5, 8, 13, 30, 60, 150),
    "agent_http_request_seconds": (0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.5, 1, 2.5, 5),
}
HELP = {
    "agent_http_requests_total": ("counter", "HTTP requests by method, route template and status"),
    "agent_http_request_seconds": ("histogram", "HTTP request latency (time to response start for SSE)"),
    "agent_chat_turns_total": ("counter", "Chat turns by route (fast/main) and outcome"),
    "agent_chat_turn_seconds": ("histogram", "Chat turn wall time, first event to final"),
    "agent_errors_total": ("counter", "Error events streamed, by code"),
    "agent_tool_calls_total": ("counter", "Tool calls by tool and status"),
    "agent_tool_seconds": ("histogram", "Tool latency by tool"),
    "agent_llm_calls_total": ("counter", "Model calls by model"),
    "agent_llm_seconds": ("histogram", "Model call latency by model"),
    "agent_llm_retries_total": ("counter", "Model call retries (429/5xx/overloaded/connection)"),
    "agent_llm_aborts_total": ("counter", "Model streams closed by a cancel"),
    "agent_tokens_total": ("counter", "Tokens by model and kind (in, out, cache_read, cache_write)"),
    "agent_cost_usd_total": ("counter", "Estimated model cost in USD by model"),
    "agent_prompt_cache_hits_total": ("counter", "Model calls that read from the prompt cache"),
    "agent_route_total": ("counter", "Routing decisions by route and reason"),
    "agent_rate_limit_hits_total": ("counter", "Chat requests refused by a rate limit, by scope"),
    "agent_runs_in_progress": ("gauge", "Chat streams running in this process"),
    "agent_db_pool_connections": ("gauge", "DB pool connections by pool and state"),
}


def _known_tool(name: str) -> str:
    try:
        from .tools.registry import TOOLS
        return name if name in TOOLS else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _labels(d: dict) -> str:
    return "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in sorted(d.items())) + "}" if d else ""


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._c: dict[tuple, float] = {}
        self._h: dict[tuple, list] = {}

    def reset(self) -> None:
        with self._lock:
            self._c.clear()
            self._h.clear()

    def inc(self, name: str, labels: Optional[dict] = None, v: float = 1.0) -> None:
        k = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._c[k] = self._c.get(k, 0.0) + v

    def observe(self, name: str, labels: Optional[dict], value: float) -> None:
        k = (name, tuple(sorted((labels or {}).items())))
        b = BUCKETS[name]
        with self._lock:
            h = self._h.get(k)
            if h is None:
                h = self._h[k] = [0] * len(b) + [0.0, 0]
            for i, le in enumerate(b):
                if value <= le:
                    h[i] += 1
            h[-2] += value
            h[-1] += 1

    def value(self, name: str, labels: Optional[dict] = None) -> float:
        return self._c.get((name, tuple(sorted((labels or {}).items()))), 0.0)

    # -- domain helpers ------------------------------------------------------------------------------------------
    def tool(self, name: str, seconds: float, ok: bool, status: str) -> None:
        t = _known_tool(name)
        st = status if status in ("ok", "ambiguous", "not_found", "forbidden", "error", "invalid", "invalid_args",
                                  "timeout", "db_unavailable", "pending_approval", "read_only", "saved",
                                  "blocked", "empty") else ("ok" if ok else "error")
        self.inc("agent_tool_calls_total", {"tool": t, "status": st})
        self.observe("agent_tool_seconds", {"tool": t}, seconds)

    def llm(self, model: str, seconds: float, tin: int, tout: int, cread: int, cwrite: int, cost: float) -> None:
        m = {"model": model}
        self.inc("agent_llm_calls_total", m)
        self.observe("agent_llm_seconds", m, seconds)
        for kind, v in (("in", tin), ("out", tout), ("cache_read", cread), ("cache_write", cwrite)):
            if v:
                self.inc("agent_tokens_total", {"model": model, "kind": kind}, v)
        if cread:
            self.inc("agent_prompt_cache_hits_total", m)
        if cost:
            self.inc("agent_cost_usd_total", m, cost)

    def route(self, route: str, reason: str) -> None:
        self.inc("agent_route_total", {"route": route, "reason": reason})

    def turn(self, route: str, seconds: float, outcome: str) -> None:
        self.inc("agent_chat_turns_total", {"route": route, "outcome": outcome})
        self.observe("agent_chat_turn_seconds", {"route": route}, seconds)

    def error(self, code: str) -> None:
        self.inc("agent_errors_total", {"code": code})

    def request(self, method: str, route: str, status: int, seconds: float) -> None:
        self.inc("agent_http_requests_total", {"method": method, "route": route, "status": str(status)})
        self.observe("agent_http_request_seconds", {"route": route}, seconds)

    # -- exposition ----------------------------------------------------------------------------------------------
    def render(self) -> str:
        with self._lock:
            counters = dict(self._c)
            hists = {k: list(v) for k, v in self._h.items()}
        lines: list[str] = []
        seen: set[str] = set()

        def head(name: str) -> None:
            if name not in seen:
                seen.add(name)
                typ, txt = HELP.get(name, ("untyped", name))
                lines.append(f"# HELP {name} {txt}")
                lines.append(f"# TYPE {name} {typ}")

        for (name, lab), v in sorted(counters.items()):
            head(name)
            lines.append(f"{name}{_labels(dict(lab))} {v:g}")
        for (name, lab), h in sorted(hists.items()):
            head(name)
            b = BUCKETS[name]
            base = dict(lab)
            for i, le in enumerate(b):
                lines.append(f"{name}_bucket{_labels({**base, 'le': f'{le:g}'})} {h[i]}")
            lines.append(f"{name}_bucket{_labels({**base, 'le': '+Inf'})} {h[-1]}")
            lines.append(f"{name}_sum{_labels(base)} {h[-2]:.6f}")
            lines.append(f"{name}_count{_labels(base)} {h[-1]}")
        try:
            from .runs import RUNS
            head("agent_runs_in_progress")
            lines.append(f"agent_runs_in_progress {RUNS.running()}")
        except Exception:  # noqa: BLE001
            pass
        try:
            from .pool import stats
            for pname, st in sorted(stats().items()):
                head("agent_db_pool_connections")
                for state in ("size", "available", "waiting"):
                    lines.append(f"agent_db_pool_connections{_labels({'pool': pname, 'state': state})} {st[state]}")
        except Exception:  # noqa: BLE001
            pass
        return "\n".join(lines) + "\n"


METRICS = Metrics()
