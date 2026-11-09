"""In-process Prometheus metrics (text format 0.0.4). Labels are fixed vocabularies (endpoint template, tool name,
model id, error code) — never user text, user ids or institute ids."""
from __future__ import annotations

import threading


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()

    def render(self) -> str:
        return "# a4ai agent metrics\n"


METRICS = Metrics()
