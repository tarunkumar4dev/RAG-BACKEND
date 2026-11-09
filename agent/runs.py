"""Running /agent/chat requests, so POST /agent/chat/{request_id}/cancel can stop one. Process-local.
A request can only be cancelled by the user + institute that started it (anything else looks like 'not found')."""
from __future__ import annotations

import threading
from typing import Optional

from .store import Owner


class RunRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, tuple[Owner, threading.Event]] = {}

    def register(self, request_id: str, owner: Owner) -> threading.Event:
        ev = threading.Event()
        with self._lock:
            self._runs[request_id] = (owner, ev)
        return ev

    def cancel(self, request_id: str, owner: Owner) -> bool:
        with self._lock:
            hit: Optional[tuple[Owner, threading.Event]] = self._runs.get(request_id)
        if hit is None or hit[0] != owner:
            return False
        hit[1].set()
        return True

    def unregister(self, request_id: str) -> None:
        with self._lock:
            self._runs.pop(request_id, None)

    def running(self) -> int:
        with self._lock:
            return len(self._runs)


RUNS = RunRegistry()
