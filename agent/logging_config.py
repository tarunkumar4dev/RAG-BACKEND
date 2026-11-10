"""Structured JSON logging for the agent. Every log line is valid JSON with `request_id` only (no PII).

AGENT_LOG_FORMAT=json (default in production) switches the agent logger to JSON. In dev mode or when unset,
the standard text format is kept. The uvicorn access log redaction (RedactAgentQueryStrings) stays regardless.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional


class JSONFormatter(logging.Formatter):
    """One JSON object per line: {ts, level, logger, message, request_id (if in extra)}. No PII."""

    def format(self, record: logging.LogRecord) -> str:
        d = {
            "ts": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # request_id is threaded through logging.extra when available
        rid = getattr(record, "request_id", None)
        if rid:
            d["request_id"] = rid
        if record.exc_info and record.exc_info[1]:
            d["error"] = type(record.exc_info[1]).__name__
        return json.dumps(d, ensure_ascii=False)


def configure(name: str = "agent") -> None:
    """Call once at startup. AGENT_LOG_FORMAT=json → JSON on the agent.* loggers; otherwise default."""
    fmt = (os.getenv("AGENT_LOG_FORMAT") or "").strip().lower()
    if fmt != "json":
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    lg = logging.getLogger(name)
    lg.handlers = [handler]
    lg.setLevel(logging.INFO)
    lg.propagate = False
