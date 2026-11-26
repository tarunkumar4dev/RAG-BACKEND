"""Secret-scrubbing logging filter. Replaces credentials, connection strings, API keys and JWTs in log
messages BEFORE handlers format them, so Render / stdout never see a secret.

Why a logging.Filter and not FastAPI middleware: credentials leak from places that are not on the request
path — provider errors, warmup threads, startup DB probes, uvicorn access lines. A FastAPI middleware only
wraps HTTP req/resp and would miss all of those. A filter attached to handlers scrubs every log line
regardless of where it was emitted.

Guarantees:
- Runs synchronously inside logging.Handler.handle(), after record formatting. Zero latency impact on the
  request path (logging is already off the hot path for /agent/chat streaming).
- Fail open: any exception inside the filter returns True and lets the record through unchanged. Scrubbing
  must never crash logging.
- Idempotent install: calling install() twice does not stack filters.

Patterns (ordered — env-var literals first so they win over the generic regexes):
  1. Env-var literal values (AGENT_REAL_DB_URL, AGENT_STORE_URL, SARVAM_API_KEY, GEMINI_API_KEY,
     SUPABASE_KEY, SUPABASE_URL): the user-set value is redacted anywhere it appears. Values shorter than
     MIN_ENV_VALUE_LEN are skipped to avoid false positives on short tokens.
  2. Connection strings: postgresql:// postgres:// sqlite:/// mysql:// redis:// amqp://
  3. Bearer tokens: "Bearer <opaque>" -> "Bearer ***REDACTED_KEY***"
  4. JWTs: eyJ...<.>...<.>...
  5. API-key prefixes: sk-* / key-* / pk-*
"""
from __future__ import annotations

import logging
import os
import re
from typing import Iterable, Optional

CONN_REDACTED = "***REDACTED_CONN***"
KEY_REDACTED = "***REDACTED_KEY***"
JWT_REDACTED = "***REDACTED_JWT***"
SECRET_REDACTED = "***REDACTED_SECRET***"

# Env vars whose exact values should be redacted in any log line.
ENV_VAR_NAMES: tuple[str, ...] = (
    "AGENT_REAL_DB_URL",
    "AGENT_STORE_URL",
    "SARVAM_API_KEY",
    "GEMINI_API_KEY",
    "SUPABASE_KEY",
    "SUPABASE_URL",
)

# Skip env-var values below this length: too likely to false-positive on short strings.
MIN_ENV_VALUE_LEN = 8

# Loggers whose handlers get the filter attached. Empty string = the root logger.
DEFAULT_LOGGERS: tuple[str, ...] = (
    "",
    "agent",
    "agent.provider",
    "uvicorn",
    "uvicorn.access",
    "uvicorn.error",
    "gunicorn.error",
    "gunicorn.access",
)


def _build_patterns(env) -> list[tuple[re.Pattern[str], str]]:
    """Ordered list of (compiled regex, replacement). Earlier patterns win."""
    patterns: list[tuple[re.Pattern[str], str]] = []

    # 1. Env-var literal values first — exact strings, longest first so a value that contains another
    # shorter secret still matches as a whole.
    values: set[str] = set()
    for name in ENV_VAR_NAMES:
        raw = env.get(name)
        if not raw:
            continue
        v = raw.strip()
        if len(v) >= MIN_ENV_VALUE_LEN:
            values.add(v)
    for v in sorted(values, key=len, reverse=True):
        patterns.append((re.compile(re.escape(v)), SECRET_REDACTED))

    # 2. Connection strings. The terminator set includes whitespace and common quoting characters so a
    # URL embedded in JSON, a Python repr or an error message stops at its boundary.
    patterns.append((
        re.compile(r'\b(?:postgresql|postgres|mysql|mariadb|redis|rediss|amqp|amqps|mongodb)://[^\s"\'<>`\\]+'),
        CONN_REDACTED,
    ))
    patterns.append((
        re.compile(r'\bsqlite:///?[^\s"\'<>`\\]*'),
        CONN_REDACTED,
    ))

    # 3. Bearer tokens — keep the "Bearer" word visible for log readability.
    patterns.append((
        re.compile(r'(?i)\bBearer\s+[A-Za-z0-9._\-+/=]{8,}'),
        f"Bearer {KEY_REDACTED}",
    ))

    # 4. JWTs — three base64url segments joined by '.'. Minimum segment length 4 keeps the match tight
    # enough that a word that happens to start with "eyJ" won't match.
    patterns.append((
        re.compile(r'\beyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\b'),
        JWT_REDACTED,
    ))

    # 5. Common API-key prefixes. Length >= 10 avoids matching e.g. "sk-test" in prose.
    patterns.append((
        re.compile(r'\b(?:sk|key|pk|api_key|apikey)-[A-Za-z0-9_\-]{10,}'),
        KEY_REDACTED,
    ))

    return patterns


def _scrub(text: str, patterns: Iterable[tuple[re.Pattern[str], str]]) -> str:
    for pat, repl in patterns:
        text = pat.sub(repl, text)
    return text


class SecretScrubFilter(logging.Filter):
    """Mutates record.msg to a scrubbed, pre-formatted string. Never drops a record."""

    def __init__(self, env: Optional[dict] = None):
        super().__init__()
        self._patterns = _build_patterns(env if env is not None else os.environ)

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        try:
            try:
                msg = record.getMessage()
            except Exception:
                return True  # bad args — let the formatter deal with it, don't crash
            scrubbed = _scrub(msg, self._patterns)
            if scrubbed != msg:
                record.msg = scrubbed
                record.args = ()
        except Exception:
            # Fail open: logging must never break because of scrubbing.
            return True
        return True


_INSTALLED: list[SecretScrubFilter] = []


def install(env: Optional[dict] = None, loggers: Optional[Iterable[str]] = None) -> SecretScrubFilter:
    """Attach the filter to every handler of the named loggers. Idempotent per handler.

    Call AFTER logging is configured (uvicorn/gunicorn handlers must already be in place). Any handler
    added later can be covered by calling install() again — it re-walks the loggers and only attaches to
    handlers that don't already have the filter.
    """
    f = SecretScrubFilter(env)
    names = tuple(loggers) if loggers is not None else DEFAULT_LOGGERS
    for name in names:
        lg = logging.getLogger(name)
        for h in lg.handlers:
            if not any(isinstance(x, SecretScrubFilter) for x in h.filters):
                h.addFilter(f)
        # Also attach to the logger itself so records emitted directly (not propagated) still pass through.
        if not any(isinstance(x, SecretScrubFilter) for x in lg.filters):
            lg.addFilter(f)
    _INSTALLED.append(f)
    return f


def uninstall() -> None:
    """Remove every scrub filter we installed. Test-only helper."""
    for name in DEFAULT_LOGGERS:
        lg = logging.getLogger(name)
        lg.filters = [x for x in lg.filters if not isinstance(x, SecretScrubFilter)]
        for h in lg.handlers:
            h.filters = [x for x in h.filters if not isinstance(x, SecretScrubFilter)]
    _INSTALLED.clear()
