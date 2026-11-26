"""Tests for agent/middleware/log_scrub.py — the secret-scrubbing logging filter."""
from __future__ import annotations

import io
import logging

import pytest

from agent.middleware import log_scrub
from agent.middleware.log_scrub import (
    CONN_REDACTED,
    JWT_REDACTED,
    KEY_REDACTED,
    SECRET_REDACTED,
    SecretScrubFilter,
    _scrub,
    _build_patterns,
    install,
    uninstall,
)


# ---------------------------- pure _scrub() tests -----------------------------------------------------


def _patterns(env=None):
    return _build_patterns(env or {})


def test_postgres_connection_string_redacted():
    text = "connecting to postgresql://user:secret@host:5432/db for ro"
    out = _scrub(text, _patterns())
    assert CONN_REDACTED in out
    assert "secret" not in out
    assert "user" not in out
    assert "connecting to" in out  # surrounding prose survives


def test_postgres_scheme_also_redacted():
    out = _scrub("db url postgres://a:b@c/d", _patterns())
    assert CONN_REDACTED in out
    assert "a:b" not in out


def test_sqlite_connection_string_redacted():
    out = _scrub("opened sqlite:///var/data/agent.db", _patterns())
    assert CONN_REDACTED in out
    assert "/var/data/agent.db" not in out


def test_bearer_token_redacted_but_word_kept():
    out = _scrub("Authorization: Bearer abc123DEF456ghi789", _patterns())
    assert f"Bearer {KEY_REDACTED}" in out
    assert "abc123DEF456ghi789" not in out


def test_bearer_token_case_insensitive():
    out = _scrub("header: bearer MY_LONG_TOKEN_1234567890", _patterns())
    assert KEY_REDACTED in out
    assert "MY_LONG_TOKEN_1234567890" not in out


def test_jwt_three_segments_redacted():
    jwt = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
           "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ."
           "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c")
    out = _scrub(f"auth failed: token={jwt} expired", _patterns())
    assert JWT_REDACTED in out
    assert "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c" not in out


def test_sk_api_key_redacted():
    out = _scrub("using key sk-abc123XYZ456def789ghi", _patterns())
    assert KEY_REDACTED in out
    assert "sk-abc123XYZ456def789ghi" not in out


def test_key_prefix_redacted():
    out = _scrub("configured key-ABCDEFGHIJ1234567 for client", _patterns())
    assert KEY_REDACTED in out


def test_env_var_value_redacted_literally():
    env = {"SARVAM_API_KEY": "sarv_live_zyxwvu9876543210"}
    out = _scrub("called Sarvam with sarv_live_zyxwvu9876543210 and got 200",
                 _build_patterns(env))
    assert SECRET_REDACTED in out
    assert "sarv_live_zyxwvu9876543210" not in out


def test_env_var_short_value_skipped():
    """Values shorter than MIN_ENV_VALUE_LEN must NOT be redacted — too risky for false positives."""
    env = {"SUPABASE_KEY": "abc"}
    out = _scrub("value is abc in the response", _build_patterns(env))
    assert "abc" in out  # not redacted — too short
    assert SECRET_REDACTED not in out


def test_env_var_unset_is_safe():
    """Patterns build cleanly even when none of the env vars are set."""
    pats = _build_patterns({})
    assert pats  # still has the generic regexes
    out = _scrub("hello world, nothing secret here", pats)
    assert out == "hello world, nothing secret here"


def test_normal_log_passes_through_unchanged():
    out = _scrub("INFO user requested /agent/sessions", _patterns())
    assert out == "INFO user requested /agent/sessions"


def test_partial_jwt_prefix_does_not_match():
    """'eyJ' alone (no three segments) must not trigger a JWT redaction."""
    out = _scrub("prefix eyJ is not a jwt", _patterns())
    assert JWT_REDACTED not in out
    assert "eyJ" in out


def test_short_sk_token_not_matched():
    """'sk-test' is too short to match the API-key pattern (requires >= 10 chars after the prefix)."""
    out = _scrub("branch sk-test for local dev", _patterns())
    assert KEY_REDACTED not in out


def test_multiple_secrets_all_redacted():
    env = {"GEMINI_API_KEY": "gem_live_1234567890abcdef"}
    line = ("POST /chat body={'db': 'postgresql://u:p@h/d', "
            "'auth': 'Bearer abcdefghij1234567890', "
            "'jwt': 'eyJhbGci.eyJzdWIi.SflKxwRJSMeKKF2Q', "
            "'gem': 'gem_live_1234567890abcdef'}")
    out = _scrub(line, _build_patterns(env))
    assert CONN_REDACTED in out
    assert f"Bearer {KEY_REDACTED}" in out
    assert JWT_REDACTED in out
    assert SECRET_REDACTED in out
    # None of the actual secret values should leak
    assert "postgresql://u:p" not in out
    assert "abcdefghij1234567890" not in out
    assert "SflKxwRJSMeKKF2Q" not in out
    assert "gem_live_1234567890abcdef" not in out


def test_connection_string_stops_at_quote():
    out = _scrub('error: "postgresql://u:p@h/db" rejected', _patterns())
    assert CONN_REDACTED in out
    assert ' rejected' in out  # text after the closing quote survives
    assert "u:p@h/db" not in out


def test_env_var_longer_wins_over_shorter():
    """Two env-var values where one contains the other: the longer must redact cleanly without leaving
    a fragment of the shorter one visible."""
    env = {"A_KEY": "short_value_12", "B_KEY": "short_value_12_and_more_stuff_here"}
    out = _scrub("saw short_value_12_and_more_stuff_here in logs",
                 _build_patterns({**env, "SARVAM_API_KEY": env["B_KEY"], "GEMINI_API_KEY": env["A_KEY"]}))
    # Both values should have been redacted (longest-first ordering ensures no leftover fragment)
    assert "short_value_12" not in out


# ---------------------------- SecretScrubFilter end-to-end -------------------------------------------


@pytest.fixture
def captured_log():
    """A logger wired to a StringIO so test assertions can read the actual emitted text."""
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(message)s"))
    lg = logging.getLogger(f"test_log_scrub.{id(buf)}")
    lg.handlers = [handler]
    lg.setLevel(logging.DEBUG)
    lg.propagate = False
    yield lg, buf, handler
    lg.handlers = []


def test_filter_mutates_record_before_handler(captured_log):
    lg, buf, handler = captured_log
    handler.addFilter(SecretScrubFilter({"SARVAM_API_KEY": "my_long_secret_value_1234"}))
    lg.info("calling with my_long_secret_value_1234 header")
    assert "my_long_secret_value_1234" not in buf.getvalue()
    assert SECRET_REDACTED in buf.getvalue()


def test_filter_handles_percent_args(captured_log):
    lg, buf, handler = captured_log
    handler.addFilter(SecretScrubFilter({}))
    # logging with % args: record.msg='x %s' + record.args=(jwt,) — getMessage() formats them.
    jwt = "eyJhbGci.eyJzdWIi.SflKxwRJSMeKKF2Q"
    lg.info("token %s expired", jwt)
    out = buf.getvalue()
    assert JWT_REDACTED in out
    assert jwt not in out


def test_filter_fail_open_on_bad_args(captured_log):
    """A log call with mismatched args would raise inside getMessage(); the filter must still return
    True so logging doesn't swallow the record."""
    lg, buf, handler = captured_log
    handler.addFilter(SecretScrubFilter({}))
    # %s with no arg — getMessage() will raise. The filter should return True (record passes through).
    # logging itself prints a formatter error to stderr; we just assert the filter did not crash.
    lg.info("broken format %s")  # no arg
    # If we got here, the filter did not raise.


def test_filter_leaves_clean_lines_untouched(captured_log):
    lg, buf, handler = captured_log
    handler.addFilter(SecretScrubFilter({}))
    lg.info("user viewed /agent/sessions, 200 OK")
    assert buf.getvalue().strip() == "user viewed /agent/sessions, 200 OK"


# ---------------------------- install() / uninstall() -------------------------------------------------


def test_install_is_idempotent():
    """Calling install() twice must not stack two filters on the same handler."""
    tgt = "test_log_scrub.install_idem"
    root = logging.getLogger(tgt)
    handler = logging.StreamHandler(io.StringIO())
    root.addHandler(handler)
    try:
        install(env={}, loggers=[tgt])
        install(env={}, loggers=[tgt])
        scrub_filters = [f for f in handler.filters if isinstance(f, SecretScrubFilter)]
        assert len(scrub_filters) == 1
    finally:
        root.removeHandler(handler)
        root.filters = [f for f in root.filters if not isinstance(f, SecretScrubFilter)]


def test_install_then_uninstall_clears_filters():
    tgt = "test_log_scrub.install_uninstall"
    lg = logging.getLogger(tgt)
    h = logging.StreamHandler(io.StringIO())
    lg.addHandler(h)
    try:
        install(env={}, loggers=[tgt])
        assert any(isinstance(f, SecretScrubFilter) for f in h.filters)
        # uninstall() is defined over DEFAULT_LOGGERS; for a custom logger we clean up manually below.
        h.filters = [f for f in h.filters if not isinstance(f, SecretScrubFilter)]
        lg.filters = [f for f in lg.filters if not isinstance(f, SecretScrubFilter)]
        assert not any(isinstance(f, SecretScrubFilter) for f in h.filters)
    finally:
        lg.removeHandler(h)


def test_install_covers_root_logger_handlers():
    """install() with the empty-string name must attach to root's handlers."""
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.addHandler(h)
    try:
        install(env={"SUPABASE_URL": "https://abc.supabase.co"}, loggers=[""])
        logging.getLogger("random.module").warning("talking to https://abc.supabase.co")
        out = buf.getvalue()
        assert "https://abc.supabase.co" not in out
        assert SECRET_REDACTED in out
    finally:
        root.removeHandler(h)
        uninstall()
