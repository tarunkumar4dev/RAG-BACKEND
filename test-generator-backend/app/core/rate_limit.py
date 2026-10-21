"""
Rate limiting — who is counted and how much.

Key  = route TEMPLATE (e.g. /api/v1/copy-checker/sheets/{sheet_id}/check, never the raw path — raw paths with
       ids gave every id its own bucket, i.e. no limit at all)
     + the user id for requests with a bearer token (school WiFi = one IP for a whole class, so authenticated
       routes must never be limited per IP), else the client IP.
The user id is read from the JWT WITHOUT verifying it: it only picks the bucket. The route still verifies the
token, so a forged `sub` gets its own bucket and then a 401.

Limits are per minute. In-memory per instance (best effort on serverless; the money-relevant limits — daily
checker quota, generator plan quota — live in the database).
"""

import base64
import json
import re
from functools import lru_cache, partial
from typing import Optional

from starlette.routing import Match

WINDOW_S = 60

# Never limited: provider webhooks (signature-checked), internal worker calls (secret header), health
EXEMPT_PREFIXES = ("/api/v1/whatsapp", "/internal/", "/api/v1/internal/")
EXEMPT_PATHS = {"/health", "/", "/openapi.json", "/docs", "/redoc", "/favicon.ico", "/api/v1/payment/webhook"}

# Per user (authenticated) or per IP (anonymous), requests per minute. Keys: "METHOD template".
ROUTE_LIMITS = {
    # AI / Gemini — cost money
    "POST /api/v1/test-generator/generate-frontend": 20,
    "POST /api/v1/test-generator/generate": 20,
    "POST /api/v1/test-generator/verify-answers": 20,
    "POST /api/v1/test-generator/feedback": 30,
    "POST /api/v1/test-generator/export": 30,
    "POST /api/v1/test-generator/export-answer-key": 30,
    "POST /api/v1/modules/process": 10,
    "POST /api/v1/modules/{module_id}/generate-test": 20,
    "POST /api/v1/modules/{module_id}/generate-worksheet": 20,
    "POST /api/v1/worksheet/generate-direct": 20,
    "POST /modules/process": 10,
    "POST /modules/{module_id}/generate-test": 20,
    "POST /modules/{module_id}/generate-worksheet": 20,
    "POST /worksheet/generate-direct": 20,
    "POST /api/v1/community-quizzes/preview-video": 20,
    "POST /api/v1/community-quizzes": 20,
    "POST /api/v1/test-checker/extract-key": 10,
    "POST /api/v1/test-checker/check": 20,
    "POST /api/v1/test-checker/check-with-test": 20,
    "POST /api/v1/chat": 20,
    "POST /api/v1/smart-assignment-checker/check": 20,
    "POST /api/v1/smart-assignment-checker/assignments/{assignment_id}/rubric": 10,
    # a class of 40 submissions / copies checked 3 at a time from one teacher's browser
    "POST /api/v1/smart-assignment-checker/assignments/{assignment_id}/submissions/{submission_id}/check": 60,
    "POST /api/v1/copy-checker/sessions/{session_id}/rubric": 10,
    "POST /api/v1/copy-checker/sheets/{sheet_id}/check": 60,
    "PATCH /api/v1/copy-checker/sheets/{sheet_id}/questions/{q_no}/marks": 120,
    # payments
    "POST /api/v1/payment/create-order": 10,
    "POST /api/v1/payment/verify-payment": 20,
    # public, read-heavy (NCERT browser) — per IP
    "GET /api/v1/test-generator/ncert-questions": 1200,
    "GET /api/v1/test-generator/ncert-question-stats": 1200,
    "GET /api/v1/test-generator/chapters": 1200,
    "GET /api/v1/test-generator/templates": 1200,
    # students taking a contest / quiz, often a whole class behind one school IP
    "GET /api/v1/contests/{short_code}/info": 1200,
    "POST /api/v1/contests/{short_code}/start": 1200,
    "POST /api/v1/contests/{contest_id}/submit": 1200,
    "GET /api/v1/contests/{contest_id}/leaderboard": 1200,
    "GET /api/v1/community-quizzes/q/{slug}": 1200,
    "POST /api/v1/community-quizzes/q/{slug}/start": 1200,
    "POST /api/v1/community-quizzes/q/{slug}/submit": 1200,
    "GET /api/v1/community-quizzes/q/{slug}/leaderboard": 1200,
    # public share page (G6): token guessing protection
    "GET /r/{token}": 30,
}

UNMATCHED = "<unmatched>"
_SUB_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def route_template(app, method: str, path: str) -> str:
    """The matched route's path template, or UNMATCHED (404s share one bucket per client). Cached per app."""
    cache = getattr(app.state, "route_template_cache", None)
    if cache is None:
        cache = app.state.route_template_cache = lru_cache(maxsize=4096)(partial(_match, app))
    return cache(method, path)


def _match(app, method: str, path: str) -> str:
    scope = {"type": "http", "method": method, "path": path, "root_path": "", "headers": [], "query_string": b""}
    partial = None
    for route in app.router.routes:
        match, _ = route.matches(scope)
        if match == Match.FULL:
            return getattr(route, "path", UNMATCHED)
        if match == Match.PARTIAL and partial is None:        # right path, other method
            partial = getattr(route, "path", None)
    return partial or UNMATCHED


def jwt_subject(authorization: Optional[str]) -> Optional[str]:
    """`sub` of a bearer JWT, decoded without verification (bucket key only). None if absent/malformed."""
    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or token.count(".") != 2 or len(token) > 4096:
        return None
    payload = token.split(".")[1]
    try:
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (ValueError, TypeError):
        return None
    sub = data.get("sub") if isinstance(data, dict) else None
    return sub if isinstance(sub, str) and _SUB_RE.match(sub) else None


def is_exempt(path: str) -> bool:
    return path in EXEMPT_PATHS or path.startswith(EXEMPT_PREFIXES)


def limit_for(method: str, template: str, default: int) -> int:
    return ROUTE_LIMITS.get(f"{method} {template}", default)


def bucket_key(method: str, template: str, authorization: Optional[str], client_ip: str) -> str:
    sub = jwt_subject(authorization)
    who = f"u:{sub}" if sub else f"ip:{client_ip}"
    return f"{who}|{method} {template}"
