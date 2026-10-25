"""
Rate limiting — who is counted and how much.

Key  = route TEMPLATE (e.g. /api/v1/copy-checker/sheets/{sheet_id}/check, never the raw path — raw paths with
       ids gave every id its own bucket, i.e. no limit at all)
     + WHO, first match wins (school WiFi = one IP for a whole class, so a student or teacher identity always
       beats the IP):
         1. the user id of a bearer JWT
         2. the X-Attempt-Id header (students without a login: contest / community-quiz attempt) — these
            requests ALSO count against a per-IP ceiling (ATTEMPT_IP_CEILING) so made-up ids can't flood
         3. the client IP
The user id / attempt id is NOT verified here: it only picks the bucket. The route still verifies it, so a
forged value gets its own bucket and then a 401 / 404.

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
    # PDF → contest questions (daily limit of 10 lives in the database); the review screen polls the job
    "POST /api/v1/contests/import-pdf": 10,
    "GET /api/v1/contests/import-pdf/{job_id}": 120,
    # payments
    "POST /api/v1/payment/create-order": 10,
    "POST /api/v1/payment/verify-payment": 20,
    # public, read-heavy (NCERT browser) — per IP
    "GET /api/v1/test-generator/ncert-questions": 1200,
    "GET /api/v1/test-generator/ncert-question-stats": 1200,
    "GET /api/v1/test-generator/chapters": 1200,
    "GET /api/v1/test-generator/templates": 1200,
    # students taking a contest / quiz, often a whole class behind one school IP.
    # No identity yet (join / fetch / results): per IP, high.
    "GET /api/v1/contests/{short_code}/info": 1200,
    "POST /api/v1/contests/{short_code}/start": 1200,
    "GET /api/v1/contests/{contest_id}/leaderboard": 1200,
    "GET /api/v1/community-quizzes/q/{slug}": 1200,
    "POST /api/v1/community-quizzes/q/{slug}/start": 1200,
    "GET /api/v1/community-quizzes/q/{slug}/leaderboard": 1200,
    # submit: per attempt (X-Attempt-Id) — one student, a few tries; per IP (old clients without the header): high
    "POST /api/v1/contests/{contest_id}/submit": 30,
    "POST /api/v1/community-quizzes/q/{slug}/submit": 30,
    # progress autosave every ~15 s: ~1 per 3 s per attempt allowed (a resumed tab + retries)
    "PUT /api/v1/contests/{contest_id}/attempts/{attempt_id}/progress": 20,
    # public share page (G6): token guessing protection
    "GET /r/{token}": 30,
}

# Requests keyed by IP on routes whose per-identity limit is low (old clients that don't send X-Attempt-Id)
IP_FALLBACK_LIMITS = {
    "POST /api/v1/contests/{contest_id}/submit": 1200,
    "PUT /api/v1/contests/{contest_id}/attempts/{attempt_id}/progress": 3000,
    "POST /api/v1/community-quizzes/q/{slug}/submit": 1200,
}
# All attempt-keyed requests from one IP, per route per minute (made-up attempt ids can't bypass the limit)
ATTEMPT_IP_CEILING = 3000

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


def attempt_id(header: Optional[str]) -> Optional[str]:
    value = (header or "").strip()
    return value if _SUB_RE.match(value) else None


def identity(authorization: Optional[str], attempt_header: Optional[str], client_ip: str) -> str:
    """'u:<user>' | 'att:<attempt>' | 'ip:<ip>' — who a request is counted against."""
    sub = jwt_subject(authorization)
    if sub:
        return f"u:{sub}"
    att = attempt_id(attempt_header)
    if att:
        return f"att:{att}"
    return f"ip:{client_ip}"


def limit_for(method: str, template: str, default: int, who: str = "") -> int:
    route = f"{method} {template}"
    if who.startswith("ip:") and route in IP_FALLBACK_LIMITS:
        return IP_FALLBACK_LIMITS[route]
    return ROUTE_LIMITS.get(route, default)


def bucket_key(method: str, template: str, authorization: Optional[str], client_ip: str,
               attempt_header: Optional[str] = None) -> str:
    return f"{identity(authorization, attempt_header, client_ip)}|{method} {template}"


def checks(method: str, template: str, authorization: Optional[str], client_ip: str,
           attempt_header: Optional[str], default: int) -> list:
    """(bucket key, limit) pairs a request must pass: its identity bucket, plus the per-IP ceiling for
    attempt-keyed requests."""
    who = identity(authorization, attempt_header, client_ip)
    out = [(f"{who}|{method} {template}", limit_for(method, template, default, who))]
    if who.startswith("att:"):
        out.append((f"attip:{client_ip}|{method} {template}", ATTEMPT_IP_CEILING))
    return out
