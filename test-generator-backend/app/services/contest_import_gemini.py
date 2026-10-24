"""
Gemini over plain REST (httpx) for the contest import only — no google-genai SDK.

Production pins google-genai 0.8.0 for the copy checker; that SDK cannot cap thinking (its ThinkingConfig has no
thinking_budget and unknown keys are forbidden), and uncapped the 50-MCQ golden paper used 63k thinking tokens and
hit MAX_TOKENS at 201 s. REST takes generationConfig.thinkingConfig.thinkingBudget directly (the field SDK 1.14's
thinking_budget maps to; thinking dropped to 0 on gemini-3-flash-preview and gemini-3.8-flash).

Same rules as smart_assignment_service._generate / _generate_json (the checker):
  * same key: CHECKER_GEMINI_API_KEY, else GEMINI_API_KEY
  * one call per model in order: 404 (model gone) / 503 (overloaded) → next model; 429 / 500 / 504 → GeminiBusy
    (with the Retry-After seconds Gemini asked for); every model 503 → GeminiBusy(503)
  * the caller retries once (429 → Retry-After capped at CHECKER_MAX_RETRY_WAIT_S, else 2 s) — see
    contest_import_service._parse_chunk
  * every attempt counts against the usage meter's call cap and is logged like a checker call
  * GEMINI_MOCK=1 → the recorded reply GEMINI_MOCK_DIR/<TAG>[_n].json, no network
Each model call gets min(max_timeout, call_deadline - now) — a slow first model can never push the job past its
deadline. The PDF goes inline when it fits the checker's inline budget, else through the Files API (deleted after).
"""

import base64
import json
import logging
import os
import time
from types import SimpleNamespace
from typing import Optional

import httpx

from app.core.config import settings
from app.services import smart_assignment_service as svc

logger = logging.getLogger(__name__)

API = "https://generativelanguage.googleapis.com"
_FALLBACK_CODES = {404, 503}
_BUSY_CODES = {429, 500, 504}
MIN_CALL_S = 10          # don't start a model call with less time than this

_http = httpx.Client()


class GeminiError(RuntimeError):
    """A non-retryable API error (400 bad request, 403 key, every model 404)."""

    def __init__(self, code: int, message: str):
        super().__init__(f"Gemini {code}: {message}")
        self.code = code


def _key() -> str:
    key = svc.CHECKER_API_KEY or settings.GEMINI_API_KEY
    if not key:
        raise ValueError("No Gemini API key. Set CHECKER_GEMINI_API_KEY in .env")
    return key


def _retry_after(r: httpx.Response) -> Optional[float]:
    """Seconds from the Retry-After header, else from the error's RetryInfo ("retryDelay": "27s")."""
    try:
        if r.headers.get("retry-after") is not None:
            return max(0.0, float(r.headers["retry-after"]))
    except ValueError:
        pass
    try:
        details = (r.json().get("error") or {}).get("details") or []
    except Exception:
        return None
    for d in details:
        if isinstance(d, dict) and str(d.get("@type", "")).endswith("RetryInfo"):
            try:
                return max(0.0, float(str(d.get("retryDelay", "")).rstrip("s")))
            except ValueError:
                return None
    return None


def _error_message(r: httpx.Response) -> str:
    try:
        return str((r.json().get("error") or {}).get("message") or r.text)[:200]
    except Exception:
        return r.text[:200]


def _reply(data: dict) -> tuple:
    """(text, usage namespace, finish reason) from a generateContent response."""
    cands = data.get("candidates") or []
    cand = cands[0] if cands else {}
    parts = (cand.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    um = data.get("usageMetadata") or {}
    usage = SimpleNamespace(prompt_token_count=um.get("promptTokenCount", 0),
                            candidates_token_count=um.get("candidatesTokenCount", 0),
                            total_token_count=um.get("totalTokenCount", 0),
                            thoughts_token_count=um.get("thoughtsTokenCount", 0))
    return text, usage, cand.get("finishReason")


def _loads(text: str, tag: str, finish=None) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"{tag}: unreadable reply (finish={finish})")
    if not isinstance(data, dict):
        raise ValueError(f"{tag}: model returned JSON that is not an object")
    return data


def generate_json(parts: list, tag: str, schema: dict, *, models: list, call_deadline: float, max_timeout: float,
                  max_output_tokens: int, thinking_budget: int) -> dict:
    """ONE attempt (one call per model, see module doc) → the reply's JSON object.
    Raises GeminiBusy, ValueError (unreadable / truncated reply), GeminiError, httpx.TimeoutException,
    TimeoutError (no time left for any model)."""
    if svc.GEMINI_MOCK:
        svc._count_call(tag)
        response = svc._mock_response(tag)
        svc._log_usage(tag, "mock", response, 0.0)
        return _loads(response.text, tag)

    body = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": schema,
            # temperature 0 + fixed seed, as the checker: the same paper should read the same way every time
            "temperature": 0,
            "seed": 7,
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {"thinkingBudget": thinking_budget},
        },
    }
    last: Optional[httpx.Response] = None
    for model in models:
        timeout = min(max_timeout, call_deadline - time.monotonic())
        if timeout < MIN_CALL_S:
            break
        svc._count_call(tag)
        t_call = time.time()
        try:
            r = _http.post(f"{API}/v1beta/models/{model}:generateContent", json=body, timeout=timeout + 5,
                           headers={"x-goog-api-key": _key(), "X-Server-Timeout": str(int(timeout))})
        except httpx.TimeoutException:
            logger.warning(f"[checker] {svc._tag_prefix()}{tag.lower()} call to {model} timed out after "
                           f"{time.time() - t_call:.1f}s")
            raise
        if r.status_code == 200:
            text, usage, finish = _reply(r.json())
            svc._log_usage(tag, model, SimpleNamespace(usage_metadata=usage,
                                                       candidates=[SimpleNamespace(finish_reason=finish)]),
                           time.time() - t_call)
            return _loads(text, tag, finish)
        logger.warning(f"[checker] {svc._tag_prefix()}{tag.lower()} call to {model} failed after "
                       f"{time.time() - t_call:.1f}s: HTTP {r.status_code}")
        if r.status_code in _FALLBACK_CODES:
            last = r
            continue
        if r.status_code in _BUSY_CODES:
            raise svc.GeminiBusy(r.status_code, retry_after=_retry_after(r))
        raise GeminiError(r.status_code, _error_message(r))
    if last is None:
        raise TimeoutError(f"{tag}: no time left for a Gemini call")
    if last.status_code == 503:
        raise svc.GeminiBusy(503)
    raise GeminiError(last.status_code, _error_message(last))


# ── the PDF part: inline bytes, or the Files API when too big ────────────────

def pdf_part(path: str, deadline: float) -> tuple:
    """({"inlineData"…} or {"fileData"…}, uploaded file name or None — pass it to delete_file afterwards)."""
    size = os.path.getsize(path)
    if svc.GEMINI_MOCK or svc.b64_size(size) <= svc.INLINE_FILES_BUDGET:
        data = base64.b64encode(open(path, "rb").read()).decode("ascii")
        return {"inlineData": {"mimeType": "application/pdf", "data": data}}, None
    timeout = max(MIN_CALL_S, min(60.0, deadline - time.monotonic()))
    start = _http.post(f"{API}/upload/v1beta/files", timeout=timeout, json={"file": {"display_name": "contest-import"}},
                       headers={"x-goog-api-key": _key(), "X-Goog-Upload-Protocol": "resumable",
                                "X-Goog-Upload-Command": "start", "X-Goog-Upload-Header-Content-Length": str(size),
                                "X-Goog-Upload-Header-Content-Type": "application/pdf"})
    upload_url = start.headers.get("x-goog-upload-url")
    if start.status_code != 200 or not upload_url:
        raise _upload_error(start)
    done = _http.post(upload_url, timeout=timeout, content=open(path, "rb").read(),
                      headers={"X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"})
    if done.status_code != 200:
        raise _upload_error(done)
    f = done.json().get("file") or {}
    name, uri = f.get("name"), f.get("uri")
    while f.get("state") == "PROCESSING" and time.monotonic() < deadline - MIN_CALL_S:
        time.sleep(1)
        f = _http.get(f"{API}/v1beta/{name}", headers={"x-goog-api-key": _key()}, timeout=MIN_CALL_S).json()
    if f.get("state") not in (None, "ACTIVE"):
        delete_file(name)
        raise GeminiError(0, f"uploaded file is {f.get('state')}")
    return {"fileData": {"mimeType": "application/pdf", "fileUri": uri}}, name


def _upload_error(r: httpx.Response) -> Exception:
    if r.status_code in _BUSY_CODES or r.status_code == 503:
        return svc.GeminiBusy(r.status_code, retry_after=_retry_after(r))
    return GeminiError(r.status_code, _error_message(r))


def delete_file(name: Optional[str]) -> None:
    if not name:
        return
    try:
        _http.delete(f"{API}/v1beta/{name}", headers={"x-goog-api-key": _key()}, timeout=MIN_CALL_S)
    except Exception as e:
        logger.warning(f"Could not delete Gemini file {name}: {type(e).__name__}")
