"""contest_import_gemini (REST, no SDK): request body (schema + thinking cap), the checker's model / 429 rules,
deadline-bounded calls, Files API for big PDFs. httpx.MockTransport, no network."""

import json
import time

import httpx
import pytest

from app.services import contest_import_gemini as g
from app.services import smart_assignment_service as svc

SCHEMA = {"type": "OBJECT", "properties": {"ok": {"type": "INTEGER"}}, "required": ["ok"]}


def _ok(payload, thoughts=False):
    parts = ([{"text": "thinking…", "thought": True}] if thoughts else []) + [{"text": json.dumps(payload)}]
    return httpx.Response(200, json={"candidates": [{"content": {"parts": parts}, "finishReason": "STOP"}],
                                     "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5,
                                                       "totalTokenCount": 15, "thoughtsTokenCount": 0}})


@pytest.fixture
def http(monkeypatch):
    calls = []
    replies = []

    def handler(request):
        calls.append(request)
        r = replies.pop(0)
        return r(request) if callable(r) else r
    monkeypatch.setattr(svc, "GEMINI_MOCK", False)
    monkeypatch.setattr(svc, "CHECKER_API_KEY", "k")
    monkeypatch.setattr(g, "_http", httpx.Client(transport=httpx.MockTransport(handler)))
    return calls, replies


def _call(**kw):
    args = dict(models=["m1", "m2"], call_deadline=time.monotonic() + 200, max_timeout=150,
                max_output_tokens=1000, thinking_budget=2048)
    args.update(kw)
    return g.generate_json([{"text": "hi"}], "IMPORT-PARSE", SCHEMA, **args)


def test_request_body_and_thought_parts_skipped(http):
    calls, replies = http
    replies.append(_ok({"ok": 1}, thoughts=True))
    assert _call() == {"ok": 1}
    req = calls[0]
    assert req.url.path == "/v1beta/models/m1:generateContent"
    assert req.headers["x-goog-api-key"] == "k"
    body = json.loads(req.content)
    cfg = body["generationConfig"]
    assert cfg["responseMimeType"] == "application/json" and cfg["responseSchema"] == SCHEMA
    assert cfg["thinkingConfig"] == {"thinkingBudget": 2048} and cfg["maxOutputTokens"] == 1000
    assert cfg["temperature"] == 0


@pytest.mark.parametrize("code", [404, 503])
def test_404_503_go_to_the_next_model(http, code):
    calls, replies = http
    replies += [httpx.Response(code, json={"error": {"message": "x"}}), _ok({"ok": 2})]
    assert _call() == {"ok": 2}
    assert [c.url.path.split("/")[-1] for c in calls] == ["m1:generateContent", "m2:generateContent"]


def test_429_is_busy_with_retry_after_and_no_fallback(http):
    calls, replies = http
    replies.append(httpx.Response(429, headers={"retry-after": "12"}, json={"error": {"message": "quota"}}))
    with pytest.raises(svc.GeminiBusy) as e:
        _call()
    assert e.value.code == 429 and e.value.retry_after == 12 and len(calls) == 1


def test_retry_info_in_body(http):
    _, replies = http
    replies.append(httpx.Response(429, json={"error": {"details": [
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "27s"}]}}))
    with pytest.raises(svc.GeminiBusy) as e:
        _call()
    assert e.value.retry_after == 27


def test_every_model_503_is_busy(http):
    _, replies = http
    replies += [httpx.Response(503), httpx.Response(503)]
    with pytest.raises(svc.GeminiBusy) as e:
        _call()
    assert e.value.code == 503


def test_400_is_an_error_not_busy(http):
    _, replies = http
    replies.append(httpx.Response(400, json={"error": {"message": "bad schema"}}))
    with pytest.raises(g.GeminiError, match="bad schema"):
        _call()


def test_truncated_json_is_value_error(http):
    _, replies = http
    replies.append(httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"ok": '}]},
                                                             "finishReason": "MAX_TOKENS"}]}))
    with pytest.raises(ValueError, match="MAX_TOKENS"):
        _call()


def test_each_call_is_bounded_by_the_deadline(http):
    calls, replies = http
    replies += [httpx.Response(503), _ok({"ok": 3})]
    _call(call_deadline=time.monotonic() + 40, max_timeout=150)
    assert all(int(c.headers["x-server-timeout"]) <= 40 for c in calls)


def test_no_time_left_makes_no_call(http):
    calls, _ = http
    with pytest.raises(TimeoutError):
        _call(call_deadline=time.monotonic() + 5)
    assert calls == []


def test_big_pdf_goes_through_files_api(http, tmp_path, monkeypatch):
    calls, replies = http
    monkeypatch.setattr(svc, "INLINE_FILES_BUDGET", 10)
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 " + b"x" * 100)
    replies += [httpx.Response(200, headers={"x-goog-upload-url": "https://upload.example/u1"}),
                httpx.Response(200, json={"file": {"name": "files/abc", "uri": "https://g/files/abc",
                                                   "state": "PROCESSING"}}),
                httpx.Response(200, json={"name": "files/abc", "uri": "https://g/files/abc", "state": "ACTIVE"}),
                httpx.Response(200, json={})]
    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    part, name = g.pdf_part(str(pdf), time.monotonic() + 100)
    assert part == {"fileData": {"mimeType": "application/pdf", "fileUri": "https://g/files/abc"}}
    assert name == "files/abc"
    assert calls[0].headers["x-goog-upload-command"] == "start"
    assert calls[1].headers["x-goog-upload-command"] == "upload, finalize" and calls[1].content == pdf.read_bytes()
    g.delete_file(name)
    assert calls[-1].method == "DELETE" and calls[-1].url.path == "/v1beta/files/abc"


def test_small_pdf_is_inline(tmp_path, monkeypatch):
    monkeypatch.setattr(svc, "GEMINI_MOCK", False)
    pdf = tmp_path / "p.pdf"
    pdf.write_bytes(b"%PDF-1.4 small")
    part, name = g.pdf_part(str(pdf), time.monotonic() + 100)
    assert name is None and part["inlineData"]["mimeType"] == "application/pdf"
