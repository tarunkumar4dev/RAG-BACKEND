"""Retry-storm fix (rule 2): one call per model, fallback only on 404/503, at most one JSON retry, hard call cap."""

import json
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from app.services import smart_assignment_service as svc


def api_error(code: int, headers: dict = None, retry_delay: str = None) -> genai_errors.APIError:
    err = {"code": code, "message": "x", "status": "X"}
    if retry_delay:
        err["details"] = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay}]
    return genai_errors.APIError(code, {"error": err}, SimpleNamespace(headers=headers or {}))


class FakeModels:
    def __init__(self, script):
        self.script = list(script)      # each item: an int (error code) or a str (reply text)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        item = self.script.pop(0)
        if isinstance(item, int):
            raise api_error(item)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item, usage_metadata=SimpleNamespace(
            prompt_token_count=100, candidates_token_count=20, total_token_count=130, thoughts_token_count=10),
            candidates=[])


@pytest.fixture
def live(monkeypatch):
    """Real (non-mock) code path with a fake client; sleeps are recorded, not slept."""
    monkeypatch.setattr(svc, "GEMINI_MOCK", False)
    monkeypatch.setattr(svc, "CHECKER_MODEL", "primary")
    monkeypatch.setattr(svc, "CHECKER_FALLBACK_MODELS", ["fb1", "fb2"])
    slept = []
    monkeypatch.setattr(svc.time, "sleep", lambda s: slept.append(s))

    def make(script):
        models = FakeModels(script)
        return SimpleNamespace(models=models), models, slept
    return make


def test_429_one_retry_after_default_5s_never_to_fallbacks(live):
    client, models, slept = live([429, 429])
    with pytest.raises(svc.GeminiBusy) as e:
        svc._generate_json(client, ["x"], "GRADE", max_retries=5)
    assert e.value.code == 429
    assert models.calls == ["primary", "primary"]
    assert slept == [5.0]


def test_429_then_success(live):
    client, models, slept = live([429, json.dumps({"ok": 1})])
    assert svc._generate_json(client, ["x"], "GRADE") == {"ok": 1}
    assert slept == [5.0]


@pytest.mark.parametrize("err,wait", [
    (api_error(429, headers={"retry-after": "7"}), 7.0),
    (api_error(429, retry_delay="12s"), 12.0),
    (api_error(429, headers={"retry-after": "120"}), 30.0),          # capped (CHECKER_MAX_RETRY_WAIT_S)
])
def test_429_waits_what_gemini_asks(live, err, wait):
    client, models, slept = live([err, err])
    with pytest.raises(svc.GeminiBusy):
        svc._generate_json(client, ["x"], "GRADE")
    assert slept == [wait]
    assert len(models.calls) == 2


def test_503_falls_back_once_per_model(live):
    client, models, _ = live([503, json.dumps({"ok": 1})])
    assert svc._generate_json(client, ["x"], "GRADE") == {"ok": 1}
    assert models.calls == ["primary", "fb1"]


def test_404_everywhere_is_raised_not_retried(live):
    client, models, _ = live([404, 404, 404])
    with pytest.raises(genai_errors.APIError):
        svc._generate(client, ["x"], "GRADE")
    assert models.calls == ["primary", "fb1", "fb2"]


def test_500_gets_exactly_one_retry(live):
    client, models, slept = live([500, 500])
    with pytest.raises(svc.GeminiBusy):
        svc._generate_json(client, ["x"], "GRADE", max_retries=10)
    assert models.calls == ["primary", "primary"]
    assert slept == [2]


def test_bad_json_gets_exactly_one_retry(live):
    client, models, _ = live(["not json", "still not json"])
    with pytest.raises(ValueError):
        svc._generate_json(client, ["x"], "GRADE", max_retries=10)
    assert len(models.calls) == 2

    client, models, _ = live(["not json", json.dumps({"fine": True})])
    assert svc._generate_json(client, ["x"], "GRADE") == {"fine": True}


def test_call_cap_stops_before_the_next_call(live):
    client, models, _ = live([503, 503, 503, 503, 503, 503])
    with svc.usage_meter(cap=4) as meter:
        with pytest.raises(svc.CallCapExceeded):
            svc._generate_json(client, ["x"], "GRADE")     # 3 models, then the retry hits the cap
    assert meter.calls == 4
    assert len(models.calls) == 4


def test_usage_is_counted(live):
    client, _, _ = live([json.dumps({"a": 1}), json.dumps({"b": 2})])
    with svc.usage_meter() as meter:
        svc._generate_json(client, ["x"], "RUBRIC")
        svc._generate_json(client, ["x"], "GRADE")
    u = meter.as_dict()
    assert (u["gemini_calls"], u["input_tokens"], u["output_tokens"], u["thinking_tokens"]) == (2, 200, 40, 20)
    assert set(u["ms_by_tag"]) == {"RUBRIC", "GRADE"}


def test_inline_parts_instead_of_uploads(tmp_path):
    p = tmp_path / "a.jpg"
    p.write_bytes(b"\xff\xd8\xff" + b"0" * 100)
    parts = svc.upload_files([str(p)])
    assert parts[0].inline_data.mime_type == "image/jpeg"
    svc.delete_files(parts)          # inline parts: nothing to delete, no client needed
