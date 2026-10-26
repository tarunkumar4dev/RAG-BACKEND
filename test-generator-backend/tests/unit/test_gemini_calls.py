"""Gemini call rules (fix/checker-p0): the primary model up to 3 calls on 429 / 5xx / timeout within the deadline,
a backup model only with CHECKER_ALLOW_FALLBACK=true (and then flagged), key / model errors at once, one JSON retry,
hard call cap."""

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
    monkeypatch.setattr(svc, "CHECKER_ALLOW_FALLBACK", False)
    monkeypatch.setattr(svc, "CHECKER_PRIMARY_ATTEMPTS", 3)
    slept = []
    monkeypatch.setattr(svc.time, "sleep", lambda s: slept.append(s))

    def make(script):
        models = FakeModels(script)
        return SimpleNamespace(models=models), models, slept
    return make


def test_429_primary_three_times_then_busy_never_to_fallbacks(live):
    client, models, slept = live([429, 429, 429])
    with pytest.raises(svc.GeminiBusy) as e:
        svc._generate_json(client, ["x"], "GRADE", max_retries=5)
    assert e.value.code == 429
    assert models.calls == ["primary"] * 3
    assert slept == [5.0, 5.0]           # Gemini didn't say how long: 5 s


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
    client, models, slept = live([err, err, err])
    with pytest.raises(svc.GeminiBusy):
        svc._generate_json(client, ["x"], "GRADE")
    assert slept == [wait, wait]
    assert len(models.calls) == 3


def test_503_retries_the_primary_and_never_falls_back_by_default(live):
    client, models, slept = live([503, 503, 503])
    with svc.usage_meter() as meter, pytest.raises(svc.GeminiBusy) as e:
        svc._generate_json(client, ["x"], "GRADE")
    assert e.value.code == 503
    assert models.calls == ["primary"] * 3
    assert slept == [2.0, 5.0]
    assert meter.fallback_used is False


def test_503_then_primary_answers(live):
    client, models, _ = live([503, 503, json.dumps({"ok": 1})])
    with svc.usage_meter() as meter:
        assert svc._generate_json(client, ["x"], "GRADE") == {"ok": 1}
    assert models.calls == ["primary"] * 3
    u = meter.as_dict()
    assert u["fallback_used"] is False and u["models_by_tag"] == {"GRADE": "primary"}


def test_fallback_only_when_allowed_and_then_it_is_flagged(live, monkeypatch):
    monkeypatch.setattr(svc, "CHECKER_ALLOW_FALLBACK", True)
    client, models, _ = live([503, 503, 503, json.dumps({"ok": 1})])
    with svc.usage_meter() as meter:
        assert svc._generate_json(client, ["x"], "GRADE") == {"ok": 1}
    assert models.calls == ["primary", "primary", "primary", "fb1"]
    u = meter.as_dict()
    assert u["fallback_used"] is True and u["models_by_tag"] == {"GRADE": "fb1"} and u["primary_model"] == "primary"


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_key_or_model_problem_is_a_config_error_at_once(live, code):
    client, models, slept = live([code, code, code])
    with pytest.raises(svc.GeminiConfigError) as e:
        svc._generate(client, ["x"], "GRADE")
    assert e.value.code == code and e.value.model == "primary"
    assert models.calls == ["primary"] and slept == []


def test_404_with_fallback_allowed_tries_the_backups_once(live, monkeypatch):
    monkeypatch.setattr(svc, "CHECKER_ALLOW_FALLBACK", True)
    client, models, _ = live([404, 404, 404])
    with pytest.raises(svc.GeminiConfigError):
        svc._generate(client, ["x"], "GRADE")
    assert models.calls == ["primary", "fb1", "fb2"]


def test_timeouts_are_retried_then_raised(live):
    client, models, slept = live([TimeoutError("read"), TimeoutError("read"), TimeoutError("read")])
    with pytest.raises(TimeoutError):
        svc._generate_json(client, ["x"], "GRADE")
    assert models.calls == ["primary"] * 3 and slept == [2.0, 5.0]


def test_no_retry_when_a_whole_call_no_longer_fits_the_deadline(live, monkeypatch):
    client, models, slept = live([503, 503, 503])
    now = svc.time.monotonic()
    # 150 s calls, 100 s left: the first call is made, a retry (2 s + 150 s) doesn't fit
    with svc.call_settings(timeout_s=150, deadline=now + 100), pytest.raises(svc.GeminiBusy):
        svc._generate_json(client, ["x"], "GRADE")
    assert models.calls == ["primary"] and slept == []

    client, models, slept = live([503, json.dumps({"ok": 1})])
    with svc.call_settings(timeout_s=150, deadline=now + 400):
        assert svc._generate_json(client, ["x"], "GRADE") == {"ok": 1}
    assert models.calls == ["primary", "primary"]


def test_other_api_errors_are_raised_as_is(live):
    client, models, _ = live([409])
    with pytest.raises(genai_errors.APIError):
        svc._generate(client, ["x"], "GRADE")
    assert models.calls == ["primary"]


def test_bad_json_gets_exactly_one_retry(live):
    client, models, _ = live(["not json", "still not json"])
    with pytest.raises(ValueError):
        svc._generate_json(client, ["x"], "GRADE", max_retries=10)
    assert len(models.calls) == 2

    client, models, _ = live(["not json", json.dumps({"fine": True})])
    assert svc._generate_json(client, ["x"], "GRADE") == {"fine": True}


def test_call_cap_stops_before_the_next_call(live):
    client, models, _ = live([503, 503, 503, 503, 503, 503])
    with svc.usage_meter(cap=2) as meter:
        with pytest.raises(svc.CallCapExceeded):
            svc._generate_json(client, ["x"], "GRADE")     # 2 primary calls, the 3rd hits the cap
    assert meter.calls == 2
    assert len(models.calls) == 2


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
