"""GeminiProvider behaviour with the SDK client stubbed — no network, no real API calls."""

import json
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest
from google.genai import errors as genai_errors
from google.genai import types

from app.config import Settings
from app.llm.base import (
    LLMAuthError,
    LLMBadRequestError,
    LLMModelNotAllowedError,
    LLMModelUnavailableError,
    LLMNotConfiguredError,
    LLMOutputError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.gemini_client import FREE_TIER_MODELS, GeminiProvider, next_quota_reset

FAKE_KEY = "AIzaFAKE-test-key-not-real-123"


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        out = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(out, BaseException):
            raise out
        return out


def ok_response(data=None, finish="STOP", text=None, block=None):
    return SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=getattr(types.FinishReason, finish))],
        text=text if text is not None else json.dumps(data if data is not None else {"ok": True}),
        prompt_feedback=SimpleNamespace(block_reason=block) if block else None,
        usage_metadata=SimpleNamespace(prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5),
        model_version="gemini-3.8-flash",
    )


def api_error(code, status, message="error", details=None):
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "status": status, "message": message, "details": details or []}})


def per_minute_429(delay="7s"):
    return api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}])


def daily_429():
    return api_error(429, "RESOURCE_EXHAUSTED", "Quota exceeded", [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "40s"}])


def make(outcomes, **overrides):
    s = Settings(_env_file=None, gemini_api_key=FAKE_KEY, **overrides)
    models = FakeModels(outcomes)
    sleeps = []
    p = GeminiProvider(s, client=SimpleNamespace(models=models), sleep=sleeps.append)
    return p, models, sleeps


def call(p):
    return p.generate_json(system="sys", user="question", schema={"type": "object"})


def test_default_model_is_on_free_tier_allowlist():
    assert Settings(_env_file=None).llm_model == "gemini-3.8-flash"
    assert "gemini-3.8-flash" in FREE_TIER_MODELS


@pytest.mark.parametrize("key", [None, "", "   ", "your-gemini-api-key-here"])
def test_missing_or_placeholder_key(key):
    with pytest.raises(LLMNotConfiguredError):
        GeminiProvider(Settings(_env_file=None, gemini_api_key=key), client=SimpleNamespace(models=None))


@pytest.mark.parametrize("model", ["gemini-3.1-pro-preview", "gemini-4-argon", "claude-sonnet-5-5", "gemini-2.5-flash-image"])
def test_non_free_models_refused_before_any_request(model):
    models = FakeModels([ok_response()])
    with pytest.raises(LLMModelNotAllowedError) as info:
        GeminiProvider(Settings(_env_file=None, gemini_api_key=FAKE_KEY, llm_model=model),
                       client=SimpleNamespace(models=models))
    assert models.calls == [] and model in info.value.user_message


def test_success_request_shape_has_no_tools():
    p, models, _ = make([ok_response({"answer": "x"})])
    out = call(p)
    assert out.data == {"answer": "x"} and out.usage["input_tokens"] == 100
    sent = models.calls[0]
    cfg = sent["config"]
    assert sent["model"] == "gemini-3.8-flash" and sent["contents"] == "question"
    assert cfg.response_mime_type == "application/json" and cfg.response_json_schema == {"type": "object"}
    assert cfg.system_instruction == "sys" and cfg.thinking_config.thinking_level.lower() == "low"
    assert not cfg.tools  # no Google Search grounding or other tools


def test_invalid_key_and_permission_errors():
    bad_key = api_error(400, "INVALID_ARGUMENT", f"API key not valid. key={FAKE_KEY}",
                        [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}])
    p, models, _ = make([bad_key])
    with pytest.raises(LLMAuthError) as info:
        call(p)
    assert FAKE_KEY not in str(info.value) and FAKE_KEY not in info.value.user_message
    assert len(models.calls) == 1  # never retried
    p, _, _ = make([api_error(403, "PERMISSION_DENIED", "denied")])
    with pytest.raises(LLMAuthError):
        call(p)


def test_per_minute_429_retries_with_server_delay_then_succeeds():
    p, models, sleeps = make([per_minute_429("7s"), ok_response()])
    assert call(p).data == {"ok": True}
    assert len(models.calls) == 2 and sleeps == [7.0]


def test_per_minute_429_retries_are_bounded_and_capped():
    p, models, sleeps = make([per_minute_429("300s")], llm_max_retries=2, llm_retry_max_wait_seconds=60)
    with pytest.raises(LLMRateLimitError) as info:
        call(p)
    assert not isinstance(info.value, LLMQuotaExhaustedError)
    assert len(models.calls) == 3 and sleeps == [60.0, 60.0]


def test_daily_quota_stops_immediately_and_blocks_further_requests():
    p, models, sleeps = make([daily_429()])
    with pytest.raises(LLMQuotaExhaustedError) as info:
        call(p)
    assert len(models.calls) == 1 and sleeps == []  # no retry
    assert "daily quota" in info.value.user_message and "No paid tier" in info.value.user_message
    with pytest.raises(LLMQuotaExhaustedError):
        call(p)
    assert len(models.calls) == 1  # circuit breaker: nothing sent after exhaustion


def test_model_unavailable_and_bad_request():
    p, _, _ = make([api_error(404, "NOT_FOUND", "models/gemini-3.8-flash is not found")])
    with pytest.raises(LLMModelUnavailableError):
        call(p)
    p, _, _ = make([api_error(400, "INVALID_ARGUMENT", "bad schema")])
    with pytest.raises(LLMBadRequestError):
        call(p)


def overload(delay=None):
    details = [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}] if delay else []
    return api_error(503, "UNAVAILABLE", "This model is currently experiencing high demand.", details)


def test_503_succeeds_after_one_backoff():
    p, models, sleeps = make([overload(), ok_response()])
    assert call(p).data == {"ok": True}
    assert len(models.calls) == 2 and len(sleeps) == 1 and 3.0 <= sleeps[0] <= 3.5


def test_503_initial_attempt_plus_three_retries_with_3_8_15_schedule():
    from app.llm.base import LLMServerError
    p, models, sleeps = make([overload()])
    with pytest.raises(LLMServerError) as info:
        call(p)
    assert len(models.calls) == 4  # 1 initial + 3 retries
    assert len(sleeps) == 3 and sum(sleeps) <= 30
    assert all(lo <= s <= lo + 0.5 for s, lo in zip(sleeps, (3.0, 8.0, 15.0)))  # ~3, 8, 15 s + jitter
    exc = info.value
    assert exc.category == "http_5xx" and exc.retryable is True
    assert "high demand (HTTP 503 UNAVAILABLE)" in exc.user_message and "4 attempt(s)" in exc.user_message


def test_503_respects_retry_delay_within_total_budget():
    p, models, sleeps = make([overload("10s"), ok_response()])
    call(p)
    assert sleeps == [10.0]  # server's retryDelay (10 s) exceeds the 3 s schedule step
    from app.llm.base import LLMServerError
    p, models, sleeps = make([overload("40s")])  # 40 s > 30 s total budget: give up, don't sleep
    with pytest.raises(LLMServerError):
        call(p)
    assert len(models.calls) == 1 and sleeps == []
    p, models, sleeps = make([overload("12s")])  # 12 + 12 = 24 fits; a third 15 s step (39 s) would not
    with pytest.raises(LLMServerError):
        call(p)
    assert sleeps == [12.0, 12.0] and len(models.calls) == 3


def test_overload_retries_configurable_and_zero_disables():
    from app.llm.base import LLMServerError
    p, models, sleeps = make([overload()], llm_overload_max_retries=0)
    with pytest.raises(LLMServerError):
        call(p)
    assert len(models.calls) == 1 and sleeps == []


@pytest.mark.parametrize("code,status", [(500, "INTERNAL"), (502, "BAD_GATEWAY"), (504, "DEADLINE_EXCEEDED")])
def test_other_5xx_are_never_retried(code, status):
    from app.llm.base import LLMServerError
    p, models, sleeps = make([api_error(code, status, "server error"), ok_response()])
    with pytest.raises(LLMServerError) as info:
        call(p)
    assert len(models.calls) == 1 and sleeps == []
    assert info.value.retryable is True and f"HTTP {code}" in info.value.user_message
    assert "not retried automatically" in info.value.user_message


def test_rate_limit_and_overload_retry_budgets_are_independent():
    p, models, sleeps = make([per_minute_429("5s"), overload(), ok_response()])
    assert call(p).data == {"ok": True}
    assert len(models.calls) == 3 and sleeps[0] == 5.0 and 3.0 <= sleeps[1] <= 3.5


def test_retryable_flags_by_category():
    from app.llm.base import LLMServerError
    assert LLMServerError.retryable and LLMRateLimitError.retryable
    for cls in (LLMQuotaExhaustedError, LLMAuthError, LLMModelUnavailableError, LLMTimeoutError,
                LLMNotConfiguredError, LLMBadRequestError, LLMOutputError, LLMRefusalError):
        assert cls.retryable is False, cls


def test_timeout_and_connection_errors():
    p, _, _ = make([httpx.ReadTimeout("slow")])
    with pytest.raises(LLMTimeoutError):
        call(p)
    p, _, _ = make([httpx.ConnectError("down")])
    with pytest.raises(LLMUnavailableError):
        call(p)


def test_blocked_truncated_and_malformed_outputs():
    p, _, _ = make([ok_response(finish="SAFETY")])
    with pytest.raises(LLMRefusalError):
        call(p)
    p, _, _ = make([ok_response(block="PROHIBITED_CONTENT")])
    with pytest.raises(LLMRefusalError):
        call(p)
    p, _, _ = make([ok_response(finish="MAX_TOKENS")])
    with pytest.raises(LLMOutputError):
        call(p)
    p, _, _ = make([ok_response(text="not json")])
    with pytest.raises(LLMOutputError):
        call(p)
    p, _, _ = make([ok_response(text="[1, 2]")])
    with pytest.raises(LLMOutputError):
        call(p)


def test_next_quota_reset_is_midnight_pacific():
    pac = ZoneInfo("America/Los_Angeles")
    reset = next_quota_reset(datetime(2026, 10, 30, 23, 59, tzinfo=pac))
    assert reset == datetime(2026, 10, 31, 0, 0, tzinfo=pac)
