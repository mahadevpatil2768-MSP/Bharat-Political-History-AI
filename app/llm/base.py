"""Provider-neutral LLM interface. Ingestion and retrieval never import a provider."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    """Base class. ``user_message`` is safe to show in the UI (never contains secrets)."""

    user_message = "The language model request failed."
    category = "llm_error"  # stable, log-friendly error category
    retryable = False  # whether the UI may offer a manual Retry (HTTP 5xx and per-minute rate limits only)

    def __init__(self, detail: str = "", user_message: str | None = None):
        super().__init__(detail or self.user_message)
        if user_message:
            self.user_message = user_message


class LLMNotConfiguredError(LLMError):
    category = "not_configured"
    user_message = ("No GEMINI_API_KEY is configured. Add it to the .env file "
                    "(see .env.example) to enable generated answers.")


class LLMModelNotAllowedError(LLMError):
    category = "model_not_allowed"
    user_message = ("The configured LLM_MODEL is not on the verified free-tier allowlist, so no request "
                    "was sent. Choose a model listed in app/llm/gemini_client.py FREE_TIER_MODELS.")


class LLMAuthError(LLMError):
    category = "auth"
    user_message = ("The Gemini API rejected the API key (invalid key or permission denied). "
                    "Check GEMINI_API_KEY in .env.")


class LLMRateLimitError(LLMError):
    category = "rate_limit"
    retryable = True
    user_message = ("The Gemini free-tier per-minute rate limit was reached. "
                    "Please wait about a minute and try again.")


class LLMQuotaExhaustedError(LLMRateLimitError):
    category = "quota_daily"
    retryable = False  # retrying cannot help until the daily reset
    user_message = ("The Gemini free-tier daily quota for this model is used up. Requests are stopped "
                    "until the quota resets (midnight Pacific time). No paid tier or other provider is used.")


class LLMTimeoutError(LLMError):
    category = "timeout"
    user_message = "The language model did not respond in time. Please try again."


class LLMUnavailableError(LLMError):
    user_message = "The language model service is temporarily unavailable. Please try again later."
    category = "unavailable"


class LLMServerError(LLMUnavailableError):
    """Google returned HTTP 5xx on every attempt."""
    category = "http_5xx"
    retryable = True


class LLMNetworkError(LLMUnavailableError):
    user_message = ("Could not reach the Gemini API (network error). Check the internet connection, "
                    "firewall or proxy, then try again.")
    category = "network"


class LLMProtocolError(LLMUnavailableError):
    user_message = ("The connection to the Gemini API was interrupted mid-response (protocol error). "
                    "Please try again.")
    category = "protocol"


class LLMModelUnavailableError(LLMError):
    category = "model_unavailable"
    user_message = ("The configured Gemini model is not available to this API key "
                    "(not found or not enabled). Check LLM_MODEL.")


class LLMBadRequestError(LLMError):
    category = "bad_request"
    user_message = "The language model request was invalid (check LLM_MODEL and settings)."


class LLMRefusalError(LLMError):
    category = "refusal"
    user_message = "The model declined to answer this request (blocked by the provider's safety filters)."


class LLMOutputError(LLMError):
    user_message = "The model's response could not be parsed."
    category = "output"


class LLMUnexpectedError(LLMError):
    user_message = "The language model request failed with an unexpected error. See logs/backend.log."
    category = "unexpected"


@dataclass
class LLMResult:
    text: str
    data: dict[str, Any] | None
    model: str
    stop_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_seconds: float = 0.0


class LLMProvider(Protocol):
    model: str

    def generate_json(self, *, system: str, user: str, schema: dict[str, Any]) -> LLMResult: ...
