"""Google Gemini provider (official ``google-genai`` SDK), free tier only.

Zero-cost safeguards:
* Only models verified as "Free of charge" on the Gemini API free tier may be used
  (``FREE_TIER_MODELS``); any other LLM_MODEL is refused before a request is sent.
* No tools are sent, so no Google Search grounding or other billable features.
* No fallback to another model or provider, ever.
* HTTP 429: per-minute limits get a small, bounded number of retries (honouring the
  server's retryDelay, capped); a daily quota violation stops immediately.

Free-tier pricing applies only while the Google project behind the key has no billing
account linked — that cannot be detected from the API, so it is the user's responsibility
(see README → Gemini free tier). The API key is read from settings, never logged, and
redacted from any error text.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.config import Settings
from app.llm.base import (
    LLMAuthError,
    LLMBadRequestError,
    LLMModelNotAllowedError,
    LLMModelUnavailableError,
    LLMNetworkError,
    LLMNotConfiguredError,
    LLMOutputError,
    LLMProtocolError,
    LLMQuotaExhaustedError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMResult,
    LLMServerError,
    LLMTimeoutError,
    LLMUnexpectedError,
)
from app.logging_setup import redact, register_secret

log = logging.getLogger(__name__)

# Text-generation models listed with Free Tier input and output "Free of charge" on
# https://ai.google.dev/gemini-api/docs/pricing (page last updated 2026-10-28 UTC).
# Re-verify against that page before adding a model; pricing pages change.
FREE_TIER_MODELS_VERIFIED_ON = "2026-10-28"
FREE_TIER_MODELS = frozenset({
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
})

# Backoff schedule for HTTP 503 UNAVAILABLE (model overloaded). Jitter of up to 0.5 s is added.
OVERLOAD_BACKOFF_SECONDS = (3.0, 8.0, 15.0)

_BLOCK_FINISH_REASONS = {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"}
_OK_FINISH_REASONS = {"STOP", "FINISH_REASON_UNSPECIFIED", "", "None"}

try:  # the SDK may use either HTTP stack
    import httpx
    _TIMEOUT_EXC: tuple[type[BaseException], ...] = (httpx.TimeoutException,)
    _CONNECT_EXC: tuple[type[BaseException], ...] = (httpx.TransportError,)
except ImportError:  # pragma: no cover
    _TIMEOUT_EXC, _CONNECT_EXC = (), ()
try:
    import httpx2
    _TIMEOUT_EXC += (httpx2.TimeoutException,)
    _CONNECT_EXC += (httpx2.TransportError,)
except (ImportError, AttributeError):  # pragma: no cover
    pass


def _error_body(exc: genai_errors.APIError) -> dict[str, Any]:
    d = exc.details if isinstance(exc.details, dict) else {}
    return d.get("error", d) if isinstance(d, dict) else {}


def _detail_items(exc: genai_errors.APIError) -> list[dict[str, Any]]:
    items = _error_body(exc).get("details") or []
    return [i for i in items if isinstance(i, dict)]


def quota_ids(exc: genai_errors.APIError) -> list[str]:
    ids = []
    for item in _detail_items(exc):
        for v in item.get("violations") or []:
            if isinstance(v, dict) and v.get("quotaId"):
                ids.append(str(v["quotaId"]))
    return ids


def is_daily_quota(exc: genai_errors.APIError) -> bool:
    ids = quota_ids(exc)
    if ids:
        return any("perday" in q.lower() for q in ids)
    return bool(re.search(r"per\s*day|daily", str(exc.message or ""), re.I))


def retry_delay_seconds(exc: genai_errors.APIError) -> float | None:
    for item in _detail_items(exc):
        if str(item.get("@type", "")).endswith("RetryInfo") and item.get("retryDelay"):
            m = re.match(r"^\s*([\d.]+)s\s*$", str(item["retryDelay"]))
            if m:
                return float(m.group(1))
    return None


def next_quota_reset(now: datetime | None = None) -> datetime:
    """Free-tier daily (RPD) quotas reset at midnight Pacific time (Gemini rate-limits docs)."""
    pacific = ZoneInfo("America/Los_Angeles")
    now = (now or datetime.now(pacific)).astimezone(pacific)
    return datetime.combine(now.date() + timedelta(days=1), datetime.min.time(), tzinfo=pacific)


def _is_invalid_key(exc: genai_errors.APIError) -> bool:
    reasons = {str(i.get("reason", "")) for i in _detail_items(exc)}
    return "API_KEY_INVALID" in reasons or "api key not valid" in str(exc.message or "").lower()


def transport_category(exc: BaseException) -> str:
    """'protocol' for HTTP protocol violations (e.g. server closed the connection mid-response),
    'network' for everything else at the transport layer (connect/read/write/proxy failures)."""
    return "protocol" if any(c.__name__.endswith("ProtocolError") for c in type(exc).__mro__) else "network"


def api_error_summary(exc: genai_errors.APIError, limit: int = 500) -> dict[str, Any]:
    """Safe, structured view of a Gemini API error: no request content, key redacted, message truncated."""
    items = _detail_items(exc)
    summary: dict[str, Any] = {
        "http_status": exc.code,
        "google_status": exc.status,
        "message": redact(str(exc.message or ""))[:limit],
        "detail_types": [str(i.get("@type", "")).rsplit(".", 1)[-1] for i in items if i.get("@type")],
    }
    reasons = [str(i["reason"]) for i in items if i.get("reason")]
    if reasons:
        summary["reasons"] = reasons
    qids = quota_ids(exc)
    if qids:
        summary["quota_ids"] = qids
    delay = retry_delay_seconds(exc)
    if delay is not None:
        summary["retry_delay_s"] = delay
    fields = [str(v.get("field")) for i in items for v in (i.get("fieldViolations") or [])
              if isinstance(v, dict) and v.get("field")]
    if fields:
        summary["field_violations"] = fields  # field paths only, never values
    return summary


class GeminiProvider:
    def __init__(self, settings: Settings, client: Any | None = None, sleep=time.sleep):
        if not settings.llm_configured:
            raise LLMNotConfiguredError("GEMINI_API_KEY missing")
        if settings.llm_model not in FREE_TIER_MODELS:
            raise LLMModelNotAllowedError(
                f"model {settings.llm_model!r} not in FREE_TIER_MODELS",
                user_message=(f"LLM_MODEL '{settings.llm_model}' is not on the verified free-tier allowlist "
                              f"(verified {FREE_TIER_MODELS_VERIFIED_ON}), so no request was sent. Allowed: "
                              + ", ".join(sorted(FREE_TIER_MODELS)) + "."))
        self.model = settings.llm_model
        self.max_tokens = settings.llm_max_tokens
        self.thinking_level = settings.llm_thinking_level
        self.max_retries = settings.llm_max_retries
        self.max_retry_wait = settings.llm_retry_max_wait_seconds
        self.overload_max_retries = min(settings.llm_overload_max_retries, len(OVERLOAD_BACKOFF_SECONDS))
        self.overload_max_total_wait = settings.llm_overload_max_total_wait_seconds
        self._key = settings.gemini_api_key.get_secret_value().strip()
        register_secret(self._key)  # scrubbed from every log line, including tracebacks
        self._sleep = sleep
        # Circuit breaker: after a daily-quota 429, send nothing until the quota resets.
        self._quota_blocked_until: datetime | None = None
        self._client = client or genai.Client(
            api_key=self._key,
            # SDK retries are disabled (retry_options=None); retries are handled below so that
            # daily-quota errors are never retried. Timeout is in milliseconds.
            http_options=types.HttpOptions(timeout=int(settings.llm_timeout_seconds * 1000)),
        )

    def _redact(self, text: str) -> str:
        text = text.replace(self._key, "[REDACTED]") if self._key else text
        return redact(text)

    def _log_api_error(self, category: str, exc: genai_errors.APIError, attempt: int, action: str) -> None:
        # Structured and safe: status, Google's status/message, detail types - never prompts or passages.
        log.warning("gemini_api_error category=%s model=%s attempt=%d action=%s %s", category, self.model,
                    attempt, action, json.dumps(api_error_summary(exc), ensure_ascii=False))

    def _log_transport_error(self, category: str, exc: BaseException, attempt: int) -> None:
        # Full traceback (redacted by the log formatter) so the exact failure point is visible.
        log.error("gemini_transport_error category=%s model=%s attempt=%d exception=%s.%s message=%s",
                  category, self.model, attempt, type(exc).__module__,
                  type(exc).__name__, self._redact(str(exc))[:500], exc_info=exc)

    def _config(self, system: str, schema: dict[str, Any]) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            max_output_tokens=self.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level),
            # No tools: no Google Search grounding or other billable features.
        )

    def _overload_wait(self, exc: genai_errors.APIError, retries_done: int, waited: float) -> float | None:
        """Seconds to wait before the next 503 retry, or None to give up.

        Follows OVERLOAD_BACKOFF_SECONDS (~3, 8, 15 s + jitter), waits at least Google's retryDelay when
        given, and gives up rather than exceed the total wait budget.
        """
        if retries_done >= self.overload_max_retries:
            return None
        wait = OVERLOAD_BACKOFF_SECONDS[retries_done] + random.uniform(0, 0.5)
        server_delay = retry_delay_seconds(exc)
        if server_delay is not None:
            wait = max(wait, server_delay)
        if waited + wait > self.overload_max_total_wait:
            return None
        return wait

    def _quota_message(self) -> str:
        reset = self._quota_blocked_until
        local = reset.astimezone().strftime("%Y-%m-%d %H:%M %Z") if reset else "midnight Pacific time"
        return (f"The Gemini free-tier daily quota for {self.model} is used up. No further requests will be "
                f"sent until it resets (about {local}). No paid tier or other provider is used.")

    def generate_json(self, *, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        if self._quota_blocked_until and datetime.now(self._quota_blocked_until.tzinfo) < self._quota_blocked_until:
            raise LLMQuotaExhaustedError("daily quota exhausted (request not sent)", user_message=self._quota_message())
        config = self._config(system, schema)
        attempt = 0
        rate_limit_retries = overload_retries = 0
        overload_waited = 0.0
        t0 = time.perf_counter()
        while True:
            attempt += 1
            try:
                response = self._client.models.generate_content(model=self.model, contents=user, config=config)
                break
            except genai_errors.APIError as exc:
                code, status = exc.code, str(exc.status or "")
                detail = self._redact(f"{code} {status}: {exc.message}")
                if code == 429:
                    if is_daily_quota(exc):
                        self._quota_blocked_until = next_quota_reset()
                        self._log_api_error("quota_daily", exc, attempt,
                                            "not retried; requests blocked until "
                                            f"{self._quota_blocked_until.isoformat()}")
                        raise LLMQuotaExhaustedError(detail, user_message=self._quota_message()) from exc
                    if rate_limit_retries >= self.max_retries:
                        self._log_api_error("rate_limit", exc, attempt, "giving up")
                        raise LLMRateLimitError(detail) from exc
                    rate_limit_retries += 1
                    wait = min(retry_delay_seconds(exc) or 2.0 ** rate_limit_retries + random.uniform(0, 1),
                               self.max_retry_wait)
                    self._log_api_error("rate_limit", exc, attempt, f"retry in {wait:.1f}s")
                    self._sleep(wait)
                    continue
                if code == 503:  # UNAVAILABLE: model overloaded — the only 5xx retried automatically
                    wait = self._overload_wait(exc, overload_retries, overload_waited)
                    if wait is None:
                        self._log_api_error("http_5xx", exc, attempt,
                                            f"giving up after {overload_retries} retr(ies), "
                                            f"waited {overload_waited:.1f}s")
                        raise LLMServerError(detail, user_message=(
                            f"Gemini is experiencing high demand (HTTP 503 {status or 'UNAVAILABLE'}); "
                            f"{attempt} attempt(s), with {overload_waited:.0f}s of waiting between them, "
                            "did not succeed. This is a temporary server-side condition — use Retry in a "
                            "minute or two.")) from exc
                    overload_retries += 1
                    overload_waited += wait
                    self._log_api_error("http_5xx", exc, attempt,
                                        f"overload retry {overload_retries}/{self.overload_max_retries} "
                                        f"in {wait:.1f}s")
                    self._sleep(wait)
                    continue
                if code is not None and 500 <= code <= 599:  # other 5xx: never retried automatically
                    self._log_api_error("http_5xx", exc, attempt, "not retried (only 503 is retried)")
                    raise LLMServerError(detail, user_message=(
                        f"Google's Gemini service returned HTTP {code} {status}. This is a server-side "
                        "error; it was not retried automatically. Use Retry to try again.")) from exc
                if code in (401, 403) or (code == 400 and _is_invalid_key(exc)):
                    self._log_api_error("auth", exc, attempt, "not retried")
                    raise LLMAuthError(detail) from exc
                if code == 404:
                    self._log_api_error("model_unavailable", exc, attempt, "not retried")
                    raise LLMModelUnavailableError(detail) from exc
                self._log_api_error("bad_request", exc, attempt, "not retried")
                raise LLMBadRequestError(detail) from exc
            except _TIMEOUT_EXC as exc:
                self._log_transport_error("timeout", exc, attempt)
                raise LLMTimeoutError(f"request timed out: {type(exc).__name__}") from exc
            except _CONNECT_EXC as exc:
                category = transport_category(exc)
                self._log_transport_error(category, exc, attempt)
                err = LLMProtocolError if category == "protocol" else LLMNetworkError
                raise err(f"{category} error: {type(exc).__name__}: {self._redact(str(exc))[:300]}") from exc
            except Exception as exc:  # anything else from the SDK: never swallow silently
                self._log_transport_error("unexpected", exc, attempt)
                raise LLMUnexpectedError(f"unexpected {type(exc).__name__}: {self._redact(str(exc))[:300]}") from exc
        latency = time.perf_counter() - t0

        um = getattr(response, "usage_metadata", None)
        usage = {"input_tokens": getattr(um, "prompt_token_count", None),
                 "output_tokens": getattr(um, "candidates_token_count", None),
                 "thinking_tokens": getattr(um, "thoughts_token_count", None)}
        pf = getattr(response, "prompt_feedback", None)
        if pf is not None and getattr(pf, "block_reason", None):
            raise LLMRefusalError(f"prompt blocked: {pf.block_reason}")
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise LLMOutputError("no candidates returned")
        finish = getattr(candidates[0].finish_reason, "name", str(candidates[0].finish_reason or ""))
        log.info("llm call provider=gemini model=%s served=%s finish=%s attempts=%d latency=%.1fs usage=%s",
                 self.model, getattr(response, "model_version", None), finish, attempt, latency, usage)
        if finish == "MAX_TOKENS":
            raise LLMOutputError("output truncated at max tokens",
                                 user_message="The answer was cut off (LLM_MAX_TOKENS too low).")
        if finish in _BLOCK_FINISH_REASONS:
            raise LLMRefusalError(f"generation stopped: {finish}")
        if finish not in _OK_FINISH_REASONS:
            raise LLMOutputError(f"unexpected finish reason: {finish}")
        text = response.text or ""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"invalid JSON from model: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMOutputError("model JSON was not an object")
        return LLMResult(text=text, data=data, model=getattr(response, "model_version", None) or self.model,
                         stop_reason=finish, usage=usage, latency_seconds=round(latency, 2))
