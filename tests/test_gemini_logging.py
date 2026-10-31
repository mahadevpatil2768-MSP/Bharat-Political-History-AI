"""Safe, categorised Gemini error logging — SDK stubbed, no network, no real API calls."""

import io
import json
import logging
from logging.handlers import RotatingFileHandler
from types import SimpleNamespace

import httpx
import httpx2
import pytest
from google.genai import errors as genai_errors

from app import logging_setup
from app.config import Settings
from app.generation.answer import AnswerGenerator
from app.llm.base import (
    LLMAuthError,
    LLMNetworkError,
    LLMProtocolError,
    LLMQuotaExhaustedError,
    LLMServerError,
    LLMTimeoutError,
    LLMUnavailableError,
    LLMUnexpectedError,
)
from app.llm.gemini_client import GeminiProvider, api_error_summary, transport_category
from tests.test_gemini_client import FakeModels, api_error, daily_429, ok_response

KEY = "AQ.FAKE-new-format-key.not-real-0123456789abcdef"
PROMPT = "PRIVATE-PROMPT-MARKER Article 14 passage text that must never be logged"
REQ = httpx2.Request("POST", f"https://generativelanguage.googleapis.com/v1beta/models/x:generateContent?key={KEY}")


@pytest.fixture
def captured():
    """Formatted log output exactly as the real handlers would write it (redacting formatter)."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging_setup.RedactingFormatter(logging_setup.LOG_FORMAT))
    logger = logging.getLogger("app.llm.gemini_client")
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield stream
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def provider(outcomes, **overrides):
    models = FakeModels(outcomes)
    p = GeminiProvider(Settings(_env_file=None, gemini_api_key=KEY, **overrides),
                       client=SimpleNamespace(models=models), sleep=lambda _: None)
    return p, models


def run(p):
    return p.generate_json(system="SYSTEM-PROMPT-MARKER", user=PROMPT, schema={"type": "object"})


def assert_safe(text: str):
    assert KEY not in text and "AQ.FAKE" not in text
    assert "PRIVATE-PROMPT-MARKER" not in text and "SYSTEM-PROMPT-MARKER" not in text


def api_lines(text: str) -> list[dict]:
    out = []
    for line in text.splitlines():
        if "gemini_api_error" in line:
            out.append({"line": line, "json": json.loads(line[line.index("{"):])})
    return out


def test_http_5xx_logged_per_attempt_and_distinguished(captured):
    err = api_error(503, "UNAVAILABLE", f"The model is overloaded. Please try again later. key={KEY}")
    p, models = provider([err])
    with pytest.raises(LLMServerError) as info:
        run(p)
    exc = info.value
    assert isinstance(exc, LLMUnavailableError) and exc.category == "http_5xx" and exc.retryable
    assert "HTTP 503 UNAVAILABLE" in exc.user_message and "4 attempt(s)" in exc.user_message
    out = captured.getvalue()
    lines = api_lines(out)
    assert len(lines) == 4 and len(models.calls) == 4
    assert [f"attempt={i}" in x["line"] for i, x in enumerate(lines, start=1)] == [True] * 4
    assert "action=overload retry 1/3 in 3." in lines[0]["line"]
    assert "action=overload retry 3/3 in 15." in lines[2]["line"]
    assert "action=giving up after 3 retr(ies)" in lines[3]["line"]
    j = lines[3]["json"]
    assert j["http_status"] == 503 and j["google_status"] == "UNAVAILABLE"
    assert j["message"].startswith("The model is overloaded") and "[REDACTED]" in j["message"]
    assert all("category=http_5xx" in x["line"] for x in lines)
    assert_safe(out)


def test_other_5xx_logged_as_not_retried(captured):
    p, models = provider([api_error(500, "INTERNAL", "Internal error encountered.")])
    with pytest.raises(LLMServerError):
        run(p)
    line = api_lines(captured.getvalue())[0]
    assert len(models.calls) == 1 and "not retried (only 503 is retried)" in line["line"]
    assert line["json"]["http_status"] == 500


@pytest.mark.parametrize("exc_factory,expected,category", [
    (lambda: httpx2.ConnectError(f"[Errno 11001] getaddrinfo failed for {REQ.url}", request=REQ),
     LLMNetworkError, "network"),
    (lambda: httpx.ConnectError("connection refused"), LLMNetworkError, "network"),
    (lambda: httpx2.ReadError("connection reset by peer", request=REQ), LLMNetworkError, "network"),
    (lambda: httpx2.RemoteProtocolError("Server disconnected without sending a response.", request=REQ),
     LLMProtocolError, "protocol"),
    (lambda: httpx.RemoteProtocolError("peer closed connection without sending complete message body"),
     LLMProtocolError, "protocol"),
])
def test_transport_errors_are_categorised_with_traceback(captured, exc_factory, expected, category):
    p, models = provider([exc_factory()])
    with pytest.raises(expected) as info:
        run(p)
    assert info.value.category == category and len(models.calls) == 1
    out = captured.getvalue()
    assert f"gemini_transport_error category={category}" in out
    assert "Traceback (most recent call last)" in out  # full traceback captured
    assert type(info.value.__cause__).__name__ in out   # original exception class visible
    assert_safe(out)
    assert KEY not in str(info.value)


def test_timeout_logged_with_traceback(captured):
    p, _ = provider([httpx2.ReadTimeout("timed out", request=REQ)])
    with pytest.raises(LLMTimeoutError):
        run(p)
    out = captured.getvalue()
    assert "category=timeout" in out and "Traceback" in out
    assert_safe(out)


def test_quota_and_rate_limit_logged(captured):
    p, _ = provider([daily_429()])
    with pytest.raises(LLMQuotaExhaustedError) as info:
        run(p)
    assert info.value.category == "quota_daily"
    line = api_lines(captured.getvalue())[0]
    assert "category=quota_daily" in line["line"] and "not retried" in line["line"]
    assert line["json"]["quota_ids"] == ["GenerateRequestsPerDayPerProjectPerModel-FreeTier"]
    assert line["json"]["retry_delay_s"] == 40.0
    assert_safe(captured.getvalue())


def test_auth_error_logged_with_reason(captured):
    bad = api_error(400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.",
                    [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID",
                      "domain": "googleapis.com"}])
    p, _ = provider([bad])
    with pytest.raises(LLMAuthError) as info:
        run(p)
    assert info.value.category == "auth"
    line = api_lines(captured.getvalue())[0]
    assert "category=auth" in line["line"] and line["json"]["reasons"] == ["API_KEY_INVALID"]
    assert line["json"]["detail_types"] == ["ErrorInfo"]


def test_unexpected_sdk_exception_is_not_hidden(captured):
    p, _ = provider([ValueError(f"unexpected payload shape near key={KEY}")])
    with pytest.raises(LLMUnexpectedError) as info:
        run(p)
    assert info.value.category == "unexpected"
    out = captured.getvalue()
    assert "category=unexpected" in out and "ValueError" in out and "Traceback" in out
    assert_safe(out)


def test_bad_request_logs_field_paths_only():
    exc = api_error(400, "INVALID_ARGUMENT", "Invalid JSON payload", [
        {"@type": "type.googleapis.com/google.rpc.BadRequest",
         "fieldViolations": [{"field": "generation_config.response_json_schema", "description": "bad"}]}])
    s = api_error_summary(exc)
    assert s["field_violations"] == ["generation_config.response_json_schema"] and "description" not in json.dumps(s)


def test_transport_category_helper():
    assert transport_category(httpx2.LocalProtocolError("bad header")) == "protocol"
    assert transport_category(httpx2.ProxyError("proxy refused")) == "network"
    assert transport_category(httpx2.ConnectError("refused")) == "network"


def test_redactor_scrubs_registered_and_pattern_secrets():
    logging_setup.register_secret(KEY)
    text = (f"hdr x-goog-api-key: {KEY} url ?key=abc123SECRET AIza{'B' * 35} "
            f"Authorization: Bearer abcdefghijklmnopqrstuvwxyz")
    red = logging_setup.redact(text)
    assert KEY not in red and "abc123SECRET" not in red and "AIza" not in red and "abcdefghijklmnop" not in red


def test_rotating_file_logging_writes_redacted_lines(tmp_path, monkeypatch):
    root = logging.getLogger()
    before = list(root.handlers)
    monkeypatch.setattr(logging_setup, "_CONFIGURED", False)
    log_path = tmp_path / "logs" / "backend.log"
    try:
        logging_setup.setup_logging(log_file=log_path)
        new = [h for h in root.handlers if h not in before]
        fh = next(h for h in new if isinstance(h, RotatingFileHandler))
        assert fh.maxBytes == 5 * 1024 * 1024 and fh.backupCount == 5
        logging_setup.register_secret(KEY)
        logging.getLogger("app.test").warning("probe key=%s", KEY)
        fh.flush()
        content = log_path.read_text(encoding="utf-8")
        assert "probe key=[REDACTED]" in content and KEY not in content
    finally:
        for h in [h for h in root.handlers if h not in before]:
            root.removeHandler(h)
            h.close()


def test_log_file_setting_resolves_and_can_be_disabled():
    assert Settings(_env_file=None, log_file="logs/backend.log").log_file_path.name == "backend.log"
    assert Settings(_env_file=None, log_file="").log_file_path is None


def test_category_reaches_answer_response(services, caplog):
    from tests.conftest import CHARTER_MD, CHARTER_META, write_doc
    write_doc(services.settings.sources_dir, "charter.md", CHARTER_MD, CHARTER_META)
    services.pipeline().run()
    p, _ = provider([httpx2.RemoteProtocolError("Server disconnected", request=REQ)])
    gen = AnswerGenerator(services.retriever, p, top_k=3)
    with caplog.at_level(logging.WARNING, logger="app.generation.answer"):
        res = gen.answer("What does the Assembly consist of?")
    assert res.status == "error" and "interrupted mid-response (protocol error)" in res.answer and res.retrieved
    assert any("category=protocol" in r.getMessage() and "LLMProtocolError" in r.getMessage()
               for r in caplog.records)
