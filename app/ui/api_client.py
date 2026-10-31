"""HTTP client for the FastAPI backend, used by the Streamlit UI.

The UI never talks to Gemini, Chroma or the index directly — every answer comes from
``POST /query`` so the RAG pipeline (retrieval, grounding, citation validation) is the
single source of truth. Failures are mapped to ``ApiError`` with a user-friendly message;
raw response bodies are never shown to users.
"""

from __future__ import annotations

import os
from typing import Any

import requests
from pydantic import ValidationError

from app.schemas import AnswerResponse

DEFAULT_API = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
START_HINT = "Start it with:  uvicorn app.api.main:app --host 127.0.0.1 --port 8000"


class ApiError(RuntimeError):
    """kind: connection | timeout | validation | server | invalid_response | http"""

    def __init__(self, kind: str, user_message: str):
        super().__init__(user_message)
        self.kind = kind
        self.user_message = user_message


class ApiClient:
    def __init__(self, base_url: str = DEFAULT_API, session: Any = None,
                 query_timeout: float = 300.0, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.session = session or requests  # anything with requests-style .get/.post (e.g. FastAPI TestClient)
        self.query_timeout = query_timeout
        self.timeout = timeout

    # -- transport -------------------------------------------------------
    def _request(self, method: str, path: str, timeout: float, **kwargs) -> Any:
        url = f"{self.base_url}{path}"
        try:
            r = getattr(self.session, method)(url, timeout=timeout, **kwargs)
        except requests.ConnectTimeout as exc:  # subclass of both ConnectionError and Timeout
            raise ApiError("connection", f"Can't reach the research backend at {self.base_url}. "
                                         f"{START_HINT}") from exc
        except requests.Timeout as exc:
            raise ApiError("timeout", "The research backend took too long to respond. "
                                      "Please try again in a moment.") from exc
        except requests.ConnectionError as exc:
            raise ApiError("connection", f"Can't reach the research backend at {self.base_url}. "
                                         f"{START_HINT}") from exc
        if r.status_code == 422:
            raise ApiError("validation", f"The question couldn't be processed: {_detail(r)}")
        if r.status_code >= 500:
            raise ApiError("server", "The research backend hit an internal error. "
                                     "Check the API terminal for details and try again.")
        if r.status_code >= 400:
            raise ApiError("http", f"The backend rejected the request ({r.status_code}): {_detail(r)}")
        try:
            return r.json()
        except ValueError as exc:
            raise ApiError("invalid_response", "The backend returned a response the interface "
                                               "couldn't read.") from exc

    def get(self, path: str) -> Any:
        return self._request("get", path, self.timeout)

    def post(self, path: str, payload: dict, timeout: float | None = None) -> Any:
        return self._request("post", path, timeout or self.timeout, json=payload)

    # -- endpoints -------------------------------------------------------
    def health(self) -> dict:
        return self.get("/health")

    def filter_options(self) -> dict:
        return self.get("/meta/filters")

    def documents(self) -> list[dict]:
        return self.get("/documents").get("documents", [])

    def query(self, question: str, filters: dict | None = None) -> dict:
        """POST /query and validate the response against the backend's AnswerResponse schema."""
        data = self.post("/query", {"question": question, "filters": filters or {}}, timeout=self.query_timeout)
        try:
            return AnswerResponse.model_validate(data).model_dump(mode="json")
        except ValidationError as exc:
            raise ApiError("invalid_response", "The backend returned an answer in an unexpected format, "
                                               "so it was not displayed.") from exc


def _detail(r: Any) -> str:
    try:
        d = r.json().get("detail")
    except (ValueError, AttributeError):
        return "no details"
    if isinstance(d, list):  # FastAPI validation errors
        return "; ".join(str(e.get("msg", "")) for e in d if isinstance(e, dict)) or "invalid input"
    return str(d)[:300] if d else "no details"
