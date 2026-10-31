"""Scripted provider for tests and offline demos. Never used unless explicitly constructed."""

from __future__ import annotations

from typing import Any, Callable

from app.llm.base import LLMError, LLMResult


class FakeProvider:
    def __init__(self, responder: Callable[[str, str], dict[str, Any]] | dict[str, Any] | LLMError,
                 model: str = "fake-model"):
        self.responder = responder
        self.model = model
        self.calls: list[dict[str, str]] = []

    def generate_json(self, *, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        self.calls.append({"system": system, "user": user})
        if isinstance(self.responder, LLMError):
            raise self.responder
        data = self.responder(system, user) if callable(self.responder) else self.responder
        return LLMResult(text="", data=data, model=self.model, stop_reason="end_turn")
