"""Session-only conversation state and the follow-up context strategy.

The backend's POST /query takes a single question (no history), and every answer must be
grounded only in evidence retrieved for that request. For follow-ups that refer back to the
previous turn ("What about his view on railways?"), the UI may prepend the *previous question*
— never previous answers — so retrieval can resolve the reference. The exact text sent is
stored and shown to the user ("Interpreted as …"), and the behaviour can be switched off.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any

MAX_QUESTION_CHARS = 1000  # backend QueryRequest limit
MAX_FOLLOWUP_WORDS = 14

_REFERENTIAL = re.compile(
    r"\b(it|its|this|that|these|those|he|she|him|his|her|hers|they|them|their|theirs|"
    r"the same|above|earlier|previous|former|latter|the article|the document|the text|the book)\b", re.I)
_FOLLOWUP_START = re.compile(
    r"^\s*(and|also|but|so|then|what about|how about|why|explain more|tell me more|more on|elaborate|"
    r"what else|compare (it|this|that))\b", re.I)


def new_conversation() -> dict[str, Any]:
    return {"id": uuid.uuid4().hex[:10], "title": "New chat", "created": datetime.now().strftime("%H:%M"),
            "messages": []}


def title_from(question: str, limit: int = 48) -> str:
    q = re.sub(r"\s+", " ", question).strip()
    return q if len(q) <= limit else q[: limit - 1].rstrip() + "…"


def is_follow_up(question: str) -> bool:
    words = question.split()
    if not words or len(words) > MAX_FOLLOWUP_WORDS:
        return False
    return bool(_FOLLOWUP_START.search(question) or _REFERENTIAL.search(question))


def previous_context_question(messages: list[dict]) -> str | None:
    """Most recent user question whose answer was not declined (declined topics are not carried over)."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m["role"] != "user":
            continue
        reply = messages[i + 1] if i + 1 < len(messages) else None
        status = (reply or {}).get("response", {}) or {}
        if status.get("status") == "declined":
            continue
        return m["content"]
    return None


def build_query(question: str, messages: list[dict], use_context: bool = True) -> tuple[str, bool]:
    """Return (text to send to /query, whether earlier-question context was added)."""
    question = re.sub(r"\s+", " ", question).strip()
    if not use_context or not is_follow_up(question):
        return question, False
    prev = previous_context_question(messages)
    if not prev:
        return question, False
    template = '{q} (Follow-up to my earlier question: "{p}")'
    room = MAX_QUESTION_CHARS - len(template.format(q=question, p=""))
    if room < 20:
        return question, False
    prev = prev if len(prev) <= room else prev[: room - 1].rstrip() + "…"
    return template.format(q=question, p=prev), True
