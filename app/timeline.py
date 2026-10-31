"""Timeline / event explorer.

Two kinds of entries, neither invented:
1. Curated events from data/timeline/events.json. Each must cite an indexed
   document and an ``evidence_quote``; an event is only shown when that quote is
   actually found in the cited document's indexed text (and page, if given).
2. Document coverage spans derived from document metadata (event_year_start/end).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class TimelineEvent(BaseModel):
    id: str
    date: str = Field(..., pattern=r"^-?\d{1,4}(-\d{2}(-\d{2})?)?$")
    title: str
    description: str = ""
    document_id: str
    page: int | None = None
    evidence_quote: str = Field(..., min_length=10)
    periods: list[str] = Field(default_factory=list)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_events(path: Path) -> tuple[list[TimelineEvent], list[str]]:
    if not path.exists():
        return [], []
    errors: list[str] = []
    events: list[TimelineEvent] = []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [], [f"events.json is not valid JSON: {exc}"]
    for item in raw.get("events", []):
        try:
            events.append(TimelineEvent.model_validate(item))
        except ValidationError as exc:
            errors.append(f"event {item.get('id', '?')}: {exc.errors()[0]['msg']}")
    return events, errors


def verify_events(events: list[TimelineEvent], chunks: list[dict[str, Any]]) -> tuple[list[dict], list[str]]:
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for c in chunks:
        by_doc.setdefault(c["metadata"]["document_id"], []).append(c)
    verified, problems = [], []
    for ev in events:
        doc_chunks = by_doc.get(ev.document_id)
        if not doc_chunks:
            problems.append(f"{ev.id}: document '{ev.document_id}' is not indexed")
            continue
        quote = _norm(ev.evidence_quote)
        match = None
        for c in doc_chunks:
            m = c["metadata"]
            if quote in _norm(c["text"]):
                if ev.page is None or (m.get("page_start") or 0) <= ev.page <= (m.get("page_end") or 0):
                    match = c
                    break
        if not match:
            problems.append(f"{ev.id}: evidence quote not found in '{ev.document_id}'"
                            + (f" on page {ev.page}" if ev.page else ""))
            continue
        m = match["metadata"]
        verified.append({
            "kind": "event", "id": ev.id, "date": ev.date, "year": int(ev.date.lstrip("-")[:4]) * (-1 if ev.date.startswith("-") else 1),
            "title": ev.title, "description": ev.description, "periods": ev.periods,
            "evidence_quote": ev.evidence_quote, "document_id": ev.document_id, "chunk_id": match["chunk_id"],
            "source_title": m.get("title"), "source_url": m.get("source_url"),
            "page_start": m.get("page_start"), "page_end": m.get("page_end"),
        })
    return verified, problems


def document_spans(manifest_docs: dict[str, dict[str, Any]]) -> list[dict]:
    spans = []
    for doc_id, entry in manifest_docs.items():
        meta = entry.get("metadata") or {}
        if entry.get("status") != "indexed" or meta.get("event_year_start") is None:
            continue
        spans.append({
            "kind": "document_span", "document_id": doc_id, "title": meta.get("title"),
            "year": meta["event_year_start"], "year_end": meta.get("event_year_end") or meta["event_year_start"],
            "periods": meta.get("historical_periods", []), "source_type": meta.get("source_type"),
            "source_url": meta.get("source_url"),
        })
    return sorted(spans, key=lambda s: s["year"])
