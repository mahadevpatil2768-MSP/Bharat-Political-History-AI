"""Metadata filtering.

``to_chroma_where`` pushes exact-match conditions down to Chroma; ``matches``
applies the full filter (including substring filters) in Python so that vector
and keyword retrieval honour exactly the same rules.
"""

from __future__ import annotations

from typing import Any

from app.schemas import SearchFilters


def _year(date_str: str | None) -> int | None:
    if not date_str:
        return None
    try:
        return int(str(date_str)[:4])
    except ValueError:
        return None


def to_chroma_where(f: SearchFilters) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if f.periods:
        ors = [{f"period_{p if isinstance(p, str) else p.value}": True} for p in f.periods]
        clauses.append(ors[0] if len(ors) == 1 else {"$or": ors})
    if f.source_types:
        vals = [s if isinstance(s, str) else s.value for s in f.source_types]
        clauses.append({"source_type": {"$in": vals}})
    if f.document_ids:
        clauses.append({"document_id": {"$in": list(f.document_ids)}})
    if f.languages:
        clauses.append({"language": {"$in": [l.lower() for l in f.languages]}})
    # Year-range overlap: documents without event years are excluded when a range is given,
    # because we cannot establish that they cover the requested period.
    if f.year_from is not None:
        clauses.append({"event_year_end": {"$gte": f.year_from}})
    if f.year_to is not None:
        clauses.append({"event_year_start": {"$lte": f.year_to}})
    if f.published_from:
        clauses.append({"pub_year": {"$gte": _year(f.published_from)}})
    if f.published_to:
        clauses.append({"pub_year": {"$lte": _year(f.published_to)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def has_python_only_filters(f: SearchFilters) -> bool:
    return bool(f.topic or f.person or f.institution or f.author_or_institution)


def matches(meta: dict[str, Any], f: SearchFilters) -> bool:
    if f.periods:
        keys = [p if isinstance(p, str) else p.value for p in f.periods]
        if not any(meta.get(f"period_{k}") for k in keys):
            return False
    if f.source_types:
        vals = {s if isinstance(s, str) else s.value for s in f.source_types}
        if meta.get("source_type") not in vals:
            return False
    if f.document_ids and meta.get("document_id") not in set(f.document_ids):
        return False
    if f.languages and (meta.get("language") or "").lower() not in {l.lower() for l in f.languages}:
        return False
    if f.year_from is not None:
        end = meta.get("event_year_end")
        if end is None or end < f.year_from:
            return False
    if f.year_to is not None:
        start = meta.get("event_year_start")
        if start is None or start > f.year_to:
            return False
    pub = meta.get("pub_year")
    if f.published_from and (pub is None or pub < _year(f.published_from)):
        return False
    if f.published_to and (pub is None or pub > _year(f.published_to)):
        return False
    for field, key in ((f.topic, "topics"), (f.person, "persons"), (f.institution, "institutions")):
        if field and field.lower().strip() not in (meta.get(key) or "").lower():
            return False
    if f.author_or_institution and f.author_or_institution.lower() not in (
            meta.get("author_or_institution") or "").lower():
        return False
    return True
