"""Retrieval metrics over labelled queries.

A label is ``{"document_id": ..., "contains": ["evidence phrase", ...]}``. A retrieved
chunk satisfies a label when it belongs to that document and contains at least one
of the phrases (whitespace/case-insensitive). Labels are therefore independent of
chunk IDs, so they survive changes to chunk size or overlap.

* Recall@k    = labels satisfied by the top-k chunks / number of labels
* Precision@k = top-k chunks satisfying any label / k
* MRR         = 1 / rank of the first chunk satisfying any label within the full
                retrieved list (0 if none); computed separately from the @k cut-offs
"""

from __future__ import annotations

import re
from typing import Any


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def chunk_matches(chunk: dict[str, Any], label: dict[str, Any]) -> bool:
    if chunk["document_id"] != label["document_id"]:
        return False
    phrases = label.get("contains") or []
    if not phrases:
        return True
    text = _norm(chunk["text"])
    return any(_norm(p) in text for p in phrases)


def retrieval_metrics(retrieved: list[dict[str, Any]], labels: list[dict[str, Any]], k: int) -> dict[str, float]:
    top = retrieved[:k]
    if not labels:
        return {}
    satisfied = sum(1 for lab in labels if any(chunk_matches(c, lab) for c in top))
    relevant_flags = [any(chunk_matches(c, lab) for lab in labels) for c in top]
    return {f"recall@{k}": satisfied / len(labels), f"precision@{k}": sum(relevant_flags) / k}


def precision_of_returned(retrieved: list[dict[str, Any]], labels: list[dict[str, Any]], k: int) -> float | None:
    """Share of the passages actually returned (up to k) that are relevant — unlike precision@k, it is not
    capped by k when the retriever deliberately returns fewer than k passages."""
    top = retrieved[:k]
    if not labels or not top:
        return None
    return sum(1 for c in top if any(chunk_matches(c, lab) for lab in labels)) / len(top)


def reciprocal_rank(retrieved: list[dict[str, Any]], labels: list[dict[str, Any]]) -> float | None:
    if not labels:
        return None
    for i, c in enumerate(retrieved, start=1):
        if any(chunk_matches(c, lab) for lab in labels):
            return 1.0 / i
    return 0.0


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None
