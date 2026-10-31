"""Optional reranking behind a configurable interface (RERANKER=none|cross-encoder)."""

from __future__ import annotations

import logging
from typing import Protocol

from app.schemas import RetrievedChunk

log = logging.getLogger(__name__)


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]: ...


class NoOpReranker:
    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return chunks


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        log.info("Loading reranker %s", model_name)
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        scores = self._model.predict([(query, c.text) for c in chunks])
        for c, s in zip(chunks, scores):
            c.rerank_score = float(s)
        return sorted(chunks, key=lambda c: c.rerank_score, reverse=True)


def build_reranker(kind: str, model_name: str, device: str = "cpu") -> Reranker:
    if kind == "cross-encoder":
        return CrossEncoderReranker(model_name, device)
    return NoOpReranker()
