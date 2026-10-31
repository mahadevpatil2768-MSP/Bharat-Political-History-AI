"""Wires configuration into long-lived components (index, retriever, LLM, answerer)."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.ingestion.manifest import Manifest
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.embeddings import get_embedder
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rerank import build_reranker
from app.retrieval.vector_store import ChromaIndex


@dataclass
class Services:
    settings: Settings
    index: ChromaIndex
    retriever: HybridRetriever

    def pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(self.settings, self.index, Manifest(self.settings.manifest_path))

    def manifest(self) -> Manifest:
        return Manifest(self.settings.manifest_path)


_lock = threading.Lock()
_services: Services | None = None


def build_services(settings: Settings | None = None) -> Services:
    settings = settings or get_settings()
    embedder = get_embedder(settings.embedding_model, settings.embedding_device,
                            settings.embedding_batch_size)
    index = ChromaIndex(settings.index_dir, settings.collection_name, embedder)
    retriever = HybridRetriever(
        index, candidate_k=settings.retrieval_candidate_k, rrf_k=settings.rrf_k,
        min_vector_similarity=settings.min_vector_similarity,
        reranker=build_reranker(settings.reranker, settings.reranker_model, settings.embedding_device),
        manifest_path=settings.manifest_path,
        article_max_continuation=settings.article_max_continuation,
        article_max_related=settings.article_max_related,
    )
    return Services(settings=settings, index=index, retriever=retriever)


def get_services() -> Services:
    global _services
    with _lock:
        if _services is None:
            _services = build_services()
        return _services


def set_services(services: Services | None) -> None:
    """Override (used by tests)."""
    global _services
    _services = services
