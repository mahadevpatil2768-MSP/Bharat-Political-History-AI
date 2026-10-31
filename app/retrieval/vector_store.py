"""Persistent Chroma vector store. Implements the ingestion ``Indexer`` protocol."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.retrieval.embeddings import Embedder
from app.retrieval.legal_structure import annotate_document
from app.schemas import Chunk, IngestedDocument

log = logging.getLogger(__name__)

ADD_BATCH = 128


class EmbeddingModelMismatchError(RuntimeError):
    pass


def _year(date_str: str | None) -> int | None:
    try:
        return int(date_str[:4]) if date_str else None
    except ValueError:
        return None


def embedding_text(chunk_text: str, section_heading: str | None) -> str:
    """Text used for embedding / keyword indexing: section heading gives context."""
    return f"{section_heading}\n{chunk_text}" if section_heading else chunk_text


def chunk_metadata(doc: IngestedDocument, chunk: Chunk) -> dict[str, Any]:
    """Flatten document + chunk metadata into Chroma-compatible scalars (no None values)."""
    meta: dict[str, Any] = {
        "document_id": doc.document_id,
        "chunk_index": chunk.chunk_index,
        "title": doc.title,
        "author_or_institution": doc.author_or_institution,
        "source_url": doc.source_url,
        "source_type": doc.source_type,
        "publication_date": doc.publication_date,
        "pub_year": _year(doc.publication_date),
        "date_accessed": doc.date_accessed,
        "language": (doc.language or "").lower() or None,
        "document_version": doc.document_version,
        "rights_statement": doc.rights_statement,
        "event_year_start": doc.event_year_start,
        "event_year_end": doc.event_year_end,
        "periods": "|".join(doc.historical_periods) or None,
        "topics": "|".join(doc.topics) or None,
        "persons": "|".join(doc.persons) or None,
        "institutions": "|".join(doc.institutions) or None,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_heading": chunk.section_heading,
        "contains_table": chunk.contains_table,
        "content_hash": doc.content_hash,
        "ingestion_timestamp": doc.ingestion_timestamp,
        "file_path": doc.file_path,
    }
    for p in doc.historical_periods:
        meta[f"period_{p}"] = True
    return {k: v for k, v in meta.items() if v is not None}


class ChromaIndex:
    def __init__(self, persist_dir: Path, collection_name: str, embedder: Embedder):
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.embedding_model_name = embedder.name
        self._client = chromadb.PersistentClient(
            path=str(persist_dir), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": embedder.name},
            embedding_function=None,
        )
        stored = (self.collection.metadata or {}).get("embedding_model")
        if stored and stored != embedder.name:
            raise EmbeddingModelMismatchError(
                f"Index was built with embedding model '{stored}' but '{embedder.name}' is configured. "
                "Use the same EMBEDDING_MODEL, or rebuild: python -m scripts.ingest --rebuild"
            )

    # -- Indexer protocol ---------------------------------------------------
    def replace_document(self, doc: IngestedDocument, chunks: list[Chunk]) -> None:
        texts = [embedding_text(c.text, c.section_heading) for c in chunks]
        vectors = self.embedder.embed_documents(texts)
        legal = annotate_document([{"chunk_id": c.chunk_id, "chunk_index": c.chunk_index, "text": c.text,
                                    "page_start": c.page_start, "section_heading": c.section_heading}
                                   for c in chunks])
        self.delete_document(doc.document_id)
        for i in range(0, len(chunks), ADD_BATCH):
            batch = chunks[i:i + ADD_BATCH]
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                embeddings=vectors[i:i + ADD_BATCH],
                metadatas=[chunk_metadata(doc, c) | legal.get(c.chunk_id, {}) for c in batch],
            )
        log.info("Indexed %s: %d chunks", doc.document_id, len(chunks))

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    # -- Queries ------------------------------------------------------------
    def count(self) -> int:
        return self.collection.count()

    def query(self, text: str, n_results: int, where: dict | None = None) -> list[dict[str, Any]]:
        if self.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=[self.embedder.embed_query(text)],
            n_results=min(n_results, self.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            # Chroma cosine distance = 1 - cosine similarity.
            hits.append({"chunk_id": cid, "text": doc, "metadata": meta, "similarity": 1.0 - dist})
        return hits

    def all_chunks(self) -> list[dict[str, Any]]:
        out = []
        total = self.count()
        page = 1000
        for offset in range(0, total, page):
            res = self.collection.get(include=["documents", "metadatas"], limit=page, offset=offset)
            out.extend({"chunk_id": cid, "text": d, "metadata": m}
                       for cid, d, m in zip(res["ids"], res["documents"], res["metadatas"]))
        return out

    def get_chunks(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        res = self.collection.get(ids=ids, include=["documents", "metadatas"])
        return [{"chunk_id": cid, "text": d, "metadata": m}
                for cid, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def reset(self) -> None:
        name = self.collection.name
        self._client.delete_collection(name)
        self.collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine", "embedding_model": self.embedder.name},
            embedding_function=None,
        )
