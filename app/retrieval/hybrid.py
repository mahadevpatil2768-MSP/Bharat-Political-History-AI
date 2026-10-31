"""Hybrid retrieval: vector (Chroma) + keyword (BM25) fused with reciprocal-rank fusion.

RRF (Cormack, Clarke & Büttcher, SIGIR 2009): score(d) = Σ_r 1 / (k + rank_r(d)).
The fused score is only an ordering signal — it is not a probability or a
calibrated relevance estimate, and is never presented as one.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any

from app.retrieval.bm25 import BM25Index
from app.retrieval.filters import has_python_only_filters, matches, to_chroma_where
from app.retrieval.legal_structure import content_terms, has, inline_headings, query_articles
from app.retrieval.rerank import NoOpReranker, Reranker
from app.retrieval.vector_store import ChromaIndex
from app.schemas import RetrievedChunk, SearchFilters

log = logging.getLogger(__name__)


class QueryValidationError(ValueError):
    pass


def preprocess_query(q: str) -> str:
    q = unicodedata.normalize("NFC", q or "")
    q = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) < 3:
        raise QueryValidationError("Query is too short (minimum 3 characters).")
    if len(q) > 1000:
        raise QueryValidationError("Query is too long (maximum 1000 characters).")
    if not re.search(r"[\wऀ-෿]", q):
        raise QueryValidationError("Query contains no searchable words.")
    return q


def _text_key(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.sha1(norm.encode()).hexdigest()


class HybridRetriever:
    def __init__(self, index: ChromaIndex, *, candidate_k: int = 30, rrf_k: int = 60,
                 min_vector_similarity: float = 0.0, reranker: Reranker | None = None,
                 manifest_path: Path | None = None, article_max_continuation: int = 3,
                 article_max_related: int = 2):
        self.index = index
        self.article_max_continuation = article_max_continuation
        self.article_max_related = article_max_related
        self.manifest_path = manifest_path
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.min_vector_similarity = min_vector_similarity
        self.reranker = reranker or NoOpReranker()
        self._bm25: BM25Index | None = None
        self._bm25_key: Any = None

    def _bm25_index(self) -> BM25Index:
        # Rebuild when the collection changes (e.g. ingestion ran in another process).
        key = (self.index.count(), self._manifest_mtime())
        if self._bm25 is None or key != self._bm25_key:
            self._bm25 = BM25Index(self.index.all_chunks())
            self._bm25_key = key
            log.info("BM25 index built over %d chunks", len(self._bm25.chunks))
        return self._bm25

    def _manifest_mtime(self) -> float | None:
        if self.manifest_path and self.manifest_path.exists():
            return self.manifest_path.stat().st_mtime
        return None

    def invalidate(self) -> None:
        self._bm25 = None

    def retrieve(self, query: str, filters: SearchFilters | None = None, top_k: int = 8,
                 mode: str = "hybrid") -> list[RetrievedChunk]:
        q = preprocess_query(query)
        filters = filters or SearchFilters()
        articles = query_articles(q) if mode == "hybrid" else []
        if articles:
            structured = self._article_retrieve(q, articles, filters, top_k)
            if structured is not None:
                self._log(q, structured)
                return structured
        ranked = self._fused(q, filters, mode)
        # Table-of-contents chunks list provision titles and match many queries; rank them after content.
        ranked = [c for c in ranked if not c.metadata.get("is_contents")] + \
                 [c for c in ranked if c.metadata.get("is_contents")]
        results = self.reranker.rerank(q, ranked[: max(top_k * 3, top_k)])[:top_k]
        self._log(q, results)
        return results

    def _fused(self, q: str, filters: SearchFilters, mode: str) -> list[RetrievedChunk]:
        """Vector + BM25 candidates fused with reciprocal-rank fusion, duplicate texts removed."""
        n = self.candidate_k * (4 if has_python_only_filters(filters) else 1)

        vector_hits: list[dict] = []
        if mode in ("hybrid", "vector"):
            raw = self.index.query(q, n, where=to_chroma_where(filters))
            vector_hits = [h for h in raw if matches(h["metadata"], filters)
                           and h["similarity"] >= self.min_vector_similarity][: self.candidate_k]

        bm25_hits: list[dict] = []
        if mode in ("hybrid", "keyword"):
            bm25_hits = self._bm25_index().search(q, self.candidate_k,
                                                  predicate=lambda m: matches(m, filters))

        fused: dict[str, RetrievedChunk] = {}
        for rank, h in enumerate(vector_hits, start=1):
            rc = fused.setdefault(h["chunk_id"], RetrievedChunk(
                chunk_id=h["chunk_id"], document_id=h["metadata"]["document_id"],
                text=h["text"], metadata=h["metadata"]))
            rc.vector_rank, rc.vector_similarity = rank, round(h["similarity"], 4)
            rc.fused_score += 1.0 / (self.rrf_k + rank)
        for rank, h in enumerate(bm25_hits, start=1):
            rc = fused.setdefault(h["chunk_id"], RetrievedChunk(
                chunk_id=h["chunk_id"], document_id=h["metadata"]["document_id"],
                text=h["text"], metadata=h["metadata"]))
            rc.bm25_rank, rc.bm25_score = rank, round(h["score"], 4)
            rc.fused_score += 1.0 / (self.rrf_k + rank)

        ranked = sorted(fused.values(), key=lambda c: c.fused_score, reverse=True)

        # Remove duplicate text (e.g. the same passage in two editions of a document).
        seen: set[str] = set()
        deduped: list[RetrievedChunk] = []
        for c in ranked:
            key = _text_key(c.text)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(c)
        return deduped

    def _article_retrieve(self, q: str, articles: list[str], filters: SearchFilters,
                          top_k: int) -> list[RetrievedChunk] | None:
        """Structure-aware retrieval for queries naming an article ("What is Article 14?").

        Tiers: (1) chunks containing the article's own heading (inside a PART, not a Schedule or the
        contents pages); (2) its continuation chunks; (3) up to ``article_max_related`` provisions that
        substantively qualify it ("notwithstanding …", "inconsistent with … article 14"); (4) only when
        the question has topical words beyond the article reference, further hybrid results. Chunks that
        merely mention the article, and contents pages, are excluded. Returns None (fall back to plain
        hybrid retrieval) when the article's heading is not in the index.
        """
        pool = [c for c in self._bm25_index().chunks if matches(c["metadata"], filters)]
        order = lambda c: (c["metadata"]["document_id"], c["metadata"].get("chunk_index", 0))  # noqa: E731
        heading = sorted((c for c in pool if not c["metadata"].get("is_contents")
                          and any(has(c["metadata"], "article_headings", a) for a in articles)), key=order)
        if not heading:
            return None
        continuation = []
        for a in articles:
            cont = sorted((c for c in pool if has(c["metadata"], "article_spans", a)
                           and not has(c["metadata"], "article_headings", a)), key=order)
            continuation.extend(cont[: self.article_max_continuation])

        titles = []
        for c in heading:
            for _, num, title in inline_headings(c["text"]):
                if num in articles:
                    titles.append(f"Article {num} {title}")
        expanded = " ".join([q] + titles)[:1000]
        fused = self._fused(expanded, filters, "hybrid")
        fused_by_id = {c.chunk_id: c for c in fused}
        fused_pos = {c.chunk_id: i for i, c in enumerate(fused)}

        def as_result(c: dict, reason: str) -> RetrievedChunk:
            rc = fused_by_id.get(c["chunk_id"]) or RetrievedChunk(
                chunk_id=c["chunk_id"], document_id=c["metadata"]["document_id"], text=c["text"],
                metadata=c["metadata"])
            rc.match_reason = reason
            return rc

        def mention_only(meta: dict) -> bool:
            return any(has(meta, "article_refs", a) and not has(meta, "article_refs_substantive", a)
                       and not has(meta, "article_spans", a) for a in articles)

        chosen: list[RetrievedChunk] = [as_result(c, "article_heading") for c in heading]
        chosen += [as_result(c, "article_text") for c in continuation]
        taken = {c.chunk_id for c in chosen}
        related = [c for c in pool if c["chunk_id"] not in taken and not c["metadata"].get("is_contents")
                   and any(has(c["metadata"], "article_refs_substantive", a) for a in articles)]
        related.sort(key=lambda c: (fused_pos.get(c["chunk_id"], len(fused_pos)), order(c)))
        chosen += [as_result(c, "related_provision") for c in related[: self.article_max_related]]

        if content_terms(q):  # e.g. "How does Article 21 apply to prisoners?" — add topical context
            taken = {c.chunk_id for c in chosen}
            for rc in fused:
                if len(chosen) >= top_k:
                    break
                if rc.chunk_id in taken or rc.metadata.get("is_contents") or mention_only(rc.metadata):
                    continue
                rc.match_reason = "hybrid"
                chosen.append(rc)

        seen_text: set[str] = set()
        results: list[RetrievedChunk] = []
        for rc in chosen:
            key = _text_key(rc.text)
            if rc.chunk_id in {r.chunk_id for r in results} or key in seen_text:
                continue
            seen_text.add(key)
            results.append(rc)
        return results[:top_k]

    def _log(self, q: str, results: list[RetrievedChunk]) -> None:
        for c in results:
            log.info("retrieval q=%r chunk=%s reason=%s fused=%.5f vec_rank=%s vec_sim=%s bm25_rank=%s bm25=%s "
                     "rerank=%s", q[:80], c.chunk_id, c.match_reason, c.fused_score, c.vector_rank,
                     c.vector_similarity, c.bm25_rank, c.bm25_score, c.rerank_score)
