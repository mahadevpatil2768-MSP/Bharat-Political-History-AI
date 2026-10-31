"""BM25 keyword retrieval over the chunks stored in Chroma."""

from __future__ import annotations

import re
from typing import Any, Callable

from rank_bm25 import BM25Okapi

from app.retrieval.vector_store import embedding_text

# Latin word chars plus the Indic Unicode blocks (Devanagari … Malayalam/Sinhala),
# so vowel signs (combining marks) stay inside their word.
_TOKEN_RE = re.compile(r"[\wऀ-෿]+")

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have he her his i in is it its of on or that the their "
    "there they this to was were which who whom will with what when where why how did does do "
    "been being into than then these those not no but if so such can could would should shall "
    "may might about after before during under over between".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        corpus = [tokenize(embedding_text(c["text"], c["metadata"].get("section_heading")))
                  for c in chunks]
        self._bm25 = BM25Okapi(corpus) if chunks else None

    def search(self, query: str, n: int,
               predicate: Callable[[dict[str, Any]], bool] | None = None) -> list[dict[str, Any]]:
        if not self._bm25:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order:
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            if predicate and not predicate(c["metadata"]):
                continue
            hits.append({**c, "score": float(scores[i])})
            if len(hits) >= n:
                break
        return hits
