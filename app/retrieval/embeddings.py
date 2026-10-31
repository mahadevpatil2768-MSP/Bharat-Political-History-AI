"""Embedding models behind a small interface.

The same embedder (same model name) must be used for indexing and querying;
the vector store enforces this by recording the model name in the collection.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

log = logging.getLogger(__name__)


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Sentence Transformers wrapper. Handles E5-style 'query:'/'passage:' prefixes."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16):
        from sentence_transformers import SentenceTransformer

        log.info("Loading embedding model %s on %s", model_name, device)
        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, device=device)
        self._e5 = "e5" in model_name.lower()

    def _prefix(self, texts: list[str], kind: str) -> list[str]:
        return [f"{kind}: {t}" for t in texts] if self._e5 else texts

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(self._prefix(texts, "passage"), batch_size=self.batch_size,
                                  normalize_embeddings=True, show_progress_bar=len(texts) > 64)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(self._prefix([text], "query"), normalize_embeddings=True)[0].tolist()


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder.

    For tests and offline smoke checks only — it captures lexical overlap, not meaning.
    """

    def __init__(self, dim: int = 256, name: str = "hashing-test-embedder"):
        self.dim = dim
        self.name = name

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in re.findall(r"[\wऀ-෿]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) % 2 else -1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


_cache: dict[tuple[str, str], SentenceTransformerEmbedder] = {}


def get_embedder(model_name: str, device: str = "cpu", batch_size: int = 16) -> Embedder:
    if model_name == "hashing-test-embedder":
        return HashingEmbedder()
    key = (model_name, device)
    if key not in _cache:
        _cache[key] = SentenceTransformerEmbedder(model_name, device, batch_size)
    return _cache[key]
