"""Embedding provider protocol and the offline fake implementation.

The provider is a small protocol because tests must run offline and because
the model will change eventually. The Gemini implementation lands in Phase 4;
the fake provider derives deterministic, L2-normalized vectors from SHA-256
of the text so the pipeline can be exercised without any external dependency.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

from app.config import Settings


class EmbeddingError(Exception):
    """Embedding failure with a worker error-code classification."""

    code = "embedding_failed"
    retryable = True


class EmbeddingRequestError(EmbeddingError):
    """Transient provider failure (429/5xx/network); safe to retry."""


class EmbeddingDimensionError(EmbeddingError):
    """Provider returned a wrong-sized or non-finite vector; never retry."""

    code = "embedding_dimension_mismatch"
    retryable = False


@runtime_checkable
class EmbeddingProvider(Protocol):
    model: str
    dimension: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def _validate_vector(vector: list[float], dimension: int) -> list[float]:
    if len(vector) != dimension:
        raise EmbeddingDimensionError(
            f"Embedding provider returned {len(vector)} dimensions, expected {dimension}."
        )
    if not all(math.isfinite(value) for value in vector):
        raise EmbeddingDimensionError("Embedding provider returned a non-finite value.")
    return vector


class FakeEmbeddingProvider:
    """Deterministic vectors derived from SHA-256, L2-normalized.

    Different texts produce different vectors; the same text always produces
    the same vector. Used for unit/integration tests and the Phase 3 pipeline.
    """

    model = "fake-embedding-001"

    def __init__(self, dimension: int = 1536) -> None:
        self.dimension = dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_validate_vector(self._embed(text), self.dimension) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return _validate_vector(self._embed(text), self.dimension)

    def _embed(self, text: str) -> list[float]:
        seed = hashlib.sha256(text.encode("utf-8")).digest()
        raw: list[float] = []
        counter = 0
        while len(raw) < self.dimension:
            digest = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
            for start in range(0, len(digest) - 7, 8):
                value = int.from_bytes(digest[start : start + 8], "big")
                raw.append(value / 2**63 - 1.0)  # uniform in [-1, 1)
                if len(raw) >= self.dimension:
                    break
            counter += 1
        norm = math.sqrt(sum(value * value for value in raw))
        if norm == 0.0:  # pragma: no cover - requires an all-zero digest
            raw[0] = 1.0
            norm = 1.0
        return [value / norm for value in raw]


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Provider factory; keep `EMBEDDING_PROVIDER=fake` in tests and CI."""
    if settings.embedding_provider == "fake":
        return FakeEmbeddingProvider(settings.embedding_dimension)
    raise NotImplementedError("The Gemini provider lands in Phase 4.")
