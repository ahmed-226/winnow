"""Unit tests for the fake embedding provider."""

from __future__ import annotations

import math

from app.services.embeddings import FakeEmbeddingProvider


async def test_vectors_are_deterministic() -> None:
    provider = FakeEmbeddingProvider(1536)
    first = await provider.embed_query("same text")
    second = await provider.embed_query("same text")
    assert first == second


async def test_different_texts_differ() -> None:
    provider = FakeEmbeddingProvider(1536)
    a = await provider.embed_query("alpha")
    b = await provider.embed_query("beta")
    assert a != b


async def test_vectors_are_l2_normalized_and_finite() -> None:
    provider = FakeEmbeddingProvider(1536)
    vector = await provider.embed_query("normalized")
    assert len(vector) == 1536
    assert all(math.isfinite(value) for value in vector)
    assert math.isclose(sum(value * value for value in vector), 1.0, rel_tol=1e-9)


async def test_embed_documents_matches_embed_query() -> None:
    provider = FakeEmbeddingProvider(1536)
    documents = await provider.embed_documents(["hello", "world"])
    assert len(documents) == 2
    assert documents[0] == await provider.embed_query("hello")
