"""Unit tests for the Gemini embedding provider (offline, via the embed_fn seam)."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable

import pytest
from google.genai.errors import APIError

from app.config import Settings
from app.services.embeddings import (
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingRequestError,
    GeminiEmbeddingProvider,
    RateLimiter,
    _retry_delay,
    build_embedding_provider,
)

DIM = 1536


def make_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "gemini_api_key": "test-key",
        "embedding_provider": "gemini",
        "embedding_max_retries": 3,
        "embedding_batch_size": 16,
        "embedding_requests_per_minute": 10_000,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


class RecordingSleeper:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def unit_vector(first: float, second: float) -> list[float]:
    """A valid 1536-dim vector; (3, 4) -> normalized to (0.6, 0.8, 0...)."""
    norm = math.sqrt(first * first + second * second)
    return [first / norm, second / norm] + [0.0] * (DIM - 2)


def stub_embed(
    *results: object,
) -> Callable[[list[str], str], Awaitable[list[list[float]]]]:
    """embed_fn that raises or returns each queued result in order."""
    queue = list(results)

    async def embed(texts: list[str], task_type: str) -> list[list[float]]:
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item  # type: ignore[return-value]

    return embed


class FakeAPIError(APIError):
    """Real APIError subclass so the provider's classifier sees `code`."""

    def __init__(self, code: int) -> None:
        super().__init__(code, {"error": {"message": "test"}})


def make_provider(
    settings: Settings, embed_fn: Callable[[list[str], str], Awaitable[list[list[float]]]]
) -> tuple[GeminiEmbeddingProvider, RecordingSleeper]:
    sleeper = RecordingSleeper()
    provider = GeminiEmbeddingProvider(settings, embed_fn=embed_fn, sleep=sleeper)
    return provider, sleeper


async def test_success_returns_validated_normalized_vectors() -> None:
    provider, _ = make_provider(make_settings(), stub_embed([unit_vector(3.0, 4.0), [0.0] * DIM]))
    vectors = await provider.embed_documents(["a", "b"])
    assert len(vectors) == 2
    for vector in vectors:
        assert len(vector) == DIM
        assert all(math.isfinite(value) for value in vector)
    assert vectors[0][0] == pytest.approx(0.6)
    assert vectors[0][1] == pytest.approx(0.8)
    assert math.isclose(sum(v * v for v in vectors[0]), 1.0, rel_tol=1e-9)
    # A zero vector passes validation unchanged (nothing to normalize).
    assert vectors[1] == [0.0] * DIM


async def test_task_types_are_passed_through() -> None:
    seen: list[str] = []

    async def embed(texts: list[str], task_type: str) -> list[list[float]]:
        seen.append(task_type)
        return [unit_vector(1.0, 0.0) for _ in texts]

    provider, _ = make_provider(make_settings(), embed)
    await provider.embed_documents(["x"])
    await provider.embed_query("q")
    assert seen == ["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]


async def test_query_cache_roundtrip() -> None:
    calls: list[str] = []

    class FakeCache:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}
            self.ttls: list[int] = []

        async def get(self, key: str) -> str | None:
            return self.store.get(key)

        async def set(self, key: str, value: str, ex: int) -> None:
            self.store[key] = value
            self.ttls.append(ex)

    async def embed(texts: list[str], task_type: str) -> list[list[float]]:
        calls.append(task_type)
        return [unit_vector(1.0, 0.0)]

    cache = FakeCache()
    settings = make_settings(embedding_query_cache_ttl_seconds=77)
    provider, _ = make_provider(settings, embed)
    provider._cache = cache  # type: ignore[attr-defined]

    first = await provider.embed_query("hello")
    second = await provider.embed_query("hello")
    assert first == second == unit_vector(1.0, 0.0)
    assert len(calls) == 1  # second call served from the cache
    assert cache.ttls == [77]


async def test_429_then_success_retries() -> None:
    provider, sleeper = make_provider(
        make_settings(),
        stub_embed(FakeAPIError(429), FakeAPIError(500), [unit_vector(3.0, 4.0)]),
    )
    vectors = await provider.embed_documents(["ok"])
    assert vectors == [unit_vector(3.0, 4.0)]
    assert len(sleeper.calls) == 2
    assert sleeper.calls[0] == pytest.approx(1.0, rel=0.25)
    assert sleeper.calls[1] == pytest.approx(2.0, rel=0.25)


async def test_exhausted_429_raises_retryable_request_error() -> None:
    provider, sleeper = make_provider(
        make_settings(embedding_max_retries=2),
        stub_embed(FakeAPIError(429), FakeAPIError(429)),
    )
    with pytest.raises(EmbeddingRequestError) as info:
        await provider.embed_documents(["ok"])
    assert info.value.retryable is True
    assert len(sleeper.calls) == 1


async def test_permanent_api_error_fails_fast() -> None:
    provider, sleeper = make_provider(make_settings(), stub_embed(FakeAPIError(403)))
    with pytest.raises(EmbeddingError) as info:
        await provider.embed_documents(["ok"])
    assert info.value.retryable is False
    assert sleeper.calls == []


async def test_wrong_dimension_is_never_retried() -> None:
    provider, sleeper = make_provider(make_settings(), stub_embed([[0.0] * (DIM + 1)]))
    with pytest.raises(EmbeddingDimensionError):
        await provider.embed_documents(["ok"])
    assert sleeper.calls == []


async def test_non_finite_value_is_rejected() -> None:
    bad = [0.0] * DIM
    bad[7] = float("inf")
    provider, _ = make_provider(make_settings(), stub_embed([bad]))
    with pytest.raises(EmbeddingDimensionError):
        await provider.embed_documents(["ok"])


def test_retry_delay_schedule_and_cap() -> None:
    assert _retry_delay(1, None) == pytest.approx(1.0, rel=0.25)
    assert _retry_delay(2, None) == pytest.approx(2.0, rel=0.25)
    assert _retry_delay(3, None) == pytest.approx(4.0, rel=0.25)
    for attempt in range(1, 10):
        assert 0.0 <= _retry_delay(attempt, None) <= 60.0 * 1.2


def test_retry_delay_respects_retry_after() -> None:
    assert _retry_delay(1, "30") == 30.0
    assert _retry_delay(1, "120") == 60.0  # capped
    assert _retry_delay(1, "not-a-number") == pytest.approx(1.0, rel=0.25)


async def test_rate_limiter_blocks_over_quota() -> None:
    waits: list[float] = []
    now = [0.0]

    def clock() -> float:
        return now[0]

    async def sleep(seconds: float) -> None:
        waits.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2, clock=clock, sleep=sleep)
    for _ in range(4):
        await limiter.acquire()
    # Slots 1-2 are free; slot 3 waits out the window (60.01 s); slot 4 reuses
    # the slot freed by purging the expired first entry.
    assert waits == [pytest.approx(60.01)]


def test_factory_builds_gemini_provider() -> None:
    provider = build_embedding_provider(make_settings())
    assert isinstance(provider, GeminiEmbeddingProvider)
    assert provider.model == "gemini-embedding-001"
    assert provider.dimension == DIM
