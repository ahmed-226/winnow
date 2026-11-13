"""Embedding providers: protocol, offline fake, and the Gemini implementation.

The provider is a small protocol because tests must run offline and because
the model will change eventually. The fake provider derives deterministic,
L2-normalized vectors from SHA-256 of the text. The Gemini provider calls
`gemini-embedding-001` with fixed task types and `output_dimensionality=1536`,
validates every vector (dimension + finiteness), L2-normalizes, retries
transient failures with backoff/jitter and `Retry-After`, and rate-limits
itself below the configured requests-per-minute quota.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Protocol, cast, runtime_checkable

from google.genai import Client as GenaiClient
from google.genai.errors import APIError as GenaiAPIError
from google.genai.types import EmbedContentConfig as GenaiEmbedContentConfig
from google.genai.types import HttpOptions as GenaiHttpOptions
from google.genai.types import HttpRetryOptions as GenaiHttpRetryOptions

from app.config import Settings

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_BACKOFF_INITIAL_SECONDS = 1.0
_BACKOFF_CAP_SECONDS = 60.0
_JITTER_FRACTION = 0.2
_QUERY_CACHE_PREFIX = "rag:embq"


class EmbeddingError(Exception):
    """Embedding failure with a worker error-code classification."""

    code = "embedding_failed"
    retryable = True

    def __init__(self, message: str, *, retryable: bool | None = None) -> None:
        super().__init__(message)
        if retryable is not None:
            self.retryable = retryable


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


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """Delay before retry `attempt` (1-based): backoff with jitter, or `Retry-After`.

    Schedule per plan §9.4: 1s, 2s, 4s, 8s, ... capped at 60s. A server-provided
    `Retry-After` (whole seconds) wins and is capped at the same ceiling.
    """
    if retry_after is not None:
        try:
            requested = float(retry_after)
        except ValueError:
            pass
        else:
            return float(max(0.0, min(requested, _BACKOFF_CAP_SECONDS)))
    delay = float(min(_BACKOFF_INITIAL_SECONDS * 2 ** (attempt - 1), _BACKOFF_CAP_SECONDS))
    return delay * random.uniform(1.0 - _JITTER_FRACTION, 1.0 + _JITTER_FRACTION)


class RateLimiter:
    """Client-side requests-per-minute limiter (in-process sliding window).

    §11.3: keeps a single worker from exhausting the free-tier minute quota.
    The clock is injectable so tests can exercise the window without sleeping.
    """

    _WINDOW_SECONDS = 60.0

    def __init__(
        self,
        requests_per_minute: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._limit = requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._sent_at: deque[float] = deque()

    async def acquire(self) -> None:
        """Block (cooperatively) until a request slot is available."""
        while True:
            now = self._clock()
            while self._sent_at and now - self._sent_at[0] >= self._WINDOW_SECONDS:
                self._sent_at.popleft()
            if len(self._sent_at) < self._limit:
                self._sent_at.append(now)
                return
            wait = self._WINDOW_SECONDS - (now - self._sent_at[0]) + 0.01
            await self._sleep(wait)


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


def _l2_normalize(vector: list[float]) -> list[float]:
    """Normalize in place-free; cosine is scale-invariant but we store unit vectors."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _finalize(vector: list[float], dimension: int) -> list[float]:
    """Validate a raw provider vector (dimension + finite), then L2-normalize.

    `gemini-embedding-001` only auto-normalizes the 3072-dim output, so 1536-dim
    vectors must be normalized manually; the validation never pads or truncates.
    """
    _validate_vector(vector, dimension)
    return _l2_normalize(vector)


def _is_transient(exc: BaseException) -> bool:
    """429/5xx API errors and transport/network failures may be retried.

    Network errors surface as transport-library exceptions (httpx, httpx2, or
    aiohttp depending on the backend `google-genai` selected), so anything that
    is not an APIError is treated as transient and goes through the same retry
    path; already-classified EmbeddingError exceptions never reach this check.
    """
    if isinstance(exc, GenaiAPIError):
        return exc.code in RETRYABLE_STATUS_CODES
    return True


def _retry_after_header(exc: BaseException) -> str | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    return str(value) if value is not None else None


@runtime_checkable
class QueryCache(Protocol):
    """Minimal async Redis surface used to cache query embeddings."""

    async def get(self, name: str) -> str | None: ...

    async def set(self, name: str, value: str, ex: int) -> None: ...


class GeminiEmbeddingProvider:
    """`gemini-embedding-001` through `google-genai`, fixed contract (§11.2).

    - one API request per `embed_documents` call (callers batch via
      `EMBEDDING_BATCH_SIZE`) and one per `embed_query`
    - always `task_type=RETRIEVAL_DOCUMENT|RETRIEVAL_QUERY` and
      `output_dimensionality=1536`; the API default of 3072 would violate the
      schema, so it is never relied on
    - every returned vector is validated (dimension + finiteness) and then
      L2-normalized; a mismatch is a hard failure, never padded or truncated
    - transient failures (429/5xx/network) are retried with exponential
      backoff + jitter, capped at 60 s and honoring `Retry-After`; other API
      errors (bad key, bad request) fail fast as non-retryable
    - a client-side per-minute limiter keeps requests under the configured
      free-tier quota
    - optional Redis cache for query embeddings only; document embeddings are
      never cached (each document is processed once)
    """

    def __init__(
        self,
        settings: Settings,
        *,
        embed_fn: Callable[[list[str], str], Awaitable[list[list[float]]]] | None = None,
        cache: QueryCache | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.model = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self._settings = settings
        # Test seam: an async (texts, task_type) -> vectors callable.
        self._embed_fn = embed_fn
        self._cache = cache
        self._sleep = sleep
        self._limiter = RateLimiter(
            settings.embedding_requests_per_minute, clock=clock, sleep=sleep
        )
        self._client: GenaiClient | None = None

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raw = await self._request(texts, self._settings.embedding_document_task)
        return [_finalize(vector, self.dimension) for vector in raw]

    async def embed_query(self, text: str) -> list[float]:
        task = self._settings.embedding_query_task
        key = (
            f"{_QUERY_CACHE_PREFIX}:{self.model}:{self.dimension}:"
            f"{hashlib.sha256(f'{task}|{text}'.encode()).hexdigest()}"
        )
        if self._cache is not None:
            cached = await self._cache.get(key)
            if cached is not None:
                return cast("list[float]", json.loads(cached))
        raw = (await self._request([text], task))[0]
        vector = _finalize(raw, self.dimension)
        if self._cache is not None:
            await self._cache.set(
                key,
                json.dumps(vector),
                ex=self._settings.embedding_query_cache_ttl_seconds,
            )
        return vector

    # ------------------------------------------------------------------ internals

    async def _request(self, contents: list[str], task_type: str) -> list[list[float]]:
        await self._limiter.acquire()
        max_retries = self._settings.embedding_max_retries
        for attempt in range(1, max_retries + 1):
            try:
                return await self._call(contents, task_type)
            except EmbeddingError:
                raise  # already classified (validation, contract violation)
            except Exception as exc:
                if not _is_transient(exc):
                    raise EmbeddingError(
                        f"Embedding API rejected the request ({type(exc).__name__}); not retrying.",
                        retryable=False,
                    ) from exc
                if attempt >= max_retries:
                    raise EmbeddingRequestError(
                        f"Embedding API failed after {attempt} attempts ({type(exc).__name__})."
                    ) from exc
                delay = _retry_delay(attempt, _retry_after_header(exc))
                await self._sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _call(self, contents: list[str], task_type: str) -> list[list[float]]:
        if self._embed_fn is not None:
            return await self._embed_fn(contents, task_type)
        client = self._client
        if client is None:
            api_key = self._settings.gemini_api_key
            if not api_key:
                raise EmbeddingError("GEMINI_API_KEY is not configured.", retryable=False)
            client = self._client = GenaiClient(
                api_key=api_key,
                http_options=GenaiHttpOptions(
                    timeout=self._settings.embedding_timeout_seconds * 1_000,
                    # Our own retry loop owns backoff; the SDK must not retry too.
                    retry_options=GenaiHttpRetryOptions(attempts=1),
                ),
            )
        response = await client.aio.models.embed_content(
            model=self.model,
            contents=contents,
            config=GenaiEmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimension,
            ),
        )
        embeddings = response.embeddings
        if not embeddings:
            raise EmbeddingRequestError("Embedding API returned no embeddings.")
        vectors: list[list[float]] = []
        for item in embeddings:
            if item.values is None:
                raise EmbeddingRequestError("Embedding API returned an empty vector for one input.")
            vectors.append(list(item.values))
        if len(vectors) != len(contents):
            raise EmbeddingError(
                f"Embedding API returned {len(vectors)} vectors for {len(contents)} inputs.",
                retryable=False,
            )
        return vectors


def build_embedding_provider(
    settings: Settings, *, cache: QueryCache | None = None
) -> EmbeddingProvider:
    """Provider factory; keep `EMBEDDING_PROVIDER=fake` in tests and CI.

    `cache` is the optional Redis query-embedding cache; the worker passes its
    shared Redis client.
    """
    if settings.embedding_provider == "gemini":
        return GeminiEmbeddingProvider(settings, cache=cache)
    if settings.embedding_provider == "fake":
        return FakeEmbeddingProvider(settings.embedding_dimension)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {settings.embedding_provider!r}")
