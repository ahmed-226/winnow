from __future__ import annotations

import fakeredis.aioredis
import pytest

from app.errors import RateLimitedError
from app.services.rate_limit import enforce_rate_limit


async def test_rate_limit_allows_until_limit_then_rejects() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        for _ in range(2):
            await enforce_rate_limit(redis, scope="test", key="rl:test", limit=2, window_seconds=60)

        with pytest.raises(RateLimitedError) as exc_info:
            await enforce_rate_limit(redis, scope="test", key="rl:test", limit=2, window_seconds=60)
        assert int(exc_info.value.headers["Retry-After"]) >= 1  # type: ignore[index]
    finally:
        await redis.aclose()


async def test_rate_limit_is_disabled_when_limit_is_zero() -> None:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    try:
        await enforce_rate_limit(redis, scope="test", key="rl:off", limit=0, window_seconds=60)
        assert await redis.exists("rl:off") == 0
    finally:
        await redis.aclose()
