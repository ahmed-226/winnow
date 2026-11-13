from __future__ import annotations

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.errors import RateLimitedError
from app.metrics import RATE_LIMIT_REJECTIONS

log = structlog.get_logger(__name__)


async def enforce_rate_limit(
    redis: Redis,
    *,
    scope: str,
    key: str,
    limit: int,
    window_seconds: int,
) -> None:
    """Increment a fixed-window counter, raising 429 when the limit is exceeded.

    A Redis outage fails open: authentication still requires a valid password or
    token, and a cache outage must not lock every user out of the service.
    """
    if limit <= 0:
        return

    try:
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, window_seconds)
        ttl = await redis.ttl(key) if count > limit else window_seconds
    except RedisError:
        log.warning("rate_limit.unavailable", scope=scope)
        return

    if count > limit:
        retry_after = ttl if ttl and ttl > 0 else window_seconds
        RATE_LIMIT_REJECTIONS.labels(scope=scope).inc()
        raise RateLimitedError(retry_after=retry_after)
