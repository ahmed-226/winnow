"""Redis Streams producer for ingestion jobs.

The message is a hint, not the source of truth. The Phase 3 worker always
loads the document and job rows from PostgreSQL, so duplicate or lost
messages are harmless (the sweeper recovers lost ones).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from redis.asyncio import Redis

INGEST_MESSAGE_VERSION = "1"


async def enqueue_ingest(
    redis: Redis,
    *,
    stream: str,
    document_id: uuid.UUID,
    tenant_id: uuid.UUID,
    job_id: uuid.UUID,
    max_len: int = 10_000,
) -> None:
    """Add an ingest message to the stream. Raises on Redis failure."""
    await redis.xadd(
        stream,
        {
            "version": INGEST_MESSAGE_VERSION,
            "job_id": str(job_id or uuid.uuid4()),
            "document_id": str(document_id),
            "tenant_id": str(tenant_id),
            "enqueued_at": datetime.now(UTC).isoformat(),
        },
        maxlen=max_len,
        approximate=True,
    )
