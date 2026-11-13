"""Ingestion worker: Redis Streams consumer, sweeper, and DLQ entrypoint.

Runs as `python -m app.workers.ingest`; the API and worker share one image and
differ only by this command. The DB job row is authoritative: messages are
hints, the sweeper recovers lost ones, and the dead-letter stream records jobs
that exhausted their attempts.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import socket
import time
import uuid
from datetime import UTC, datetime
from typing import cast

import structlog
from redis.asyncio import Redis
from redis.exceptions import ResponseError
from redis.exceptions import TimeoutError as RedisTimeoutError

from app.config import Settings
from app.db import create_engine, create_sessionmaker
from app.logging import configure_logging
from app.services.embeddings import build_embedding_provider
from app.services.ingestion import (
    Outcome,
    WorkerContext,
    process_job,
    reset_stale_running_job,
    sweeper_due_jobs,
)
from app.services.queue import enqueue_ingest
from app.services.storage import StorageService

log = structlog.get_logger(__name__)

_BLOCK_MS = 5_000
_RECLAIM_COUNT = 10


async def ensure_group(redis: Redis, *, stream: str, group: str) -> None:
    try:
        await redis.xgroup_create(stream, group, id="0", mkstream=True)
    except ResponseError as error:
        if "BUSYGROUP" not in str(error):
            raise


def consumer_name() -> str:
    """Unique per process so stale consumers can be reclaimed via XAUTOCLAIM."""
    return f"{socket.gethostname()}-{os.getpid()}-{secrets.token_hex(4)}"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


async def handle_message(
    ctx: WorkerContext, redis: Redis, message_id: str, fields: dict[str, str]
) -> None:
    """Process one stream message; always acknowledges at the end."""
    stream = ctx.settings.ingest_stream
    group = ctx.settings.ingest_group
    try:
        job_id = uuid.UUID(fields["job_id"])
        tenant_id = uuid.UUID(fields["tenant_id"])
    except (KeyError, ValueError):
        log.warning("ingest.malformed_message", message_id=message_id)
        await redis.xack(stream, group, message_id)
        return

    outcome = await process_job(ctx, job_id=job_id, tenant_id=tenant_id)
    if outcome is Outcome.RETRIED:
        # Job reset to queued with attempts remaining; re-enqueue it.
        try:
            await enqueue_ingest(
                redis,
                stream=stream,
                document_id=uuid.UUID(fields.get("document_id", "")),
                tenant_id=tenant_id,
                job_id=job_id,
            )
        except Exception as exc:
            log.warning("ingest.reenqueue_failed", error=type(exc).__name__)
    elif outcome is Outcome.TERMINAL:
        try:
            dead_fields: dict[str, str] = {
                **fields,
                "failed_at": _now_iso(),
                "job_id": str(job_id),
            }
            await redis.xadd(ctx.settings.ingest_dead_stream, dead_fields)  # type: ignore[arg-type]
        except Exception as exc:
            log.warning("ingest.dead_letter_failed", error=type(exc).__name__)
    await redis.xack(stream, group, message_id)


async def run_sweep(
    ctx: WorkerContext,
    redis: Redis,
    *,
    queued_stale_seconds: int = 120,
    running_stale_seconds: int = 900,
) -> int:
    """Recover lost and crashed jobs: re-enqueue each due job."""
    enqueued = 0
    for job in await sweeper_due_jobs(
        ctx, queued_stale_seconds=queued_stale_seconds, running_stale_seconds=running_stale_seconds
    ):
        if job.status == "running":
            # The worker died mid-job; move it back to queued so it can be claimed.
            await reset_stale_running_job(ctx, job_id=job.job_id, tenant_id=job.tenant_id)
        try:
            await enqueue_ingest(
                redis,
                stream=ctx.settings.ingest_stream,
                document_id=job.document_id,
                tenant_id=job.tenant_id,
                job_id=job.job_id,
            )
            enqueued += 1
        except Exception as exc:
            log.warning("ingest.sweep_enqueue_failed", error=type(exc).__name__)
    if enqueued:
        log.info("ingest.sweeper_reenqueued", count=enqueued)
    return enqueued


async def _reclaim_idle(
    ctx: WorkerContext, redis: Redis, consumer: str
) -> list[tuple[str, dict[str, str]]]:
    """XAUTOCLAIM messages abandoned by dead workers; returns (id, fields)."""
    result = await redis.xautoclaim(
        ctx.settings.ingest_stream,
        ctx.settings.ingest_group,
        consumer,
        min_idle_time=ctx.settings.ingest_claim_idle_ms,
        start_id="0-0",
        count=_RECLAIM_COUNT,
    )
    # redis-py returns [next_cursor, messages, deleted_ids] for XAUTOCLAIM.
    messages = result[1] if result and isinstance(result[0], str) else result
    entries = cast("list[tuple[str, dict[str, str]]]", messages)
    return [(message_id, fields) for message_id, fields in entries]


async def _worker_loop(ctx: WorkerContext, redis: Redis) -> None:
    settings = ctx.settings
    consumer = consumer_name()
    log.info("worker.started", consumer=consumer)
    await ensure_group(redis, stream=settings.ingest_stream, group=settings.ingest_group)
    last_sweep = 0.0
    while True:
        now = time.monotonic()
        if now - last_sweep >= settings.ingest_sweeper_interval_seconds:
            last_sweep = now
            try:
                await run_sweep(ctx, redis)
            except Exception as exc:
                log.warning("ingest.sweep_failed", error=type(exc).__name__)
        try:
            reclaimed = await _reclaim_idle(ctx, redis, consumer)
            for message_id, fields in reclaimed:
                await handle_message(ctx, redis, message_id, fields)
        except Exception as exc:
            log.warning("ingest.reclaim_failed", error=type(exc).__name__)

        try:
            batches = cast(
                "list[tuple[str, list[tuple[str, dict[str, str]]]]]",
                await redis.xreadgroup(
                    settings.ingest_group,
                    consumer,
                    {settings.ingest_stream: ">"},
                    count=1,
                    block=_BLOCK_MS,
                ),
            )
        except (TimeoutError, RedisTimeoutError):
            # Blocking read expired with no messages; normal idle behavior.
            continue
        except Exception as exc:
            log.warning("ingest.xreadgroup_failed", error=type(exc).__name__)
            await asyncio.sleep(1)
            continue
        for _stream_name, batch_entries in batches:
            for message_id, message_fields in batch_entries:
                await handle_message(ctx, redis, message_id, message_fields)


async def run() -> None:
    settings = Settings()  # type: ignore[call-arg]
    engine = create_engine(settings, application_name="rag-worker", statement_timeout_ms=120_000)
    ctx = WorkerContext(
        settings=settings,
        sessionmaker=create_sessionmaker(engine),
        storage=StorageService(settings),
        embeddings=build_embedding_provider(settings),
    )
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await _worker_loop(ctx, redis)
    finally:
        await redis.aclose()
        await engine.dispose()


def main() -> None:
    configure_logging(Settings())  # type: ignore[call-arg]
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("worker.stopped")


if __name__ == "__main__":
    main()
