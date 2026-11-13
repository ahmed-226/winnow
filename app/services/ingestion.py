"""Ingestion pipeline: job claim, processing, and chunk writes (plan §9).

The DB job row is authoritative; queue messages are hints. Claiming is a
conditional transition so duplicate delivery is harmless, chunks are written
only after every embedding succeeded in one transaction under an advisory
lock, and failures are classified into the §9.4 error taxonomy.
"""

from __future__ import annotations

import asyncio
import enum
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import structlog
from sqlalchemy import delete, insert, select, text, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings
from app.errors import StorageUnavailableError
from app.models import Document, DocumentChunk, DocumentStatus, IngestionJob, JobStatus
from app.services.chunking import Chunk, chunk_blocks, split_for_token_limit
from app.services.embeddings import (
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingProvider,
)
from app.services.parsing import (
    NoExtractableTextError,
    ParseFailure,
    parse,
    total_text_length,
)
from app.services.storage import StorageService

log = structlog.get_logger(__name__)

_INSERT_BATCH_ROWS = 500
_MIN_TOTAL_TEXT_CHARS = 50
_LAST_ERROR_MAX_CHARS = 500


class IngestionFailure(Exception):
    """A classified pipeline failure (§9.4); carries the error code."""

    def __init__(self, code: str, detail: str, *, retryable: bool) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail
        self.retryable = retryable


class Outcome(enum.Enum):
    """What happened to one queue message."""

    DONE = "done"  # chunks committed
    DOC_GONE_DONE = "doc_gone_done"  # document deleted meanwhile; job marked done
    SKIPPED = "skipped"  # job not in queued state (done/failed/running elsewhere)
    LOCK_HELD = "lock_held"  # another worker is writing this document's chunks
    RETRIED = "retried"  # attempts remain; re-enqueued
    TERMINAL = "terminal"  # attempts exhausted; dead-lettered


@dataclass(frozen=True, slots=True)
class WorkerContext:
    """Everything the worker needs to process jobs."""

    settings: Settings
    sessionmaker: async_sessionmaker[AsyncSession]
    storage: StorageService
    embeddings: EmbeddingProvider


@dataclass(frozen=True, slots=True)
class SweeperJob:
    job_id: uuid.UUID
    document_id: uuid.UUID
    tenant_id: uuid.UUID
    status: str
    attempts: int
    max_attempts: int


async def set_tenant(session: AsyncSession, tenant_id: uuid.UUID) -> None:
    """Transaction-local RLS context; never session-level `SET`."""
    await session.execute(
        text("SELECT set_config('app.tenant_id', :tid, true)"),
        {"tid": str(tenant_id)},
    )


# --------------------------------------------------------------------- claim


@dataclass(frozen=True, slots=True)
class _Claim:
    document_id: uuid.UUID
    storage_key: str
    source_type: str


async def claim_job(
    ctx: WorkerContext, *, job_id: uuid.UUID, tenant_id: uuid.UUID
) -> _Claim | Outcome:
    """Atomically move queued -> running and the document -> processing.

    Returns `_Claim` on success or an `Outcome` when the message must not
    be processed (not claimable, or the document vanished).
    """
    async with ctx.sessionmaker() as session:
        async with session.begin():
            await set_tenant(session, tenant_id)
            job = (
                await session.execute(
                    select(IngestionJob)
                    .where(
                        IngestionJob.id == job_id,
                        IngestionJob.tenant_id == tenant_id,
                        IngestionJob.status == JobStatus.QUEUED,
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if job is None:
                return Outcome.SKIPPED
            document = (
                await session.execute(
                    select(Document)
                    .where(
                        Document.id == job.document_id,
                        Document.tenant_id == tenant_id,
                        Document.deleted_at.is_(None),
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if document is None:
                # Deleted while queued: do not resurrect it through a stale message.
                job.status = JobStatus.DONE
                job.finished_at = datetime.now(UTC)
                return Outcome.DOC_GONE_DONE
            document.status = DocumentStatus.PROCESSING
            job.status = JobStatus.RUNNING
            job.attempts += 1
            job.started_at = datetime.now(UTC)
            return _Claim(
                document_id=document.id,
                storage_key=document.storage_key,
                source_type=document.source_type,
            )


# ----------------------------------------------------------------- processing


async def process_job(ctx: WorkerContext, *, job_id: uuid.UUID, tenant_id: uuid.UUID) -> Outcome:
    """Run one job end to end: claim, download, parse, chunk, embed, commit."""
    claim = await claim_job(ctx, job_id=job_id, tenant_id=tenant_id)
    if isinstance(claim, Outcome):
        return claim

    started = time.monotonic()
    try:
        prepared = await _prepare_chunks(ctx, claim)
        outcome = await _write_chunks(ctx, claim.document_id, tenant_id, prepared)
    except ParseFailure as failure:
        return await _handle_failure(ctx, job_id=job_id, tenant_id=tenant_id, failure=failure)
    except EmbeddingError as failure:
        return await _handle_failure(ctx, job_id=job_id, tenant_id=tenant_id, failure=failure)
    except StorageUnavailableError:
        return await _handle_failure(
            ctx,
            job_id=job_id,
            tenant_id=tenant_id,
            failure=IngestionFailure("download_failed", "Object storage error.", retryable=True),
        )
    except Exception as failure:
        log.warning("ingest.unexpected_error", error_type=type(failure).__name__)
        return await _handle_failure(
            ctx,
            job_id=job_id,
            tenant_id=tenant_id,
            failure=IngestionFailure(
                "parse_failed", f"Unexpected {type(failure).__name__}.", retryable=True
            ),
        )

    log.info(
        "document_ingested",
        document_id=str(claim.document_id),
        tenant_id=str(tenant_id),
        chunk_count=len(prepared),
        duration_ms=round((time.monotonic() - started) * 1000),
    )
    return outcome


async def _prepare_chunks(ctx: WorkerContext, claim: _Claim) -> list[tuple[Chunk, list[float]]]:
    settings = ctx.settings
    with tempfile.TemporaryDirectory(prefix="rag-ingest-") as directory:
        path = os.path.join(directory, "original")
        await ctx.storage.download_object(key=claim.storage_key, file_path=path)
        data = await asyncio.to_thread(_read_limited, path, settings.max_upload_bytes)

    blocks = await asyncio.to_thread(
        parse, data, source_type=claim.source_type, pdf_max_pages=settings.pdf_max_pages
    )
    if total_text_length(blocks) < _MIN_TOTAL_TEXT_CHARS:
        raise NoExtractableTextError(
            "The document contains no extractable text (OCR is not enabled)."
        )

    chunks = await asyncio.to_thread(
        chunk_blocks,
        blocks,
        target_chars=settings.chunk_target_chars,
        overlap_chars=settings.chunk_overlap_chars,
        max_chars=settings.chunk_max_chars,
        min_chars=settings.chunk_min_chars,
    )
    chunks = split_for_token_limit(chunks)
    return await _embed_chunks(ctx, chunks)


def _read_limited(path: str, max_bytes: int) -> bytes:
    size = os.path.getsize(path)
    if size > max_bytes:
        raise IngestionFailure(
            "object_too_large", "Stored object exceeds the size limit.", retryable=False
        )
    with open(path, "rb") as handle:
        return handle.read()


async def _embed_chunks(ctx: WorkerContext, chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]:
    """Embed every chunk in batches; a single failure aborts the whole job."""
    if not chunks:
        raise NoExtractableTextError("The document produced no chunks.")
    batch_size = ctx.settings.embedding_batch_size
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_vectors = await ctx.embeddings.embed_documents([chunk.content for chunk in batch])
        for vector in batch_vectors:
            if len(vector) != ctx.settings.embedding_dimension or not all(
                math.isfinite(value) for value in vector
            ):
                raise EmbeddingDimensionError(
                    "Embedding provider returned an invalid vector; refusing to store it."
                )
        vectors.extend(batch_vectors)
    return list(zip(chunks, vectors, strict=True))


async def _write_chunks(
    ctx: WorkerContext,
    document_id: uuid.UUID,
    tenant_id: uuid.UUID,
    prepared: list[tuple[Chunk, list[float]]],
) -> Outcome:
    """The all-or-nothing chunk write (worker algorithm steps 11-17)."""
    model = ctx.embeddings.model
    async with ctx.sessionmaker() as session:
        async with session.begin():
            await set_tenant(session, tenant_id)
            lock = (
                await session.execute(
                    text(
                        "SELECT pg_try_advisory_xact_lock("
                        "hashtextextended(CAST(:document_id AS uuid)::text, 0))"
                    ),
                    {"document_id": str(document_id)},
                )
            ).scalar_one()
            if not lock:
                # Another worker is processing this document; skip quietly.
                return Outcome.LOCK_HELD
            document = (
                await session.execute(
                    select(Document)
                    .where(
                        Document.id == document_id,
                        Document.tenant_id == tenant_id,
                        Document.deleted_at.is_(None),
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if document is None:
                job_update = (
                    update(IngestionJob)
                    .where(
                        IngestionJob.document_id == document_id,
                        IngestionJob.tenant_id == tenant_id,
                    )
                    .values(status=JobStatus.DONE, finished_at=datetime.now(UTC))
                )
                await session.execute(job_update)
                return Outcome.DOC_GONE_DONE

            # Reprocess safety: delete-then-insert keeps (document_id, chunk_index)
            # unique so a retried ingest cannot duplicate chunks.
            await session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_id == document_id,
                    DocumentChunk.tenant_id == tenant_id,
                )
            )
            rows = [
                {
                    "document_id": document_id,
                    "tenant_id": tenant_id,
                    "chunk_index": index,
                    "content": chunk.content,
                    "meta": _chunk_metadata(chunk),
                    "embedding": vector,
                    "embedding_model": model,
                    "token_estimate": chunk.token_estimate,
                }
                for index, (chunk, vector) in enumerate(prepared)
            ]
            for start in range(0, len(rows), _INSERT_BATCH_ROWS):
                await session.execute(
                    insert(DocumentChunk), rows[start : start + _INSERT_BATCH_ROWS]
                )

            document.status = DocumentStatus.READY
            document.chunk_count = len(prepared)
            document.embedding_model = model
            document.error_message = None
            await session.execute(
                update(IngestionJob)
                .where(
                    IngestionJob.document_id == document_id,
                    IngestionJob.tenant_id == tenant_id,
                )
                .values(status=JobStatus.DONE, finished_at=datetime.now(UTC))
            )
        return Outcome.DONE


def _chunk_metadata(chunk: Chunk) -> dict[str, object]:
    return {
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "heading_path": list(chunk.heading_path),
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
    }


# -------------------------------------------------------------------- failure


async def _handle_failure(
    ctx: WorkerContext, *, job_id: uuid.UUID, tenant_id: uuid.UUID, failure: Exception
) -> Outcome:
    code = getattr(failure, "code", "parse_failed")
    retryable = getattr(failure, "retryable", True)
    detail = getattr(failure, "detail", str(failure)) or str(failure)
    last_error = f"{code}: {detail}"[:_LAST_ERROR_MAX_CHARS]
    log.warning("ingest.failed", code=code, retryable=retryable, job_id=str(job_id))
    async with ctx.sessionmaker() as session:
        async with session.begin():
            await set_tenant(session, tenant_id)
            job = (
                await session.execute(
                    select(IngestionJob)
                    .where(
                        IngestionJob.id == job_id,
                        IngestionJob.tenant_id == tenant_id,
                        IngestionJob.status == JobStatus.RUNNING,
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if job is None:
                return Outcome.SKIPPED
            document = (
                await session.execute(
                    select(Document)
                    .where(
                        Document.id == job.document_id,
                        Document.tenant_id == tenant_id,
                        Document.deleted_at.is_(None),
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if document is None:
                job.status = JobStatus.DONE
                job.finished_at = datetime.now(UTC)
                return Outcome.DOC_GONE_DONE
            document.status = DocumentStatus.FAILED
            document.error_message = last_error
            if retryable and job.attempts < job.max_attempts:
                # Document stays failed until processing actually restarts.
                job.status = JobStatus.QUEUED
                return Outcome.RETRIED
            job.status = JobStatus.FAILED
            job.last_error = last_error
            job.finished_at = datetime.now(UTC)
            return Outcome.TERMINAL


# -------------------------------------------------------------------- sweeper


async def sweeper_due_jobs(
    ctx: WorkerContext,
    *,
    queued_stale_seconds: int = 120,
    running_stale_seconds: int = 900,
) -> list[SweeperJob]:
    """Find lost/stuck jobs via the security-definer function (RLS-safe)."""
    async with ctx.sessionmaker() as session:
        result = await session.execute(
            text(
                "SELECT job_id, document_id, tenant_id, status, attempts, max_attempts "
                "FROM ingest_sweeper_due_jobs("
                "CAST(:queued AS integer) * interval '1 second', "
                "CAST(:running AS integer) * interval '1 second')"
            ),
            {"queued": queued_stale_seconds, "running": running_stale_seconds},
        )
        return [
            SweeperJob(
                job_id=row.job_id,
                document_id=row.document_id,
                tenant_id=row.tenant_id,
                status=row.status,
                attempts=row.attempts,
                max_attempts=row.max_attempts,
            )
            for row in result
        ]


async def reset_stale_running_job(
    ctx: WorkerContext, *, job_id: uuid.UUID, tenant_id: uuid.UUID
) -> bool:
    """Move a worker-crashed `running` job back to `queued` so it can be claimed."""
    async with ctx.sessionmaker() as session:
        async with session.begin():
            await set_tenant(session, tenant_id)
            result = await session.execute(
                update(IngestionJob)
                .where(
                    IngestionJob.id == job_id,
                    IngestionJob.tenant_id == tenant_id,
                    IngestionJob.status == JobStatus.RUNNING,
                )
                .values(status=JobStatus.QUEUED)
            )
            return bool(cast("CursorResult[Any]", result).rowcount)


async def reset_job_for_reprocess(
    session: AsyncSession, *, document_id: uuid.UUID, tenant_id: uuid.UUID
) -> None:
    """Reset the document's job to queued (admin reprocess endpoint)."""
    result = await session.execute(
        select(IngestionJob).where(
            IngestionJob.document_id == document_id,
            IngestionJob.tenant_id == tenant_id,
        )
    )
    job = result.scalar_one_or_none()
    if job is None:
        session.add(IngestionJob(document_id=document_id, tenant_id=tenant_id))
        return
    job.status = JobStatus.QUEUED
    job.attempts = 0
    job.last_error = None
    job.started_at = None
    job.finished_at = None
    job.queued_at = datetime.now(UTC)
