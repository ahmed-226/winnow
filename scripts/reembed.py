"""Re-embed every chunk with the configured provider/model (plan §11.4).

The only supported migration path when `EMBEDDING_MODEL` or the embedding
dimension changes: a different model is not drop-in compatible, so every chunk
must be re-embedded. Each document is replaced atomically (chunk vectors +
`embedding_model` in one transaction under the document's advisory lock), so
an interrupted run never leaves a document half-embedded; re-running it
resumes because completed documents are skipped.

The script connects as `rag_owner` (BYPASSRLS) because it must reach every
tenant; every query still carries an explicit `tenant_id` predicate, and the
provider is built from the normal settings (Gemini key or fake).

Usage:
  uv run python scripts/reembed.py [--tenant acme] [--dry-run] [--force]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import structlog
from sqlalchemy import delete, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import Settings
from app.models import Document, DocumentChunk, DocumentStatus
from app.services.embeddings import (
    EmbeddingError,
    EmbeddingProvider,
    build_embedding_provider,
)

log = structlog.get_logger(__name__)

_INSERT_BATCH_ROWS = 500


@dataclass(frozen=True, slots=True)
class Target:
    document_id: uuid.UUID
    tenant_id: uuid.UUID
    title: str
    chunk_count: int
    embedding_model: str


async def select_targets(
    session: AsyncSession,
    *,
    model: str,
    tenant_slug: str | None,
    force: bool,
) -> list[Target]:
    """Documents whose stored vectors do not match the configured model."""
    stmt = (
        select(
            Document.id,
            Document.tenant_id,
            Document.title,
            Document.chunk_count,
            Document.embedding_model,
        )
        .where(
            Document.deleted_at.is_(None),
            Document.status == DocumentStatus.READY,
        )
        .order_by(Document.created_at, Document.id)
    )
    if not force:
        # NULL model (old rows) counts as a mismatch too.
        stmt = stmt.where(
            (Document.embedding_model != model) | (Document.embedding_model.is_(None))
        )
    if tenant_slug is not None:
        stmt = stmt.join(
            text("tenants t"),
            text("t.id = documents.tenant_id AND t.slug = :tenant_slug"),
        ).params(tenant_slug=tenant_slug)
    rows = (await session.execute(stmt)).all()
    return [
        Target(
            document_id=row[0],
            tenant_id=row[1],
            title=row[2],
            chunk_count=row[3],
            embedding_model=row[4],
        )
        for row in rows
    ]


@dataclass(frozen=True, slots=True)
class ChunkRow:
    chunk_index: int
    content: str
    meta: dict[str, object]
    token_estimate: int | None


async def load_chunks(
    session: AsyncSession, *, document_id: uuid.UUID, tenant_id: uuid.UUID
) -> list[ChunkRow]:
    stmt = (
        select(
            DocumentChunk.chunk_index,
            DocumentChunk.content,
            DocumentChunk.meta,
            DocumentChunk.token_estimate,
        )
        .where(
            DocumentChunk.document_id == document_id,
            DocumentChunk.tenant_id == tenant_id,
        )
        .order_by(DocumentChunk.chunk_index)
    )
    return [
        ChunkRow(chunk_index=row[0], content=row[1], meta=row[2], token_estimate=row[3])
        for row in (await session.execute(stmt)).all()
    ]


async def reembed_document(
    sessionmaker: async_sessionmaker[AsyncSession],
    provider: EmbeddingProvider,
    settings: Settings,
    target: Target,
) -> bool:
    """Re-embed one document's chunks atomically; False when the lock is held."""
    async with sessionmaker() as session:
        async with session.begin():
            chunks = await load_chunks(
                session, document_id=target.document_id, tenant_id=target.tenant_id
            )
    if len(chunks) != target.chunk_count:
        log.warning(
            "reembed.chunk_count_mismatch",
            document_id=str(target.document_id),
            expected=target.chunk_count,
            actual=len(chunks),
        )

    batch_size = settings.embedding_batch_size
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors.extend(await provider.embed_documents([row.content for row in batch]))

    async with sessionmaker() as session:
        async with session.begin():
            lock = (
                await session.execute(
                    text(
                        "SELECT pg_try_advisory_xact_lock("
                        "hashtextextended(CAST(:document_id AS uuid)::text, 0))"
                    ),
                    {"document_id": str(target.document_id)},
                )
            ).scalar_one()
            if not lock:
                return False
            # The chunk set must not have drifted (e.g. concurrent reprocess)
            # between the read and this write; otherwise refuse, never write
            # vectors for a stale chunk list.
            current = await load_chunks(
                session, document_id=target.document_id, tenant_id=target.tenant_id
            )
            if [(row.chunk_index, row.content) for row in current] != [
                (row.chunk_index, row.content) for row in chunks
            ]:
                raise EmbeddingError(
                    f"Chunk set for {target.document_id} changed during "
                    "re-embedding; refusing to write a partial document.",
                    retryable=False,
                )
            # Same pattern as the ingestion pipeline: delete-then-insert keeps
            # (document_id, chunk_index) unique and never writes partial vectors.
            await session.execute(
                delete(DocumentChunk).where(
                    DocumentChunk.document_id == target.document_id,
                    DocumentChunk.tenant_id == target.tenant_id,
                )
            )
            rows = [
                {
                    "document_id": target.document_id,
                    "tenant_id": target.tenant_id,
                    "chunk_index": row.chunk_index,
                    "content": row.content,
                    "meta": row.meta,
                    "embedding": vector,
                    "embedding_model": provider.model,
                    "token_estimate": row.token_estimate,
                }
                for row, vector in zip(chunks, vectors, strict=True)
            ]
            for start in range(0, len(rows), _INSERT_BATCH_ROWS):
                await session.execute(
                    insert(DocumentChunk), rows[start : start + _INSERT_BATCH_ROWS]
                )
            await session.execute(
                update(Document)
                .where(
                    Document.id == target.document_id,
                    Document.tenant_id == target.tenant_id,
                )
                .values(embedding_model=provider.model, error_message=None)
            )
    return True


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenant", help="only re-embed this tenant slug")
    parser.add_argument("--dry-run", action="store_true", help="list targets, do nothing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-embed even documents already at the configured model",
    )
    args = parser.parse_args()

    settings = Settings()  # type: ignore[call-arg]
    provider = build_embedding_provider(settings)
    # Owner role (BYPASSRLS): the script must reach every tenant; the runtime
    # rag_app credentials are never used for maintenance.
    engine = create_async_engine(settings.database_migration_url, pool_pre_ping=True)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    total_chunks = 0
    failed = False
    try:
        async with sessionmaker() as session:
            targets = await select_targets(
                session,
                model=provider.model,
                tenant_slug=args.tenant,
                force=args.force,
            )
        log.info(
            "reembed.targets",
            count=len(targets),
            model=provider.model,
            dry_run=args.dry_run,
        )
        for target in targets:
            log.info(
                "reembed.document",
                document_id=str(target.document_id),
                tenant_id=str(target.tenant_id),
                title_length=len(target.title),
                chunks=target.chunk_count,
                previous_model=target.embedding_model,
            )
            if args.dry_run:
                total_chunks += target.chunk_count
                continue
            written = await reembed_document(sessionmaker, provider, settings, target)
            if not written:
                log.warning("reembed.lock_held", document_id=str(target.document_id))
                continue
            total_chunks += target.chunk_count
    except EmbeddingError as exc:
        log.error("reembed.embedding_failed", code=exc.code, detail=str(exc))
        failed = True
    finally:
        await engine.dispose()
    log.info("reembed.summary", chunks=total_chunks, dry_run=args.dry_run)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
