"""Maintenance jobs: soft-delete purge and orphan-object scan (plan §19 Phase 6).

Every function here connects with `DATABASE_MIGRATION_URL` (`rag_owner`,
BYPASSRLS): purging must reach soft-deleted rows of every tenant, which the
runtime role `rag_app` cannot under RLS. This is operator tooling, never a
request path.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import Settings
from app.errors import StorageUnavailableError
from app.services.storage import StorageService

log = structlog.get_logger(__name__)

_OBJECT_KEY_PREFIX = "tenant/"


@dataclass(frozen=True, slots=True)
class PurgeReport:
    purged_documents: int
    object_deletes_failed: int


async def purge_expired_documents(
    settings: Settings, *, storage: StorageService, now: datetime | None = None
) -> PurgeReport:
    """Hard-delete soft-deleted documents past the retention window.

    Rows go first (cascading to chunks and jobs); the stored object is deleted
    afterwards, best effort. A failed object delete leaves an orphan that the
    orphan scan will find on a later run.
    """
    cutoff = (now or datetime.now(UTC)) - timedelta(days=settings.soft_delete_retention_days)
    engine = create_async_engine(settings.database_migration_url)
    try:
        async with engine.begin() as connection:
            rows = (
                await connection.execute(
                    text(
                        "DELETE FROM documents "
                        "WHERE deleted_at IS NOT NULL AND deleted_at < :cutoff "
                        "RETURNING storage_key"
                    ),
                    {"cutoff": cutoff},
                )
            ).all()
    finally:
        await engine.dispose()

    failed = 0
    for (storage_key,) in rows:
        try:
            await storage.delete_object(key=storage_key)
        except StorageUnavailableError:
            failed += 1
            log.warning("maintenance.purge_object_delete_failed", storage_key=storage_key)
    if rows:
        log.info("maintenance.purged", count=len(rows), object_deletes_failed=failed)
    return PurgeReport(purged_documents=len(rows), object_deletes_failed=failed)


async def find_orphan_objects(settings: Settings, *, storage: StorageService) -> list[str]:
    """Object keys in the bucket that no documents row references.

    These are leftovers from purged documents, failed uploads, or objects whose
    delete failed. Soft-deleted documents still own their objects by design
    (retention), so their keys are not orphans.
    """
    object_keys = await storage.list_object_keys(prefix=_OBJECT_KEY_PREFIX)
    if not object_keys:
        return []

    engine = create_async_engine(settings.database_migration_url)
    try:
        async with engine.connect() as connection:
            known = {
                key
                for (key,) in await connection.execute(text("SELECT storage_key FROM documents"))
            }
    finally:
        await engine.dispose()

    return [key for key in object_keys if key not in known]


async def delete_orphan_objects(storage: StorageService, keys: list[str]) -> int:
    """Best-effort delete for orphan keys; returns the number deleted."""
    deleted = 0
    for key in keys:
        try:
            await storage.delete_object(key=key)
            deleted += 1
        except StorageUnavailableError:
            log.warning("maintenance.orphan_delete_failed", storage_key=key)
    return deleted
