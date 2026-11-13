"""Purge soft-deleted documents past the retention window; scan for orphan objects.

Operator tooling (plan §19 Phase 6). Schedule it after deployment (cron /
scheduled task). Uses `DATABASE_MIGRATION_URL` (rag_owner, BYPASSRLS) so it
reaches every tenant's soft-deleted rows.

Usage:
  uv run python scripts/purge.py                  # purge + orphan report, no orphan deletes
  uv run python scripts/purge.py --delete-orphans # also delete the orphan objects
"""

from __future__ import annotations

import argparse
import asyncio

from app.config import get_settings
from app.services.maintenance import (
    delete_orphan_objects,
    find_orphan_objects,
    purge_expired_documents,
)
from app.services.storage import StorageService


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--delete-orphans",
        action="store_true",
        help="Delete orphan objects (default: report them only).",
    )
    args = parser.parse_args()

    settings = get_settings()
    storage = StorageService(settings)

    report = await purge_expired_documents(settings, storage=storage)
    print(f"purged documents: {report.purged_documents}")
    print(f"object deletes failed: {report.object_deletes_failed}")

    orphans = await find_orphan_objects(settings, storage=storage)
    if not orphans:
        print("orphan objects: none")
        return
    print(f"orphan objects: {len(orphans)}")
    for key in orphans:
        print(f"  {key}")
    if args.delete_orphans:
        deleted = await delete_orphan_objects(storage, orphans)
        print(f"orphan objects deleted: {deleted}")
    else:
        print("(dry run; pass --delete-orphans to delete them)")


if __name__ == "__main__":
    asyncio.run(main())
