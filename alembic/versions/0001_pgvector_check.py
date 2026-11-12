"""Verify pgvector is installed and new enough for filtered HNSW scans.

Phase 0 creates no domain tables. The extension itself is created by the
Postgres init script as the superuser; migrations only verify it.

Revision ID: 0001_pgvector_check
Revises:
Create Date: 2026-11-12

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001_pgvector_check"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

MIN_VERSION = (0, 8)


def _parse_version(raw: str) -> tuple[int, int]:
    parts = raw.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"Unrecognised pgvector version: {raw!r}") from exc


def upgrade() -> None:
    bind = op.get_bind()
    version = bind.execute(
        sa.text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar_one_or_none()

    if version is None:
        raise RuntimeError(
            "pgvector extension is not installed. Start Postgres with the "
            "docker/postgres/init script (runs as superuser) before migrating."
        )

    if _parse_version(version) < MIN_VERSION:
        raise RuntimeError(
            f"pgvector >= {'.'.join(map(str, MIN_VERSION))} required, found {version}. "
            "Upgrade the Postgres image before running migrations."
        )


def downgrade() -> None:
    pass
