"""Mark the FTS match operator function leakproof so the GIN index is usable
under row-level security.

PostgreSQL treats RLS policy predicates as security-barrier quals: quals that
are not leakproof may not be evaluated *below* the barrier as index conditions.
Core marks `tsvector @@ tsquery` (`ts_match_vq`) non-leakproof, so once the
tenant-isolation policy exists on `document_chunks` the planner refuses the
`chunks_fts_gin` bitmap path and falls back to scanning the tenant's chunks
through `chunks_tenant_idx`. `ts_match_vq` is a pure function of its two
arguments with no side channels, so marking it leakproof is safe and restores
GIN index scans for the sparse retriever under RLS. (pgvector marks its own
distance operators leakproof, which is why the HNSW index is unaffected.)

Setting LEAKPROOF requires a superuser. Every supported setup applies this
before Alembic runs:

- Docker: `docker/postgres/init/01-roles-and-extension.sql` (first init).
- Tests: `tests/integration/conftest.py` `_setup_statements`.

When the migration runs as a superuser it applies the flag itself; when the
flag is already set it is a no-op; otherwise it fails with instructions.

Revision ID: 0005_fts_leakproof
Revises: 0004_ingestion
Create Date: 2026-11-13

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0005_fts_leakproof"
down_revision: str | None = "0004_ingestion"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SET_LEAKPROOF = "ALTER FUNCTION pg_catalog.ts_match_vq(tsvector, tsquery) LEAKPROOF;"

_GUARDED = """
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_proc
        WHERE oid = 'pg_catalog.ts_match_vq(tsvector, tsquery)'::regprocedure
          AND proleakproof
    ) THEN
        RETURN;  -- already applied by the init script or a previous run
    END IF;
    IF (SELECT rolsuper FROM pg_roles WHERE rolname = session_user) THEN
        ALTER FUNCTION pg_catalog.ts_match_vq(tsvector, tsquery) LEAKPROOF;
    ELSE
        RAISE EXCEPTION USING
            MESSAGE = 'ts_match_vq must be marked LEAKPROOF for the sparse '
                      'retriever to use the GIN index under RLS',
            HINT = 'Run the migration as a superuser, or apply once: '
                   || 'ALTER FUNCTION pg_catalog.ts_match_vq(tsvector, tsquery) '
                   || 'LEAKPROOF;';
    END IF;
END
$$;
"""


def upgrade() -> None:
    op.execute(_GUARDED)


def downgrade() -> None:
    # Reverting only matters for environments that applied it here; setups
    # that got the flag from the init script keep it.
    op.execute(_SET_LEAKPROOF.replace("LEAKPROOF;", "NOT LEAKPROOF;"))
