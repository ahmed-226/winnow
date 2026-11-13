"""Ingestion jobs and document chunks with vector and FTS indexes.

Adds the `job_status` enum, the `ingestion_jobs` and `document_chunks` tables
from plan §7.2-§7.4, the HNSW/GIN/sweeper indexes, forced RLS, the updated_at
trigger, and the security-definer sweeper function the worker uses to find
stuck jobs despite RLS.

Revision ID: 0004_ingestion
Revises: 0003_documents
Create Date: 2026-11-12

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0004_ingestion"
down_revision: str | None = "0003_documents"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

job_status = postgresql.ENUM(
    "queued",
    "running",
    "done",
    "failed",
    name="job_status",
)


def _execute(*statements: str) -> None:
    for statement in statements:
        op.execute(statement)


def upgrade() -> None:
    op.create_table(
        "ingestion_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("status", job_status, nullable=False, server_default=sa.text("'queued'")),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default=sa.text("3")),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "queued_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="ingestion_jobs_pkey"),
        sa.ForeignKeyConstraint(
            ["document_id", "tenant_id"],
            ["documents.id", "documents.tenant_id"],
            name="ingestion_jobs_document_id_tenant_id_fkey",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("document_id", name="ingestion_jobs_document_id_key"),
    )
    op.create_index(
        "ingestion_jobs_status_updated_idx",
        "ingestion_jobs",
        ["status", "updated_at"],
    )

    op.create_table(
        "document_chunks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("embedding_model", sa.String(length=64), nullable=False),
        sa.Column("token_estimate", sa.Integer(), nullable=True),
        sa.Column(
            "fts_doc",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="document_chunks_pkey"),
        sa.ForeignKeyConstraint(
            ["document_id", "tenant_id"],
            ["documents.id", "documents.tenant_id"],
            name="document_chunks_document_id_tenant_id_fkey",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("document_id", "chunk_index", name="document_chunks_index_key"),
        sa.CheckConstraint("chunk_index >= 0", name="document_chunks_index_nonnegative"),
        sa.CheckConstraint("length(content) > 0", name="document_chunks_content_nonempty"),
    )
    op.create_index("chunks_tenant_idx", "document_chunks", ["tenant_id"])
    op.create_index(
        "chunks_embedding_hnsw",
        "document_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
        postgresql_with={"m": 16, "ef_construction": 64},
    )
    op.create_index("chunks_fts_gin", "document_chunks", ["fts_doc"], postgresql_using="gin")

    _execute(
        "ALTER TABLE ingestion_jobs ENABLE ROW LEVEL SECURITY;",
        "ALTER TABLE ingestion_jobs FORCE ROW LEVEL SECURITY;",
        "ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;",
        "ALTER TABLE document_chunks FORCE ROW LEVEL SECURITY;",
        """
        CREATE POLICY tenant_isolation ON ingestion_jobs
            FOR ALL
            USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
            WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
        """,
        """
        CREATE POLICY tenant_isolation ON document_chunks
            FOR ALL
            USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
            WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
        """,
        """
        CREATE TRIGGER ingestion_jobs_set_updated_at
            BEFORE UPDATE ON ingestion_jobs
            FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """,
        # The sweeper must see stale jobs across the whole tenant despite RLS
        # (rag_app is NOBYPASSRLS). This narrowly scoped function returns only
        # the rows needed to re-enqueue; the worker then touches each job under
        # its own tenant context.
        """
        CREATE FUNCTION ingest_sweeper_due_jobs(
            p_queued_stale interval DEFAULT interval '2 minutes',
            p_running_stale interval DEFAULT interval '15 minutes'
        )
        RETURNS TABLE (
            job_id uuid, document_id uuid, tenant_id uuid,
            status job_status, attempts integer, max_attempts integer
        )
        LANGUAGE sql
        SECURITY DEFINER
        SET search_path = pg_catalog, public
        AS $$
            SELECT j.id, j.document_id, j.tenant_id,
                   j.status, j.attempts, j.max_attempts
            FROM public.ingestion_jobs j
            WHERE (j.status = 'queued' AND j.updated_at < now() - p_queued_stale)
               OR (j.status = 'running' AND j.updated_at < now() - p_running_stale);
        $$;
        """,
        "REVOKE ALL ON FUNCTION ingest_sweeper_due_jobs(interval, interval) FROM PUBLIC;",
        "GRANT EXECUTE ON FUNCTION ingest_sweeper_due_jobs(interval, interval) TO rag_app;",
    )


def downgrade() -> None:
    _execute(
        "DROP FUNCTION IF EXISTS ingest_sweeper_due_jobs(interval, interval);",
        "DROP TRIGGER IF EXISTS ingestion_jobs_set_updated_at ON ingestion_jobs;",
        "DROP POLICY IF EXISTS tenant_isolation ON document_chunks;",
        "DROP POLICY IF EXISTS tenant_isolation ON ingestion_jobs;",
        "DROP TABLE IF EXISTS document_chunks;",
        "DROP TABLE IF EXISTS ingestion_jobs;",
    )
    job_status.drop(op.get_bind(), checkfirst=True)
