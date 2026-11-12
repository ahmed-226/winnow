"""Documents table with status enum, idempotency index, and RLS.

Adds `document_status` and the `documents` table described in the
implementation plan §7.2-§7.4, including the partial unique hash index for
upload idempotency, forced RLS, and the updated_at trigger.

Revision ID: 0003_documents
Revises: 0002_auth_tenancy
Create Date: 2026-11-12

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0003_documents"
down_revision: str | None = "0002_auth_tenancy"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

document_status = postgresql.ENUM(
    "pending",
    "processing",
    "ready",
    "failed",
    name="document_status",
)


def _execute(*statements: str) -> None:
    for statement in statements:
        op.execute(statement)


def upgrade() -> None:
    # `document_status` is created automatically by the column definition below.
    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("file_name", sa.Text(), nullable=False),
        sa.Column("mime_type", sa.String(length=127), nullable=False),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("file_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("access_roles", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column(
            "status",
            document_status,
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "chunk_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("embedding_model", sa.String(length=64), nullable=True),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="documents_pkey"),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="documents_tenant_id_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name="documents_created_by_fkey",
            ondelete="SET NULL",
        ),
        sa.UniqueConstraint("id", "tenant_id", name="documents_id_tenant_id_key"),
        sa.CheckConstraint(
            "source_type IN ('pdf', 'markdown', 'html')",
            name="documents_source_type_check",
        ),
        sa.CheckConstraint("file_hash ~ '^[0-9a-f]{64}$'", name="documents_file_hash_format"),
        sa.CheckConstraint("size_bytes > 0", name="documents_size_bytes_positive"),
        sa.CheckConstraint("chunk_count >= 0", name="documents_chunk_count_nonnegative"),
        sa.CheckConstraint(
            "array_length(access_roles, 1) IS NULL OR array_length(access_roles, 1) <= 20",
            name="documents_roles_count",
        ),
        sa.CheckConstraint(
            "NOT ('tenant_admin' = ANY(access_roles))",
            name="documents_no_system_roles",
        ),
    )

    op.create_index(
        "documents_tenant_hash_uidx",
        "documents",
        ["tenant_id", "file_hash"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "documents_tenant_status_idx",
        "documents",
        ["tenant_id", "status"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )
    op.create_index(
        "documents_access_roles_gin",
        "documents",
        ["access_roles"],
        postgresql_using="gin",
    )

    _execute(
        "ALTER TABLE documents ENABLE ROW LEVEL SECURITY;",
        "ALTER TABLE documents FORCE ROW LEVEL SECURITY;",
        """
        CREATE POLICY tenant_isolation ON documents
            FOR ALL
            USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
            WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
        """,
        """
        CREATE TRIGGER documents_set_updated_at
            BEFORE UPDATE ON documents
            FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """,
    )


def downgrade() -> None:
    _execute(
        "DROP TRIGGER IF EXISTS documents_set_updated_at ON documents;",
        "DROP POLICY IF EXISTS tenant_isolation ON documents;",
        "DROP TABLE IF EXISTS documents;",
    )
    document_status.drop(op.get_bind(), checkfirst=True)
