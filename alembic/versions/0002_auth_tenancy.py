"""Auth, tenancy, and row-level security.

Creates tenants, users, and refresh_tokens plus the security-definer bootstrap
functions and forced RLS policies described in the implementation plan §7-§8.

Revision ID: 0002_auth_tenancy
Revises: 0001_pgvector_check
Create Date: 2026-11-12

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002_auth_tenancy"
down_revision: str | None = "0001_pgvector_check"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TENANT_TABLES = ("tenants", "users", "refresh_tokens")


def _execute(*statements: str) -> None:
    for statement in statements:
        op.execute(statement)


def upgrade() -> None:
    op.create_table(
        "tenants",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=63), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="tenants_pkey"),
        sa.UniqueConstraint("slug", name="tenants_slug_key"),
        sa.CheckConstraint(
            "slug ~ '^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$'",
            name="tenants_slug_format",
        ),
    )

    op.create_table(
        "users",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column(
            "roles",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'::text[]"),
            nullable=False,
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
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
        sa.PrimaryKeyConstraint("id", name="users_pkey"),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="users_tenant_id_fkey",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("id", "tenant_id", name="users_id_tenant_id_key"),
        sa.CheckConstraint("email = lower(email)", name="users_email_lowercase"),
        sa.CheckConstraint(
            "array_length(roles, 1) IS NULL OR array_length(roles, 1) <= 20",
            name="users_roles_count",
        ),
    )
    op.create_index(
        "users_tenant_email_uidx",
        "users",
        ["tenant_id", sa.text("lower(email)")],
        unique=True,
    )

    op.create_table(
        "refresh_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("family_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("token_hash", sa.CHAR(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name="refresh_tokens_pkey"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="refresh_tokens_user_id_fkey",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="refresh_tokens_tenant_id_fkey",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint("token_hash", name="refresh_tokens_token_hash_key"),
    )
    op.create_index(
        "refresh_tokens_expires_idx",
        "refresh_tokens",
        ["expires_at"],
        postgresql_where=sa.text("revoked_at IS NULL"),
    )

    _execute(
        """
        CREATE FUNCTION set_updated_at() RETURNS trigger
        LANGUAGE plpgsql
        SET search_path = pg_catalog, public
        AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$;
        """,
        """
        CREATE TRIGGER users_set_updated_at
            BEFORE UPDATE ON users
            FOR EACH ROW EXECUTE FUNCTION set_updated_at();
        """,
    )

    for table in TENANT_TABLES:
        _execute(
            f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;",
            f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY;",
        )

    _execute(
        """
        CREATE POLICY tenant_isolation ON users
            FOR ALL
            USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
            WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
        """,
        """
        CREATE POLICY tenant_isolation ON refresh_tokens
            FOR ALL
            USING (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
            WITH CHECK (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
        """,
        """
        CREATE POLICY tenant_self_select ON tenants
            FOR SELECT
            USING (id = nullif(current_setting('app.tenant_id', true), '')::uuid);
        """,
    )

    _execute(
        """
        CREATE FUNCTION auth_lookup_tenant(p_slug text)
        RETURNS TABLE (id uuid, name text, slug text)
        LANGUAGE sql
        SECURITY DEFINER
        SET search_path = pg_catalog, public
        AS $$
            SELECT t.id, t.name::text, t.slug
            FROM public.tenants t
            WHERE t.slug = p_slug;
        $$;
        """,
        """
        CREATE FUNCTION auth_get_user_for_login(p_slug text, p_email text)
        RETURNS TABLE (
            user_id uuid, tenant_id uuid, password_hash text,
            roles text[], is_active boolean
        )
        LANGUAGE sql
        SECURITY DEFINER
        SET search_path = pg_catalog, public
        AS $$
            SELECT u.id, u.tenant_id, u.password_hash, u.roles, u.is_active
            FROM public.users u
            JOIN public.tenants t ON t.id = u.tenant_id
            WHERE t.slug = p_slug AND u.email = lower(p_email);
        $$;
        """,
        """
        CREATE FUNCTION auth_create_tenant_with_admin(
            p_tenant_name text, p_slug text, p_email text, p_password_hash text
        )
        RETURNS TABLE (tenant_id uuid, user_id uuid)
        LANGUAGE plpgsql
        SECURITY DEFINER
        SET search_path = pg_catalog, public
        AS $$
        DECLARE
            v_tenant_id uuid;
            v_user_id   uuid;
        BEGIN
            INSERT INTO public.tenants (name, slug)
            VALUES (p_tenant_name, lower(p_slug))
            RETURNING id INTO v_tenant_id;

            INSERT INTO public.users (tenant_id, email, password_hash, roles)
            VALUES (v_tenant_id, lower(p_email), p_password_hash, ARRAY['tenant_admin'])
            RETURNING id INTO v_user_id;

            RETURN QUERY SELECT v_tenant_id, v_user_id;
        END;
        $$;
        """,
        """
        CREATE FUNCTION auth_get_refresh_token(p_token_hash char(64))
        RETURNS TABLE (
            token_id uuid, user_id uuid, tenant_id uuid,
            family_id uuid, expires_at timestamptz, revoked_at timestamptz
        )
        LANGUAGE sql
        SECURITY DEFINER
        SET search_path = pg_catalog, public
        AS $$
            SELECT r.id, r.user_id, r.tenant_id, r.family_id, r.expires_at, r.revoked_at
            FROM public.refresh_tokens r
            WHERE r.token_hash = p_token_hash;
        $$;
        """,
    )

    _execute(
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE tenants, users, refresh_tokens TO rag_app;",
        "ALTER DEFAULT PRIVILEGES FOR ROLE rag_owner IN SCHEMA public "
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rag_app;",
        "ALTER DEFAULT PRIVILEGES FOR ROLE rag_owner IN SCHEMA public "
        "GRANT USAGE, SELECT ON SEQUENCES TO rag_app;",
        "ALTER DEFAULT PRIVILEGES FOR ROLE rag_owner IN SCHEMA public "
        "GRANT EXECUTE ON FUNCTIONS TO rag_app;",
        "REVOKE ALL ON FUNCTION auth_lookup_tenant(text) FROM PUBLIC;",
        "REVOKE ALL ON FUNCTION auth_get_user_for_login(text, text) FROM PUBLIC;",
        "REVOKE ALL ON FUNCTION auth_create_tenant_with_admin(text, text, text, text) FROM PUBLIC;",
        "REVOKE ALL ON FUNCTION auth_get_refresh_token(char(64)) FROM PUBLIC;",
        "GRANT EXECUTE ON FUNCTION auth_lookup_tenant(text) TO rag_app;",
        "GRANT EXECUTE ON FUNCTION auth_get_user_for_login(text, text) TO rag_app;",
        "GRANT EXECUTE ON FUNCTION auth_create_tenant_with_admin(text, text, text, text) "
        "TO rag_app;",
        "GRANT EXECUTE ON FUNCTION auth_get_refresh_token(char(64)) TO rag_app;",
    )


def downgrade() -> None:
    _execute(
        "DROP FUNCTION IF EXISTS auth_get_refresh_token(char(64));",
        "DROP FUNCTION IF EXISTS auth_create_tenant_with_admin(text, text, text, text);",
        "DROP FUNCTION IF EXISTS auth_get_user_for_login(text, text);",
        "DROP FUNCTION IF EXISTS auth_lookup_tenant(text);",
        "DROP TABLE IF EXISTS refresh_tokens;",
        "DROP TABLE IF EXISTS users;",
        "DROP TABLE IF EXISTS tenants;",
        "DROP FUNCTION IF EXISTS set_updated_at();",
    )
