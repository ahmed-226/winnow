-- Runs as the Postgres superuser against the `rag` database on first init only.
-- Creates the pgvector extension and the two application roles described in
-- dev/docs/implementation-plan.md §8.4.

CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rag_owner') THEN
        CREATE ROLE rag_owner LOGIN PASSWORD 'rag_owner' BYPASSRLS;
    END IF;

    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'rag_app') THEN
        CREATE ROLE rag_app LOGIN PASSWORD 'rag_app' NOBYPASSRLS;
    END IF;
END
$$;

GRANT CONNECT ON DATABASE rag TO rag_owner, rag_app;

-- PostgreSQL 15+ no longer grants CREATE on public to everyone.
GRANT USAGE, CREATE ON SCHEMA public TO rag_owner;
GRANT USAGE ON SCHEMA public TO rag_app;

-- Tables created by rag_owner (Alembic) must be usable by rag_app without
-- per-migration grants. RLS still constrains every row.
ALTER DEFAULT PRIVILEGES FOR ROLE rag_owner IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rag_app;
ALTER DEFAULT PRIVILEGES FOR ROLE rag_owner IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO rag_app;
ALTER DEFAULT PRIVILEGES FOR ROLE rag_owner IN SCHEMA public
    GRANT EXECUTE ON FUNCTIONS TO rag_app;

ALTER DATABASE rag SET timezone TO 'UTC';
