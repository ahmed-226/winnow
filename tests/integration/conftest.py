from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from pathlib import Path

import fakeredis.aioredis
import pytest
import pytest_asyncio
from alembic.config import Config as AlembicConfig
from httpx import ASGITransport, AsyncClient
from moto import mock_aws
from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import command
from app.config import Settings
from app.main import create_app
from app.services.embeddings import build_embedding_provider

ROOT = Path(__file__).resolve().parents[2]
TEST_JWT_SECRET = "integration-test-secret-that-is-long-enough"


@dataclass(frozen=True, slots=True)
class DatabaseHandle:
    name: str
    admin_db_url: str
    owner_url: str
    app_url: str


def _setup_statements(database: str) -> list[str]:
    return [
        "CREATE EXTENSION IF NOT EXISTS vector;",
        # Mirrors docker/postgres/init/01-roles-and-extension.sql: without
        # this, the tenant-isolation RLS barrier blocks the GIN index for the
        # sparse retriever (see alembic/versions/0005_fts_leakproof.py).
        "ALTER FUNCTION pg_catalog.ts_match_vq(tsvector, tsquery) LEAKPROOF;",
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_owner') THEN
                CREATE ROLE rag_owner LOGIN PASSWORD 'rag_owner' BYPASSRLS;
            END IF;
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'rag_app') THEN
                CREATE ROLE rag_app LOGIN PASSWORD 'rag_app' NOBYPASSRLS;
            END IF;
        END
        $$;
        """,
        "ALTER ROLE rag_owner WITH LOGIN PASSWORD 'rag_owner' BYPASSRLS;",
        "ALTER ROLE rag_app WITH LOGIN PASSWORD 'rag_app' NOBYPASSRLS;",
        f'GRANT CONNECT ON DATABASE "{database}" TO rag_owner, rag_app;',
        "GRANT USAGE, CREATE ON SCHEMA public TO rag_owner;",
        "GRANT USAGE ON SCHEMA public TO rag_app;",
        "ALTER DEFAULT PRIVILEGES FOR ROLE rag_owner IN SCHEMA public "
        "GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO rag_app;",
        "ALTER DEFAULT PRIVILEGES FOR ROLE rag_owner IN SCHEMA public "
        "GRANT USAGE, SELECT ON SEQUENCES TO rag_app;",
        "ALTER DEFAULT PRIVILEGES FOR ROLE rag_owner IN SCHEMA public "
        "GRANT EXECUTE ON FUNCTIONS TO rag_app;",
    ]


async def _create_database(admin_url: URL, database: str) -> None:
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(text(f'CREATE DATABASE "{database}"'))
    finally:
        await engine.dispose()

    engine = create_async_engine(admin_url.set(database=database), isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            for statement in _setup_statements(database):
                await connection.execute(text(statement))
    finally:
        await engine.dispose()


async def _drop_database(admin_url: URL, database: str) -> None:
    if admin_url.drivername != "postgresql+asyncpg":
        admin_url = admin_url.set(drivername="postgresql+asyncpg")
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :name AND pid <> pg_backend_pid()"
                ),
                {"name": database},
            )
            await connection.execute(text(f'DROP DATABASE IF EXISTS "{database}"'))
    finally:
        await engine.dispose()


def _provision(admin_dsn: str) -> DatabaseHandle:
    admin_url = make_url(admin_dsn)
    if admin_url.drivername != "postgresql+asyncpg":
        admin_url = admin_url.set(drivername="postgresql+asyncpg")

    database = f"rag_test_{uuid.uuid4().hex[:12]}"
    asyncio.run(_create_database(admin_url, database))

    owner_url = admin_url.set(username="rag_owner", password="rag_owner", database=database)
    app_url = admin_url.set(username="rag_app", password="rag_app", database=database)

    config = AlembicConfig(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", owner_url.render_as_string(hide_password=False))
    command.upgrade(config, "head")

    return DatabaseHandle(
        name=database,
        admin_db_url=admin_url.set(database=database).render_as_string(hide_password=False),
        owner_url=owner_url.render_as_string(hide_password=False),
        app_url=app_url.render_as_string(hide_password=False),
    )


@pytest.fixture(scope="session")
def test_database() -> Iterator[DatabaseHandle]:
    admin_dsn = os.environ.get("TEST_DATABASE_ADMIN_URL")
    container = None

    if not admin_dsn:
        if shutil.which("docker") is None:
            pytest.skip("Integration tests need TEST_DATABASE_ADMIN_URL or a local Docker daemon.")
        try:
            from testcontainers.community.postgres import PostgresContainer

            container = PostgresContainer("pgvector/pgvector:pg17", driver="asyncpg")
            container.start()
            admin_dsn = container.get_connection_url()
        except Exception as exc:  # pragma: no cover - environment-dependent
            pytest.skip(f"Could not start a Postgres testcontainer: {exc}")

    database = _provision(admin_dsn)
    try:
        yield database
    finally:
        asyncio.run(_drop_database(make_url(admin_dsn), database.name))
        if container is not None:
            container.stop()


@pytest_asyncio.fixture(autouse=True)
async def reset_database(test_database: DatabaseHandle) -> AsyncIterator[None]:
    engine = create_async_engine(test_database.admin_db_url, isolation_level="AUTOCOMMIT")
    try:
        async with engine.connect() as connection:
            await connection.execute(
                text("TRUNCATE TABLE refresh_tokens, users, tenants RESTART IDENTITY CASCADE")
            )
    finally:
        await engine.dispose()
    yield


def make_settings(test_database: DatabaseHandle, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "app_env": "test",
        "database_url": test_database.app_url,
        "database_migration_url": test_database.owner_url,
        "jwt_secret": TEST_JWT_SECRET,
        "embedding_provider": "fake",
        "allow_tenant_signup": True,
        "rl_login_per_minute": 1_000,
        "rl_register_per_hour": 1_000,
        # moto accepts its canonical "test" credentials in mocked mode.
        "s3_access_key_id": "test",
        "s3_secret_access_key": "test",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


@pytest_asyncio.fixture
async def app(test_database: DatabaseHandle) -> AsyncIterator[object]:
    with mock_aws():
        application = create_app(make_settings(test_database))
        async with application.router.lifespan_context(application):
            await application.state.redis.aclose()
            fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
            application.state.redis = fake_redis
            # Rebuild the query-embedding provider so its cache points at the
            # fake Redis the tests actually use.
            application.state.embeddings = build_embedding_provider(
                application.state.settings, cache=fake_redis
            )
            try:
                yield application
            finally:
                await fake_redis.aclose()


@pytest_asyncio.fixture
async def client(app: object) -> AsyncIterator[AsyncClient]:
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
