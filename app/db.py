from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import Settings


def create_engine(
    settings: Settings,
    *,
    application_name: str = "rag-api",
    statement_timeout_ms: int | None = None,
) -> AsyncEngine:
    """Create the API/worker engine with a transaction statement timeout.

    The statement timeout is applied per connection through asyncpg so a runaway
    query cannot hold a pooled connection indefinitely. Worker sessions pass a
    higher value (plan §14) so a large chunk insert is not killed mid-transaction.
    """
    return create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
        connect_args={
            "server_settings": {
                "application_name": application_name,
                "statement_timeout": str(statement_timeout_ms or settings.db_statement_timeout_ms),
            }
        },
    )


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Sessions never expire loaded objects, so responses can be built after commit."""
    return async_sessionmaker(engine, expire_on_commit=False)
